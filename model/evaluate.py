"""
evaluate.py
Comprehensive evaluation of the trained U-Net model on the test set with XAI support.

Metrics computed:
  - Dice Coefficient (F1 for segmentation)
  - IoU / Jaccard Index
  - Pixel-wise Accuracy
  - Precision / Recall / F1 Score
  - Per-sample breakdown
  - Confusion Matrix (TP, TN, FP, FN)

XAI Features (when --xai-enabled):
  - Grad-CAM visualizations
  - Attention maps
  - Feature importance heatmaps
  - Overlay visualizations

Outputs:
  - Printed summary report to console
  - evaluation_results.json — full per-sample + aggregate results
  - evaluation_report.png  — visual comparison grid (if matplotlib available)
  - xai_visualizations/    — XAI outputs (if --xai-enabled)

Usage:
    # Basic evaluation
    python model/evaluate.py

    # With XAI enabled
    python model/evaluate.py --xai-enabled

    # Custom paths
    python model/evaluate.py --model model/saved_model/brain_tumor_model.h5 \
                             --test-dir data/test \
                             --output results/ \
                             --xai-enabled
"""

import sys
import os
import json
import argparse
import time
from typing import Tuple, List, Dict, Optional

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from saved_model import ModelSaver, CUSTOM_OBJECTS

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[!] Matplotlib not available - visual reports will be skipped")

# Optional OpenCV for advanced XAI
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# ============================================================
# CONFIG / DEFAULTS
# ============================================================
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "brain_tumor_model.h5")
DEFAULT_TEST_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "test")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
IMG_SIZE           = (256, 256)
THRESHOLD          = 0.5  # Prediction threshold for binary mask


# ============================================================
# DATA LOADING
# ============================================================
def load_test_data(test_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all images and masks from test directory.

    Expected layout:
        test_dir/
            images/   ← original MRI PNGs
            masks/    ← ground truth mask PNGs

    Returns:
        (images: np.ndarray, masks: np.ndarray, filenames: list)
        images shape: (N, 256, 256, 1)
        masks  shape: (N, 256, 256, 1)
    """
    img_dir  = os.path.join(test_dir, "images")
    mask_dir = os.path.join(test_dir, "masks")

    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        raise FileNotFoundError(
            f"Test directories not found.\n"
            f"  Expected: {img_dir}\n"
            f"            {mask_dir}\n"
            f"  Run split_data.py first."
        )

    filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

    if not filenames:
        raise ValueError(f"No PNG files found in {img_dir}")

    images, masks = [], []

    for fname in filenames:
        # Load image (grayscale, normalized)
        img = Image.open(os.path.join(img_dir, fname)).convert("L")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        images.append(img_arr)

        # Load mask (binary)
        mask = Image.open(os.path.join(mask_dir, fname)).convert("L")
        mask = mask.resize(IMG_SIZE, Image.NEAREST)
        mask_arr = (np.array(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        masks.append(mask_arr)

    images = np.array(images)[..., np.newaxis]  # (N, 256, 256, 1)
    masks  = np.array(masks)[..., np.newaxis]   # (N, 256, 256, 1)

    print(f"[✓] Loaded {len(filenames)} test samples")
    print(f"    Images shape: {images.shape}")
    print(f"    Masks  shape: {masks.shape}")

    return images, masks, filenames


# ============================================================
# METRIC FUNCTIONS
# ============================================================
def compute_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Dice coefficient for two binary arrays."""
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    union        = np.sum(y_true) + np.sum(y_pred)
    return float((2.0 * intersection + smooth) / (union + smooth))


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """IoU / Jaccard Index for two binary arrays."""
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    union        = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + smooth) / (union + smooth))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pixel-wise accuracy."""
    correct = np.sum(y_true == y_pred)
    total   = y_true.size
    return float(correct / total)


def compute_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute precision, recall, F1 from binary arrays.

    Returns:
        dict with keys: tp, tn, fp, fn, precision, recall, f1
    """
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4)
    }


# ============================================================
# XAI - GRAD-CAM IMPLEMENTATION
# ============================================================
class GradCAM:
    """
    Grad-CAM implementation for U-Net models.
    
    Generates class activation maps to visualize which regions
    the model focuses on when making predictions.
    """
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            layer_name: Name of the convolutional layer to visualize.
                       If None, uses the last Conv2D layer.
        """
        self.model = model
        
        # Find the target layer
        if layer_name is None:
            # Find last Conv2D layer
            for layer in reversed(model.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    self.layer_name = layer.name
                    break
            else:
                raise ValueError("No Conv2D layer found in model")
        else:
            self.layer_name = layer_name
        
        print(f"[XAI] Using layer for Grad-CAM: {self.layer_name}")
        
        # Create gradient model
        self.grad_model = keras.Model(
            inputs=[model.input],
            outputs=[model.get_layer(self.layer_name).output, model.output]
        )
    
    def generate_heatmap(self, image: np.ndarray) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.
        
        Args:
            image: Input image (1, H, W, 1)
            
        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            # Use mean of prediction as the score
            loss = tf.reduce_mean(predictions)
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize to [0, 1]
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        # Resize to match input size
        if HAS_OPENCV:
            heatmap = cv2.resize(heatmap, IMG_SIZE)
        else:
            # Use PIL if cv2 not available
            heatmap = np.array(Image.fromarray(heatmap).resize(IMG_SIZE, Image.BILINEAR))
        
        return heatmap


# ============================================================
# XAI - ATTENTION MAP EXTRACTION
# ============================================================
def extract_attention_maps(
    model: keras.Model,
    image: np.ndarray,
    layer_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract activation maps from specified layers.
    
    Args:
        model: Keras model
        image: Input image (1, H, W, 1)
        layer_names: List of layer names to extract. If None, extracts from
                    key layers in the encoder path.
    
    Returns:
        Dict mapping layer names to activation arrays
    """
    if layer_names is None:
        # Auto-detect encoder layers (typically named with 'conv' or 'block')
        layer_names = []
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2D):
                if 'conv' in layer.name.lower() or 'block' in layer.name.lower():
                    layer_names.append(layer.name)
        
        # Limit to first few layers to avoid too many visualizations
        layer_names = layer_names[:5]
    
    if not layer_names:
        return {}
    
    # Create model to extract activations
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = keras.Model(inputs=model.input, outputs=outputs)
    
    # Get activations
    activations = activation_model.predict(image, verbose=0)
    
    # Normalize each activation map
    attention_maps = {}
    for name, activation in zip(layer_names, activations):
        # Take mean across channels
        att_map = np.mean(activation[0], axis=-1)
        
        # Normalize
        if att_map.max() > att_map.min():
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
        
        attention_maps[name] = att_map
    
    return attention_maps


# ============================================================
# XAI - VISUALIZATION
# ============================================================
def visualize_xai(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    pred_binary: np.ndarray,
    gradcam_heatmap: np.ndarray,
    attention_maps: Dict[str, np.ndarray],
    filename: str,
    output_path: str,
    dice_score: float
):
    """
    Create comprehensive XAI visualization.
    
    Args:
        image: Original image (H, W)
        mask: Ground truth mask (H, W)
        prediction: Raw prediction (H, W)
        pred_binary: Binary prediction (H, W)
        gradcam_heatmap: Grad-CAM heatmap (H, W)
        attention_maps: Dict of attention maps
        filename: Original filename
        output_path: Path to save visualization
        dice_score: Dice score for this sample
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Setup figure
    n_attention = min(3, len(attention_maps))  # Show max 3 attention maps
    n_rows = 2
    n_cols = 4 + n_attention
    
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    fig.patch.set_facecolor('#0a0e1a')
    
    # Title
    fig.suptitle(
        f"XAI Analysis: {filename} (Dice: {dice_score:.4f})",
        color='#00e5ff', fontsize=14, fontweight='bold', y=0.98
    )
    
    # Row 1: Main visualizations
    # Original
    ax1 = plt.subplot(n_rows, n_cols, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original MRI', color='#8899b0', fontsize=10)
    ax1.axis('off')
    ax1.set_facecolor('#0a0e1a')
    
    # Ground truth
    ax2 = plt.subplot(n_rows, n_cols, 2)
    ax2.imshow(mask, cmap='Reds', alpha=0.8)
    ax2.set_title('Ground Truth', color='#8899b0', fontsize=10)
    ax2.axis('off')
    ax2.set_facecolor('#0a0e1a')
    
    # Prediction heatmap
    ax3 = plt.subplot(n_rows, n_cols, 3)
    ax3.imshow(prediction, cmap='plasma', vmin=0, vmax=1)
    ax3.set_title('Prediction Heatmap', color='#8899b0', fontsize=10)
    ax3.axis('off')
    ax3.set_facecolor('#0a0e1a')
    
    # Binary prediction overlay
    ax4 = plt.subplot(n_rows, n_cols, 4)
    overlay = np.stack([image, image, image], axis=-1)
    overlay[pred_binary == 1] = [0.95, 0.32, 0.32]
    ax4.imshow(overlay)
    ax4.set_title('Prediction Overlay', color='#8899b0', fontsize=10)
    ax4.axis('off')
    ax4.set_facecolor('#0a0e1a')
    
    # Attention maps (top 3)
    for i, (layer_name, att_map) in enumerate(list(attention_maps.items())[:n_attention]):
        ax = plt.subplot(n_rows, n_cols, 5 + i)
        
        # Resize attention map to match image size
        if att_map.shape != IMG_SIZE:
            att_map_resized = np.array(
                Image.fromarray(att_map).resize(IMG_SIZE, Image.BILINEAR)
            )
        else:
            att_map_resized = att_map
        
        ax.imshow(att_map_resized, cmap='viridis')
        ax.set_title(f'Attention: {layer_name[:15]}', color='#8899b0', fontsize=9)
        ax.axis('off')
        ax.set_facecolor('#0a0e1a')
    
    # Row 2: Grad-CAM visualizations
    # Grad-CAM heatmap
    ax5 = plt.subplot(n_rows, n_cols, n_cols + 1)
    ax5.imshow(gradcam_heatmap, cmap='jet', alpha=0.8)
    ax5.set_title('Grad-CAM Heatmap', color='#8899b0', fontsize=10)
    ax5.axis('off')
    ax5.set_facecolor('#0a0e1a')
    
    # Grad-CAM overlay on original
    ax6 = plt.subplot(n_rows, n_cols, n_cols + 2)
    ax6.imshow(image, cmap='gray')
    ax6.imshow(gradcam_heatmap, cmap='jet', alpha=0.5)
    ax6.set_title('Grad-CAM Overlay', color='#8899b0', fontsize=10)
    ax6.axis('off')
    ax6.set_facecolor('#0a0e1a')
    
    # Comparison: Grad-CAM vs Ground Truth
    ax7 = plt.subplot(n_rows, n_cols, n_cols + 3)
    comparison = np.stack([gradcam_heatmap, mask, image], axis=-1)
    ax7.imshow(comparison)
    ax7.set_title('CAM vs GT Comparison', color='#8899b0', fontsize=10)
    ax7.axis('off')
    ax7.set_facecolor('#0a0e1a')
    
    # Feature importance (combined)
    ax8 = plt.subplot(n_rows, n_cols, n_cols + 4)
    combined_importance = gradcam_heatmap * 0.7 + prediction * 0.3
    ax8.imshow(combined_importance, cmap='hot')
    ax8.set_title('Combined Importance', color='#8899b0', fontsize=10)
    ax8.axis('off')
    ax8.set_facecolor('#0a0e1a')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0a0e1a')
    plt.close(fig)


# ============================================================
# MAIN EVALUATION
# ============================================================
def evaluate(
    model_path: str = None,
    test_dir: str = None,
    output_dir: str = None,
    batch_size: int = 16,
    xai_enabled: bool = False,
    xai_samples: int = 10
) -> Dict:
    """
    Run full evaluation on the test set.

    Args:
        model_path: path to saved .h5 model
        test_dir: path to test/ directory (contains images/ and masks/)
        output_dir: where to save results JSON and report PNG
        batch_size: inference batch size
        xai_enabled: whether to generate XAI visualizations
        xai_samples: number of samples to generate XAI for (if enabled)

    Returns:
        dict with 'aggregate' and 'per_sample' results
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    test_dir   = test_dir   or DEFAULT_TEST_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 58)
    print("  NeuroScan AI — Model Evaluation")
    if xai_enabled:
        print("  [XAI Mode: ENABLED]")
    print("=" * 58)
    print()

    # --- Load model ---
    print("[1/5] Loading model...")
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        print(f"[✓] Model loaded from: {model_path}")
        print(f"    Total parameters: {model.count_params():,}")
    else:
        print(f"[!] Model not found at {model_path}")
        print("    Generating mock predictions for demo...")
        model = None  # Will use mock below

    # --- Initialize XAI if enabled ---
    gradcam = None
    if xai_enabled and model is not None:
        print("\n[2/5] Initializing XAI components...")
        try:
            gradcam = GradCAM(model)
            xai_output_dir = os.path.join(output_dir, "xai_visualizations")
            os.makedirs(xai_output_dir, exist_ok=True)
            print(f"[✓] XAI initialized - outputs will be saved to: {xai_output_dir}")
        except Exception as e:
            print(f"[!] XAI initialization failed: {e}")
            print("    Continuing without XAI...")
            xai_enabled = False
    else:
        print("\n[2/5] XAI disabled - skipping initialization")

    # --- Load test data ---
    print(f"\n[3/5] Loading test data...")
    images, masks, filenames = load_test_data(test_dir)
    n_samples = len(filenames)

    # --- Run inference ---
    print(f"\n[4/5] Running inference...")
    start_time = time.time()

    if model is not None:
        predictions = model.predict(images, batch_size=batch_size, verbose=1)
    else:
        # Mock predictions: add noise to ground truth masks
        np.random.seed(42)
        predictions = np.clip(masks + np.random.normal(0, 0.2, masks.shape), 0, 1)

    inference_time = time.time() - start_time
    print(f"[✓] Inference complete — {inference_time:.2f}s "
          f"({inference_time / n_samples * 1000:.1f} ms/sample)")

    # Binarize predictions
    pred_binary = (predictions > THRESHOLD).astype(np.float32)

    # --- Compute metrics ---
    print(f"\n[5/5] Computing metrics...")

    per_sample = []
    all_dice, all_iou, all_acc = [], [], []

    # Determine which samples to generate XAI for
    xai_indices = []
    if xai_enabled and gradcam is not None:
        # Select top, middle, and bottom samples by dice score
        # We'll compute dice first, then select
        temp_dice_scores = []
        for i in range(n_samples):
            gt = masks[i, :, :, 0]
            pred = pred_binary[i, :, :, 0]
            temp_dice_scores.append(compute_dice(gt, pred))
        
        # Sort by dice and select samples
        sorted_indices = np.argsort(temp_dice_scores)
        n_xai = min(xai_samples, n_samples)
        
        # Get diverse samples: worst, median, best
        step = max(1, len(sorted_indices) // n_xai)
        xai_indices = sorted_indices[::step][:n_xai].tolist()
        
        print(f"[XAI] Will generate visualizations for {len(xai_indices)} samples")

    for i in range(n_samples):
        gt   = masks[i, :, :, 0]         # (256, 256)
        pred = pred_binary[i, :, :, 0]   # (256, 256)
        raw  = predictions[i, :, :, 0]   # raw scores

        dice = compute_dice(gt, pred)
        iou  = compute_iou(gt, pred)
        acc  = compute_accuracy(gt, pred)
        prf  = compute_precision_recall_f1(gt, pred)

        has_tumor_gt   = bool(np.sum(gt) > 0)
        has_tumor_pred = bool(np.sum(pred) > 0)

        sample_result = {
            "filename":        filenames[i],
            "dice":            round(dice, 4),
            "iou":             round(iou, 4),
            "accuracy":        round(acc, 4),
            "precision":       prf["precision"],
            "recall":          prf["recall"],
            "f1":              prf["f1"],
            "tp":              prf["tp"],
            "tn":              prf["tn"],
            "fp":              prf["fp"],
            "fn":              prf["fn"],
            "ground_truth_has_tumor": has_tumor_gt,
            "prediction_has_tumor":   has_tumor_pred,
            "correct_detection":      has_tumor_gt == has_tumor_pred
        }
        per_sample.append(sample_result)

        all_dice.append(dice)
        all_iou.append(iou)
        all_acc.append(acc)

        # Generate XAI visualizations for selected samples
        if xai_enabled and i in xai_indices and gradcam is not None:
            try:
                print(f"  Generating XAI for sample {i+1}/{n_samples}: {filenames[i]}")
                
                # Generate Grad-CAM
                img_input = images[i:i+1]  # (1, 256, 256, 1)
                heatmap = gradcam.generate_heatmap(img_input)
                
                # Extract attention maps
                attention_maps = extract_attention_maps(model, img_input)
                
                # Create visualization
                xai_output_path = os.path.join(
                    xai_output_dir,
                    f"xai_{filenames[i]}"
                )
                
                visualize_xai(
                    image=images[i, :, :, 0],
                    mask=gt,
                    prediction=raw,
                    pred_binary=pred,
                    gradcam_heatmap=heatmap,
                    attention_maps=attention_maps,
                    filename=filenames[i],
                    output_path=xai_output_path,
                    dice_score=dice
                )
                
                sample_result["xai_generated"] = True
                sample_result["xai_path"] = xai_output_path
                
            except Exception as e:
                print(f"  [!] XAI generation failed for {filenames[i]}: {e}")
                sample_result["xai_generated"] = False

    # --- Aggregate metrics ---
    # Detection-level accuracy (tumor present/absent correctly identified)
    detection_correct = sum(1 for s in per_sample if s["correct_detection"])
    detection_acc     = detection_correct / n_samples

    # Confusion matrix at detection level
    tp_det = sum(1 for s in per_sample if s["ground_truth_has_tumor"] and s["prediction_has_tumor"])
    tn_det = sum(1 for s in per_sample if not s["ground_truth_has_tumor"] and not s["prediction_has_tumor"])
    fp_det = sum(1 for s in per_sample if not s["ground_truth_has_tumor"] and s["prediction_has_tumor"])
    fn_det = sum(1 for s in per_sample if s["ground_truth_has_tumor"] and not s["prediction_has_tumor"])

    aggregate = {
        "n_samples":          n_samples,
        "inference_time_sec": round(inference_time, 2),
        "ms_per_sample":      round(inference_time / n_samples * 1000, 1),
        "mean_dice":          round(float(np.mean(all_dice)), 4),
        "std_dice":           round(float(np.std(all_dice)),  4),
        "mean_iou":           round(float(np.mean(all_iou)),  4),
        "std_iou":            round(float(np.std(all_iou)),   4),
        "mean_accuracy":      round(float(np.mean(all_acc)),  4),
        "detection_accuracy": round(detection_acc, 4),
        "detection_confusion_matrix": {
            "TP": tp_det, "TN": tn_det,
            "FP": fp_det, "FN": fn_det
        },
        "threshold": THRESHOLD,
        "xai_enabled": xai_enabled,
        "xai_samples_generated": len(xai_indices) if xai_enabled else 0
    }

    results = {
        "aggregate":  aggregate,
        "per_sample": per_sample
    }

    # --- Print Report ---
    print_report(aggregate, per_sample)

    # --- Save JSON ---
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[✓] Results saved → {json_path}")

    # --- Generate visual report ---
    if HAS_MATPLOTLIB:
        png_path = os.path.join(output_dir, "evaluation_report.png")
        generate_visual_report(images, masks, predictions, pred_binary, per_sample, png_path)
        print(f"[✓] Visual report saved → {png_path}")

    if xai_enabled and xai_indices:
        print(f"[✓] XAI visualizations saved → {xai_output_dir}/")
        print(f"    Generated {len(xai_indices)} XAI reports")

    return results


# ============================================================
# PRINT REPORT
# ============================================================
def print_report(aggregate: Dict, per_sample: List[Dict]):
    """Pretty-print the evaluation summary."""

    cm = aggregate["detection_confusion_matrix"]

    print()
    print("┌" + "─" * 56 + "┐")
    print("│" + "  EVALUATION SUMMARY".center(56) + "│")
    print("├" + "─" * 56 + "┤")
    print(f"│  {'Samples evaluated:':<30s} {aggregate['n_samples']:<24d} │")
    print(f"│  {'Inference time:':<30s} {aggregate['inference_time_sec']:<18.2f}s   │")
    print(f"│  {'Speed:':<30s} {aggregate['ms_per_sample']:<20.1f}ms/img │")
    
    if aggregate.get("xai_enabled"):
        print(f"│  {'XAI samples generated:':<30s} {aggregate['xai_samples_generated']:<24d} │")
    
    print("├" + "─" * 56 + "┤")
    print("│" + "  SEGMENTATION METRICS".center(56) + "│")
    print("├" + "─" * 56 + "┤")
    print(f"│  {'Mean Dice Coefficient:':<30s} {aggregate['mean_dice']:<10.4f}"
          f"(±{aggregate['std_dice']:.4f})     │")
    print(f"│  {'Mean IoU (Jaccard):':<30s} {aggregate['mean_iou']:<10.4f}"
          f"(±{aggregate['std_iou']:.4f})     │")
    print(f"│  {'Mean Pixel Accuracy:':<30s} {aggregate['mean_accuracy']:<24.4f} │")
    print("├" + "─" * 56 + "┤")
    print("│" + "  DETECTION METRICS".center(56) + "│")
    print("├" + "─" * 56 + "┤")
    print(f"│  {'Detection Accuracy:':<30s} {aggregate['detection_accuracy']:<24.4f} │")
    print(f"│  {'True Positives:':<30s} {cm['TP']:<24d} │")
    print(f"│  {'True Negatives:':<30s} {cm['TN']:<24d} │")
    print(f"│  {'False Positives:':<30s} {cm['FP']:<24d} │")
    print(f"│  {'False Negatives:':<30s} {cm['FN']:<24d} │")
    print("├" + "─" * 56 + "┤")

    # Top 3 best and worst by Dice
    sorted_samples = sorted(per_sample, key=lambda s: s["dice"], reverse=True)

    print("│" + "  TOP 3 — Best Dice".center(56) + "│")
    for s in sorted_samples[:3]:
        print(f"│    {s['filename']:<38s} Dice: {s['dice']:.4f}  │")

    print("│" + "  BOTTOM 3 — Worst Dice".center(56) + "│")
    for s in sorted_samples[-3:]:
        print(f"│    {s['filename']:<38s} Dice: {s['dice']:.4f}  │")

    print("└" + "─" * 56 + "┘")


# ============================================================
# VISUAL REPORT (matplotlib grid)
# ============================================================
def generate_visual_report(
    images: np.ndarray,
    masks: np.ndarray,
    predictions: np.ndarray,
    pred_binary: np.ndarray,
    per_sample: List[Dict],
    output_path: str,
    n_rows: int = 4
):
    """
    Generate a visual comparison grid showing:
        Row per sample: [Original] [Ground Truth] [Prediction] [Overlay]

    Shows the first n_rows samples.
    """
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.2))
    fig.patch.set_facecolor('#0a0e1a')
    fig.suptitle(
        "Brain MRI — Evaluation Visual Report",
        color='#00e5ff', fontsize=15, fontweight='bold', y=0.98
    )

    col_titles = ['Original MRI', 'Ground Truth', 'Prediction', 'Overlay']

    for row in range(n_rows):
        if row >= len(per_sample):
            # Hide unused rows
            for col in range(n_cols):
                axes[row][col].set_visible(False)
            continue

        img  = images[row, :, :, 0]       # (256, 256)
        gt   = masks[row, :, :, 0]        # (256, 256)
        pred = pred_binary[row, :, :, 0]  # (256, 256)
        raw  = predictions[row, :, :, 0]  # (256, 256) raw scores
        info = per_sample[row]

        # Col 0: Original
        axes[row][0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row][0].set_facecolor('#0a0e1a')
        axes[row][0].axis('off')

        # Col 1: Ground Truth mask
        axes[row][1].imshow(gt, cmap='Reds', vmin=0, vmax=1)
        axes[row][1].set_facecolor('#0a0e1a')
        axes[row][1].axis('off')

        # Col 2: Predicted mask (raw scores as heatmap)
        axes[row][2].imshow(raw, cmap='plasma', vmin=0, vmax=1)
        axes[row][2].set_facecolor('#0a0e1a')
        axes[row][2].axis('off')

        # Col 3: Overlay (original + predicted mask in red)
        overlay = np.stack([img, img, img], axis=-1)  # grayscale → RGB
        overlay[pred == 1] = [0.95, 0.32, 0.32]       # Red for tumor
        axes[row][3].imshow(overlay, vmin=0, vmax=1)
        axes[row][3].set_facecolor('#0a0e1a')
        axes[row][3].axis('off')

        # Row label: filename + dice
        axes[row][0].set_ylabel(
            f"{info['filename']}\nDice: {info['dice']:.3f} | IoU: {info['iou']:.3f}",
            color='#8899b0', fontsize=8, labelpad=8, rotation=0,
            ha='right', va='center'
        )

    # Column titles (top row only)
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, color='#8899b0', fontsize=10, pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=140, bbox_inches='tight', facecolor='#0a0e1a')
    plt.close(fig)
    print(f"[✓] Visual report ({n_rows} samples) → {output_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained U-Net on the test set with optional XAI visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python model/evaluate.py

  # With XAI enabled
  python model/evaluate.py --xai-enabled

  # Custom paths with XAI
  python model/evaluate.py --model saved_model/model.h5 --xai-enabled --xai-samples 15

  # Full custom configuration
  python model/evaluate.py --model model.h5 --test-dir data/test --output results/ --xai-enabled
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Path to saved model .h5 (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help=f"Path to test/ directory (default: {DEFAULT_TEST_DIR})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)"
    )
    parser.add_argument(
        "--xai-enabled",
        action="store_true",
        help="Enable XAI (Explainable AI) visualizations including Grad-CAM and attention maps"
    )
    parser.add_argument(
        "--xai-samples",
        type=int,
        default=10,
        help="Number of samples to generate XAI visualizations for (default: 10)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 58)
    print("  Configuration")
    print("=" * 58)
    print(f"  Model:      {args.model or DEFAULT_MODEL_PATH}")
    print(f"  Test Dir:   {args.test_dir or DEFAULT_TEST_DIR}")
    print(f"  Output:     {args.output or DEFAULT_OUTPUT_DIR}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  XAI:        {'ENABLED' if args.xai_enabled else 'DISABLED'}")
    if args.xai_enabled:
        print(f"  XAI Samples: {args.xai_samples}")
    print("=" * 58 + "\n")

    # Run evaluation
    results = evaluate(
        model_path=args.model,
        test_dir=args.test_dir,
        output_dir=args.output,
        batch_size=args.batch_size,
        xai_enabled=args.xai_enabled,
        xai_samples=args.xai_samples
    )
    
    print("\n" + "=" * 58)
    print("  Evaluation Complete!")
    print("=" * 58)
    print()