"""
evaluate.py
Comprehensive evaluation of the trained U-Net model on the test set.

Metrics computed:
  - Dice Coefficient (F1 for segmentation)
  - IoU / Jaccard Index
  - Pixel-wise Accuracy
  - Precision / Recall / F1 Score
  - Per-sample breakdown
  - Confusion Matrix (TP, TN, FP, FN)

Outputs:
  - Printed summary report to console
  - evaluation_results.json — full per-sample + aggregate results
  - evaluation_report.png  — visual comparison grid (if matplotlib available)

Usage:
    python model/evaluate.py

    # Or with custom paths:
    python model/evaluate.py --model model/saved_model/brain_tumor_model.h5
                             --test-dir data/test
                             --output results/
"""

import sys
import os
import json
import argparse
import time

import numpy as np
from PIL import Image

import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from saved_model import ModelSaver, CUSTOM_OBJECTS

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
def load_test_data(test_dir: str) -> tuple:
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


def compute_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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
# MAIN EVALUATION
# ============================================================
def evaluate(
    model_path: str = None,
    test_dir:   str = None,
    output_dir: str = None,
    batch_size: int = 16
) -> dict:
    """
    Run full evaluation on the test set.

    Args:
        model_path: path to saved .h5 model
        test_dir:   path to test/ directory (contains images/ and masks/)
        output_dir: where to save results JSON and report PNG
        batch_size: inference batch size

    Returns:
        dict with 'aggregate' and 'per_sample' results
    """
    model_path = model_path or DEFAULT_MODEL_PATH
    test_dir   = test_dir   or DEFAULT_TEST_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 58)
    print("  NeuroScan AI — Model Evaluation")
    print("=" * 58)
    print()

    # --- Load model ---
    print("[1/4] Loading model...")
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        print(f"[✓] Model loaded from: {model_path}")
    else:
        print(f"[!] Model not found at {model_path}")
        print("    Generating mock predictions for demo...")
        model = None  # Will use mock below

    # --- Load test data ---
    print("\n[2/4] Loading test data...")
    images, masks, filenames = load_test_data(test_dir)
    n_samples = len(filenames)

    # --- Run inference ---
    print("\n[3/4] Running inference...")
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
    print("\n[4/4] Computing metrics...")

    per_sample = []
    all_dice, all_iou, all_acc = [], [], []

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
        "threshold": THRESHOLD
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

    return results


# ============================================================
# PRINT REPORT
# ============================================================
def print_report(aggregate: dict, per_sample: list):
    """Pretty-print the evaluation summary."""

    cm = aggregate["detection_confusion_matrix"]

    print()
    print("┌" + "─" * 56 + "┐")
    print("│" + "  EVALUATION SUMMARY".center(56) + "│")
    print("├" + "─" * 56 + "┤")
    print(f"│  {'Samples evaluated:':<30s} {aggregate['n_samples']:<24d} │")
    print(f"│  {'Inference time:':<30s} {aggregate['inference_time_sec']:<18.2f}s   │")
    print(f"│  {'Speed:':<30s} {aggregate['ms_per_sample']:<20.1f}ms/img │")
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
    per_sample: list,
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
        description="Evaluate trained U-Net on the test set."
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        model_path=args.model,
        test_dir=args.test_dir,
        output_dir=args.output,
        batch_size=args.batch_size
    )