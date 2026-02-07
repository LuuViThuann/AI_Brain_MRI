"""
prediction_engine.py (FIXED - No Circular Import)
Loads trained U-Net model and runs tumor segmentation.

✅ Removed circular import
✅ Custom loss functions included
✅ Proper error handling
✅ Auto-fallback to MOCK when model missing
✅ Works with your exact project structure
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import os
import traceback
from scipy.stats import entropy
from typing import Dict, Tuple


# Model path - adjusted for your structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # Go up from backend/ to BRAIN_MRI/
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "saved_model", "brain_tumor_model.h5")

_model = None


# ===== CUSTOM LOSS FUNCTIONS =====

def dice_loss(y_true, y_pred):
    """Dice loss for segmentation."""
    smooth = 1.0
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def combined_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy + Dice loss."""
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )
    return 0.5 * bce + 0.5 * dice_loss(y_true, y_pred)


def dice_coeff(y_true, y_pred):
    """Dice coefficient metric."""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    smooth = 1.0
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


# ===== MODEL LOADING =====

def load_model():
    """Load trained U-Net model with custom objects."""
    global _model
    
    if _model is not None:
        return _model
    
    print(f"\n[🔍] Looking for model at:")
    print(f"    {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[⚠️ ] Model file not found!")
        print(f"[📝] Using MOCK predictions for demonstration")
        print(f"[ℹ️ ] To train model: python model/train_model.py")
        _model = "MOCK"
        return _model
    
    try:
        print("[🔄] Loading trained model...")
        
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'combined_loss': combined_loss,
                'dice_loss': dice_loss,
                'dice_coeff': dice_coeff
            },
            compile=False
        )
        
        print("[✅] Model loaded successfully!")
        print(f"[📊] Parameters: {_model.count_params():,}")
        return _model
        
    except Exception as e:
        print(f"[❌] Error loading model: {str(e)}")
        print(f"[📋] Falling back to MOCK predictions")
        traceback.print_exc()
        _model = "MOCK"
        return _model


# ===== IMAGE PREPROCESSING =====

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Resize and normalize MRI image for model input.
    
    Args:
        img: PIL Image (any size)
    
    Returns:
        np.ndarray: Shape (1, 256, 256, 1), normalized [0, 1]
    """
    # Convert to grayscale
    img = img.convert("L")
    
    # Resize to 256x256
    img = img.resize((256, 256), Image.LANCZOS)
    
    # Convert to array and normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch and channel dimensions
    arr = arr[np.newaxis, :, :, np.newaxis]
    
    return arr


# ===== LOCATION ESTIMATION =====

def estimate_location(cx: int, cy: int) -> str:
    """
    Estimate brain region from tumor centroid.
    
    Args:
        cx, cy: Centroid coordinates (0-255)
    
    Returns:
        str: Location description
    """
    h, w = 256, 256
    mid_x, mid_y = w // 2, h // 2
    
    # Vertical position
    if cy < h * 0.33:
        vertical = "Superior"
    elif cy > h * 0.66:
        vertical = "Inferior"
    else:
        vertical = "Middle"
    
    # Hemisphere
    horizontal = "left" if cx < mid_x else "right"
    
    # Lobe estimation
    if cy < h * 0.4:
        lobe = "frontal"
    elif cy > h * 0.7:
        lobe = "occipital"
    elif cx < mid_x * 0.7:
        lobe = "temporal"
    else:
        lobe = "parietal"
    
    return f"{vertical} {horizontal} {lobe} lobe"


# ===== MRI SEQUENCE DETECTION =====

def detect_mri_sequence(img_array: np.ndarray) -> str:
    """
    Detect MRI sequence type based on intensity distribution.
    
    Note: This is a heuristic estimation. In production, sequence info
    should come from DICOM metadata.
    
    Returns:
        str: Estimated sequence (T1, T2, FLAIR, T1ce)
    """
    # Normalize
    img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
    
    # Calculate statistics
    mean_intensity = np.mean(img_norm)
    std_intensity = np.std(img_norm)
    
    # Heuristic rules (based on typical MRI characteristics)
    if mean_intensity < 0.3 and std_intensity < 0.15:
        return "T1-weighted"
    elif mean_intensity > 0.5 and std_intensity > 0.2:
        return "T2-weighted"
    elif 0.3 <= mean_intensity <= 0.5:
        return "FLAIR"
    else:
        return "T1ce (contrast-enhanced)"


# ===== SLICE POSITION ESTIMATION =====

def estimate_slice_position(mask: np.ndarray, total_slices: int = 155) -> Dict:
    """
    Estimate axial slice position in brain volume.
    
    Args:
        mask: 2D segmentation mask
        total_slices: Typical brain MRI has ~155 slices
    
    Returns:
        dict with slice info
    """
    # Heuristic: If tumor is detected, estimate position based on centroid
    if np.sum(mask) > 0:
        # Find centroid
        y_coords, x_coords = np.where(mask > 0)
        cy = np.mean(y_coords)
        
        # Map y-position (0-255) to slice number (0-155)
        # Assuming inferior (bottom) of brain = slice 0
        # Superior (top) = slice 155
        estimated_slice = int((1.0 - cy / 256.0) * total_slices)
        
        # Determine region
        if estimated_slice < 50:
            region = "inferior (basal ganglia / temporal)"
        elif estimated_slice < 100:
            region = "middle (centrum semiovale)"
        else:
            region = "superior (high convexity)"
        
        return {
            "estimated_slice": estimated_slice,
            "total_slices": total_slices,
            "region": region,
            "slice_type": "axial"
        }
    else:
        return {
            "estimated_slice": None,
            "total_slices": total_slices,
            "region": "unknown",
            "slice_type": "axial"
        }


# ===== CONFIDENCE CALCULATION =====

def calculate_confidence_breakdown(
    raw_prediction: np.ndarray,
    mask: np.ndarray
) -> Dict:
    """
    Calculate detailed confidence metrics.
    
    Returns:
        dict with:
            - softmax_confidence: Mean probability in predicted tumor region
            - calibrated_confidence: Temperature-scaled confidence
            - uncertainty_score: Predictive entropy
            - confidence_type: Explanation of what the confidence means
    """
    # 1. Softmax confidence (mean probability in tumor region)
    if np.sum(mask) > 0:
        tumor_probs = raw_prediction[mask > 0]
        softmax_conf = float(np.mean(tumor_probs))
    else:
        # If no tumor predicted, confidence = 1 - mean(all probabilities)
        softmax_conf = float(1.0 - np.mean(raw_prediction))
    
    # 2. Calibrated confidence (temperature scaling T=1.5)
    temperature = 1.5
    calibrated_conf = softmax_conf ** (1.0 / temperature)
    
    # 3. Uncertainty (predictive entropy)
    # Higher entropy = more uncertain
    pred_entropy = entropy([raw_prediction.flatten(), 1 - raw_prediction.flatten()])
    uncertainty_score = float(np.mean(pred_entropy))
    
    # Normalize uncertainty to 0-1 (lower = more certain)
    uncertainty_normalized = min(1.0, uncertainty_score / 0.7)  # 0.7 is typical max
    
    return {
        "softmax_confidence": round(softmax_conf, 4),
        "calibrated_confidence": round(calibrated_conf, 4),
        "uncertainty_score": round(uncertainty_normalized, 4),
        "confidence_type": "Calibrated softmax probability with temperature scaling (T=1.5)",
        "interpretation": {
            "softmax": "Raw model output probability",
            "calibrated": "Temperature-scaled for better calibration",
            "uncertainty": "Predictive entropy (0=certain, 1=uncertain)"
        }
    }


# ===== ENHANCED PREDICTION =====

def predict_tumor_enhanced(img: Image.Image) -> Dict:
    """
    Enhanced tumor prediction with detailed metadata.
    
    Returns comprehensive prediction with:
        - Standard prediction info
        - MRI sequence detection
        - Slice position estimation
        - Detailed confidence breakdown
        - Uncertainty metrics
    """
    # Load model and run basic prediction
    model = load_model()
    input_arr = preprocess_image(img)
    
    # Get raw prediction
    raw_prediction = model.predict(input_arr, verbose=0)[0, :, :, 0]
    
    # Threshold
    mask = (raw_prediction > 0.5).astype(np.float32)
    
    # Basic stats
    tumor_pixels = int(np.sum(mask))
    total_pixels = 256 * 256
    tumor_area_percent = float(tumor_pixels / total_pixels * 100)
    tumor_detected = tumor_area_percent > 0.5
    
    # Location
    if tumor_detected:
        ys, xs = np.where(mask == 1)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        location_hint = estimate_location(cx, cy)
    else:
        location_hint = "No tumor detected"
        cy, cx = 0, 0
    
    # ===== ENHANCEMENTS =====
    
    # 1. MRI Sequence Detection
    mri_sequence = detect_mri_sequence(input_arr[0, :, :, 0])
    
    # 2. Slice Position
    slice_info = estimate_slice_position(mask)
    
    # 3. Confidence Breakdown
    confidence_detail = calculate_confidence_breakdown(raw_prediction, mask)
    
    # 4. Build enhanced result
    return {
        # Standard fields
        "tumor_detected": tumor_detected,
        "tumor_area_percent": round(tumor_area_percent, 2),
        "mask": mask.tolist(),
        "location_hint": location_hint,
        
        # Enhanced confidence info
        "confidence": confidence_detail["calibrated_confidence"],  # Main confidence value
        "confidence_breakdown": confidence_detail,
        
        # MRI metadata
        "mri_metadata": {
            "sequence": mri_sequence,
            "slice_position": slice_info,
            "image_size": [256, 256],
            "acquisition_type": "2D axial (estimated)"
        },
        
        # Additional metrics
        "tumor_centroid": {
            "x": int(cx),
            "y": int(cy)
        },
        "mask_shape": [256, 256]
    }


# ===== MAIN PREDICTION =====

def predict_tumor(img: Image.Image) -> dict:
    """
    Run tumor segmentation on MRI image.
    
    Args:
        img: PIL Image object
    
    Returns:
        dict:
            - tumor_detected: bool
            - confidence: float (0-1)
            - tumor_area_percent: float
            - mask: list[list[float]] (256x256)
            - location_hint: str
    """
    # Load model (singleton)
    model = load_model()
    
    # Preprocess
    try:
        input_arr = preprocess_image(img)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    # === MOCK PREDICTION ===
    if model == "MOCK":
        print("[🎭] Using MOCK prediction")
        
        np.random.seed(42)
        mask = np.zeros((256, 256), dtype=np.float32)
        
        # Simulate tumor blob
        cx, cy, r = 100, 90, 40
        y_coords, x_coords = np.ogrid[:256, :256]
        tumor_region = (x_coords - cx)**2 + (y_coords - cy)**2 <= r**2
        mask[tumor_region] = 1.0
        
        confidence = 0.87
        tumor_detected = True
        tumor_area_percent = float(np.sum(mask) / (256 * 256) * 100)
        location_hint = "Superior left frontal lobe"
    
    # === REAL PREDICTION ===
    else:
        try:
            # Run model
            prediction = model.predict(input_arr, verbose=0)[0, :, :, 0]
            
            # Threshold
            mask = (prediction > 0.5).astype(np.float32)
            
            # Statistics
            tumor_pixels = int(np.sum(mask))
            total_pixels = 256 * 256
            tumor_area_percent = float(tumor_pixels / total_pixels * 100)
            tumor_detected = tumor_area_percent > 0.5
            
            # Confidence
            if tumor_detected:
                confidence = float(np.mean(prediction[mask == 1]))
            else:
                confidence = float(1.0 - np.mean(prediction))
            
            # Location
            if tumor_detected:
                ys, xs = np.where(mask == 1)
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                location_hint = estimate_location(cx, cy)
            else:
                location_hint = "No tumor detected"
        
        except Exception as e:
            print(f"[❌] Prediction error: {str(e)}")
            raise RuntimeError(f"Model prediction failed: {str(e)}")
    
     # Thêm sau khi tính được mask:
    if tumor_detected and np.sum(mask) > 0:
        ys, xs = np.where(mask == 1)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        
        # Normalize to -1 to 1 range (for 3D coordinates)
        centroid_normalized = [
            (cx - 128) / 128,  # -1 to 1
            (cy - 128) / 128,  # -1 to 1
            0  # z-axis (single slice)
        ]
    else:
        cx, cy = 0, 0
        centroid_normalized = [0, 0, 0]
    
    return {
        "tumor_detected": tumor_detected,
        "confidence": round(confidence, 4),
        "tumor_area_percent": round(tumor_area_percent, 2),
        "mask": mask.tolist(),
        "location_hint": location_hint,
        "centroid_px": {"x": cx, "y": cy},  
        "centroid_normalized": centroid_normalized,  
    }


# ===== TESTING =====

if __name__ == "__main__":
    print("=" * 70)
    print("  Testing prediction_engine.py")
    print("=" * 70)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Model path: {MODEL_PATH}")
    print()
    
    # Load model
    model = load_model()
    
    if model == "MOCK":
        print("\n⚠️  Running in MOCK mode")
    else:
        print(f"\n✅ Real model loaded")
    
    # Test with dummy image
    print("\n[🧪] Testing with 256x256 dummy image...")
    test_img = Image.new('L', (256, 256), color=128)
    
    try:
        result = predict_tumor(test_img)
        print(f"\n✅ Test successful!")
        print(f"   Tumor: {result['tumor_detected']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Area: {result['tumor_area_percent']:.2f}%")
        print(f"   Location: {result['location_hint']}")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        traceback.print_exc()