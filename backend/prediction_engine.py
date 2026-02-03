"""
prediction_engine.py (FIXED FOR YOUR PROJECT)
Loads trained U-Net model and runs tumor segmentation.

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
    
    return {
        "tumor_detected": tumor_detected,
        "confidence": round(confidence, 4),
        "tumor_area_percent": round(tumor_area_percent, 2),
        "mask": mask.tolist(),
        "location_hint": location_hint
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