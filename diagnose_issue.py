"""
diagnose_issue.py
Diagnostic script to identify the exact issue with the /api/diagnose endpoint.
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("  Brain MRI Diagnosis - Issue Diagnostic Tool")
print("=" * 70)
print()

# ===== 1. CHECK PROJECT STRUCTURE =====
print("[1/5] Checking project structure...")

# Find project root
current_dir = Path(__file__).parent
project_root = current_dir

# Expected structure
expected_files = {
    "backend/app.py": "Main FastAPI application",
    "backend/prediction_engine.py": "CNN prediction module",
    "backend/groq_client.py": "Groq AI client",
    "backend/routes/diagnosis.py": "Diagnosis endpoint",
    "backend/routes/brain3d.py": "3D brain endpoint",
    "model/saved_model/": "Trained model directory",
    "data/": "Dataset directory"
}

print(f"   Project root: {project_root}")
print(f"   Checking for expected files...")

missing_files = []
for filepath, description in expected_files.items():
    full_path = project_root / filepath
    if full_path.exists():
        print(f"   ✅ {filepath}")
    else:
        print(f"   ❌ {filepath} - MISSING!")
        missing_files.append(filepath)

if missing_files:
    print(f"\n   ⚠️  Missing {len(missing_files)} expected files/directories")
else:
    print(f"\n   ✅ All expected files found")

print()

# ===== 2. CHECK MODEL FILE =====
print("[2/5] Checking trained model...")

model_dir = project_root / "model" / "saved_model"
model_path = model_dir / "brain_tumor_model.h5"

print(f"   Model directory: {model_dir}")
print(f"   Looking for: brain_tumor_model.h5")

if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Model file found! ({size_mb:.1f} MB)")
else:
    print(f"   ❌ Model file NOT found!")
    print(f"   Expected at: {model_path}")
    print()
    print(f"   Available model files:")
    if model_dir.exists():
        model_files = list(model_dir.glob("*.h5")) + list(model_dir.glob("*.keras"))
        if model_files:
            for f in model_files:
                print(f"      • {f.name}")
        else:
            print(f"      (No .h5 or .keras files found)")
    else:
        print(f"      (Model directory doesn't exist)")

print()

# ===== 3. TEST TENSORFLOW IMPORT =====
print("[3/5] Testing TensorFlow import...")

try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow version: {tf.__version__}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✅ GPU available: {len(gpus)} device(s)")
    else:
        print(f"   ℹ️  No GPU detected (using CPU)")
    
except ImportError as e:
    print(f"   ❌ TensorFlow import failed!")
    print(f"   Error: {str(e)}")

print()

# ===== 4. TEST MODEL LOADING =====
print("[4/5] Testing model loading...")

if model_path.exists():
    try:
        import tensorflow as tf
        
        # Define custom loss functions
        def dice_loss(y_true, y_pred):
            smooth = 1.0
            y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return 1.0 - (2.0 * intersection + smooth) / (
                tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
            )
        
        def combined_loss(y_true, y_pred):
            bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
            return 0.5 * bce + 0.5 * dice_loss(y_true, y_pred)
        
        def dice_coeff(y_true, y_pred):
            y_pred = tf.cast(y_pred > 0.5, tf.float32)
            smooth = 1.0
            y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2.0 * intersection + smooth) / (
                tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
            )
        
        print(f"   Attempting to load: {model_path.name}")
        
        model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                'combined_loss': combined_loss,
                'dice_loss': dice_loss,
                'dice_coeff': dice_coeff
            },
            compile=False
        )
        
        print(f"   ✅ Model loaded successfully!")
        print(f"   Model name: {model.name}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parameters: {model.count_params():,}")
        
        # Test prediction
        import numpy as np
        test_input = np.random.rand(1, 256, 256, 1).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        print(f"   ✅ Test prediction successful!")
        print(f"   Output shape: {prediction.shape}")
        
    except Exception as e:
        print(f"   ❌ Model loading failed!")
        print(f"   Error: {str(e)}")
        import traceback
        print(f"\n   Full traceback:")
        traceback.print_exc()
else:
    print(f"   ⚠️  Skipping (model file not found)")

print()

# ===== 5. CHECK DEPENDENCIES =====
print("[5/5] Checking Python dependencies...")

required_packages = {
    'fastapi': 'FastAPI web framework',
    'uvicorn': 'ASGI server',
    'tensorflow': 'Deep learning framework',
    'pillow': 'Image processing',
    'numpy': 'Numerical computing',
    'groq': 'Groq AI API client'
}

missing_packages = []
for package, description in required_packages.items():
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - NOT INSTALLED!")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ⚠️  Missing {len(missing_packages)} required packages")
    print(f"   Install with: pip install {' '.join(missing_packages)}")
else:
    print(f"\n   ✅ All required packages installed")

print()

# ===== SUMMARY =====
print("=" * 70)
print("  DIAGNOSTIC SUMMARY")
print("=" * 70)

issues_found = []

if missing_files:
    issues_found.append(f"Missing {len(missing_files)} expected files")

if not model_path.exists():
    issues_found.append("Trained model file not found")

if missing_packages:
    issues_found.append(f"Missing {len(missing_packages)} Python packages")

if issues_found:
    print("\n❌ ISSUES FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"   {i}. {issue}")
    
    print("\n📋 RECOMMENDED ACTIONS:")
    
    if not model_path.exists():
        print(f"   1. Train the model:")
        print(f"      python model/train_model.py")
        print(f"      (or use train_model_ULTRA_FAST.py for quick testing)")
    
    if missing_packages:
        print(f"   2. Install missing packages:")
        print(f"      pip install {' '.join(missing_packages)}")
    
    if missing_files:
        print(f"   3. Ensure all project files are in place")
else:
    print("\n✅ NO CRITICAL ISSUES FOUND!")
    print("\n   Your setup appears to be correct.")
    print("   If you're still experiencing errors, check:")
    print("   • Server logs for specific error messages")
    print("   • File permissions")
    print("   • Port availability (8000)")

print("\n" + "=" * 70)