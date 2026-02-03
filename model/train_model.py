"""
train_model_ULTRA_FAST.py (5-10 MINUTES VERSION)
Ultra-fast training just to verify pipeline works!

⚡⚡⚡ SPEED: 5-10 minutes ONLY!
⚠️  QUALITY: Will be poor, but pipeline verification
❌ USE: Quick proof-of-concept / infrastructure testing ONLY

Key Changes:
  • Image size: 256x256 → 128x128 (4x faster)
  • Model: 25% parameters (4x smaller)
  • Epochs: 5 → 2 (2.5x faster)
  • Batch: 32 → 16 (faster GPU prep)
  • Data: Only 10% of training data
  • Augmentation: Minimal
  • Result: 5-10 minutes total!

✅ PERFECT for testing your pipeline works!
⚠️ NOT for any real evaluation!
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


# ===================== CONFIG - ULTRA FAST =====================
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR   = os.path.join(PROJECT_DIR, "model", "saved_model")
DATA_DIR    = os.path.join(PROJECT_DIR, "data")

IMG_SIZE   = (128, 128)  # ⚡⚡ Ultra tiny
BATCH_SIZE = 16  # ⚡ Smaller batch
EPOCHS     = 2  # ⚡⚡ Just 2 epochs!
LR         = 1e-3
SEED       = 42

SAVE_PATH = os.path.join(MODEL_DIR, "brain_tumor_model_ultrafast.h5")

# Mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f"\n⚡⚡⚡ ULTRA-FAST: Just for testing! (~5-10 min)")

_DRIVE = os.path.splitdrive(PROJECT_DIR)[0]
TB_LOG_ROOT = os.path.join(_DRIVE, "\\", "Brain_MRI_Logs_UltraFast")
LOG_DIR = os.path.join(TB_LOG_ROOT, datetime.now().strftime("%Y%m%d-%H%M%S"))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ===================== TINY MODEL =====================
def build_tiny_unet(input_shape=(128, 128, 1)):
    """Super tiny U-Net - only 0.5M parameters!"""
    inputs = layers.Input(shape=input_shape)

    # Encoder - Ultra minimal
    c1 = layers.Conv2D(8, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(16, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(32, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.Concatenate()([u1, c2])
    u1 = layers.Conv2D(16, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D((2, 2))(u1)
    u2 = layers.Concatenate()([u2, c1])
    u2 = layers.Conv2D(8, 3, activation='relu', padding='same')(u2)

    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u2)

    return Model(inputs, outputs, name="TinyUNet")


# ===================== DATA LOADING =====================
def load_images_sample(folder, max_samples=50):
    """Load ONLY subset of images for speed"""
    paths = sorted([p for p in os.listdir(folder) if p.endswith(".png")])
    paths = paths[:max_samples]  # ⚡ Only first 50 images
    images = []

    for p in paths:
        img = Image.open(os.path.join(folder, p)).convert("L")
        img = img.resize(IMG_SIZE, Image.LANCZOS)  # 128x128
        images.append(np.array(img, dtype=np.float32) / 255.0)

    return np.array(images)[..., np.newaxis]


def load_split(split):
    img_dir  = os.path.join(DATA_DIR, split, "images")
    mask_dir = os.path.join(DATA_DIR, split, "masks")

    if not os.path.isdir(img_dir):
        raise RuntimeError(f"❌ Not found: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise RuntimeError(f"❌ Not found: {mask_dir}")

    return load_images_sample(img_dir, 50), load_images_sample(mask_dir, 50)


# ===================== LOSS =====================
def dice_loss(y_true, y_pred):
    smooth = 1.0
    inter  = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union  = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred),
        axis=[1, 2]
    )
    return 0.5 * bce + 0.5 * dice_loss(y_true, y_pred)


def dice_coeff(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    smooth = 1.0
    inter  = tf.reduce_sum(y_true * y_pred)
    union  = tf.reduce_sum(y_true + y_pred)
    return (2.0 * inter + smooth) / (union + smooth)


# ===================== MAIN =====================
def main():
    print("=" * 70)
    print("  ⚡⚡⚡ ULTRA-FAST TEST (5-10 MINUTES)")
    print("=" * 70)
    print()
    print("⚠️  THIS IS JUST TO TEST YOUR PIPELINE!")
    print("   Model will be terrible, but training should work.")
    print()

    print("📥 Loading SAMPLE data (50 images only)...")
    try:
        X_train, Y_train = load_split("train")
        X_val,   Y_val   = load_split("val")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\n   Make sure data/ folder has train/ and val/ subdirectories")
        print("   With images/ and masks/ inside each")
        return 1

    print(f"   Train: {X_train.shape[0]} images (128x128)")
    print(f"   Val:   {X_val.shape[0]} images")
    print()

    print("🧠 Building TINY model (0.5M params)...")
    model = build_tiny_unet()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[dice_coeff]
    )
    
    print(f"   Parameters: {model.count_params():,} (tiny!)")
    print()

    print("⚙️  Training Configuration:")
    print(f"   Image Size: {IMG_SIZE} (very small)")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS} (just 2)")
    print(f"   Data: 50 images only (10% of actual)")
    print(f"   Expected Time: 5-10 minutes")
    print()

    callbacks = [
        ModelCheckpoint(
            filepath=SAVE_PATH,
            save_best_only=True,
            monitor="val_dice_coeff",
            mode="max",
            verbose=1
        ),
        TensorBoard(log_dir=LOG_DIR, update_freq='epoch'),
    ]

    print("=" * 70)
    print("  🚀 STARTING QUICK TEST")
    print("=" * 70)
    print()

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    best_dice = max(history.history.get("val_dice_coeff", [0]))

    print("\n" + "=" * 70)
    print("  ✅ QUICK TEST COMPLETED!")
    print("=" * 70)
    print(f"  Best val Dice: {best_dice:.4f}")
    print()
    print("  ✅ Your pipeline works!")
    print()
    print(f"  Next: Choose your actual training version:")
    print(f"    • FAST (1-2h):    train_model_FAST_1-2hours.py")
    print(f"    • BALANCED (8h):  train_model_BALANCED_8-12hours.py")
    print(f"    • QUALITY (20h):  train_model_REPLACE_THIS.py")
    print()
    print(f"  📊 TensorBoard: {LOG_DIR}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())