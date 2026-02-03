"""
saved_model.py
Utility for saving and loading trained U-Net models.

Features:
  - Save model weights only (.weights.h5) — lightweight
  - Save full model (.h5) — architecture + weights + optimizer state
  - Save training metadata as JSON (epochs, loss history, config)
  - Load model with custom objects (dice_loss, dice_coeff)
  - Auto-create output directories
  - Versioned saving (model_v1.h5, model_v2.h5, ...)
  - Summary print on load

Usage:
    from model.saved_model import ModelSaver

    saver = ModelSaver(base_dir="model/saved_model")

    # After training:
    saver.save_full(model, metadata={"epochs": 50, "best_dice": 0.89})
    saver.save_weights(model, version=2)

    # Loading:
    model = saver.load_full()
    # or
    model = saver.load_weights(architecture_fn=build_unet, version=2)
"""

import os
import sys
import json
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

# Add project root to path so we can import model_architecture
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model_architecture import build_unet


# ============================================================
# CUSTOM OBJECTS (must be passed to tf.keras when loading)
# ============================================================
def dice_loss(y_true, y_pred):
    """Dice loss for segmentation."""
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union        = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice         = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def combined_loss(y_true, y_pred):
    """BCE + Dice combined loss."""
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def dice_coeff(y_true, y_pred):
    """Dice coefficient metric (evaluation)."""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    union        = tf.reduce_sum(y_true + y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


# Registry of custom objects for deserialization
CUSTOM_OBJECTS = {
    "combined_loss": combined_loss,
    "dice_loss":     dice_loss,
    "dice_coeff":    dice_coeff,
}


# ============================================================
# MODEL SAVER CLASS
# ============================================================
class ModelSaver:
    """
    Handles all save/load operations for the trained U-Net model.

    Args:
        base_dir: directory where model files are stored
                  default → model/saved_model/
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), "saved_model")
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        # Standard file paths
        self.full_model_path   = os.path.join(self.base_dir, "brain_tumor_model.h5")
        self.weights_path      = os.path.join(self.base_dir, "brain_tumor_weights.weights.h5")
        self.metadata_path     = os.path.join(self.base_dir, "model_metadata.json")

    # ----------------------------------------------------------
    # SAVE: Full model (architecture + weights + optimizer)
    # ----------------------------------------------------------
    def save_full(self, model: tf.keras.Model, metadata: dict = None):
        """
        Save the complete Keras model to .h5 (can be loaded without
        knowing the architecture).

        Args:
            model:    trained tf.keras.Model
            metadata: optional dict with training info to save alongside
        """
        start = time.time()
        model.save(self.full_model_path)
        elapsed = time.time() - start

        print(f"[✓] Full model saved → {self.full_model_path} ({elapsed:.2f}s)")
        print(f"    Parameters: {model.count_params():,}")

        # Save metadata if provided
        if metadata is not None:
            self.save_metadata(metadata)

    # ----------------------------------------------------------
    # SAVE: Weights only (lighter, faster)
    # ----------------------------------------------------------
    def save_weights(self, model: tf.keras.Model, version: int = None):
        """
        Save only model weights (no architecture).
        Useful for checkpointing during training.

        Args:
            model:   trained tf.keras.Model
            version: if provided, saves as brain_tumor_weights_v{N}.weights.h5
        """
        if version is not None:
            path = os.path.join(
                self.base_dir,
                f"brain_tumor_weights_v{version}.weights.h5"
            )
        else:
            path = self.weights_path

        model.save_weights(path)
        print(f"[✓] Weights saved → {path}")

    # ----------------------------------------------------------
    # SAVE: Versioned full model
    # ----------------------------------------------------------
    def save_versioned(self, model: tf.keras.Model, version: int, metadata: dict = None):
        """
        Save a versioned copy of the full model.
        Example output: brain_tumor_model_v2.h5

        Args:
            model:    trained tf.keras.Model
            version:  version number
            metadata: optional training metadata
        """
        path = os.path.join(self.base_dir, f"brain_tumor_model_v{version}.h5")
        model.save(path)
        print(f"[✓] Versioned model saved → {path}")

        if metadata is not None:
            meta_path = os.path.join(self.base_dir, f"model_metadata_v{version}.json")
            self._write_metadata(metadata, meta_path)

    # ----------------------------------------------------------
    # LOAD: Full model
    # ----------------------------------------------------------
    def load_full(self, path: str = None) -> tf.keras.Model:
        """
        Load a full saved model (.h5) with custom objects registered.

        Args:
            path: optional override path (defaults to standard full model path)

        Returns:
            tf.keras.Model ready for inference
        """
        if path is None:
            path = self.full_model_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        print(f"[⬇] Loading full model from: {path}")
        model = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
        print(f"[✓] Model loaded successfully — {model.count_params():,} parameters")
        model.summary()
        return model

    # ----------------------------------------------------------
    # LOAD: Weights only (requires building architecture first)
    # ----------------------------------------------------------
    def load_weights(
        self,
        architecture_fn=None,
        version: int = None,
        input_shape: tuple = (256, 256, 1)
    ) -> tf.keras.Model:
        """
        Build a fresh model from architecture, then load saved weights.

        Args:
            architecture_fn: callable that returns a tf.keras.Model
                             defaults to build_unet()
            version:         if provided, loads versioned weights file
            input_shape:     input shape passed to architecture_fn

        Returns:
            tf.keras.Model with loaded weights
        """
        if architecture_fn is None:
            architecture_fn = build_unet

        # Build fresh model
        print(f"[⬇] Building model architecture...")
        model = architecture_fn(input_shape=input_shape)

        # Compile (required before loading weights in some TF versions)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=combined_loss,
            metrics=[dice_coeff, "accuracy"]
        )

        # Determine weights path
        if version is not None:
            path = os.path.join(
                self.base_dir,
                f"brain_tumor_weights_v{version}.weights.h5"
            )
        else:
            path = self.weights_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")

        print(f"[⬇] Loading weights from: {path}")
        model.load_weights(path)
        print(f"[✓] Weights loaded successfully")
        return model

    # ----------------------------------------------------------
    # METADATA: Save / Load training info
    # ----------------------------------------------------------
    def save_metadata(self, metadata: dict):
        """Save training metadata as JSON."""
        self._write_metadata(metadata, self.metadata_path)

    def load_metadata(self, path: str = None) -> dict:
        """Load training metadata from JSON."""
        if path is None:
            path = self.metadata_path
        if not os.path.exists(path):
            print(f"[!] Metadata file not found: {path}")
            return {}
        with open(path, "r") as f:
            meta = json.load(f)
        print(f"[✓] Metadata loaded from: {path}")
        return meta

    def _write_metadata(self, metadata: dict, path: str):
        """Internal: write metadata dict to JSON file."""
        # Add automatic fields
        metadata.setdefault("saved_at", datetime.now().isoformat())
        metadata.setdefault("tensorflow_version", tf.__version__)

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"[✓] Metadata saved → {path}")

    # ----------------------------------------------------------
    # LIST: Show all saved model files
    # ----------------------------------------------------------
    def list_saved(self):
        """Print all model/weight/metadata files in base_dir."""
        print(f"\n📁 Saved models in: {self.base_dir}")
        print("-" * 50)

        if not os.path.isdir(self.base_dir):
            print("  (directory does not exist)")
            return

        files = sorted(os.listdir(self.base_dir))
        if not files:
            print("  (no files found)")
            return

        for fname in files:
            fpath = os.path.join(self.base_dir, fname)
            size_kb = os.path.getsize(fpath) / 1024
            size_mb = size_kb / 1024

            if size_mb >= 1:
                size_str = f"{size_mb:.1f} MB"
            else:
                size_str = f"{size_kb:.1f} KB"

            # Icon by type
            if fname.endswith(".h5") and "weights" not in fname:
                icon = "🧠"  # Full model
            elif fname.endswith(".h5"):
                icon = "⚖️"   # Weights
            elif fname.endswith(".json"):
                icon = "📄"  # Metadata
            else:
                icon = "📎"

            print(f"  {icon}  {fname:<45s} {size_str:>10s}")

        print("-" * 50)


# ============================================================
# STANDALONE DEMO
# ============================================================
if __name__ == "__main__":
    print("=" * 58)
    print("  NeuroScan AI — Model Save/Load Demo")
    print("=" * 58)
    print()

    saver = ModelSaver()

    # --- Build a model ---
    print("[1/5] Building U-Net model...")
    model = build_unet(input_shape=(256, 256, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=[dice_coeff, "accuracy"]
    )
    print(f"       Parameters: {model.count_params():,}\n")

    # --- Save full model ---
    print("[2/5] Saving full model...")
    saver.save_full(model, metadata={
        "epochs_trained": 50,
        "best_val_dice": 0.8934,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 16,
        "dataset": "LGG MRI Segmentation (Kaggle)",
        "input_shape": [256, 256, 1],
        "architecture": "U-Net (4-level encoder-decoder)"
    })
    print()

    # --- Save weights ---
    print("[3/5] Saving weights (v1)...")
    saver.save_weights(model, version=1)
    print()

    # --- Load full model ---
    print("[4/5] Loading full model back...")
    loaded_model = saver.load_full()
    print()

    # --- Quick inference test ---
    print("[5/5] Running inference test...")
    dummy_input = np.random.rand(1, 256, 256, 1).astype(np.float32)
    output = loaded_model.predict(dummy_input, verbose=0)
    print(f"       Input shape:  {dummy_input.shape}")
    print(f"       Output shape: {output.shape}")
    print(f"       Output range: [{output.min():.4f}, {output.max():.4f}]")
    print()

    # --- List all saved files ---
    saver.list_saved()

    # --- Load metadata ---
    print()
    meta = saver.load_metadata()
    print(f"       Metadata content:")
    for k, v in meta.items():
        print(f"         {k}: {v}")