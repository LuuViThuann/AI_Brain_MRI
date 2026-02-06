"""
build_faiss_index.py
Build FAISS vector index for similar case retrieval.

Usage:
    python dataset_scripts/build_faiss_index.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import tempfile
import shutil

# Check for required packages
try:
    import faiss
except ImportError:
    print("ERROR: FAISS not installed. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow==2.16.2")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_DIR = DATA_DIR / "faiss"

MODEL_PATH = PROJECT_ROOT / "model" / "saved_model" / "brain_tumor_model.h5"
METADATA_PATH = PROCESSED_DIR / "dataset_metadata.json"

FAISS_INDEX_PATH = FAISS_DIR / "brain_tumor_index.faiss"
FAISS_METADATA_PATH = FAISS_DIR / "case_metadata.json"


def load_model():
    """Load trained U-Net model and create feature extractor."""
    print("[Loading model...]")
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Please train the model first: python train_model.py")
        return None
    
    # Custom objects for loading
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
    
    custom_objects = {
        'combined_loss': combined_loss,
        'dice_loss': dice_loss,
        'dice_coeff': dice_coeff
    }
    
    try:
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            custom_objects=custom_objects,
            compile=False
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Find bottleneck layer (middle of U-Net)
    bottleneck_layer = None
    for layer in model.layers:
        if hasattr(layer, 'output_shape'):
            shape = layer.output_shape
            if isinstance(shape, tuple) and len(shape) == 4:
                h, w = shape[1], shape[2]
                if h is not None and w is not None and h <= 16 and w <= 16:
                    bottleneck_layer = layer.name
                    break
    
    if bottleneck_layer is None:
        # Fallback: use layer at 50% depth
        bottleneck_layer = model.layers[len(model.layers) // 2].name
        print(f"   ⚠️  Using fallback layer")
    
    print(f"   Using layer: {bottleneck_layer}")
    
    # Create feature extractor
    try:
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(bottleneck_layer).output
        )
        return feature_model
    except Exception as e:
        print(f"❌ Error creating feature extractor: {e}")
        return None


def extract_features(model, img_path):
    """Extract feature vector from image."""
    try:
        img = Image.open(img_path).convert('L')
        img = img.resize((256, 256), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = img_array[np.newaxis, :, :, np.newaxis]
        
        features = model.predict(img_array, verbose=0)
        
        # Flatten and normalize
        features = features.flatten()
        norm = np.linalg.norm(features)
        
        if norm > 1e-8:
            features = features / norm
        else:
            features = np.zeros_like(features)
        
        return features
    except Exception as e:
        return None


def find_image_directory():
    """Find directory containing processed images."""
    possible_dirs = [
        PROCESSED_DIR / "combined_images",
        PROCESSED_DIR / "mat_images",
        PROCESSED_DIR / "parquet_images",
        DATA_DIR / "images",
        DATA_DIR / "train" / "images"
    ]
    
    for img_dir in possible_dirs:
        if img_dir.exists() and any(img_dir.glob("*.png")):
            return img_dir
    
    return None


def load_metadata():
    """Load dataset metadata."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Try alternatives
    alt_paths = [
        (PROCESSED_DIR / "mat_metadata.json", "json"),
        (PROCESSED_DIR / "parquet_metadata.json", "json"),
    ]
    
    for alt_path, file_type in alt_paths:
        if alt_path.exists():
            with open(alt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    return None


def build_index():
    """Main function to build FAISS index."""
    print("=" * 70)
    print("  Building FAISS Index for Similar Case Retrieval")
    print("=" * 70)
    print()
    
    # Create output directory with explicit error handling
    print("[0/4] Creating output directory...")
    try:
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Directory ready: {FAISS_DIR}")
        
        # Test write permission
        test_file = FAISS_DIR / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print(f"   ✅ Write permission OK")
    except Exception as e:
        print(f"   ❌ Error with directory: {e}")
        print(f"\n   Please manually create: {FAISS_DIR}")
        print(f"   Or run as administrator")
        return 1
    
    print()
    
    # Load model
    print("[1/4] Loading feature extraction model...")
    model = load_model()
    
    if model is None:
        return 1
    
    # Get feature dimension
    try:
        dummy_input = np.zeros((1, 256, 256, 1), dtype=np.float32)
        dummy_output = model.predict(dummy_input, verbose=0)
        feature_dim = dummy_output.flatten().shape[0]
        print(f"   ✅ Feature dimension: {feature_dim}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1
    
    print()
    
    # Load metadata
    print("[2/4] Loading dataset metadata...")
    metadata = load_metadata()
    
    if metadata is None or len(metadata) == 0:
        print(f"   ❌ No metadata found!")
        print(f"   Expected: {METADATA_PATH}")
        print(f"   Run: python dataset_scripts/process_mat_files.py")
        return 1
    
    print(f"   Total cases: {len(metadata)}")
    print()
    
    # Find images
    print("[3/4] Finding images...")
    img_dir = find_image_directory()
    
    if img_dir is None:
        print(f"   ❌ No image directory found!")
        return 1
    
    print(f"   Using: {img_dir}")
    print()
    
    # Extract features
    print("   Extracting features...")
    features = []
    case_metadata = []
    
    success_count = 0
    error_count = 0
    
    for item in tqdm(metadata, desc="   Progress"):
        try:
            filename = item.get('filename', '') if isinstance(item, dict) else str(item)
            if not filename:
                error_count += 1
                continue
            
            img_path = img_dir / filename
            
            if not img_path.exists():
                error_count += 1
                continue
            
            feat = extract_features(model, img_path)
            
            if feat is None:
                error_count += 1
                continue
            
            features.append(feat)
            
            # Store metadata (ensure JSON serializable)
            case_metadata.append({
                "id": success_count,
                "filename": str(filename),
                "has_tumor": bool(item.get('has_tumor', False)) if isinstance(item, dict) else False,
                "source": str(item.get('source', 'unknown')) if isinstance(item, dict) else 'unknown',
                "patient_id": str(item.get('patient_id', 'unknown')) if isinstance(item, dict) else 'unknown',
                "label": str(item.get('label', 'unknown')) if isinstance(item, dict) else 'unknown'
            })
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            continue
    
    if success_count == 0:
        print(f"\n   ❌ No features extracted!")
        print(f"   Success: {success_count} / Errors: {error_count}")
        return 1
    
    features = np.array(features, dtype='float32')
    
    print(f"\n   ✅ Features: {features.shape}")
    print(f"   Success: {success_count} / Errors: {error_count}")
    print()
    
    # Build FAISS index
    print("[4/4] Building FAISS index...")
    
    # Normalize features
    faiss.normalize_L2(features)
    
    # Choose index type
    if success_count < 1000:
        print("   Using IndexFlatL2 (exact search)")
        index = faiss.IndexFlatL2(feature_dim)
        index.add(features)
    else:
        nlist = min(100, success_count // 10)
        print(f"   Using IndexIVFFlat (nlist={nlist})")
        
        quantizer = faiss.IndexFlatL2(feature_dim)
        index = faiss.IndexIVFFlat(quantizer, feature_dim, nlist)
        
        print("   Training...")
        index.train(features)
        index.add(features)
        index.nprobe = 10
    
    print(f"   ✅ Index built: {index.ntotal} vectors")
    print()
    
    # Save with robust error handling
    print("[Saving...]")
    
    # Save FAISS index
    try:
        # Method 1: Direct save
        index_path_str = str(FAISS_INDEX_PATH.absolute())
        faiss.write_index(index, index_path_str)
        print(f"   ✅ Index saved: {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"   ⚠️  Direct save failed: {e}")
        print(f"   Trying alternative method...")
        
        try:
            # Method 2: Save to temp then move
            with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as tmp:
                temp_path = tmp.name
            
            faiss.write_index(index, temp_path)
            shutil.move(temp_path, str(FAISS_INDEX_PATH))
            print(f"   ✅ Index saved (via temp): {FAISS_INDEX_PATH}")
        except Exception as e2:
            print(f"   ❌ Failed to save index: {e2}")
            return 1
    
    # Save metadata
    try:
        with open(FAISS_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(case_metadata, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Metadata saved: {FAISS_METADATA_PATH}")
    except Exception as e:
        print(f"   ❌ Error saving metadata: {e}")
        return 1
    
    print()
    
    # Statistics
    print("=" * 70)
    print("  FAISS INDEX BUILD COMPLETE")
    print("=" * 70)
    print(f"Total vectors: {index.ntotal}")
    print(f"Feature dimension: {feature_dim}")
    
    try:
        size_mb = FAISS_INDEX_PATH.stat().st_size / (1024*1024)
        print(f"Index size: {size_mb:.1f} MB")
    except:
        pass
    
    print()
    
    # Test search
    print("[Testing search...]")
    try:
        test_query = features[0:1]
        D, I = index.search(test_query, k=min(5, len(features)))
        print(f"   ✅ Search successful")
        print(f"   Top similar IDs: {I[0]}")
        print(f"   Distances: {np.round(D[0], 4)}")
    except Exception as e:
        print(f"   ⚠️  Test failed: {e}")
    
    print()
    print("=" * 70)
    print("Done! Index ready for use.")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(build_index())