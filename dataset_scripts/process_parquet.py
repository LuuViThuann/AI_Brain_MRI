"""
process_parquet.py
Xử lý các file .parquet từ HuggingFace datasets.

Nguồn:
  - AIOmarRehan/Brain_Tumor_MRI_Dataset
  - vanhai123/Brain_tumor_detections

Files:
  - train.parquet
  - test.parquet
  - validation.parquet

Output:
  - data/processed/parquet_images/
  - data/processed/parquet_masks/ (nếu có)
  - data/processed/parquet_metadata.json
"""

import os
import sys
import json
import io
import base64
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUT_IMG_DIR = PROCESSED_DIR / "parquet_images"
OUT_MASK_DIR = PROCESSED_DIR / "parquet_masks"
METADATA_FILE = PROCESSED_DIR / "parquet_metadata.json"

TARGET_SIZE = (256, 256)


def decode_image_from_parquet(img_data):
    """
    Decode image from parquet column.
    Supports: base64 string, bytes, PIL Image dict, numpy array
    """
    try:
        # Case 1: Base64 string
        if isinstance(img_data, str):
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            return img
        
        # Case 2: Bytes directly
        elif isinstance(img_data, bytes):
            img = Image.open(io.BytesIO(img_data))
            return img
        
        # Case 3: PIL Image dict (from datasets library)
        elif isinstance(img_data, dict):
            if 'bytes' in img_data:
                img = Image.open(io.BytesIO(img_data['bytes']))
                return img
            elif 'path' in img_data:
                # Some datasets store path instead of bytes
                img = Image.open(img_data['path'])
                return img
            else:
                raise ValueError(f"Unknown dict format: {img_data.keys()}")
        
        # Case 4: Numpy array
        elif isinstance(img_data, np.ndarray):
            if img_data.dtype != np.uint8:
                img_data = (img_data * 255).astype(np.uint8)
            img = Image.fromarray(img_data)
            return img
        
        # Case 5: Already PIL Image
        elif isinstance(img_data, Image.Image):
            return img_data
        
        else:
            raise ValueError(f"Unsupported image data type: {type(img_data)}")
            
    except Exception as e:
        raise Exception(f"Error decoding image: {e}")


def inspect_parquet_structure(parquet_path: Path):
    """
    Inspect the structure of parquet file to understand data format.
    """
    print(f"\n[Inspecting] {parquet_path.name}...")
    
    try:
        df = pd.read_parquet(parquet_path)
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Show data types
        print(f"\n   Column types:")
        for col in df.columns:
            dtype = df[col].dtype
            print(f"      {col}: {dtype}")
            
            # Show sample values for non-image columns
            if df[col].dtype in ['int64', 'object', 'string']:
                sample = df[col].iloc[0] if len(df) > 0 else None
                print(f"         Sample: {sample}")
        
        # Try to inspect first row image column
        img_cols = [col for col in df.columns if 'image' in col.lower()]
        if img_cols:
            img_col = img_cols[0]
            print(f"\n   Inspecting image column '{img_col}':")
            if len(df) > 0:
                sample_data = df[img_col].iloc[0]
                print(f"      Type: {type(sample_data)}")
                
                if isinstance(sample_data, dict):
                    print(f"      Dict keys: {sample_data.keys()}")
                elif isinstance(sample_data, (int, str)):
                    print(f"      Value: {sample_data}")
        
        return df
        
    except Exception as e:
        print(f"   Error: {e}")
        return None


def process_parquet_file(parquet_path: Path, split_name: str):
    """
    Process a single parquet file.
    
    Args:
        parquet_path: Path to .parquet file
        split_name: 'train', 'test', or 'validation'
    
    Returns:
        List of metadata dicts
    """
    print(f"\n[Processing] {parquet_path.name}...")
    
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"❌ Error reading parquet: {e}")
        return []
    
    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Detect column names (may vary between datasets)
    img_col = None
    label_col = None
    mask_col = None
    
    # First, try to find exact 'image' column (not 'image_id')
    if 'image' in df.columns:
        img_col = 'image'
    
    # Then look for other patterns
    for col in df.columns:
        col_lower = col.lower()
        # Skip image_id, we want the actual image column
        if col_lower == 'image_id':
            continue
        if 'image' in col_lower and img_col is None:
            img_col = col
        elif 'label' in col_lower or 'class' in col_lower:
            label_col = col
        elif 'mask' in col_lower or 'segmentation' in col_lower:
            mask_col = col
    
    # Try to extract label from 'objects' column if no label column found
    if label_col is None and 'objects' in df.columns:
        label_col = 'objects'
    
    if img_col is None:
        print(f"❌ No image column found!")
        return []
    
    print(f"   Image column: {img_col}")
    print(f"   Label column: {label_col}")
    print(f"   Mask column: {mask_col}")
    
    # Check if image column contains actual images or just indices
    if len(df) > 0:
        sample_type = type(df[img_col].iloc[0])
        print(f"   Image data type: {sample_type}")
        
        # If it's just integers, this dataset format is not supported
        if sample_type == int:
            print(f"❌ Image column contains integers (likely indices), not actual image data!")
            print(f"   This parquet format is not supported.")
            print(f"   Please download the actual image files from the dataset source.")
            return []
    
    metadata = []
    success_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"   {split_name}"):
        try:
            # Decode image
            img = decode_image_from_parquet(row[img_col])
            
            if img is None:
                error_count += 1
                continue
            
            # Convert to grayscale and resize
            img = img.convert('L')
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            
            # Save image
            filename = f"{split_name}_{idx:05d}.png"
            img_path = OUT_IMG_DIR / filename
            img.save(img_path, 'PNG')
            
            # Extract label
            label = str(row[label_col]) if label_col else "unknown"
            has_tumor = "tumor" in label.lower() or "glioma" in label.lower()
            
            # Process mask if available
            mask_saved = False
            if mask_col and row[mask_col] is not None:
                try:
                    mask = decode_image_from_parquet(row[mask_col])
                    if mask is not None:
                        mask = mask.convert('L')
                        mask = mask.resize(TARGET_SIZE, Image.NEAREST)
                        mask_path = OUT_MASK_DIR / filename
                        mask.save(mask_path, 'PNG')
                        mask_saved = True
                except:
                    pass
            
            # Store metadata
            metadata.append({
                "filename": filename,
                "split": split_name,
                "label": label,
                "has_tumor": has_tumor,
                "has_mask": mask_saved,
                "source": "parquet",
                "original_index": int(idx)
            })
            
            success_count += 1
            
        except Exception as e:
            # Only print first few errors to avoid spam
            if error_count < 5:
                print(f"   Error processing row {idx}: {e}")
            error_count += 1
            continue
    
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Errors: {error_count}")
    
    return metadata


def main():
    print("=" * 70)
    print("  Parquet Dataset Processor")
    print("=" * 70)
    print()
    
    # Create output directories
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find parquet files
    parquet_files = list(RAW_DIR.glob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ No .parquet files found in {RAW_DIR}")
        print(f"\n   Expected files:")
        print(f"     • train.parquet")
        print(f"     • test.parquet")
        print(f"     • validation.parquet")
        return 1
    
    print(f"Found {len(parquet_files)} parquet file(s):")
    for f in parquet_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   • {f.name} ({size_mb:.1f} MB)")
    
    # First, inspect one file to understand structure
    print("\n" + "=" * 70)
    print("  INSPECTING DATASET STRUCTURE")
    print("=" * 70)
    inspect_parquet_structure(parquet_files[0])
    
    # Ask user to confirm
    print("\n" + "=" * 70)
    print("  PROCESSING FILES")
    print("=" * 70)
    
    # Process each file
    all_metadata = []
    
    for parquet_file in parquet_files:
        # Determine split name from filename
        split_name = parquet_file.stem  # 'train', 'test', 'validation'
        
        metadata = process_parquet_file(parquet_file, split_name)
        all_metadata.extend(metadata)
    
    # Check if any images were processed
    if len(all_metadata) == 0:
        print("\n" + "=" * 70)
        print("  ⚠️  NO IMAGES PROCESSED")
        print("=" * 70)
        print("\nPossible reasons:")
        print("  1. Image column contains indices instead of actual image data")
        print("  2. Parquet format is incompatible")
        print("  3. All images failed to decode")
        print("\nRecommendations:")
        print("  • Check the dataset source for actual image files")
        print("  • Try downloading from HuggingFace using datasets library:")
        print("      from datasets import load_dataset")
        print("      dataset = load_dataset('AIOmarRehan/Brain_Tumor_MRI_Dataset')")
        print("=" * 70)
        return 1
    
    # Save combined metadata
    print(f"\n[Saving metadata]...")
    with open(METADATA_FILE, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Statistics
    print("\n" + "=" * 70)
    print("  PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total images processed: {len(all_metadata)}")
    
    # Count by split
    splits = {}
    for item in all_metadata:
        split = item['split']
        splits[split] = splits.get(split, 0) + 1
    
    if splits:
        print(f"\nBy split:")
        for split, count in splits.items():
            print(f"   {split}: {count} images")
    
    # Count by tumor presence
    with_tumor = sum(1 for item in all_metadata if item['has_tumor'])
    without_tumor = len(all_metadata) - with_tumor
    
    print(f"\nTumor distribution:")
    print(f"   With tumor: {with_tumor} ({with_tumor/len(all_metadata)*100:.1f}%)")
    print(f"   Without tumor: {without_tumor} ({without_tumor/len(all_metadata)*100:.1f}%)")
    
    # Masks availability
    with_masks = sum(1 for item in all_metadata if item['has_mask'])
    print(f"\nMasks available: {with_masks} ({with_masks/len(all_metadata)*100:.1f}%)")
    
    print(f"\nOutput:")
    print(f"   📂 Images: {OUT_IMG_DIR}/")
    print(f"   📂 Masks: {OUT_MASK_DIR}/")
    print(f"   📄 Metadata: {METADATA_FILE}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())