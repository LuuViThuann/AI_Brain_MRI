"""
merge_datasets.py
Gộp tất cả nguồn dữ liệu thành một dataset thống nhất.

Nguồn:
  1. Kaggle LGG (~3000 images)
  2. HuggingFace Parquet (~7000 images)
  3. Mendeley (~3000 images)
  4. .mat files (~2500 slices)

Output:
  - data/processed/combined_images/
  - data/processed/combined_masks/
  - data/processed/dataset_metadata.json
  - data/processed/dataset_stats.json
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Input directories
KAGGLE_IMG = DATA_DIR / "images"
KAGGLE_MASK = DATA_DIR / "masks"
PARQUET_IMG = PROCESSED_DIR / "parquet_images"
PARQUET_MASK = PROCESSED_DIR / "parquet_masks"
MAT_IMG = PROCESSED_DIR / "mat_images"
MAT_MASK = PROCESSED_DIR / "mat_masks"
MENDELEY_IMG = PROCESSED_DIR / "mendeley_images"
MENDELEY_MASK = PROCESSED_DIR / "mendeley_masks"

# Metadata files
KAGGLE_META = DATA_DIR / "dataset_summary.csv"
PARQUET_META = PROCESSED_DIR / "parquet_metadata.json"
MAT_META = PROCESSED_DIR / "mat_metadata.json"
MENDELEY_META = PROCESSED_DIR / "mendeley_metadata.json"

# Output directories
OUT_IMG_DIR = PROCESSED_DIR / "combined_images"
OUT_MASK_DIR = PROCESSED_DIR / "combined_masks"
OUT_METADATA = PROCESSED_DIR / "dataset_metadata.json"
OUT_STATS = PROCESSED_DIR / "dataset_stats.json"


def load_kaggle_metadata():
    """Load Kaggle LGG dataset metadata."""
    import csv
    
    if not KAGGLE_META.exists():
        print("⚠️  Kaggle metadata not found, skipping...")
        return []
    
    metadata = []
    with open(KAGGLE_META, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append({
                "filename": row['filename'],
                "patient_id": row['patient_id'],
                "slice_num": row.get('slice_num', '0'),
                "has_tumor": row['has_tumor'] == '1',
                "has_mask": True,
                "source": "kaggle_lgg"
            })
    
    return metadata


def load_json_metadata(json_path: Path, source_name: str):
    """Load metadata from JSON file."""
    if not json_path.exists():
        print(f"⚠️  {source_name} metadata not found, skipping...")
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Ensure all entries have source field
    for item in data:
        item['source'] = source_name
    
    return data


def copy_with_rename(src_path: Path, dest_dir: Path, new_filename: str):
    """Copy file with new name."""
    if not src_path.exists():
        return False
    
    dest_path = dest_dir / new_filename
    shutil.copy2(src_path, dest_path)
    return True


def merge_datasets():
    """Main merging function."""
    print("=" * 70)
    print("  Dataset Merger - Combining All Sources")
    print("=" * 70)
    print()
    
    # Create output directories
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all metadata
    print("[1/3] Loading metadata from all sources...")
    
    all_metadata = []
    source_counts = defaultdict(int)
    
    # Kaggle LGG
    if KAGGLE_IMG.exists():
        kaggle_meta = load_kaggle_metadata()
        all_metadata.extend(kaggle_meta)
        source_counts['kaggle_lgg'] = len(kaggle_meta)
        print(f"   ✅ Kaggle LGG: {len(kaggle_meta)} images")
    
    # Parquet datasets
    if PARQUET_IMG.exists():
        parquet_meta = load_json_metadata(PARQUET_META, 'parquet')
        all_metadata.extend(parquet_meta)
        source_counts['parquet'] = len(parquet_meta)
        print(f"   ✅ Parquet: {len(parquet_meta)} images")
    
    # .mat files
    if MAT_IMG.exists():
        mat_meta = load_json_metadata(MAT_META, 'mat')
        all_metadata.extend(mat_meta)
        source_counts['mat'] = len(mat_meta)
        print(f"   ✅ MAT files: {len(mat_meta)} images")
    
    # Mendeley
    if MENDELEY_IMG.exists():
        mendeley_meta = load_json_metadata(MENDELEY_META, 'mendeley')
        all_metadata.extend(mendeley_meta)
        source_counts['mendeley'] = len(mendeley_meta)
        print(f"   ✅ Mendeley: {len(mendeley_meta)} images")
    
    if not all_metadata:
        print("\n❌ No metadata found! Run preprocessing scripts first:")
        print("   • python dataset_scripts/preprocess.py (Kaggle)")
        print("   • python dataset_scripts/process_parquet.py")
        print("   • python dataset_scripts/process_mat_files.py")
        return 1
    
    print(f"\n   Total items to merge: {len(all_metadata)}")
    
    # Copy and rename files
    print("\n[2/3] Copying and renaming files...")
    
    combined_metadata = []
    success_count = 0
    error_count = 0
    
    for idx, item in enumerate(tqdm(all_metadata, desc="   Merging")):
        source = item['source']
        old_filename = item['filename']
        
        # New unified filename format: {source}_{idx:06d}.png
        new_filename = f"{source}_{idx:06d}.png"
        
        # Determine source directories
        if source == 'kaggle_lgg':
            src_img_dir = KAGGLE_IMG
            src_mask_dir = KAGGLE_MASK
        elif source == 'parquet':
            src_img_dir = PARQUET_IMG
            src_mask_dir = PARQUET_MASK
        elif source == 'mat':
            src_img_dir = MAT_IMG
            src_mask_dir = MAT_MASK
        elif source == 'mendeley':
            src_img_dir = MENDELEY_IMG
            src_mask_dir = MENDELEY_MASK
        else:
            error_count += 1
            continue
        
        # Copy image
        src_img_path = src_img_dir / old_filename
        img_copied = copy_with_rename(src_img_path, OUT_IMG_DIR, new_filename)
        
        if not img_copied:
            error_count += 1
            continue
        
        # Copy mask if available
        mask_copied = False
        if item.get('has_mask', False):
            src_mask_path = src_mask_dir / old_filename
            mask_copied = copy_with_rename(src_mask_path, OUT_MASK_DIR, new_filename)
        
        # Update metadata
        combined_item = {
            "filename": new_filename,
            "original_filename": old_filename,
            "source": source,
            "has_tumor": item.get('has_tumor', False),
            "has_mask": mask_copied,
            "patient_id": item.get('patient_id', 'unknown'),
            "index": idx
        }
        
        # Add source-specific fields
        if 'slice_index' in item:
            combined_item['slice_index'] = item['slice_index']
        if 'label' in item:
            combined_item['label'] = item['label']
        
        combined_metadata.append(combined_item)
        success_count += 1
    
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Errors: {error_count}")
    
    # Save combined metadata
    print("\n[3/3] Saving metadata and statistics...")
    
    with open(OUT_METADATA, 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    # Compute statistics
    stats = compute_statistics(combined_metadata)
    
    with open(OUT_STATS, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print_summary(stats, source_counts)
    
    return 0


def compute_statistics(metadata: list) -> dict:
    """Compute dataset statistics."""
    total = len(metadata)
    
    # By source
    by_source = defaultdict(int)
    for item in metadata:
        by_source[item['source']] += 1
    
    # Tumor presence
    with_tumor = sum(1 for item in metadata if item['has_tumor'])
    without_tumor = total - with_tumor
    
    # Mask availability
    with_mask = sum(1 for item in metadata if item['has_mask'])
    
    # Unique patients
    unique_patients = len(set(item['patient_id'] for item in metadata if item['patient_id'] != 'unknown'))
    
    stats = {
        "total_images": total,
        "by_source": dict(by_source),
        "tumor_distribution": {
            "with_tumor": with_tumor,
            "without_tumor": without_tumor,
            "tumor_percentage": round(with_tumor / total * 100, 2)
        },
        "mask_availability": {
            "with_mask": with_mask,
            "mask_percentage": round(with_mask / total * 100, 2)
        },
        "unique_patients": unique_patients
    }
    
    return stats


def print_summary(stats: dict, source_counts: dict):
    """Print merge summary."""
    print("\n" + "=" * 70)
    print("  MERGE COMPLETE - DATASET STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal Images: {stats['total_images']:,}")
    
    print(f"\nBy Source:")
    for source, count in stats['by_source'].items():
        pct = count / stats['total_images'] * 100
        print(f"   • {source:20s}: {count:6,} ({pct:5.1f}%)")
    
    print(f"\nTumor Distribution:")
    print(f"   • With tumor:    {stats['tumor_distribution']['with_tumor']:6,} ({stats['tumor_distribution']['tumor_percentage']:.1f}%)")
    print(f"   • Without tumor: {stats['tumor_distribution']['without_tumor']:6,}")
    
    print(f"\nMask Availability:")
    print(f"   • With mask:     {stats['mask_availability']['with_mask']:6,} ({stats['mask_availability']['mask_percentage']:.1f}%)")
    
    print(f"\nUnique Patients: {stats['unique_patients']}")
    
    print(f"\nOutput:")
    print(f"   📂 Images:    {OUT_IMG_DIR}/")
    print(f"   📂 Masks:     {OUT_MASK_DIR}/")
    print(f"   📄 Metadata:  {OUT_METADATA}")
    print(f"   📊 Stats:     {OUT_STATS}")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("   1. python dataset_scripts/split_data.py --source combined")
    print("   2. python model/train_model.py")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(merge_datasets())