"""
preprocess.py
Preprocess the LGG MRI Segmentation dataset from Kaggle.

Dataset structure (after extraction):
    data/raw/
        kaggle_3m/ (hoặc lgg-mri-segmentation/)
            TCGA_CS_4941_19960909/
                TCGA_CS_4941_19960909_1.tif
                TCGA_CS_4941_19960909_1_mask.tif
                ...
            TCGA_CS_4942_19970222/
                ...

Output structure:
    data/
        images/     ← All preprocessed MRI PNGs
        masks/      ← All corresponding mask PNGs
        dataset_summary.csv

Usage:
    python dataset_scripts/preprocess.py
"""

import os
import sys
import csv
from PIL import Image
from tqdm import tqdm
import numpy as np

# Determine paths dynamically
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent of dataset_scripts
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
IMG_OUT = os.path.join(DATA_DIR, "images")
MASK_OUT = os.path.join(DATA_DIR, "masks")
SUMMARY = os.path.join(DATA_DIR, "dataset_summary.csv")

# Image settings
TARGET_SIZE = (256, 256)
TARGET_MODE = "L"  # Grayscale


def find_kaggle_folder():
    """
    Find the extracted Kaggle dataset folder.
    Looks for folders containing TCGA_* patient directories.
    """
    if not os.path.exists(RAW_DIR):
        print(f"[✗] Raw data directory not found: {RAW_DIR}")
        print("    Run download_dataset.py first.")
        sys.exit(1)
    
    print(f"[1/5] Scanning: {RAW_DIR}")
    
    # Check if RAW_DIR itself contains TCGA folders
    direct_tcga = [d for d in os.listdir(RAW_DIR) if d.startswith("TCGA_") and os.path.isdir(os.path.join(RAW_DIR, d))]
    
    if len(direct_tcga) >= 10:  # At least 10 patient folders = likely the right location
        print(f"[✓] Found {len(direct_tcga)} TCGA patient folders directly in raw/")
        return RAW_DIR
    
    # Check subdirectories for TCGA folders
    candidates = []
    for item in os.listdir(RAW_DIR):
        item_path = os.path.join(RAW_DIR, item)
        if os.path.isdir(item_path):
            subdirs = os.listdir(item_path)
            tcga_count = sum(1 for d in subdirs if d.startswith("TCGA_") and os.path.isdir(os.path.join(item_path, d)))
            if tcga_count >= 10:
                candidates.append((item_path, tcga_count))
    
    if not candidates:
        print(f"\n[✗] No Kaggle dataset found in {RAW_DIR}")
        print("\n⚠️  Expected structure:")
        print("    data/raw/")
        print("      kaggle_3m/ (or similar)")
        print("        TCGA_*/")
        print("\n    Current contents of data/raw/:")
        for item in os.listdir(RAW_DIR):
            print(f"      {item}/")
        print("\n    Please check:")
        print("    1. Dataset was downloaded successfully")
        print("    2. ZIP was extracted properly")
        print("    3. TCGA_* folders exist somewhere in data/raw/")
        sys.exit(1)
    
    # Return folder with most TCGA directories
    best_folder, count = max(candidates, key=lambda x: x[1])
    print(f"[✓] Found dataset folder: {os.path.basename(best_folder)}")
    print(f"    Contains {count} patient folders")
    return best_folder


def get_all_image_pairs(kaggle_dir):
    """
    Scan all patient folders and collect (image, mask) pairs.
    
    Returns:
        List of tuples: (image_path, mask_path, patient_id, slice_num)
    """
    pairs = []
    
    patient_folders = sorted([
        d for d in os.listdir(kaggle_dir)
        if os.path.isdir(os.path.join(kaggle_dir, d)) and d.startswith("TCGA_")
    ])
    
    if not patient_folders:
        print(f"[✗] No TCGA_* patient folders found in {kaggle_dir}")
        sys.exit(1)
    
    print(f"\n[2/5] Scanning patient folders...")
    print(f"[✓] Found {len(patient_folders)} patient folders")
    
    for patient_id in tqdm(patient_folders, desc="Scanning patients"):
        patient_path = os.path.join(kaggle_dir, patient_id)
        
        # Get all .tif files (not masks)
        image_files = sorted([
            f for f in os.listdir(patient_path)
            if f.endswith('.tif') and '_mask' not in f
        ])
        
        for img_file in image_files:
            # Derive mask filename
            mask_file = img_file.replace('.tif', '_mask.tif')
            
            img_path = os.path.join(patient_path, img_file)
            mask_path = os.path.join(patient_path, mask_file)
            
            # Verify mask exists
            if os.path.exists(mask_path):
                # Extract slice number from filename
                # Example: TCGA_CS_4941_19960909_1.tif → 1
                slice_num = img_file.split('_')[-1].replace('.tif', '')
                pairs.append((img_path, mask_path, patient_id, slice_num))
            else:
                print(f"\n[!] Warning: Mask not found for {img_file}")
    
    return pairs


def preprocess_image(img_path: str, out_path: str) -> dict:
    """
    Load, resize, and save image as PNG.
    
    Returns:
        dict with stats: has_tumor (for masks)
    """
    img = Image.open(img_path)
    
    # Convert to grayscale if needed
    if img.mode != TARGET_MODE:
        img = img.convert(TARGET_MODE)
    
    # Resize
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    
    # Save as PNG
    img.save(out_path, "PNG")
    
    # Check if it's a mask and has tumor
    is_mask = '_mask' in os.path.basename(img_path)
    has_tumor = False
    
    if is_mask:
        arr = np.array(img)
        has_tumor = np.any(arr > 0)  # Any non-zero pixel = tumor present
    
    return {
        "has_tumor": has_tumor
    }


def main():
    print("=" * 60)
    print("  LGG MRI Segmentation — Data Preprocessing")
    print("=" * 60)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}\n")
    
    # Setup output directories
    os.makedirs(IMG_OUT, exist_ok=True)
    os.makedirs(MASK_OUT, exist_ok=True)
    
    # Find dataset
    kaggle_dir = find_kaggle_folder()
    
    # Collect all image pairs
    pairs = get_all_image_pairs(kaggle_dir)
    print(f"\n[3/5] Total image-mask pairs found: {len(pairs)}")
    
    if len(pairs) == 0:
        print("\n[✗] No valid image-mask pairs found!")
        print("    Please check the dataset structure.")
        sys.exit(1)
    
    # Process all images
    print("\n[4/5] Preprocessing images...")
    summary_data = []
    
    for img_path, mask_path, patient_id, slice_num in tqdm(pairs, desc="Processing"):
        # Generate output filename: {patient_id}_{slice_num}.png
        out_filename = f"{patient_id}_{slice_num}.png"
        
        img_out_path = os.path.join(IMG_OUT, out_filename)
        mask_out_path = os.path.join(MASK_OUT, out_filename)
        
        # Process image and mask
        try:
            img_stats = preprocess_image(img_path, img_out_path)
            mask_stats = preprocess_image(mask_path, mask_out_path)
            
            # Record summary
            summary_data.append({
                "filename": out_filename,
                "patient_id": patient_id,
                "slice_num": slice_num,
                "has_tumor": "1" if mask_stats["has_tumor"] else "0",
                "original_img": os.path.basename(img_path),
                "original_mask": os.path.basename(mask_path)
            })
        except Exception as e:
            print(f"\n[!] Error processing {out_filename}: {e}")
            continue
    
    # Save summary CSV
    print("\n[5/5] Saving summary CSV...")
    with open(SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "patient_id", "slice_num", "has_tumor",
            "original_img", "original_mask"
        ])
        writer.writeheader()
        writer.writerows(summary_data)
    
    # Statistics
    total = len(summary_data)
    with_tumor = sum(1 for d in summary_data if d["has_tumor"] == "1")
    without_tumor = total - with_tumor
    unique_patients = len(set(d["patient_id"] for d in summary_data))
    
    print("\n" + "=" * 60)
    print("[✓] Preprocessing complete!")
    print("=" * 60)
    print(f"Total images:      {total}")
    print(f"With tumor:        {with_tumor} ({with_tumor/total*100:.1f}%)")
    print(f"Without tumor:     {without_tumor} ({without_tumor/total*100:.1f}%)")
    print(f"Unique patients:   {unique_patients}")
    print(f"\nOutput directories:")
    print(f"  Images: {IMG_OUT}/")
    print(f"  Masks:  {MASK_OUT}/")
    print(f"  CSV:    {SUMMARY}")
    print("\n" + "=" * 60)
    print("Next step:")
    print("  python dataset_scripts/split_data.py")
    print("=" * 60)


if __name__ == "__main__":
    main()