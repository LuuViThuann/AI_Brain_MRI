"""
preprocess_simple.py
Simple preprocessing for LGG dataset - works with existing folder structure.

Usage:
    python preprocess_simple.py
"""

import os
import sys
import csv
from PIL import Image
import numpy as np

# Get absolute paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == 'dataset_scripts' else CURRENT_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Output directories
IMG_OUT = os.path.join(DATA_DIR, "images")
MASK_OUT = os.path.join(DATA_DIR, "masks")
SUMMARY_CSV = os.path.join(DATA_DIR, "dataset_summary.csv")

# Settings
TARGET_SIZE = (256, 256)


def find_dataset():
    """Find the dataset in data/ directory."""
    print("=" * 60)
    print("  LGG MRI Segmentation — Simple Preprocessing")
    print("=" * 60)
    print(f"\n[1] Searching for dataset...")
    print(f"    Data directory: {DATA_DIR}\n")
    
    if not os.path.exists(DATA_DIR):
        print(f"[✗] Data directory not found: {DATA_DIR}")
        print("\n    Please create data/ folder first:")
        print(f"    mkdir {DATA_DIR}")
        sys.exit(1)
    
    # Look for folders with TCGA data
    candidates = []
    
    # Check all subdirectories in data/
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        
        # Skip if not a directory
        if not os.path.isdir(item_path):
            continue
        
        # Skip output directories
        if item in ['images', 'masks', 'train', 'val', 'test', 'raw']:
            continue
        
        # Check if this folder or its subdirectories contain TCGA folders
        tcga_folders = []
        
        # Check direct children
        try:
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path) and subitem.startswith('TCGA_'):
                    tcga_folders.append(subitem_path)
        except PermissionError:
            continue
        
        if len(tcga_folders) >= 10:  # Need at least 10 patient folders
            candidates.append((item_path, tcga_folders))
            print(f"    ✓ Found: {item}/ with {len(tcga_folders)} TCGA folders")
    
    if not candidates:
        print("\n[✗] No dataset found!")
        print("\n    Looking for folders like:")
        print("      data/kaggle_3m/TCGA_*/")
        print("      data/lgg-mri-segmentation/TCGA_*/")
        print("\n    Current contents of data/:")
        for item in os.listdir(DATA_DIR):
            item_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(item_path):
                print(f"      📂 {item}/")
            else:
                print(f"      📄 {item}")
        print("\n    Please extract the Kaggle dataset to data/")
        sys.exit(1)
    
    # Use the folder with most TCGA folders
    dataset_path, tcga_folders = max(candidates, key=lambda x: len(x[1]))
    
    print(f"\n[✓] Using dataset: {os.path.basename(dataset_path)}/")
    print(f"    Patient folders: {len(tcga_folders)}")
    
    return dataset_path, tcga_folders


def collect_image_pairs(tcga_folders):
    """Collect all image-mask pairs."""
    print(f"\n[2] Scanning {len(tcga_folders)} patient folders...")
    
    pairs = []
    
    for patient_folder in tcga_folders:
        patient_id = os.path.basename(patient_folder)
        
        try:
            files = os.listdir(patient_folder)
        except PermissionError:
            print(f"    ⚠ Cannot access {patient_id}")
            continue
        
        # Find all .tif images (not masks)
        for fname in files:
            if fname.endswith('.tif') and '_mask' not in fname:
                img_path = os.path.join(patient_folder, fname)
                mask_path = os.path.join(patient_folder, fname.replace('.tif', '_mask.tif'))
                
                if os.path.exists(mask_path):
                    # Extract slice number
                    slice_num = fname.split('_')[-1].replace('.tif', '')
                    pairs.append((img_path, mask_path, patient_id, slice_num))
    
    print(f"[✓] Found {len(pairs)} image-mask pairs\n")
    return pairs


def process_pairs(pairs):
    """Process and save all image pairs."""
    print(f"[3] Processing {len(pairs)} images...")
    
    # Create output directories
    os.makedirs(IMG_OUT, exist_ok=True)
    os.makedirs(MASK_OUT, exist_ok=True)
    
    summary_data = []
    success_count = 0
    error_count = 0
    
    for i, (img_path, mask_path, patient_id, slice_num) in enumerate(pairs):
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"    Processing {i + 1}/{len(pairs)}...")
        
        try:
            # Output filename
            out_filename = f"{patient_id}_{slice_num}.png"
            img_out_path = os.path.join(IMG_OUT, out_filename)
            mask_out_path = os.path.join(MASK_OUT, out_filename)
            
            # Load and process image
            img = Image.open(img_path).convert('L')
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(img_out_path, 'PNG')
            
            # Load and process mask
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize(TARGET_SIZE, Image.NEAREST)
            mask.save(mask_out_path, 'PNG')
            
            # Check if mask has tumor
            mask_arr = np.array(mask)
            has_tumor = np.any(mask_arr > 0)
            
            # Save metadata
            summary_data.append({
                'filename': out_filename,
                'patient_id': patient_id,
                'slice_num': slice_num,
                'has_tumor': '1' if has_tumor else '0',
                'original_img': os.path.basename(img_path),
                'original_mask': os.path.basename(mask_path)
            })
            
            success_count += 1
            
        except Exception as e:
            print(f"    ✗ Error processing {patient_id}_{slice_num}: {e}")
            error_count += 1
            continue
    
    print(f"\n[✓] Processing complete!")
    print(f"    Success: {success_count}")
    print(f"    Errors:  {error_count}\n")
    
    return summary_data


def save_summary(summary_data):
    """Save summary CSV."""
    print(f"[4] Saving summary CSV...")
    
    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'patient_id', 'slice_num', 'has_tumor',
            'original_img', 'original_mask'
        ])
        writer.writeheader()
        writer.writerows(summary_data)
    
    print(f"[✓] Saved to: {SUMMARY_CSV}\n")


def print_stats(summary_data):
    """Print dataset statistics."""
    total = len(summary_data)
    with_tumor = sum(1 for d in summary_data if d['has_tumor'] == '1')
    without_tumor = total - with_tumor
    unique_patients = len(set(d['patient_id'] for d in summary_data))
    
    print("=" * 60)
    print("  PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total images:      {total}")
    print(f"With tumor:        {with_tumor} ({with_tumor/total*100:.1f}%)")
    print(f"Without tumor:     {without_tumor} ({without_tumor/total*100:.1f}%)")
    print(f"Unique patients:   {unique_patients}")
    print(f"\nOutput:")
    print(f"  📂 Images: {IMG_OUT}/")
    print(f"  📂 Masks:  {MASK_OUT}/")
    print(f"  📄 CSV:    {SUMMARY_CSV}")
    print("=" * 60)
    print("\nNext step:")
    print("  python dataset_scripts/split_data.py")
    print("=" * 60)


def main():
    # Find dataset
    dataset_path, tcga_folders = find_dataset()
    
    # Collect pairs
    pairs = collect_image_pairs(tcga_folders)
    
    if len(pairs) == 0:
        print("[✗] No image-mask pairs found!")
        sys.exit(1)
    
    # Process all pairs
    summary_data = process_pairs(pairs)
    
    if len(summary_data) == 0:
        print("[✗] No images were processed successfully!")
        sys.exit(1)
    
    # Save summary
    save_summary(summary_data)
    
    # Print statistics
    print_stats(summary_data)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)