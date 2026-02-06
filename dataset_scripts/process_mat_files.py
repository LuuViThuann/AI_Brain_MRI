"""
process_mat_files.py
Xử lý 766 file .mat từ brainTumorDataPublic_1-766/.

Format .mat:
  - Chứa 3D MRI volume
  - Có thể có mask
  - Cần trích xuất slices 2D quan trọng

Output:
  - data/processed/mat_images/
  - data/processed/mat_masks/
  - data/processed/mat_metadata.json
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import h5py  # For MATLAB v7.3 files
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "brainTumorDataPublic_1-766"
PROCESSED_DIR = DATA_DIR / "processed"

OUT_IMG_DIR = PROCESSED_DIR / "mat_images"
OUT_MASK_DIR = PROCESSED_DIR / "mat_masks"
METADATA_FILE = PROCESSED_DIR / "mat_metadata.json"

TARGET_SIZE = (256, 256)


def load_mat_file(mat_path: Path):
    """
    Load .mat file (MATLAB v7.3 format using HDF5) and extract MRI data.
    
    Returns:
        {
            'volume': np.array (3D or 2D),
            'mask': np.array or None,
            'metadata': dict
        }
    """
    try:
        with h5py.File(str(mat_path), 'r') as f:
            result = {
                'volume': None,
                'mask': None,
                'metadata': {}
            }
            
            # List all datasets in the file
            datasets = []
            
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append((name, obj))
            
            f.visititems(collect_datasets)
            
            if not datasets:
                return None
            
            # Find the main volume (largest dataset)
            datasets.sort(key=lambda x: x[1].size, reverse=True)
            
            # Get volume data
            volume_name, volume_dataset = datasets[0]
            volume_data = np.array(volume_dataset)
            
            # HDF5 stores data in different order, may need transpose
            # Check if data looks reasonable
            if volume_data.ndim >= 2:
                result['volume'] = volume_data
            else:
                return None
            
            # Look for mask in other datasets
            for name, dataset in datasets[1:]:
                name_lower = name.lower()
                if any(keyword in name_lower for keyword in ['mask', 'tumor', 'seg', 'label']):
                    result['mask'] = np.array(dataset)
                    break
            
            # Store metadata
            result['metadata'] = {
                'volume_key': volume_name,
                'volume_shape': volume_data.shape,
                'volume_dtype': str(volume_data.dtype),
                'n_datasets': len(datasets)
            }
            
            return result
            
    except Exception as e:
        raise Exception(f"Error loading {mat_path.name}: {e}")


def normalize_image(img_array: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range.
    """
    if img_array.size == 0:
        return np.zeros((256, 256), dtype=np.uint8)
    
    # Remove outliers (1st and 99th percentile)
    p1, p99 = np.percentile(img_array, [1, 99])
    img_clipped = np.clip(img_array, p1, p99)
    
    # Normalize to 0-255
    img_min = img_clipped.min()
    img_max = img_clipped.max()
    
    if img_max > img_min:
        img_norm = ((img_clipped - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_norm = np.zeros_like(img_clipped, dtype=np.uint8)
    
    return img_norm


def extract_important_slices(volume: np.ndarray, mask: np.ndarray = None, max_slices: int = 5):
    """
    Extract most important slices from 3D volume.
    
    Strategy:
      1. If mask available: slices with most tumor
      2. Otherwise: slices with most non-zero content
    
    Returns:
        List of (slice_idx, slice_2d, mask_2d or None)
    """
    # Handle 2D case
    if volume.ndim == 2:
        return [(0, volume, mask if mask is not None else None)]
    
    # Handle 3D case
    slices = []
    
    # MATLAB/HDF5 format is often (slices, height, width) or (height, width, slices)
    # Find the axis with smallest dimension (likely the slice axis)
    axis_sizes = volume.shape
    
    # Try different axes to find slices
    # Common formats: (512, 512, N) or (N, 512, 512) where N is number of slices
    if axis_sizes[0] < min(axis_sizes[1], axis_sizes[2]):
        # Format: (slices, height, width)
        axis = 0
        n_slices = axis_sizes[0]
    elif axis_sizes[2] < min(axis_sizes[0], axis_sizes[1]):
        # Format: (height, width, slices)
        axis = 2
        n_slices = axis_sizes[2]
    else:
        # Assume last axis
        axis = 2
        n_slices = axis_sizes[2]
    
    # Limit number of slices to process
    n_slices = min(n_slices, 100)
    
    # Score each slice
    scores = []
    for i in range(n_slices):
        try:
            if axis == 0:
                slice_2d = volume[i, :, :]
                mask_2d = mask[i, :, :] if mask is not None and mask.ndim == 3 else None
            elif axis == 1:
                slice_2d = volume[:, i, :]
                mask_2d = mask[:, i, :] if mask is not None and mask.ndim == 3 else None
            else:  # axis == 2
                slice_2d = volume[:, :, i]
                mask_2d = mask[:, :, i] if mask is not None and mask.ndim == 3 else None
            
            # Ensure 2D
            if slice_2d.ndim > 2:
                slice_2d = slice_2d.squeeze()
            if mask_2d is not None and mask_2d.ndim > 2:
                mask_2d = mask_2d.squeeze()
            
            # Skip if still not 2D
            if slice_2d.ndim != 2:
                continue
            
            # Score = amount of tumor (if mask) or non-zero content
            if mask_2d is not None and mask_2d.ndim == 2:
                score = np.sum(mask_2d > 0)
            else:
                # Score based on variance (more interesting slices have more variation)
                score = np.var(slice_2d)
            
            scores.append((i, score, slice_2d, mask_2d))
            
        except Exception as e:
            continue
    
    if not scores:
        return []
    
    # Sort by score, take top slices
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, score, slice_2d, mask_2d in scores[:max_slices]:
        if score > 0:  # Only include slices with content
            slices.append((i, slice_2d, mask_2d))
    
    return slices


def process_mat_file(mat_path: Path, patient_id: str):
    """
    Process a single .mat file.
    
    Returns:
        List of metadata dicts (one per extracted slice)
    """
    # Load .mat file
    mat_data = load_mat_file(mat_path)
    
    if mat_data is None or mat_data['volume'] is None:
        return []
    
    volume = mat_data['volume']
    mask = mat_data['mask']
    
    # Extract important slices
    slices = extract_important_slices(volume, mask, max_slices=5)
    
    if not slices:
        return []
    
    metadata_list = []
    
    for slice_idx, slice_2d, mask_2d in slices:
        try:
            # Normalize and convert to image
            img_norm = normalize_image(slice_2d)
            img = Image.fromarray(img_norm, mode='L')
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            
            # Save image
            filename = f"{patient_id}_slice{slice_idx:03d}.png"
            img_path = OUT_IMG_DIR / filename
            img.save(img_path, 'PNG')
            
            # Process mask if available
            has_mask = False
            has_tumor = False
            
            if mask_2d is not None and mask_2d.ndim == 2:
                try:
                    mask_norm = normalize_image(mask_2d)
                    mask_img = Image.fromarray(mask_norm, mode='L')
                    mask_img = mask_img.resize(TARGET_SIZE, Image.NEAREST)
                    
                    mask_path = OUT_MASK_DIR / filename
                    mask_img.save(mask_path, 'PNG')
                    
                    has_mask = True
                    has_tumor = np.sum(mask_norm > 0) > 100  # Threshold for tumor presence
                except:
                    pass
            
            # Store metadata (convert numpy types to Python types)
            metadata_list.append({
                "filename": filename,
                "patient_id": patient_id,
                "slice_index": int(slice_idx),
                "has_tumor": bool(has_tumor),
                "has_mask": bool(has_mask),
                "source": "mat",
                "volume_shape": [int(x) for x in volume.shape],
                "original_file": mat_path.name
            })
            
        except Exception as e:
            # Only print first few errors
            if len(metadata_list) < 3:
                print(f"   Error processing slice {slice_idx}: {e}")
            continue
    
    return metadata_list


def main():
    print("=" * 70)
    print("  .MAT Files Processor (Brain Tumor Dataset)")
    print("=" * 70)
    print()
    
    # Check if directory exists
    if not RAW_DIR.exists():
        print(f"❌ Directory not found: {RAW_DIR}")
        print(f"\n   Expected structure:")
        print(f"     data/raw/brainTumorDataPublic_1-766/")
        print(f"       ├── 1.mat")
        print(f"       ├── 2.mat")
        print(f"       └── ...")
        return 1
    
    # Create output directories
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all .mat files
    mat_files = list(RAW_DIR.glob("*.mat"))
    
    if not mat_files:
        print(f"❌ No .mat files found in {RAW_DIR}")
        return 1
    
    print(f"Found {len(mat_files)} .mat files")
    print(f"Processing directory: {RAW_DIR}")
    print()
    
    # Process all files
    all_metadata = []
    success_count = 0
    error_count = 0
    
    for mat_file in tqdm(mat_files, desc="Processing .mat files"):
        # Patient ID from filename
        patient_id = mat_file.stem
        
        try:
            metadata = process_mat_file(mat_file, patient_id)
            
            if metadata:
                all_metadata.extend(metadata)
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            # Only print first 5 errors to avoid spam
            if error_count < 5:
                print(f"\n❌ Error with {mat_file.name}: {e}")
            error_count += 1
            continue
    
    # Check if any data was processed
    if len(all_metadata) == 0:
        print("\n" + "=" * 70)
        print("  ⚠️  NO SLICES EXTRACTED")
        print("=" * 70)
        print(f"\nProcessed {len(mat_files)} files but extracted 0 slices.")
        print("Possible reasons:")
        print("  • .mat files may be empty or corrupted")
        print("  • Data format not recognized")
        print("  • All slices filtered out due to low content")
        print("\nRecommendations:")
        print("  • Check one .mat file manually with h5py or MATLAB")
        print("  • Verify dataset source and documentation")
        print("=" * 70)
        return 1
    
    # Save metadata
    print(f"\n[Saving metadata]...")
    with open(METADATA_FILE, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Statistics
    print("\n" + "=" * 70)
    print("  PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Files processed: {success_count} success / {error_count} errors")
    print(f"Total slices extracted: {len(all_metadata)}")
    
    # Unique patients
    unique_patients = len(set(item['patient_id'] for item in all_metadata))
    print(f"Unique patients: {unique_patients}")
    
    # Tumor distribution
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