"""
download_dataset.py
Downloads the LGG MRI Segmentation dataset from Kaggle.

Prerequisites:
    1. Install kaggle CLI:  pip install kaggle
    2. Place kaggle.json in ~/.kaggle/  (download from Kaggle account settings)

Usage:
    python dataset_scripts/download_dataset.py
"""

import os
import subprocess
import sys
import zipfile
import shutil

# Determine paths based on where script is run from
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # Parent of dataset_scripts
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

DATASET_OWNER = "mateuszbuda"
DATASET_NAME  = "lgg-mri-segmentation"
ZIP_FILE = os.path.join(RAW_DIR, f"{DATASET_NAME}.zip")


def check_kaggle():
    """Verify kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, shell=True
        )
        print(f"[✓] Kaggle CLI: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("[✗] Kaggle CLI not found.")
        print("\n⚠️  Install Kaggle CLI first:")
        print("    pip install kaggle")
        return False

    # Check for kaggle.json on Windows
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print(f"[✗] kaggle.json not found at {kaggle_json}")
        print("\n⚠️  Setup Kaggle API credentials:")
        print("    1. Go to https://www.kaggle.com/settings/api")
        print("    2. Click 'Create New Token' to download kaggle.json")
        print(f"    3. Move kaggle.json to: {kaggle_dir}")
        print(f"    4. Create folder if needed: mkdir {kaggle_dir}")
        return False
    
    print(f"[✓] kaggle.json found at: {kaggle_json}")
    return True


def download():
    """Download dataset zip from Kaggle."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(ZIP_FILE):
        print(f"\n[!] ZIP already exists at:")
        print(f"    {ZIP_FILE}")
        
        response = input("\n    Delete and re-download? (y/n): ").lower()
        if response != 'y':
            print("    Skipping download, using existing file.")
            return
        else:
            os.remove(ZIP_FILE)

    print(f"\n[⬇] Downloading {DATASET_OWNER}/{DATASET_NAME}...")
    print(f"    Target: {RAW_DIR}")
    
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", f"{DATASET_OWNER}/{DATASET_NAME}",
            "-p", RAW_DIR
        ], check=True, shell=True)
        print(f"\n[✓] Download complete: {ZIP_FILE}")
    except subprocess.CalledProcessError as e:
        print(f"\n[✗] Download failed: {e}")
        print("\n⚠️  Troubleshooting:")
        print("    1. Check internet connection")
        print("    2. Verify kaggle.json is correct")
        print("    3. Try manual download from:")
        print(f"       https://www.kaggle.com/datasets/{DATASET_OWNER}/{DATASET_NAME}")
        sys.exit(1)


def extract():
    """Extract ZIP into data/raw/."""
    if not os.path.exists(ZIP_FILE):
        print(f"[✗] ZIP file not found at: {ZIP_FILE}")
        print("    Run download step first or check if file exists.")
        sys.exit(1)

    print(f"\n[📦] Extracting {os.path.basename(ZIP_FILE)}...")
    print(f"    To: {RAW_DIR}")
    
    try:
        with zipfile.ZipFile(ZIP_FILE, "r") as z:
            z.extractall(RAW_DIR)
        print(f"[✓] Extraction complete!")
    except Exception as e:
        print(f"[✗] Extraction failed: {e}")
        sys.exit(1)

    # List extracted contents
    print("\n[📁] Extracted contents:")
    items = sorted(os.listdir(RAW_DIR))
    for item in items[:10]:  # Show first 10 items
        item_path = os.path.join(RAW_DIR, item)
        if os.path.isdir(item_path):
            print(f"    📂 {item}/")
        else:
            size_mb = os.path.getsize(item_path) / (1024 * 1024)
            print(f"    📄 {item} ({size_mb:.1f} MB)")
    
    if len(items) > 10:
        print(f"    ... and {len(items) - 10} more items")
    
    # Check for TCGA folders (patient data)
    tcga_folders = [f for f in os.listdir(RAW_DIR) if f.startswith('TCGA_')]
    
    # If no TCGA folders at root, check subdirectories
    if not tcga_folders:
        for subdir in os.listdir(RAW_DIR):
            subdir_path = os.path.join(RAW_DIR, subdir)
            if os.path.isdir(subdir_path):
                tcga_in_sub = [f for f in os.listdir(subdir_path) if f.startswith('TCGA_')]
                if tcga_in_sub:
                    print(f"\n[✓] Found {len(tcga_in_sub)} patient folders in: {subdir}/")
                    break
    else:
        print(f"\n[✓] Found {len(tcga_folders)} patient folders at root level")


def main():
    print("=" * 60)
    print("  LGG MRI Segmentation — Dataset Download")
    print("=" * 60)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Raw directory: {RAW_DIR}")
    print()
    
    # Check if Kaggle CLI is available
    if not check_kaggle():
        print("\n⚠️  Cannot proceed without Kaggle CLI setup.")
        print("    Please install and configure Kaggle first.")
        sys.exit(1)
    
    # Download
    download()
    
    # Extract
    extract()
    
    print("\n" + "=" * 60)
    print("[✓] Done! Next step:")
    print("    python dataset_scripts/preprocess.py")
    print("=" * 60)


if __name__ == "__main__":
    main()