"""
split_data.py
Splits preprocessed data into Train / Val / Test sets.
Stratified by tumor presence to ensure balanced splits.

Ratios: Train 70% | Val 15% | Test 15%

Usage:
    python dataset_scripts/split_data.py
"""

import os
import sys
import csv
import shutil
import random

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
IMG_DIR     = os.path.join(DATA_DIR, "images")
MASK_DIR    = os.path.join(DATA_DIR, "masks")
SUMMARY_CSV = os.path.join(DATA_DIR, "dataset_summary.csv")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

random.seed(SEED)


def load_summary() -> list:
    """Load dataset summary CSV."""
    with open(SUMMARY_CSV, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def stratified_split(data: list):
    """Split data stratified by has_tumor field."""
    tumor     = [d for d in data if d["has_tumor"] == "1"]
    no_tumor  = [d for d in data if d["has_tumor"] == "0"]

    random.shuffle(tumor)
    random.shuffle(no_tumor)

    def split_list(lst):
        n = len(lst)
        t = int(n * TRAIN_RATIO)
        v = int(n * VAL_RATIO)
        return lst[:t], lst[t:t+v], lst[t+v:]

    t_tumor, v_tumor, te_tumor       = split_list(tumor)
    t_no,    v_no,    te_no          = split_list(no_tumor)

    train = t_tumor  + t_no
    val   = v_tumor  + v_no
    test  = te_tumor + te_no

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def copy_split(split_data: list, split_name: str):
    """Copy images and masks into split directories."""
    img_out  = os.path.join(DATA_DIR, split_name, "images")
    mask_out = os.path.join(DATA_DIR, split_name, "masks")
    os.makedirs(img_out,  exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    for item in split_data:
        fname = item["filename"]
        shutil.copy(
            os.path.join(IMG_DIR,  fname),
            os.path.join(img_out,  fname)
        )
        shutil.copy(
            os.path.join(MASK_DIR, fname),
            os.path.join(mask_out, fname)
        )


def main():
    print("=" * 50)
    print(" LGG MRI — Train/Val/Test Split")
    print("=" * 50)

    if not os.path.exists(SUMMARY_CSV):
        print(f"[✗] Summary CSV not found. Run preprocess.py first.")
        sys.exit(1)

    print("[1/3] Loading summary...")
    data = load_summary()
    print(f"       Total samples: {len(data)}")

    print("[2/3] Stratified splitting...")
    train, val, test = stratified_split(data)
    print(f"       Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Tumor distribution check
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        t = sum(1 for d in split if d["has_tumor"] == "1")
        print(f"         {name}: {t} tumor / {len(split)-t} no-tumor")

    print("[3/3] Copying files...")
    copy_split(train, "train")
    copy_split(val,   "val")
    copy_split(test,  "test")

    print(f"\n[✓] Split complete!")
    print(f"       data/train/  → {len(train)} samples")
    print(f"       data/val/    → {len(val)} samples")
    print(f"       data/test/   → {len(test)} samples")
    print(f"\n    Run: python model/train_model.py")


if __name__ == "__main__":
    main()