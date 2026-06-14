"""
Split synthetic dataset into train/val/test splits (80/10/10).
Copies images and labels into split directories for YOLO training.

Usage: venv/Scripts/python.exe scripts/training/split_dataset.py
"""

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "dataset"
IMAGES = DATASET / "images"
LABELS = DATASET / "labels"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42

random.seed(SEED)


def main():
    image_files = sorted(IMAGES.glob("*.png"))
    n = len(image_files)
    print(f"Found {n} total images")

    indices = list(range(n))
    random.shuffle(indices)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    for split_name, split_indices in splits.items():
        img_dir = DATASET / split_name / "images"
        lbl_dir = DATASET / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in split_indices:
            img_file = image_files[idx]
            stem = img_file.stem
            lbl_file = LABELS / f"{stem}.txt"

            shutil.copy2(img_file, img_dir / img_file.name)
            if lbl_file.exists():
                shutil.copy2(lbl_file, lbl_dir / lbl_file.name)

        print(f"  {split_name}: {len(split_indices)} images → {img_dir}")

    # Write data.yaml for YOLO training
    yaml_path = DATASET / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {DATASET.as_posix()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")
        f.write("nc: 2\n")
        f.write("names: ['glass', 'liquid']\n")
    print(f"\nCreated {yaml_path}")


if __name__ == "__main__":
    main()
