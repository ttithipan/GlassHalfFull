"""
Statistical dataset integrity check.
Run with: venv/Scripts/python.exe scripts/blender/check_dataset.py
"""

import json
from collections import Counter
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"


def check_labels():
    labels_dir = DATASET_DIR / "labels"
    images_dir = DATASET_DIR / "images"

    label_files = sorted(labels_dir.glob("*.txt"))
    image_files = sorted(images_dir.glob("*.png"))

    print(f"Image count: {len(image_files)}")
    print(f"Label count: {len(label_files)}")
    print()

    if not label_files:
        print("ERROR: No labels found!")
        return

    # Check pairing
    label_stems = {f.stem for f in label_files}
    image_stems = {f.stem for f in image_files}
    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images have no labels")
    if missing_images:
        print(f"WARNING: {len(missing_images)} labels have no images")
    print()

    # Analyze labels
    fill_ratios = []
    cls_counts = Counter()
    bbox_out_of_bounds = 0
    zero_area = 0
    glass_only = 0
    liquid_only = 0
    both_present = 0
    neither = 0

    for lf in label_files:
        glass_bbox = None
        liquid_bbox = None

        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                cls_counts[cls_id] += 1

                # Validate range
                if not (
                    0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 <= w <= 1 and 0 <= h <= 1
                ):
                    bbox_out_of_bounds += 1
                if w <= 0 or h <= 0:
                    zero_area += 1

                bbox = (x_c, y_c, w, h)
                if cls_id == 0:
                    glass_bbox = bbox
                elif cls_id == 1:
                    liquid_bbox = bbox

        # Class presence
        if glass_bbox and liquid_bbox:
            both_present += 1
            fill_ratios.append(liquid_bbox[3] / glass_bbox[3])
        elif glass_bbox:
            glass_only += 1
        elif liquid_bbox:
            liquid_only += 1
        else:
            neither += 1

    # Print stats
    print("=== Class Distribution ===")
    print(f"  Glass (cls 0):  {cls_counts.get(0, 0)} instances")
    print(f"  Liquid (cls 1): {cls_counts.get(1, 0)} instances")
    print()

    print("=== Per-Frame Presence ===")
    total = len(label_files)
    print(f"  Both:    {both_present} ({100 * both_present / total:.1f}%)")
    print(f"  Glass only: {glass_only} ({100 * glass_only / total:.1f}%)")
    print(f"  Liquid only: {liquid_only} ({100 * liquid_only / total:.1f}%)")
    print(f"  Neither: {neither} ({100 * neither / total:.1f}%)")
    print()

    print("=== Bbox Quality ===")
    print(f"  Out of bounds: {bbox_out_of_bounds}")
    print(f"  Zero area:     {zero_area}")
    print()

    if fill_ratios:
        fr = sorted(fill_ratios)
        print(f"=== Fill Ratio Distribution (n={len(fr)}) ===")
        print(f"  Min:    {fr[0]:.3f}")
        print(f"  10th:   {fr[len(fr) // 10]:.3f}")
        print(f"  25th:   {fr[len(fr) // 4]:.3f}")
        print(f"  Median: {fr[len(fr) // 2]:.3f}")
        print(f"  75th:   {fr[3 * len(fr) // 4]:.3f}")
        print(f"  90th:   {fr[9 * len(fr) // 10]:.3f}")
        print(f"  Max:    {fr[-1]:.3f}")

        # Count by verdict category
        verdicts = Counter()
        for r in fill_ratios:
            if r < 0.15:
                verdicts["Void"] += 1
            elif r < 0.40:
                verdicts["Pessimistic"] += 1
            elif r <= 0.60:
                verdicts["Philosophy Zone"] += 1
            elif r <= 0.85:
                verdicts["Optimistic"] += 1
            else:
                verdicts["Hubris"] += 1

        print()
        print("=== Verdict Breakdown ===")
        for k in ["Void", "Pessimistic", "Philosophy Zone", "Optimistic", "Hubris"]:
            v = verdicts.get(k, 0)
            print(f"  {k:20s}: {v:4d} ({100 * v / len(fr):5.1f}%)")

    # Check file sizes
    sizes = [f.stat().st_size for f in image_files]
    if sizes:
        print()
        print("=== Image File Sizes ===")
        print(f"  Min:    {min(sizes):,} bytes")
        print(f"  Max:    {max(sizes):,} bytes")
        print(f"  Avg:    {sum(sizes) // len(sizes):,} bytes")
        print(f"  Total:  {sum(sizes):,} bytes ({sum(sizes) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    check_labels()
