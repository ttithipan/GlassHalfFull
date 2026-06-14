"""
Validate synthetic dataset by overlaying YOLO bounding boxes on sample images.
Run with: venv/Scripts/python.exe scripts/blender/validate_dataset.py
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"
OUTPUT_DIR = DATASET_DIR / "previews"
OUTPUT_DIR.mkdir(exist_ok=True)

# Colors: BGR-ish but PIL uses RGB — glass=blue, liquid=orange
COLORS = {
    0: (30, 100, 255),  # glass: blue
    1: (255, 140, 30),  # liquid: orange
}

CLASS_NAMES = {0: "glass", 1: "liquid"}


def draw_bboxes(image_path, label_path, output_path):
    """Read image + YOLO labels, draw bboxes, save preview."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    if not label_path.exists():
        print(f"  WARN: No label for {image_path.name}")
        img.save(output_path)
        return

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_center, y_center, bw, bh = map(float, parts[1:5])

            # YOLO normalized → pixel coords
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            color = COLORS.get(cls_id, (255, 255, 255))
            name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

            # Draw rectangle
            for offset in range(2):  # Thicker stroke
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=color,
                    width=1,
                )
            # Draw label
            draw.text((x1 + 2, y1 + 2), name, fill=color)

    img.save(output_path)
    print(f"  → {output_path.name}")


def main():
    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels"
    image_files = sorted(images_dir.glob("*.png"))

    print(f"Validating {len(image_files)} images...")
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        out_path = OUTPUT_DIR / f"preview_{img_path.stem}.png"
        draw_bboxes(img_path, label_path, out_path)

    print(f"\nPreviews saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
