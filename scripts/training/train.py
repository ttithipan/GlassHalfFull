"""
Train StudentDetectorV2 — same architecture as the deployed model.
- 2 scales: stride 16 (16x16), stride 32 (8x8)
- MobileNetV3-Small backbone, ~1.66M params
- Seeded, augmentation, early stopping, TensorBoard, checkpoint resume

Usage:
  venv/Scripts/python.exe scripts/training/train.py              # fresh start
  venv/Scripts/python.exe scripts/training/train.py --resume     # resume
  tensorboard --logdir=runs
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_model import StudentDetectorV2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET = ROOT / "dataset"
TRAIN_IMG = DATASET / "train" / "images"
TRAIN_LBL = DATASET / "train" / "labels"
VAL_IMG = DATASET / "val" / "images"
VAL_LBL = DATASET / "val" / "labels"

BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
INPUT_SIZE = 256
NUM_CLASSES = 2
SEED = 42
EARLY_STOP_PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR = ROOT / "runs" / "glass_detector_v2"
STATE_PATH = CHECKPOINT_DIR / "training_state_v2.pt"
STATS_PATH = LOG_DIR.parent / "training_stats_v2.json"

GRID_S16 = 16
GRID_S32 = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------


class Augment:
    @staticmethod
    def apply(img):
        img = img.copy()
        if random.random() < 0.6:
            factor = random.uniform(0.7, 1.3)
            img = Image.eval(img, lambda x: min(255, max(0, int(x * factor))))
        if random.random() < 0.3:
            radius = random.uniform(0.3, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        if random.random() < 0.5:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        return img


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GlassDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, augment=False):
        self.img_files = sorted(Path(img_dir).glob("*.png"))
        self.lbl_dir = Path(lbl_dir)
        self.augment = augment

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        if self.augment:
            img = Augment.apply(img)
        img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img, dtype="float32") / 255.0).permute(
            2, 0, 1
        )

        lbl_path = self.lbl_dir / f"{self.img_files[idx].stem}.txt"
        boxes, classes = [], []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        boxes.append([x_c, y_c, w, h])
                        classes.append(cls_id)

        # Build targets for 2 scales (s16, s32)
        def make_targets(G):
            tb = torch.full((G, G, 4), -1.0)
            tc = torch.full((G, G), -1, dtype=torch.long)
            return tb, tc

        t16b, t16c = make_targets(GRID_S16)
        t32b, t32c = make_targets(GRID_S32)

        for bbox, cls_id in zip(boxes, classes):
            x_c, y_c, w, h = bbox
            # Assign to s32 grid
            gx32 = min(int(x_c * GRID_S32), GRID_S32 - 1)
            gy32 = min(int(y_c * GRID_S32), GRID_S32 - 1)
            t32b[gy32, gx32] = torch.tensor([x_c, y_c, w, h])
            t32c[gy32, gx32] = cls_id
            # s16: 2x2 sub-cells
            for dy in range(2):
                for dx in range(2):
                    gx16 = gx32 * 2 + dx
                    gy16 = gy32 * 2 + dy
                    if gx16 < GRID_S16 and gy16 < GRID_S16:
                        t16b[gy16, gx16] = torch.tensor([x_c, y_c, w, h])
                        t16c[gy16, gx16] = cls_id

        return img_tensor, t16b, t16c, t32b, t32c


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def giou_loss(pred_boxes, target_boxes):
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    tx1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    ty1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tx2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    ty2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    p_area = (px2 - px1) * (py2 - py1)
    t_area = (tx2 - tx1) * (ty2 - ty1)
    union = p_area + t_area - inter
    iou = inter / (union + 1e-7)
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)
    ex2 = torch.max(px2, tx2)
    ey2 = torch.max(py2, ty2)
    e_area = (ex2 - ex1) * (ey2 - ey1)
    giou = iou - (e_area - union) / (e_area + 1e-7)
    return (1.0 - giou).mean()


def scale_loss(pred, t_bbox, t_cls):
    B, _, G, _ = pred.shape
    pred_bbox = pred[:, :4].permute(0, 2, 3, 1)
    pred_cls = pred[:, 4:].permute(0, 2, 3, 1)
    obj_mask = t_cls >= 0
    noobj_mask = t_cls < 0
    loss = 0.0
    if obj_mask.sum() > 0:
        loss += giou_loss(pred_bbox[obj_mask], t_bbox[obj_mask])
        loss += nn.functional.cross_entropy(pred_cls[obj_mask], t_cls[obj_mask])
    if noobj_mask.sum() > 0:
        loss += 0.5 * torch.mean(torch.sigmoid(pred_cls[noobj_mask]) ** 2)
    return loss


def detection_loss(out_s16, out_s32, t16b, t16c, t32b, t32c):
    return scale_loss(out_s16, t16b, t16c) + scale_loss(out_s32, t32b, t32c)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_state(
    path, model, optimizer, scheduler, epoch, best_val, best_epoch, patience_counter
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "rng_state": random.getstate(),
            "np_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state()
            if torch.cuda.is_available()
            else None,
        },
        str(path),
    )


def load_state(path, model, optimizer, scheduler, device):
    ck = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    try:
        random.setstate(ck["rng_state"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
    except Exception:
        pass  # RNG state format varies across PyTorch versions
    return ck["epoch"], ck["best_val"], ck["best_epoch"], ck["patience_counter"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(resume=False):
    print(f"Seed: {SEED} | Device: {DEVICE} | Architecture: V2 (2-scale)")
    print(f"Scales: {GRID_S16}x{GRID_S16} + {GRID_S32}x{GRID_S32}")

    train_ds = GlassDataset(TRAIN_IMG, TRAIN_LBL, augment=True)
    val_ds = GlassDataset(VAL_IMG, VAL_LBL, augment=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, pin_memory=True)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = StudentDetectorV2(NUM_CLASSES).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    start_epoch = 0
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    if resume and STATE_PATH.exists():
        print(f"Resuming from {STATE_PATH}")
        start_epoch, best_val, best_epoch, patience_counter = load_state(
            STATE_PATH, model, optimizer, scheduler, DEVICE
        )
        print(
            f"  epoch={start_epoch}  best_val={best_val:.4f}@{best_epoch}  "
            f"patience={patience_counter}/{EARLY_STOP_PATIENCE}"
        )

    stats = []
    if STATS_PATH.exists():
        try:
            stats = json.loads(STATS_PATH.read_text())
        except Exception:
            pass

    writer = SummaryWriter(log_dir=str(LOG_DIR))
    if not resume:
        try:
            writer.add_graph(model, torch.randn(1, 3, 256, 256).to(DEVICE))
        except Exception:
            pass
    print(f"TensorBoard: tensorboard --logdir={LOG_DIR.parent}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, t16b, t16c, t32b, t32c in train_loader:
            imgs = imgs.to(DEVICE)
            t16b, t16c = t16b.to(DEVICE), t16c.to(DEVICE)
            t32b, t32c = t32b.to(DEVICE), t32c.to(DEVICE)

            out_s16, out_s32 = model(imgs)
            loss = detection_loss(out_s16, out_s32, t16b, t16c, t32b, t32c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, t16b, t16c, t32b, t32c in val_loader:
                imgs = imgs.to(DEVICE)
                t16b, t16c = t16b.to(DEVICE), t16c.to(DEVICE)
                t32b, t32c = t32b.to(DEVICE), t32c.to(DEVICE)
                out_s16, out_s32 = model(imgs)
                loss = detection_loss(out_s16, out_s32, t16b, t16c, t32b, t32c)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("LR", lr_now, epoch + 1)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), str(CHECKPOINT_DIR / "student_best.pt"))
            marker = " ✓ saved"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={lr_now:.6f} | best={best_val:.4f}@{best_epoch}{marker}"
        )

        stats.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": round(lr_now, 8),
                "best_val": round(best_val, 6),
                "best_epoch": best_epoch,
            }
        )
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(json.dumps(stats, indent=2))

        save_state(
            STATE_PATH,
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_val,
            best_epoch,
            patience_counter,
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    writer.close()
    torch.save(model.state_dict(), str(CHECKPOINT_DIR / "student_final.pt"))
    print(f"\nDone → best val={best_val:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    train(resume=resume)
