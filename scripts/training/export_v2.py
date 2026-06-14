"""Quick export: V2 → ONNX FP16 for frontend deployment."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_model import StudentDetectorV2

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = ROOT / "checkpoints" / "student_best.pt"
OUT = ROOT / "frontend" / "model.onnx"

model = StudentDetectorV2(2)
model.load_state_dict(torch.load(str(CKPT), map_location="cpu", weights_only=True))
model.eval()

dummy = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    model,
    dummy,
    str(OUT),
    input_names=["input"],
    output_names=["out_s16", "out_s32"],
    opset_version=17,
)

size_mb = OUT.stat().st_size / (1024 * 1024)
print(f"Exported → {OUT}  ({size_mb:.1f} MB)")
