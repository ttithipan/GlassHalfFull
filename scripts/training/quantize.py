"""
Export QAT-trained StudentDetectorV3 to ONNX and benchmark.

Uses the QAT-converted int8 model from training output.
Then applies ONNX Runtime INT8 static quantization for deployment.

Usage: venv/Scripts/python.exe scripts/training/quantize.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization as quant
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_model import StudentDetectorV2, StudentDetectorV3
from train import GlassDataset, detection_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_QAT = ROOT / "checkpoints" / "student_qat_int8.pt"
CHECKPOINT_BEST = ROOT / "checkpoints" / "student_best_v3.pt"
CHECKPOINT_V2 = ROOT / "checkpoints" / "student_best.pt"  # 1.66M params
EXPORT_DIR = ROOT / "onnx_models"
EXPORT_DIR.mkdir(exist_ok=True)

INPUT_SIZE = 256
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Export FP32 baseline
# ---------------------------------------------------------------------------


def export_fp32():
    print("=" * 60)
    print("1. Exporting FP32 ONNX (best checkpoint)...")

    model = StudentDetectorV3(NUM_CLASSES)
    model.load_state_dict(
        torch.load(str(CHECKPOINT_BEST), map_location="cpu", weights_only=True)
    )
    model.eval()

    path = str(EXPORT_DIR / "model_fp32_v3.onnx")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["out_s8", "out_s16", "out_s32"],
        opset_version=17,
    )
    kb = Path(path).stat().st_size / 1024
    print(f"  {kb:.0f} KB → {path}")
    return path, kb


# ---------------------------------------------------------------------------
# 2a. V2 FP16 (1.66M params → ~3.3 MB)
# ---------------------------------------------------------------------------


def export_v2_fp16():
    print("\n" + "=" * 60)
    print("2a. Exporting V2 FP16 ONNX (1.66M params)...")

    model = StudentDetectorV2(NUM_CLASSES)
    model.load_state_dict(
        torch.load(str(CHECKPOINT_V2), map_location="cpu", weights_only=True)
    )
    model.half()
    model.eval()

    path = str(EXPORT_DIR / "model_v2_fp16.onnx")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).half()

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["out_s16", "out_s32"],
        opset_version=17,
    )
    kb = Path(path).stat().st_size / 1024
    print(f"  {kb:.0f} KB → {path}")
    return path, kb


# ---------------------------------------------------------------------------
# 2b. V3 FP16 (2.5M params → ~5 MB)
# ---------------------------------------------------------------------------


def export_fp16():
    print("\n" + "=" * 60)
    print("2. Exporting FP16 ONNX (best checkpoint)...")

    model = StudentDetectorV3(NUM_CLASSES)
    model.load_state_dict(
        torch.load(str(CHECKPOINT_BEST), map_location="cpu", weights_only=True)
    )
    model.half()  # convert to float16
    model.eval()

    path = str(EXPORT_DIR / "model_fp16_v3.onnx")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).half()

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["out_s8", "out_s16", "out_s32"],
        opset_version=17,
    )
    kb = Path(path).stat().st_size / 1024
    print(f"  {kb:.0f} KB → {path}")
    return path, kb


# ---------------------------------------------------------------------------
# 3. Export QAT-converted INT8 model
# ---------------------------------------------------------------------------


def export_qat_int8():
    print("\n" + "=" * 60)
    print("3. Loading QAT-converted INT8 model...")

    model = StudentDetectorV3(NUM_CLASSES)
    model.load_state_dict(
        torch.load(str(CHECKPOINT_QAT), map_location="cpu", weights_only=True)
    )
    model.eval()

    path = str(EXPORT_DIR / "model_qat_int8_v3.onnx")
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["out_s8", "out_s16", "out_s32"],
        opset_version=17,
    )
    kb = Path(path).stat().st_size / 1024
    print(f"  {kb:.0f} KB → {path}")
    return path, kb


# ---------------------------------------------------------------------------
# 4. ONNX Runtime static INT8 quantization (post-processing)
# ---------------------------------------------------------------------------


def quantize_ort_static(fp32_path):
    print("\n" + "=" * 60)
    print("4. ORT INT8 static quantization (post-QAT polish)...")

    from onnxruntime.quantization import QuantType, quantize_static
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    class GlassCalibrationReader(CalibrationDataReader):
        def __init__(self, dataset, num_samples=100):
            self.dataset = dataset
            self.indices = list(range(min(num_samples, len(dataset))))
            self.iter_idx = 0

        def get_next(self):
            if self.iter_idx >= len(self.indices):
                return None
            img_tensor, _, _, _, _, _ = self.dataset[self.indices[self.iter_idx]]
            self.iter_idx += 1
            return {"input": img_tensor.unsqueeze(0).numpy()}

        def rewind(self):
            self.iter_idx = 0

    test_ds = GlassDataset(
        ROOT / "dataset" / "test" / "images",
        ROOT / "dataset" / "test" / "labels",
        augment=False,
    )
    reader = GlassCalibrationReader(test_ds, num_samples=100)

    path = str(EXPORT_DIR / "model_ort_int8_v3.onnx")
    quantize_static(
        model_input=fp32_path,
        model_output=path,
        calibration_data_reader=reader,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
    )
    kb = Path(path).stat().st_size / 1024
    print(f"  {kb:.0f} KB → {path}")
    return path, kb


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark_loss(onnx_path, num_samples=200):
    import onnxruntime as ort

    test_ds = GlassDataset(
        ROOT / "dataset" / "test" / "images",
        ROOT / "dataset" / "test" / "labels",
        augment=False,
    )
    indices = np.random.choice(
        len(test_ds), min(num_samples, len(test_ds)), replace=False
    )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    total_loss = 0.0
    times = []

    for idx in indices:
        img_tensor, t8b, t8c, t16b, t16c, t32b, t32c = test_ds[idx]
        img_np = img_tensor.unsqueeze(0).numpy()

        t0 = time.perf_counter()
        outputs = session.run(None, {"input": img_np})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

        out_s8 = torch.from_numpy(outputs[0])
        out_s16 = torch.from_numpy(outputs[1])
        out_s32 = torch.from_numpy(outputs[2])
        loss = detection_loss(
            out_s8,
            out_s16,
            out_s32,
            t8b.unsqueeze(0),
            t8c.unsqueeze(0),
            t16b.unsqueeze(0),
            t16c.unsqueeze(0),
            t32b.unsqueeze(0),
            t32c.unsqueeze(0),
        )
        total_loss += loss.item()

    return total_loss / len(indices), np.mean(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"QAT checkpoint: {CHECKPOINT_QAT}")
    print(f"Best V3:        {CHECKPOINT_BEST}")
    print(f"Best V2:        {CHECKPOINT_V2}")

    fp32_path, fp32_kb = export_fp32()
    v2_fp16_path, v2_fp16_kb = export_v2_fp16()
    fp16_path, fp16_kb = export_fp16()
    qat_path, qat_kb = export_qat_int8()
    ort_path, ort_kb = quantize_ort_static(fp32_path)

    print("\n" + "=" * 60)
    print("5. Benchmarking on test set (200 samples)...\n")

    # 2-scale loss for V2
    def loss_2scale(out_s16, out_s32, t16b, t16c, t32b, t32c):
        from train import scale_loss

        return scale_loss(out_s16, t16b, t16c) + scale_loss(out_s32, t32b, t32c)

    import onnxruntime as ort

    test_ds = GlassDataset(
        ROOT / "dataset" / "test" / "images",
        ROOT / "dataset" / "test" / "labels",
        augment=False,
    )

    results = {}
    for label, path, kb, is_v2 in [
        ("FP32 (v3)", fp32_path, fp32_kb, False),
        ("V2-FP16", v2_fp16_path, v2_fp16_kb, True),
        ("V3-FP16", fp16_path, fp16_kb, False),
        ("QAT-INT8", qat_path, qat_kb, False),
        ("ORT-INT8", ort_path, ort_kb, False),
    ]:
        indices = np.random.choice(len(test_ds), min(200, len(test_ds)), replace=False)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        total_loss, times = 0.0, []
        for idx in indices:
            img, t8b, t8c, t16b, t16c, t32b, t32c = test_ds[idx]
            img_np = img.unsqueeze(0).numpy()
            t0 = time.perf_counter()
            outputs = sess.run(None, {"input": img_np})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            b16 = t16b.unsqueeze(0)
            b32 = t32b.unsqueeze(0)
            c16 = t16c.unsqueeze(0)
            c32 = t32c.unsqueeze(0)
            if is_v2:
                loss = loss_2scale(
                    torch.from_numpy(outputs[0]),
                    torch.from_numpy(outputs[1]),
                    b16,
                    c16,
                    b32,
                    c32,
                )
            else:
                loss = detection_loss(
                    torch.from_numpy(outputs[0]),
                    torch.from_numpy(outputs[1]),
                    torch.from_numpy(outputs[2]),
                    t8b.unsqueeze(0),
                    t8c.unsqueeze(0),
                    b16,
                    c16,
                    b32,
                    c32,
                )
            total_loss += loss.item()
        avg_loss = total_loss / len(indices)
        avg_ms = np.mean(times)
        results[label] = {"size_kb": kb, "loss": avg_loss, "latency_ms": avg_ms}
        print(f"  {label:12s} | {kb:7.0f} KB | loss={avg_loss:.4f} | {avg_ms:.2f} ms")

    out = {
        "results": {
            k: {
                "size_kb": round(v["size_kb"], 1),
                "loss": round(v["loss"], 6),
                "latency_ms": round(v["latency_ms"], 2),
            }
            for k, v in results.items()
        }
    }
    (EXPORT_DIR / "benchmark_v3.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {EXPORT_DIR / 'benchmark_v3.json'}")

    print("\nBudget check (target: ≤ 3.0 MB):")
    for label, r in results.items():
        mb = r["size_kb"] / 1024
        ok = "✅" if mb <= 3.0 else "❌"
        print(f"  {label:12s}: {mb:.2f} MB {ok}")


if __name__ == "__main__":
    main()
