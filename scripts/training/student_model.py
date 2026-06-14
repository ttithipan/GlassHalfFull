"""
Student Model v3: MobileNetV3-Small + FPN, QAT-ready, ~2.5M params.
- Stride 8 (32x32), stride 16 (16x16), stride 32 (8x8) detection
- QuantStub/DeQuantStub for QAT
- Increased channel widths for more capacity

After QAT training, use torch.ao.quantization.convert() to get int8 model.
"""

import torch
import torch.ao.quantization as quant
import torch.nn as nn
import torchvision


class DetectionHead(nn.Module):
    """QAT-compatible detection head."""

    def __init__(self, in_channels, num_classes=2, mid=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
        )
        self.bn = nn.BatchNorm2d(mid)
        self.relu = nn.ReLU(inplace=True)
        self.bbox = nn.Conv2d(mid, 4, 1)
        self.cls = nn.Conv2d(mid, num_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        bbox = torch.sigmoid(self.bbox(x))
        return torch.cat([bbox, self.cls(x)], dim=1)

    def fuse_model(self):
        """Fuse Conv+BN+ReLU for QAT conversion."""
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                torch.ao.quantization.fuse_modules(
                    self.conv, [["0", "1", "2"]], inplace=True
                )
                break
        torch.ao.quantization.fuse_modules(self, ["bn", "relu"], inplace=True)


class StudentDetectorV3(nn.Module):
    """3-scale detector with QAT support, ~2.5M params."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.quant = quant.QuantStub()
        self.dequant_s16 = quant.DeQuantStub()
        self.dequant_s32 = quant.DeQuantStub()

        backbone = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        feats = backbone.features

        # Split features at stride boundaries
        # indices: 0-3 → stride 8 (32x32, 24ch), 4-7 → stride 16 (16x16, 48ch),
        #           8-12 → stride 32 (8x8, 576ch)
        self.s8 = feats[:4]  # output: 24ch, 32x32
        self.s16 = feats[4:8]  # output: 48ch, 16x16
        self.s32 = feats[8:]  # output: 576ch, 8x8

        # FPN lateral connections
        self.lat_s16 = nn.Conv2d(576, 48, 1)  # s32 → s16
        self.lat_s8 = nn.Conv2d(48, 24, 1)  # s16 → s8

        # Detection heads (wider channels)
        self.head_s8 = DetectionHead(24, num_classes, mid=128)
        self.head_s16 = DetectionHead(48, num_classes, mid=160)
        self.head_s32 = DetectionHead(576, num_classes, mid=160)

    def forward(self, x):
        x = self.quant(x)

        f8 = self.s8(x)  # (B, 24, 32, 32)
        f16 = self.s16(f8)  # (B, 48, 16, 16)
        f32 = self.s32(f16)  # (B, 576, 8, 8)

        # FPN top-down
        f32_up = nn.functional.interpolate(
            self.lat_s16(f32), size=(16, 16), mode="bilinear", align_corners=False
        )
        f16_merged = f16 + f32_up

        f16_up = nn.functional.interpolate(
            self.lat_s8(f16_merged), size=(32, 32), mode="bilinear", align_corners=False
        )
        f8_merged = f8 + f16_up

        out_s8 = self.head_s8(f8_merged)
        out_s16 = self.head_s16(f16_merged)
        out_s32 = self.head_s32(f32)

        out_s16 = self.dequant_s16(out_s16)
        out_s32 = self.dequant_s32(out_s32)
        return out_s8, out_s16, out_s32

    def fuse_model(self):
        """Fuse Conv+BN+ReLU in all heads before QAT conversion."""
        self.head_s8.fuse_model()
        self.head_s16.fuse_model()
        self.head_s32.fuse_model()


# ---------------------------------------------------------------------------
# V2 model (kept for FP16 export comparison)
# ---------------------------------------------------------------------------


class StudentDetectorV2(nn.Module):
    """Original V2: 2-scale, 1.66M params, mid=96 heads (matches student_best.pt)."""

    class Head(nn.Module):
        def __init__(self, in_ch, nc=2):
            super().__init__()
            m = 96
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, m, 3, padding=1),
                nn.BatchNorm2d(m),
                nn.ReLU(inplace=True),
                nn.Conv2d(m, m, 3, padding=1),
                nn.BatchNorm2d(m),
                nn.ReLU(inplace=True),
            )
            self.bbox = nn.Conv2d(m, 4, 1)
            self.cls = nn.Conv2d(m, nc, 1)

        def forward(self, x):
            x = self.conv(x)
            return torch.cat([torch.sigmoid(self.bbox(x)), self.cls(x)], dim=1)

    def __init__(self, num_classes=2):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features  # load into 'features' first
        self.backbone_s16 = self.features[:8]
        self.backbone_s32 = self.features[8:]
        self.head_s16 = self.Head(48, num_classes)
        self.head_s32 = self.Head(576, num_classes)
        self.lateral = nn.Conv2d(576, 48, 1)

    def forward(self, x):
        f16 = self.backbone_s16(x)
        f32 = self.backbone_s32(f16)
        f32_up = nn.functional.interpolate(
            self.lateral(f32), size=(16, 16), mode="bilinear", align_corners=False
        )
        return self.head_s16(f16 + f32_up), self.head_s32(f32)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = StudentDetectorV3()
    n = count_parameters(model)
    print(f"StudentDetectorV3 params: {n:,}")
    dummy = torch.randn(1, 3, 256, 256)
    out_s8, out_s16, out_s32 = model(dummy)
    print(f"s8: {out_s8.shape}  s16: {out_s16.shape}  s32: {out_s32.shape}")
    print(f"Target: ~2.5M — {'OK' if 2_300_000 < n < 2_800_000 else 'ADJUST'}")
