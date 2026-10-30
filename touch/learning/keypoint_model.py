"""Keypoint heatmap 模型、损失与解码。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

HM_W, HM_H = 160, 120
INPUT_W, INPUT_H = 320, 240

# 与 build_keypoint_urdf.py 的 KEYPOINTS 顺序一致
KEYPOINT_NAMES = [
    "基座",
    "肩pan",
    "肩lift",
    "大臂中点",
    "肘",
    "小臂中点",
    "腕",
    "底座左点",
    "底座右点",
]


def normalize(images):
    return (images - 0.5) / 0.5


class TinyHeatmapNet(nn.Module):
    def __init__(
        self, n_keypoints: int = 6, out_size: tuple[int, int] = (120, 160)
    ):
        super().__init__()
        self.out_size = out_size
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(32, n_keypoints, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        d2 = F.interpolate(
            e3, size=e2.shape[-2:], mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(
            d2, size=e1.shape[-2:], mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = F.interpolate(
            d1, size=self.out_size, mode="bilinear", align_corners=False
        )
        return torch.sigmoid(self.head(d1))


class ResNetHeatmapNet(nn.Module):
    """ResNet-18 encoder + U-Net decoder，输出 6×240×320 heatmap。"""

    def __init__(
        self, n_keypoints: int = 6, out_size: tuple[int, int] = (240, 320)
    ):
        super().__init__()
        self.out_size = out_size
        backbone = models.resnet18(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.up3 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU()
        )
        self.head = nn.Conv2d(64, n_keypoints, 1)

    def forward(self, x):
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        d3 = F.interpolate(
            c4, size=c3.shape[-2:], mode="bilinear", align_corners=False
        )
        d3 = self.up3(torch.cat([d3, c3], dim=1))
        d2 = F.interpolate(
            d3, size=c2.shape[-2:], mode="bilinear", align_corners=False
        )
        d2 = self.up2(torch.cat([d2, c2], dim=1))
        d1 = F.interpolate(
            d2, size=c1.shape[-2:], mode="bilinear", align_corners=False
        )
        d1 = self.up1(torch.cat([d1, c1], dim=1))
        d1 = F.interpolate(
            d1, size=self.out_size, mode="bilinear", align_corners=False
        )
        return torch.sigmoid(self.head(d1))


class MobileNetHeatmapNet(nn.Module):
    """MobileNetV3-Large + DeepLabV3 头，输出 n_keypoints 张 heatmap。"""

    def __init__(
        self, n_keypoints: int = 6, out_size: tuple[int, int] = (120, 160)
    ):
        super().__init__()
        self.out_size = out_size
        self.net = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=n_keypoints,
            aux=False,
        )

    def forward(self, x):
        out = self.net(x)["out"]
        out = F.interpolate(
            out, size=self.out_size, mode="bilinear", align_corners=False
        )
        return torch.sigmoid(out)


def collate(batch):
    # permute 只改 strides，得到的是 NHWC 内存布局的 NCHW 张量；不补齐连续内存时
    # cuDNN 每步都要绕路，实测 320 输入 85 -> 244 img/s
    images = torch.from_numpy(
        np.ascontiguousarray(
            np.stack([b["image"] for b in batch]).transpose(0, 3, 1, 2)
        )
    )
    heatmaps = torch.from_numpy(np.stack([b["heatmaps"] for b in batch]))
    keypoints = torch.from_numpy(np.stack([b["keypoints"] for b in batch]))
    return images, heatmaps, keypoints


def heatmap_argmax(heatmaps):
    b, k, h, w = heatmaps.shape
    idx = heatmaps.reshape(b, k, -1).argmax(-1)
    y = (idx // w).float()
    x = (idx % w).float()
    return torch.stack([x, y], dim=-1)


def argmax_subpixel(heatmaps):
    """峰值附近三点抛物线拟合，返回热图坐标；原图缩放由调用端负责。

    仅对有双侧邻点且曲率为负的轴作修正，边界或平坦轴保留整格峰值。
    拟合用 float32；它不保证修复多峰歧义，真实数据收益需要单独评测。
    """
    if heatmaps.ndim != 4 or min(heatmaps.shape) < 1:
        raise ValueError("热图必须是非空的 B×K×H×W 张量")
    if not torch.isfinite(heatmaps).all():
        raise ValueError("热图包含非有限值")
    heatmaps = heatmaps.float()
    b, k, h, w = heatmaps.shape
    idx = heatmap_argmax(heatmaps)
    x = idx[..., 0].long()
    y = idx[..., 1].long()
    bi = torch.arange(b, device=heatmaps.device)[:, None]
    ki = torch.arange(k, device=heatmaps.device)[None, :]
    c = heatmaps[bi, ki, y, x]
    lx = heatmaps[bi, ki, y, (x - 1).clamp(min=0)]
    rx = heatmaps[bi, ki, y, (x + 1).clamp(max=w - 1)]
    ty = heatmaps[bi, ki, (y - 1).clamp(min=0), x]
    by = heatmaps[bi, ki, (y + 1).clamp(max=h - 1), x]

    def off(a, c0, d, interior):
        den = a - 2.0 * c0 + d
        valid = interior & (den < -1e-6)
        safe_den = torch.where(valid, den, torch.ones_like(den))
        delta = (0.5 * (a - d) / safe_den).clamp(-0.5, 0.5)
        return torch.where(valid, delta, torch.zeros_like(delta))

    return torch.stack(
        [x.float() + off(lx, c, rx, (x > 0) & (x < w - 1)),
         y.float() + off(ty, c, by, (y > 0) & (y < h - 1))], dim=-1
    )


def soft_argmax(heatmaps):
    b, k, h, w = heatmaps.shape
    xs = torch.arange(w, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.arange(h, device=heatmaps.device, dtype=heatmaps.dtype)
    prob = heatmaps / (heatmaps.sum(dim=(2, 3), keepdim=True) + 1e-6)
    ex = (prob.sum(dim=2) * xs).sum(dim=-1)
    ey = (prob.sum(dim=3) * ys).sum(dim=-1)
    return torch.stack([ex, ey], dim=-1)


def keypoint_loss(pred, target, coord_weight: float = 0.05):
    bce = F.binary_cross_entropy(
        pred.clamp(1e-6, 1.0 - 1e-6), target, reduction="none"
    )
    heat = (bce * (1.0 + 50.0 * target)).mean()
    coord = F.l1_loss(soft_argmax(pred), heatmap_argmax(target).float())
    return heat + coord_weight * coord
