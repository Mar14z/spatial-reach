"""M0 合成帧读取与 keypoint heatmap 标签。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_heatmaps(
    keypoints: np.ndarray,
    width: int,
    height: int,
    stride: int = 4,
    sigma: float = 2.0,
) -> np.ndarray:
    """(N,2) 像素 keypoints → (N, H/stride, W/stride) 高斯热图。"""
    hm_w, hm_h = width // stride, height // stride
    ys, xs = np.mgrid[0:hm_h, 0:hm_w]
    heatmaps = np.zeros((len(keypoints), hm_h, hm_w), dtype=np.float32)
    for i, (u, v) in enumerate(keypoints):
        cx, cy = u / stride, v / stride
        heatmaps[i] = np.exp(
            -((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2)
        )
    return heatmaps


class M0Dataset:
    """读取 data/synthetic/m0/<frame_id>/ 下的一个样本。"""

    def __init__(
        self, root: str | Path, stride: int = 4, sigma: float = 2.0
    ):
        self.root = Path(root)
        self.dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        self.stride = stride
        self.sigma = sigma
        if not self.dirs:
            raise FileNotFoundError(f"no frames under {self.root}")

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, index: int) -> dict:
        frame_dir = self.dirs[index]
        meta = json.loads(
            (frame_dir / "meta.json").read_text(encoding="utf-8")
        )
        image = np.asarray(
            Image.open(frame_dir / "rgb.png").convert("RGB"),
            dtype=np.float32,
        ) / 255.0
        keypoints = np.asarray(meta["keypoints_2d"], dtype=np.float32)
        heatmaps = make_heatmaps(
            keypoints,
            meta["width"],
            meta["height"],
            self.stride,
            self.sigma,
        )
        return {
            "frame": frame_dir.name,
            "image": image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "K": np.asarray(meta["K"], dtype=np.float32),
            "R_cb": np.asarray(meta["R_cb"], dtype=np.float32),
            "t_cb": np.asarray(meta["t_cb"], dtype=np.float32),
        }
