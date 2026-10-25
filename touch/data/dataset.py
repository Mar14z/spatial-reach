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


def _pre_gray(x: np.ndarray) -> np.ndarray:
    g = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.repeat(g[..., None], 3, axis=2)


def _pre_gamma(x: np.ndarray, gain: float = 0.45) -> np.ndarray:
    return np.clip(x, 0.0, 1.0) ** gain


def _pre_blur(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            out += p[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return out / 9.0


def _pre_norm(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for c in range(x.shape[2]):
        ch = x[:, :, c]
        out[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-6) * 0.25 + 0.5
    return np.clip(out, 0.0, 1.0)


def _pre_equalize(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    n = x.shape[0] * x.shape[1]
    ramp = np.arange(n, dtype=np.float32) / (n - 1)
    for c in range(x.shape[2]):
        flat = x[:, :, c].ravel()
        ranks = np.empty(n, dtype=np.float32)
        ranks[np.argsort(flat, kind="stable")] = ramp
        out[:, :, c] = ranks.reshape(x.shape[0], x.shape[1])
    return out


PREPROCESS = {
    "none": lambda x: x,
    "gray": _pre_gray,
    "gamma": _pre_gamma,
    "blur": _pre_blur,
    "norm": _pre_norm,
    "equalize": _pre_equalize,
}


class M0Dataset:
    """读取 data/synthetic/m0/<frame_id>/ 下的一个样本。

    cache=True 时在初始化阶段把图像/热图读进内存，适合小数据集上喂满 GPU；
    此时建议 DataLoader 使用 num_workers=0，避免每个 worker 复制缓存。

    preprocess 选 PREPROCESS 里的键，在 /255 之后、缓存之前作用于图像。
    extra_labels 指向 make_finger_labels.py 生成的 npz，把额外的 2D 点追加到 keypoints。
    """

    def __init__(
        self,
        root: str | Path,
        stride: int = 4,
        sigma: float = 2.0,
        image_size: tuple[int, int] | None = None,
        cache: bool = False,
        preprocess: str = "none",
        extra_labels: str | Path | None = None,
    ):
        self.root = Path(root)
        self.dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        self.stride = stride
        self.sigma = sigma
        self.image_size = image_size
        self.preprocess = PREPROCESS[preprocess]
        self._extra_uv: dict[str, np.ndarray] = {}
        if extra_labels is not None:
            z = np.load(extra_labels)
            for name, uv in zip(z["frames"], z["uv"]):
                self._extra_uv[str(name)] = uv.astype(np.float32)
        if not self.dirs:
            raise FileNotFoundError(f"no frames under {self.root}")
        self.cache = cache
        self._images: list[np.ndarray] = []
        self._keypoints: list[np.ndarray] = []
        self._heatmaps: list[np.ndarray] = []
        self._K: list[np.ndarray] = []
        self._R: list[np.ndarray] = []
        self._t: list[np.ndarray] = []
        self._frames: list[str] = []
        if cache:
            self._load_cache()

    def _load_one(self, frame_dir: Path) -> dict:
        meta = json.loads(
            (frame_dir / "meta.json").read_text(encoding="utf-8")
        )
        image = Image.open(frame_dir / "rgb.png").convert("RGB")
        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.BILINEAR)
        image = self.preprocess(np.asarray(image, dtype=np.float32) / 255.0)
        keypoints = np.asarray(meta["keypoints_2d"], dtype=np.float32)
        if self._extra_uv:
            keypoints = np.vstack([keypoints, self._extra_uv[frame_dir.name]])
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

    def _load_cache(self) -> None:
        for frame_dir in self.dirs:
            sample = self._load_one(frame_dir)
            self._frames.append(sample["frame"])
            self._images.append(sample["image"])
            self._keypoints.append(sample["keypoints"])
            self._heatmaps.append(sample["heatmaps"].astype(np.float16))
            self._K.append(sample["K"])
            self._R.append(sample["R_cb"])
            self._t.append(sample["t_cb"])

    def __len__(self) -> int:
        return len(self.dirs)

    @property
    def n_keypoints(self) -> int:
        if self.cache:
            return len(self._keypoints[0])
        meta = json.loads(
            (self.dirs[0] / "meta.json").read_text(encoding="utf-8")
        )
        extra = len(next(iter(self._extra_uv.values()))) if self._extra_uv else 0
        return len(meta["keypoints_2d"]) + extra

    def __getitem__(self, index: int) -> dict:
        if self.cache:
            return {
                "frame": self._frames[index],
                "image": self._images[index],
                "keypoints": self._keypoints[index],
                "heatmaps": self._heatmaps[index].astype(np.float32),
                "K": self._K[index],
                "R_cb": self._R[index],
                "t_cb": self._t[index],
            }
        return self._load_one(self.dirs[index])
