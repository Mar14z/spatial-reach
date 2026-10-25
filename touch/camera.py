"""真实相机：标定参数加载、UVC 采集与模型输入预处理。

这个模块只负责“相机”这一层：
- 从 `configs/camera_intrinsics.yaml` 读取 K / dist
- 打开 Insta360 Ace Pro 的 UVC 流
- 按需去畸变
- 把 16:9 画面中心裁成 4:3，并同步修正 K，供 640x480 的 keypoint 模型使用

注意：cv2 是延迟导入的，所以不装 OpenCV 的环境仍然可以 import 本模块，
但调用 load_intrinsics / AceProCamera 时需要 cv2。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = ROOT / "configs/camera_intrinsics.yaml"
DEFAULT_CONFIG = ROOT / "configs/camera_acepro.yaml"


@dataclass(frozen=True)
class CameraIntrinsics:
    """相机内参 + 畸变。K 为 3x3，dist 按 OpenCV 顺序。"""

    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    model: str = "pinhole"
    rms: float | None = None

    def scaled(self, sx: float, sy: float | None = None) -> "CameraIntrinsics":
        """按比例缩放 K；dist 是无量纲的，保持不变。"""
        sy = sx if sy is None else sy
        K = self.K.astype(np.float64).copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return replace(
            self,
            K=K,
            width=int(round(self.width * sx)),
            height=int(round(self.height * sy)),
        )

    def cropped(self, x0: int, y0: int, width: int, height: int) -> "CameraIntrinsics":
        """裁剪左上角在 (x0, y0) 的窗口，平移主点。"""
        K = self.K.astype(np.float64).copy()
        K[0, 2] -= x0
        K[1, 2] -= y0
        return replace(self, K=K, width=width, height=height)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_intrinsics(path: str | Path = DEFAULT_CALIBRATION) -> CameraIntrinsics:
    """读取 OpenCV FileStorage 格式的标定 YAML。"""
    import cv2

    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"calibration not found: {p}")
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
    try:
        K = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("distortion_coefficients").mat()
        width = int(fs.getNode("image_width").real())
        height = int(fs.getNode("image_height").real())
        model_node = fs.getNode("model")
        model = model_node.string() if not model_node.empty() else "pinhole"
        rms_node = fs.getNode("rms_reprojection_error")
        rms = float(rms_node.real()) if not rms_node.empty() else None
    finally:
        fs.release()

    if K is None or K.shape != (3, 3):
        raise ValueError(f"invalid camera_matrix in {p}")
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)
    return CameraIntrinsics(
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        width=width,
        height=height,
        model=model or "pinhole",
        rms=rms,
    )


def load_camera_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """读取 `configs/camera_acepro.yaml`（标准 YAML）。"""
    import yaml

    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"camera config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid camera config: {p}")
    return data


def undistort_frame(frame_rgb: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """对 RGB 帧去畸变，返回 RGB。"""
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if intr.model == "fisheye":
        out = cv2.fisheye.undistortImage(bgr, intr.K, intr.dist.reshape(4, 1))
    else:
        out = cv2.undistort(bgr, intr.K, intr.dist)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def prepare_model_input(
    frame_rgb: np.ndarray,
    intr: CameraIntrinsics,
    model_w: int = 640,
    model_h: int = 480,
) -> tuple[np.ndarray, CameraIntrinsics]:
    """把真实画面中心裁成模型宽高比，再缩放，并同步修正 K。

    训练数据的宽高比是 4:3（640x480），Ace Pro 输出是 16:9（1280x720）。
    直接 resize 会把人拉扁；这里先中心裁成 4:3，再缩放到模型输入尺寸。

    要求 frame_rgb 已经去畸变（或者 dist 全 0）；返回的 K 对应返回图像。
    """
    import cv2

    h, w = frame_rgb.shape[:2]
    if (w, h) != (intr.width, intr.height):
        raise ValueError(
            f"frame size {w}x{h} does not match intrinsics {intr.width}x{intr.height}"
        )

    target_aspect = model_w / model_h
    if w / h > target_aspect:
        crop_w = int(round(h * target_aspect))
        crop_h = h
        x0 = (w - crop_w) // 2
        y0 = 0
    else:
        crop_w = w
        crop_h = int(round(w / target_aspect))
        x0 = 0
        y0 = (h - crop_h) // 2

    cropped = frame_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    resized = cv2.resize(cropped, (model_w, model_h), interpolation=cv2.INTER_AREA)

    K = intr.K.astype(np.float64).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    sx, sy = model_w / crop_w, model_h / crop_h
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy

    return resized, replace(intr, K=K, width=model_w, height=model_h)




def model_intrinsics(
    intr: CameraIntrinsics,
    model_w: int = 640,
    model_h: int = 480,
) -> CameraIntrinsics:
    """只计算中心裁切 + 缩放后的 K，不处理图像。"""
    w, h = intr.width, intr.height
    target_aspect = model_w / model_h
    if w / h > target_aspect:
        crop_w = int(round(h * target_aspect))
        crop_h = h
        x0 = (w - crop_w) // 2
        y0 = 0
    else:
        crop_w = w
        crop_h = int(round(w / target_aspect))
        x0 = 0
        y0 = (h - crop_h) // 2
    K = intr.K.astype(np.float64).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    sx, sy = model_w / crop_w, model_h / crop_h
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return replace(intr, K=K, width=model_w, height=model_h)


class AceProCamera:
    """Insta360 Ace Pro 的 UVC 相机源，实现 CameraSource 协议。"""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        calibration_path: str | Path | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        backend: str | None = None,
        undistort: bool | None = None,
    ):
        import cv2

        self.config = load_camera_config(config_path)
        self.width = int(width or self.config["width"])
        self.height = int(height or self.config["height"])
        self.fps = int(fps or self.config.get("fps", 30))
        self.backend = backend or self.config.get("backend", "msmf")
        self.undistort = (
            bool(self.config.get("undistort", True))
            if undistort is None
            else bool(undistort)
        )

        cal = calibration_path or self.config.get(
            "calibration", "configs/camera_intrinsics.yaml"
        )
        self.intr = load_intrinsics(cal)
        if (self.width, self.height) != (self.intr.width, self.intr.height):
            sx = self.width / self.intr.width
            sy = self.height / self.intr.height
            if abs(sx - sy) > 1e-6:
                raise ValueError(
                    "requested resolution has a different aspect ratio from calibration; "
                    "recalibrate or crop explicitly"
                )
            self.intr = self.intr.scaled(sx)

        self._cv2 = cv2
        backend_id = cv2.CAP_MSMF if self.backend == "msmf" else cv2.CAP_DSHOW
        index = int(self.config.get("camera_index", 0))
        self.cap = cv2.VideoCapture(index, backend_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index} with {self.backend}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.backend == "dshow":
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (self.width, self.height):
            self.cap.release()
            raise RuntimeError(
                f"camera negotiated {actual_w}x{actual_h}, expected {self.width}x{self.height}"
            )

    def intrinsics(self) -> np.ndarray:
        """3x3 内参 K（对应 capture 返回的图像尺寸）。"""
        return self.intr.K.copy()

    def distortion(self) -> np.ndarray:
        """畸变系数。若 capture 已去畸变，PnP 时应传全 0。"""
        return self.intr.dist.copy()

    @property
    def is_undistorted(self) -> bool:
        return self.undistort

    def capture(self) -> np.ndarray:
        """返回一帧 RGB uint8，形状 (H, W, 3)。"""
        ok = False
        frame = None
        for _ in range(10):
            ok, frame = self.cap.read()
            if ok:
                break
            time.sleep(0.05)
        if not ok:
            raise RuntimeError("camera frame read failed")
        if self.undistort:
            frame = undistort_frame(
                self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB), self.intr
            )
        else:
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return frame

    def capture_model_input(
        self, model_w: int = 640, model_h: int = 480
    ) -> tuple[np.ndarray, CameraIntrinsics]:
        """返回 (RGB 模型输入, 对应的 K)。"""
        if not self.undistort and np.any(self.intr.dist):
            raise ValueError(
                "capture_model_input requires undistorted frames; "
                "set undistort=true in the camera config"
            )
        return prepare_model_input(self.capture(), self.intr, model_w, model_h)

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "AceProCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
