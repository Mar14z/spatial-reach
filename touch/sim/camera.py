"""可任意摆放的虚拟相机：内参、外参与投影。"""

from __future__ import annotations

import math

import numpy as np


def intrinsics_from_fov(
    width: int, height: int, fov_y_deg: float
) -> np.ndarray:
    """3x3 针孔内参；FOV 取垂直，方形像素，主点在图像中心。"""
    f = (height / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)
    return np.array(
        [
            [f, 0.0, width / 2.0],
            [0.0, f, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def project_camera_point(
    K: np.ndarray, p_cam: np.ndarray
) -> tuple[float, float]:
    """相机坐标 (X, Y, Z) → 像素 (u, v)；Z <= 0 抛 ValueError。"""
    x, y, z = (float(v) for v in p_cam)
    if z <= 0.0:
        raise ValueError("point is behind the camera (Z <= 0)")
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return u, v


def project_base_points(
    K: np.ndarray,
    R_cb: np.ndarray,
    t_cb: np.ndarray,
    points_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) 基座点 → (N,2) 像素；R_cb, t_cb 为 base→camera。

    返回 (uv, valid)：相机后方的点 uv 为 NaN，valid 为 False。
    """
    points = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    points_cam = points @ np.asarray(R_cb, dtype=np.float64).T + np.asarray(
        t_cb, dtype=np.float64
    )
    z = points_cam[:, 2]
    valid = z > 0.0
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = K[0, 0] * points_cam[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * points_cam[valid, 1] / z[valid] + K[1, 2]
    return uv, valid


def projection_from_intrinsics(
    K: np.ndarray,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 5.0,
) -> list[float]:
    """由相机 K 构造 PyBullet/OpenGL 列主序投影矩阵。"""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * fx / width
    P[0, 2] = 1.0 - 2.0 * cx / width
    P[1, 1] = 2.0 * fy / height
    P[1, 2] = 1.0 - 2.0 * cy / height
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P.flatten(order="F").tolist()
