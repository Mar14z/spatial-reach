"""几何工具：投影、反投影、射线-平面求交、位姿误差。

坐标约定统一为 OpenCV：X_cam = R_cb @ X_base + t_cb，R_cb/t_cb 由 PnP 解出。
"""

from __future__ import annotations

import numpy as np


def pixel_to_ray(K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """像素坐标 (N,2) → 相机系射线方向 (3,N)，未归一化（z 分量为 1）。"""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    rays = np.linalg.inv(np.asarray(K, dtype=np.float64)) @ np.hstack(
        [uv, np.ones((len(uv), 1))]
    ).T
    return rays / rays[2:3]


def ray_plane_intersect(
    origin: np.ndarray,
    dirs: np.ndarray,
    plane_normal: np.ndarray | None = None,
    plane_d: float = 0.0,
) -> np.ndarray:
    """射线 origin + s*dirs 与平面 n·X = d 求交，返回 (N,3)。"""
    n = np.array([0.0, 0.0, 1.0]) if plane_normal is None else np.asarray(plane_normal, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(3, 1)
    denom = n @ dirs
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (plane_d - float(n @ origin[:, 0])) / denom
    return (origin + dirs * s).T


def backproject_to_plane(
    K: np.ndarray,
    uv: np.ndarray,
    R_cb: np.ndarray,
    t_cb: np.ndarray,
    plane_z: float = 0.0,
) -> np.ndarray:
    """像素 → 基座系中 z=plane_z 平面上的 3D 点 (N,3)。

    射线在相机系由 K 给出，转到基座系后再与水平面求交。
    若某条射线与平面平行（分母≈0），返回 nan。
    """
    R = np.asarray(R_cb, dtype=np.float64)
    t = np.asarray(t_cb, dtype=np.float64).reshape(3)
    center = -R.T @ t
    dirs_base = R.T @ pixel_to_ray(K, uv)
    return ray_plane_intersect(center, dirs_base, np.array([0.0, 0.0, 1.0]), plane_z)


def pose_error(
    R_cb: np.ndarray, t_cb: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray
) -> tuple[float, float]:
    """返回 (相机光心平移误差 cm, 旋转误差 deg)。"""
    R = np.asarray(R_cb, dtype=np.float64)
    t = np.asarray(t_cb, dtype=np.float64).reshape(3)
    Rg = np.asarray(R_gt, dtype=np.float64)
    tg = np.asarray(t_gt, dtype=np.float64).reshape(3)
    trans_cm = 100.0 * float(np.linalg.norm(-R.T @ t - (-Rg.T @ tg)))
    cos = (np.trace(R.T @ Rg) - 1.0) / 2.0
    rot_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return trans_cm, rot_deg
