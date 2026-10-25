"""PnP：已知 3D 点和 2D 像素，求相机相对基座的位姿。"""

from __future__ import annotations

import cv2
import numpy as np


def solve_pnp(
    K: np.ndarray,
    points3d: np.ndarray,
    points2d: np.ndarray,
    reproj_threshold: float = 3.0,
    use_ransac: bool = True,
    method: int = cv2.SOLVEPNP_SQPNP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (R_cb, t_cb, inliers)。

    OpenCV 约定：X_cam = R_cb @ X_base + t_cb。
    默认用 SQPNP：迭代法在近共面点上会系统性翻转到镜像解。
    """
    obj = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(points2d, dtype=np.float64).reshape(-1, 2)
    if len(obj) < 4:
        raise ValueError("PnP needs at least 4 points for this solver")

    if use_ransac:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            K,
            None,
            flags=method,
            reprojectionError=reproj_threshold,
            iterationsCount=200,
            confidence=0.99,
        )
        if not ok:
            raise ValueError("solvePnPRansac failed")
        inliers = (
            np.arange(len(obj))
            if inliers is None
            else inliers.reshape(-1).astype(int)
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, None, flags=method
        )
        if not ok:
            raise ValueError("solvePnP failed")
        inliers = np.arange(len(obj))

    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.reshape(3).astype(np.float64), inliers


def planarity_ratio(points3d: np.ndarray) -> float:
    """3D 点集的平面度 = 最小/最大奇异值比，≈0 表示严格共面。"""
    p = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    s = np.linalg.svd(p - p.mean(axis=0), compute_uv=False)
    return float(s[2] / s[0]) if s[0] > 1e-12 else 0.0


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """光心在基座系下的位置 C = -R^T t。t 本身不是相机位置。"""
    return -np.asarray(R, np.float64).T @ np.asarray(t, np.float64)


def _pose_gap(a, b) -> float:
    """两位姿差异取「旋转角(deg)」与「光心距离(cm)」的较大者。"""
    ang = np.degrees(np.arccos(np.clip((np.trace(a[0].T @ b[0]) - 1) / 2, -1, 1)))
    gap_cm = np.linalg.norm(camera_center(a[0], a[1]) - camera_center(b[0], b[1])) * 100
    return max(float(ang), float(gap_cm))


def solve_pnp_candidates(
    K: np.ndarray,
    points3d: np.ndarray,
    points2d: np.ndarray,
    reproj_threshold: float = 3.0,
    max_candidates: int = 4,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """枚举候选位姿 [(R_cb, t_cb, inliers, rms_px), ...]，按 rms 升序。

    近共面点会有镜像双解；重投影 RMS 对镜像几乎不敏感，不能只用它选解。
    """
    obj = np.asarray(points3d, np.float64).reshape(-1, 3)
    img = np.asarray(points2d, np.float64).reshape(-1, 2)
    found = []
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP):
        try:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, None, flags=flag)
        except cv2.error:
            continue
        for rvec, tvec in zip(rvecs[:n], tvecs[:n]):
            uv, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            rms = float(np.sqrt(np.mean(np.sum((uv.reshape(-1, 2) - img) ** 2, axis=1))))
            R, _ = cv2.Rodrigues(rvec)
            found.append((R, np.asarray(tvec, np.float64).reshape(3), np.arange(len(obj)), rms))
    if not found:
        R, t, inl = solve_pnp(K, obj, img, reproj_threshold, use_ransac=False)
        return [(R, t, inl, float("nan"))]
    uniq: list = []
    for cand in sorted(found, key=lambda c: c[3]):
        if all(_pose_gap(cand, u) > 1.0 for u in uniq):
            uniq.append(cand)
    return uniq[:max_candidates]


def select_candidate(cands, z_min: float = 0.0, dist_cm: tuple = (15.0, 250.0)):
    """按物理先验选解：相机在桌面上方、与基座距离合理；先筛合法再取 RMS 最小。"""
    ok = []
    for c in cands:
        C = camera_center(c[0], c[1])
        if C[2] >= z_min and dist_cm[0] <= np.linalg.norm(C) * 100 <= dist_cm[1]:
            ok.append(c)
    return (ok or cands)[0]
