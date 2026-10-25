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
    method: int = cv2.SOLVEPNP_ITERATIVE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (R_cb, t_cb, inliers)。

    OpenCV 约定：X_cam = R_cb @ X_base + t_cb。
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
