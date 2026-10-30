"""两帧联合解的平移不确定度：把「标定帧间隔多少度最好」从扫参变成可算的量。

高斯近似：观测 = 各帧 3D 点在共享位姿 (R_cb, t_cb) 下的投影，
H = JᵀJ（像素噪声 σ=1px），C = H⁻¹，σ_trans = sqrt(tr C_tt)。
δ=0（单帧、9 点共面）时 H 奇异——这正是平面镜像病的来源，返回 inf。
"""
from __future__ import annotations

import cv2
import numpy as np


def rot_z(deg: float) -> np.ndarray:
    """pan 转动 = 臂上各点绕基座竖轴旋转（pan 轴过基座中心竖直线）。"""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose_sigma_1px(
    points3d: list, R_cb, t_cb, K, h: float = 1e-6
) -> tuple[float, float]:
    """返回 (σ_rot_deg, σ_trans_m)：1px 像素噪声经线性化传播后的位姿不确定度。

    对 6 维位姿参数 (rvec, tvec) 做数值差分得雅可比。注意：σ 只在
    线性化可信时有意义；共面病态方向上 H 接近奇异，σ 巨大——那正是
    线性化失效、出现镜像双峰的区域。σ 本身不是翻转率，是「病态探测器」。
    """
    frames = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in points3d]
    K = np.asarray(K, dtype=np.float64)
    r0 = cv2.Rodrigues(np.asarray(R_cb, dtype=np.float64))[0]
    t0 = np.asarray(t_cb, dtype=np.float64).reshape(3)

    def uv_all(rvec, t):
        return np.vstack(
            [cv2.projectPoints(P, rvec, t, K, None)[0].reshape(-1, 2) for P in frames]
        )

    uv0 = uv_all(r0, t0)
    J = np.zeros((uv0.size, 6))
    p0 = np.concatenate([r0.reshape(3), t0])
    for i in range(6):
        dp = np.zeros(6)
        dp[i] = h
        rvec = cv2.Rodrigues(p0[:3] + dp[:3])[0]
        J[:, i] = ((uv_all(rvec, p0[3:] + dp[3:]) - uv0) / h).reshape(-1)
    H = J.T @ J
    try:
        C = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return float("inf"), float("inf")
    rot = float(np.degrees(np.sqrt(np.trace(C[:3, :3]))))
    trans = float(np.sqrt(np.trace(C[3:6, 3:6])))
    return rot, trans
