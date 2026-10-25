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


def select_candidate_multiframe(
    K: np.ndarray,
    cands,
    observations,
) -> tuple[tuple, np.ndarray]:
    """跨帧仲裁：用其余各帧的重投影残差，在单帧候选里挑一个。

    相机不动、机械臂动 pan 时，每帧的 3D 点不同而相机位姿是同一个，
    镜像解只对原来那张平面成立，臂平面一转就对不上。

    但实测收益有限：它只能在候选里挑，而单帧错帧里多数是「候选集本身就
    偏离真值 20–30°」，挑不出不存在的东西（500 帧上 97.2–98.2%，与单帧
    持平甚至略低）。要真正的增益请用 solve_pnp_multiframe。

    observations: [(points3d_i, points2d_i), ...]，每帧各自的点集与观测。
    返回 (best, scores)，scores[i] 为第 i 个候选的跨帧平均 RMS(px)。
    """
    K = np.asarray(K, dtype=np.float64)
    scores = []
    for R, t in ((c[0], c[1]) for c in cands):
        rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
        per_frame = []
        for p3, p2 in observations:
            obj = np.asarray(p3, np.float64).reshape(-1, 3)
            img = np.asarray(p2, np.float64).reshape(-1, 2)
            uv, _ = cv2.projectPoints(obj, rvec, np.asarray(t, np.float64), K, None)
            per_frame.append(np.sum((uv.reshape(-1, 2) - img) ** 2, axis=1).mean())
        scores.append(float(np.sqrt(np.mean(per_frame))))
    scores = np.asarray(scores)
    return cands[int(np.argmin(scores))], scores


def solve_pnp_multiframe(
    K: np.ndarray,
    observations,
    seeds=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """多帧联合求同一个相机位姿，返回 (R_cb, t_cb, rms_px)。

    把各帧的 3D 点拼成一个点集，它们来自不同臂姿态，合并后不再共面：
    镜像解失去立足点，同时各帧的 2D 噪声被平均。seeds 给多个初值
    （一般就是单帧双解），取联合残差最小的那个——单初值会陷局部极小。

    2 帧（pan 转 5–30°）实测 98.2% → 99.0–99.8%，两帧全合成时 100%。
    """
    obj = np.vstack([np.asarray(p3, float).reshape(-1, 3) for p3, _ in observations])
    img = np.vstack([np.asarray(p2, float).reshape(-1, 2) for _, p2 in observations])
    if seeds is None:
        seeds = [solve_pnp(K, *observations[0], use_ransac=False)[:2]]

    best = None
    for R0, t0 in seeds:
        rvec, _ = cv2.Rodrigues(np.asarray(R0, float))
        tvec = np.asarray(t0, float).reshape(3)
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, None, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue
        uv, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        rms = float(np.sqrt(np.mean(np.sum((uv.reshape(-1, 2) - img) ** 2, axis=1))))
        if best is None or rms < best[2]:
            best = (cv2.Rodrigues(rvec)[0], tvec.reshape(3), rms)
    if best is None:
        raise ValueError("solve_pnp_multiframe failed")
    return best


def _reproj_rms(obj, img, K, rvec, tvec) -> float:
    uv, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    d = uv.reshape(-1, 2) - np.asarray(img, np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))


def solve_pnp_sequence(K, observations, seeds, reject_k: float = 3.0):
    """序列联合解：相机不动、各帧臂姿态不同，联合求同一个 (R_cb, t_cb)。

    臂一转 pan 各帧点集就不再共面，镜像解失去立足点；同时多帧 2D 噪声被
    平均。与 solve_pnp_multiframe 的区别是：seeds 给**全部单帧解**（单初值
    会陷局部极小），且解完按逐帧残差剔除离群帧再解一次——一帧预测崩了不
    该把整组带歪。

    observations: [(points3d_i, points2d_i), ...]，同一序列的每帧。
    seeds: [(R, t), ...]，候选初值。
    返回 (R_cb, t_cb, joint_rms_px, per_frame_rms)。
    """
    obs = [
        (np.asarray(p3, np.float64).reshape(-1, 3), np.asarray(p2, np.float64).reshape(-1, 2))
        for p3, p2 in observations
    ]
    keep = list(range(len(obs)))
    best = None
    per = np.full(len(obs), np.inf)
    while True:
        sub = [obs[i] for i in keep]
        obj = np.vstack([o[0] for o in sub])
        img = np.vstack([o[1] for o in sub])
        best = None
        for R0, t0 in seeds:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, None, cv2.Rodrigues(np.asarray(R0, np.float64))[0],
                np.asarray(t0, np.float64).reshape(3), True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                continue
            rms = _reproj_rms(obj, img, K, rvec, tvec)
            if best is None or rms < best[2]:
                best = (cv2.Rodrigues(rvec)[0], tvec.reshape(3), rms)
        if best is None:
            raise ValueError("solve_pnp_sequence failed")
        rr = cv2.Rodrigues(best[0])[0]
        per = np.array([_reproj_rms(o[0], o[1], K, rr, best[1]) for o in obs])
        if len(keep) <= 2:
            break
        worst = keep[int(np.argmax(per[keep]))]
        if per[worst] <= reject_k * float(np.median(per[keep])):
            break
        keep.remove(worst)
    return best[0], best[1], best[2], per
