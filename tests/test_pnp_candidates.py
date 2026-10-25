"""多候选 PnP 与物理先验选解的回归测试。"""

import cv2
import numpy as np

from touch.perception.pose import (
    camera_center,
    planarity_ratio,
    select_candidate,
    select_candidate_multiframe,
    solve_pnp_candidates,
    solve_pnp_multiframe,
)

K = np.array([[415.0, 0.0, 320.0], [0.0, 415.0, 240.0], [0.0, 0.0, 1.0]])
# 9 个严格共面的点，模拟贴在同一平面上的标记（臂平面）
PLANAR = np.array([[0.0, 0.0, 0.1], [0.05, 0.0, 0.1], [0.1, 0.0, 0.1],
                   [0.0, 0.05, 0.2], [0.05, 0.05, 0.2], [0.1, 0.05, 0.2],
                   [0.0, 0.1, 0.3], [0.05, 0.1, 0.3], [0.1, 0.1, 0.3]])
REFINE = np.array([0.0, 0.02, 0.05])  # 两个点离面 5cm，作为凸耳
GT_RVEC = np.array([1.2, 0.1, -0.4])
GT_TVEC = np.array([0.05, -0.04, 0.55])


def test_planarity_ratio_is_zero_for_exact_plane():
    assert planarity_ratio(PLANAR) < 1e-9
    lugs = PLANAR.copy()
    lugs[:2] += REFINE
    # 只要两个点离面，平面度就从 0 跳到非零；它不会很大，但不为零就已消除镜像等价
    assert planarity_ratio(lugs) > 0.01
    assert planarity_ratio(lugs) > 1e4 * max(planarity_ratio(PLANAR), 1e-12)


def test_candidates_contain_the_true_pose():
    uv, _ = cv2.projectPoints(PLANAR, GT_RVEC, GT_TVEC, K, None)
    cands = solve_pnp_candidates(K, PLANAR, uv.reshape(-1, 2))
    R_gt, _ = cv2.Rodrigues(GT_RVEC)
    gaps = [np.degrees(np.arccos(np.clip((np.trace(c[0].T @ R_gt) - 1) / 2, -1, 1)))
            for c in cands]
    assert len(cands) >= 1
    assert min(gaps) < 1.0
    assert min(c[3] for c in cands) < 0.5


def test_select_candidate_rejects_camera_below_table():
    # X_cam = R @ X_base + t，光心 C = -R^T t：t 不是相机位置，符号容易写反
    good = (np.eye(3), np.array([0.0, 0.0, -0.6]), np.arange(9), 0.30)
    below = (np.eye(3), np.array([0.0, 0.0, 0.5]), np.arange(9), 0.01)
    assert camera_center(*good[:2])[2] > 0 > camera_center(*below[:2])[2]
    picked = select_candidate([below, good])
    assert picked is good


def test_fallback_without_any_valid_candidate():
    below = (np.eye(3), np.array([0.0, 0.0, 0.5]), np.arange(9), 0.01)
    assert select_candidate([below]) is below


def test_solve_pnp_default_is_not_iterative():
    """默认求解器必须不是 ITERATIVE：它在近共面点上会系统性翻到镜像解。"""
    import inspect

    from touch.perception import pose

    sig = inspect.signature(pose.solve_pnp)
    assert sig.parameters["method"].default != cv2.SOLVEPNP_ITERATIVE
    assert sig.parameters["method"].default == cv2.SOLVEPNP_SQPNP


def _rz(deg: float) -> np.ndarray:
    """绕基座竖直轴转 pan。"""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0],
                     [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _planar_pair():
    """返回 (候选集, 真值旋转, 两解与真值的角度差)。"""
    uv, _ = cv2.projectPoints(PLANAR, GT_RVEC, GT_TVEC, K, None)
    cands = solve_pnp_candidates(K, PLANAR, uv.reshape(-1, 2))
    R_gt, _ = cv2.Rodrigues(GT_RVEC)
    gaps = [float(np.degrees(np.arccos(np.clip((np.trace(c[0].T @ R_gt) - 1) / 2, -1, 1))))
            for c in cands]
    return cands, R_gt, gaps


def test_planar_points_really_produce_a_mirror_pair():
    """前提验证：第二解在 2D 上只差几像素，在 3D 上却差几十度。"""
    cands, _, gaps = _planar_pair()
    assert max(gaps) > 10.0, "没有镜像双解，这个测试就失去意义"
    # 错的那个解在本帧的重投影误差也就几像素，肉眼和 RMS 都分不出来
    assert max(c[3] for c in cands[:2]) < 15.0


def test_multiframe_selects_true_pose_after_pan_turn():
    """臂平面转 30° 后，镜像解对不上新位置，跨帧残差把它票出去。"""
    cands, R_gt, _ = _planar_pair()
    P2 = PLANAR @ _rz(30.0).T                       # 机械臂转了 pan，相机没动
    uv2, _ = cv2.projectPoints(P2, GT_RVEC, GT_TVEC, K, None)

    best, scores = select_candidate_multiframe(K, cands, [(P2, uv2.reshape(-1, 2))])
    ang = float(np.degrees(np.arccos(np.clip((np.trace(best[0].T @ R_gt) - 1) / 2, -1, 1))))
    assert ang < 1.0
    assert scores.min() < 0.3 * scores.max()


def test_multiframe_joint_solve_beats_selection():
    """联合求解不局限于候选集：多初值出发，取联合残差最小的那个。"""
    cands, R_gt, gaps = _planar_pair()
    P2 = PLANAR @ _rz(30.0).T
    uv1, _ = cv2.projectPoints(PLANAR, GT_RVEC, GT_TVEC, K, None)
    uv2, _ = cv2.projectPoints(P2, GT_RVEC, GT_TVEC, K, None)
    obs = [(PLANAR, uv1.reshape(-1, 2)), (P2, uv2.reshape(-1, 2))]

    R, t, rms = solve_pnp_multiframe(K, obs, seeds=[c[:2] for c in cands])
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1, 1))))
    assert ang < 1.0
    assert rms < 0.5


def test_multiframe_joint_solve_needs_multiple_seeds():
    """单初值不可靠：从镜像解出发会陷在局部极小（所以调用方必须传全部候选）。"""
    cands, R_gt, gaps = _planar_pair()
    P2 = PLANAR @ _rz(30.0).T
    uv1, _ = cv2.projectPoints(PLANAR, GT_RVEC, GT_TVEC, K, None)
    uv2, _ = cv2.projectPoints(P2, GT_RVEC, GT_TVEC, K, None)
    obs = [(PLANAR, uv1.reshape(-1, 2)), (P2, uv2.reshape(-1, 2))]

    mirror = max(range(len(cands)), key=lambda i: gaps[i])
    R, _, _ = solve_pnp_multiframe(K, obs, seeds=[cands[mirror][:2]])
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1, 1))))
    assert ang > 10.0, "单初值居然收敛到了真解，说明这个测试的假设需要重写"
