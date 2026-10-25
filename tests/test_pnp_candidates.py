"""多候选 PnP 与物理先验选解的回归测试。"""

import cv2
import numpy as np

from touch.perception.pose import (
    camera_center,
    planarity_ratio,
    select_candidate,
    solve_pnp_candidates,
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
