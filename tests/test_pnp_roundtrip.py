import cv2
import numpy as np

from touch.perception.geometry import backproject_to_plane, pose_error
from touch.perception.pose import solve_pnp, solve_pnp_sequence

K = np.array([[420.0, 0.0, 320.0], [0.0, 415.0, 240.0], [0.0, 0.0, 1.0]])


def scene():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-0.1, -0.1, 0.05], [0.2, 0.1, 0.35], size=(9, 3))
    R_gt, _ = cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))
    t_gt = np.array([0.0, 0.0, 0.45])
    uv, _ = cv2.projectPoints(pts, cv2.Rodrigues(R_gt)[0], t_gt, K, None)
    return pts, uv.reshape(-1, 2), R_gt, t_gt


def test_pnp_roundtrip_gt_2d():
    pts, uv, R_gt, t_gt = scene()
    R, t, _ = solve_pnp(K, pts, uv, 3.0, False, cv2.SOLVEPNP_SQPNP)
    trans_cm, rot_deg = pose_error(R, t, R_gt, t_gt)
    assert trans_cm < 0.05, trans_cm
    assert rot_deg < 0.05, rot_deg


def test_backproject_to_table_plane():
    X = np.array([[0.2, -0.05, 0.0]])
    R_gt, _ = cv2.Rodrigues(np.array([0.2, -0.1, 0.0]))
    t_gt = np.array([0.0, 0.0, 0.45])
    uv, _ = cv2.projectPoints(X, cv2.Rodrigues(R_gt)[0], t_gt, K, None)
    est = backproject_to_plane(K, uv.reshape(-1, 2), R_gt, t_gt, 0.0)
    assert np.linalg.norm(est - X) < 1e-9


def test_sequence_solve_coplanar_points():
    """共面 9 点跨 4 个 pan 角联解：相机位姿应恢复到亚厘米、亚度。

    单帧解当种子。只验证联解的精度和管线，不预设单帧一定翻。
    """
    rng = np.random.default_rng(1)
    base = np.array([
        [0.00, 0.0, 0.00], [0.02, 0.0, 0.09], [0.02, 0.0, 0.20],
        [0.10, 0.0, 0.25], [0.15, 0.0, 0.31], [0.20, 0.0, 0.37],
        [0.22, 0.0, 0.42], [0.23, 0.0, 0.45], [0.24, 0.0, 0.47],
    ])
    R_gt, _ = cv2.Rodrigues(np.array([0.4, -0.3, 0.2]))
    t_gt = np.array([0.0, 0.0, 0.45])
    obs, seeds = [], []
    for pan in np.deg2rad([0.0, 10.0, 20.0, 30.0]):
        Rz, _ = cv2.Rodrigues(np.array([0.0, 0.0, pan]))
        p3 = base @ Rz.T
        uv, _ = cv2.projectPoints(p3, cv2.Rodrigues(R_gt)[0], t_gt, K, None)
        img = uv.reshape(-1, 2) + rng.normal(0.0, 1.0, (len(p3), 2))
        obs.append((p3, img))
        R1, t1, _ = solve_pnp(K, p3, img, 3.0, False, cv2.SOLVEPNP_SQPNP)
        seeds.append((R1, t1))
    R, t, _, _ = solve_pnp_sequence(K, obs, seeds)
    trans_cm, rot_deg = pose_error(R, t, R_gt, t_gt)
    assert trans_cm < 0.5, trans_cm
    assert rot_deg < 1.0, rot_deg
