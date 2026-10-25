import cv2
import numpy as np

from touch.perception.geometry import backproject_to_plane, pose_error
from touch.perception.pose import solve_pnp

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
