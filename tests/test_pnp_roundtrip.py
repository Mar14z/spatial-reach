import cv2
import numpy as np
import pytest

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


@pytest.mark.parametrize("scores", [(5.0, 1.0), (1.0, 5.0)])
@pytest.mark.parametrize("return_solutions", [False, True])
def test_sequence_selects_lowest_rms(monkeypatch, scores, return_solutions):
    """固定候选、保留真实投影：两种顺序和返回格式都应选中 1px 解。"""
    p3 = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                   [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]])
    obs = []
    for points in (p3, p3 + [0.02, 0.0, 0.0]):
        uv, _ = cv2.projectPoints(points, np.zeros(3), np.array([0., 0., 1.]), K, None)
        obs.append((points, uv.reshape(-1, 2)))
    seeds = [(np.eye(3), np.array([score / K[0, 0], 0.0, 1.0])) for score in scores]

    def keep_seed(obj, img, intr, dist, rvec, tvec, use_guess, flags):
        assert use_guess and flags == cv2.SOLVEPNP_ITERATIVE
        return True, rvec.copy(), tvec.copy()

    monkeypatch.setattr(cv2, "solvePnP", keep_seed)
    result = solve_pnp_sequence(K, obs, seeds, return_solutions=return_solutions)
    R, t, rms, per_frame = result[:4]
    assert len(result) == (5 if return_solutions else 4)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(t, [1.0 / K[0, 0], 0.0, 1.0], atol=1e-12)
    assert rms == pytest.approx(1.0)
    np.testing.assert_allclose(per_frame, [1.0, 1.0], atol=1e-10)
    if return_solutions:
        sols = result[4]
        assert len(sols) == len(seeds)
        for (candidate_R, candidate_t, candidate_rms), (seed_R, seed_t), score in zip(
                sols, seeds, scores):
            np.testing.assert_allclose(candidate_R, seed_R, atol=1e-12)
            np.testing.assert_allclose(candidate_t, seed_t, atol=1e-12)
            assert candidate_rms == pytest.approx(score)


@pytest.mark.parametrize("bad", ["false", "exception", "rotation", "translation", "rms"])
@pytest.mark.parametrize("has_good", [False, True])
def test_sequence_skips_invalid_candidates(monkeypatch, bad, has_good):
    from touch.perception import pose
    pts, uv, R, t = scene()
    rv = cv2.Rodrigues(R)[0]
    count = 0

    def solver(*args, **kwargs):
        nonlocal count
        count += 1
        if count > 1:
            return True, rv.copy(), t.copy()
        if bad == "exception":
            raise cv2.error("无效候选")
        if bad == "false":
            return False, None, None
        return (True, rv * np.nan if bad == "rotation" else rv.copy(),
                np.full_like(t, np.inf) if bad == "translation" else t.copy())

    monkeypatch.setattr(cv2, "solvePnP", solver)
    reproj = pose._reproj_rms
    monkeypatch.setattr(pose, "_reproj_rms", lambda *a:
                        np.nan if bad == "rms" and count == 1 else reproj(*a))
    seeds = [(R, t)] * (2 if has_good else 1)
    if not has_good:
        with pytest.raises(ValueError, match="failed"):
            solve_pnp_sequence(K, [(pts, uv)] * 2, seeds)
    else:
        result = solve_pnp_sequence(K, [(pts, uv)] * 2, seeds, return_solutions=True)
        assert len(result[4]) == 1
        np.testing.assert_allclose(result[1], t)
        assert np.isfinite(result[2])


def test_sequence_equal_scores_and_empty_seeds(monkeypatch):
    pts, uv, R, t = scene()
    monkeypatch.setattr(cv2, "solvePnP", lambda *a, **kw: (True, a[4], a[5]))
    result = solve_pnp_sequence(K, [(pts, uv)] * 2, [(R, t), (R, t)], return_solutions=True)
    assert len(result[4]) == 2
    np.testing.assert_allclose(result[1], t)
    with pytest.raises(ValueError, match="failed"):
        solve_pnp_sequence(K, [(pts, uv)], [])
