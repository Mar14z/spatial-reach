"""标一次 T：在线帧的位姿必须来自标定对，不能来自自己那帧的单帧解。"""

import json

import cv2
import numpy as np

from touch.learning.evaluate import PROTOCOL, apply_calibrate_once
from touch.perception.pose import solve_pnp

K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
R_GT = np.diag([1.0, -1.0, -1.0])  # 相机俯视桌面（绕 x 转 180°）
T_GT = np.array([0.0, 0.0, 0.8])
CFG = dict(PROTOCOL, image_hw=(480, 640))


def _seq_points(frame_in_seq, rng):
    """臂姿态不同 → 3D 点集不同；共用的只有相机位姿。"""
    ang = 0.17 * frame_in_seq
    c, s = np.cos(ang), np.sin(ang)
    base = np.c_[np.linspace(0.05, 0.25, 9), np.zeros(9), np.linspace(0.0, 0.30, 9)]
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return base @ rot.T


def _make(tmp_path, n_seq=2, n_frames=4, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for s in range(n_seq):
        for k in range(n_frames):
            p3 = _seq_points(k, rng)
            uv, _ = cv2.projectPoints(p3, cv2.Rodrigues(R_GT)[0], T_GT, K, None)
            p2 = uv.reshape(-1, 2) + rng.normal(0, 0.3, (9, 2))
            name = f"{s * n_frames + k:06d}"
            d = tmp_path / name
            d.mkdir()
            (d / "meta.json").write_text(json.dumps({
                "frame_in_seq": k, "sequence": s, "R_cb": R_GT.tolist(), "t_cb": T_GT.tolist(),
            }), encoding="utf-8")
            R1, t1, _ = solve_pnp(K, p3, p2, use_ransac=False)
            records.append({
                "frame": name, "pnp_ok": True, "_seq": s, "_k": k, "_K": K.tolist(),
                "_p3": p3.tolist(), "_p2": p2.tolist(),
                "_R": R1.ravel().tolist(), "_t": t1.ravel().tolist(),
                "trans_cm": 99.0, "rot_deg": 99.0, "flip": True, "table_cm": 99.0,
                "mean_px": 0.0, "px_vec": [0.0] * 9, "dv_vec": [0.0] * 9,
                "nme": 0.0, "spread2d": 1.0,
            })
    return records


def test_online_frames_use_calibrated_pose(tmp_path):
    records = _make(tmp_path)
    st = apply_calibrate_once(records, tmp_path, CFG, (0, 3))
    assert st["n_seq"] == 2 and st["n_online"] == 4
    for s in (0, 1):
        grp = [r for r in records if r["_seq"] == s]
        # 序列内所有帧共用同一个 T → 误差必须完全一致
        assert len({round(r["trans_cm"], 9) for r in grp}) == 1
        assert {r["_calib"] for r in grp} == {"calib", "online"}
        assert all(r["trans_cm"] < 1.0 for r in grp)
    assert st["n_within"] == st["n_seq"]


def test_short_sequence_is_skipped(tmp_path):
    records = _make(tmp_path, n_frames=2)
    st = apply_calibrate_once(records, tmp_path, CFG, (0, 3))
    assert st["n_seq"] == 0 and st["n_online"] == 0


def test_missing_calib_frame_is_skipped_not_shifted(tmp_path):
    """首帧缺失时只能显式跳过：按位次挪成 (1,3) 会把 30° 基线降成 20°。"""
    records = [r for r in _make(tmp_path) if not (r["_seq"] == 0 and r["_k"] == 0)]
    st = apply_calibrate_once(records, tmp_path, CFG, (0, 3))
    assert st["n_skipped"] == 1 and st["n_seq"] == 1
    assert sum(1 for r in records if "_calib" in r) == 4  # 只处理了完整的那个序列


def test_joint_failure_is_counted_and_old_pose_cleared(tmp_path, monkeypatch):
    from touch.learning import evaluate
    records = _make(tmp_path)
    evaluate.apply_calibrate_once(records, tmp_path, CFG)

    def fail(*a, **kw):
        raise ValueError("没有有效候选")

    monkeypatch.setattr(evaluate, "solve_pnp_sequence", fail)
    st = evaluate.apply_calibrate_once(records, tmp_path, CFG)
    assert st["n_planned"] == st["n_failed"] == 2
    assert st["n_seq"] == 0
    assert all("_Rc" not in r and r["_calib_status"] == "failed" for r in records)


def test_calibrated_online_pose_does_not_require_single_frame_success(tmp_path):
    records = _make(tmp_path)
    for r in records:
        if r["_k"] == 1:
            r["pnp_ok"] = False
    apply_calibrate_once(records, tmp_path, CFG)
    from touch.learning.evaluate import summarize
    online = [r for r in records if r["_calib"] == "online"]
    assert summarize(online, CFG)["n_pnp_ok"] == 4


def test_physical_rejection_is_counted(tmp_path, monkeypatch):
    from touch.learning import evaluate
    records = _make(tmp_path)
    monkeypatch.setattr(evaluate, "solve_pnp_sequence", lambda *a, **kw:
                        (np.eye(3), np.array([0., 0., 0.8]), 0., [], []))
    st = evaluate.apply_calibrate_once(records, tmp_path, CFG)
    assert st["n_planned"] == st["n_rejected"] == 2
    assert st["n_seq"] == st["n_failed"] == st["n_skipped"] == 0
    assert all(r["_calib_status"] == "rejected" and "_Rc" not in r for r in records)
