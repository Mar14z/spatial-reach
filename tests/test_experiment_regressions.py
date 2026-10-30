"""实验控制输入、任务配对和缺失数据的回归，不驱动真实机械臂。"""
import json
import sys

import numpy as np
import pytest


def test_planning_uses_command_not_moved_object(monkeypatch):
    from scripts import mujoco_pick as mp
    captured = []

    def stop_at_planner(model, data, wid, cmd, azimuth, n):
        captured.append((cmd.copy(), azimuth))
        raise RuntimeError("测试在规划入口停止，不进行运动")

    monkeypatch.setattr(mp, "find_grasp_pose", stop_at_planner)
    for radius, az in [(0.10, -1.5), (0.14, -1.2)]:
        with pytest.raises(RuntimeError, match="规划入口"):
            mp.pick(radius=radius, azimuth=az, command_xy=[0.01, -0.12], verbose=False)
    np.testing.assert_allclose(captured[0][0], captured[1][0])
    assert captured[0][1] == pytest.approx(np.arctan2(-0.12, 0.01))
    assert captured[0][1] == captured[1][1]


def test_missing_grasp_pose_is_reported(monkeypatch):
    from scripts import mujoco_pick as mp
    monkeypatch.setattr(mp, "find_grasp_pose", lambda *args: None)
    with pytest.raises(ValueError, match="未找到满足约束的抓取姿态"):
        mp.pick(command_xy=[0.01, -0.12], verbose=False)


def test_missing_predictions_stay_in_dataset_denominator(tmp_path):
    from touch.learning.evaluate import evaluate_frames, summarize
    for k in range(2):
        d = tmp_path / str(k)
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({
            "height": 480, "width": 640, "sequence": 0, "frame_in_seq": k,
            "K": np.eye(3).tolist(), "keypoints_2d": [[0., 0.]] * 4,
            "keypoints_3d_base": [[0., 0., 0.]] * 4,
        }), encoding="utf-8")
    records, cfg = evaluate_frames(tmp_path, {})
    assert len(records) == 2
    assert all(r["error"] == "missing_or_invalid_prediction" for r in records)
    assert summarize(records, cfg)["usable"]["3cm/5°"] == 0.0


def test_rejection_does_not_shift_scene_or_noise(tmp_path, monkeypatch):
    from scripts import mujoco_closed_loop as cl
    K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
    R, t = np.diag([1., -1., -1.]), np.array([0., 0., .8])
    records = []
    for seq in range(3):
        d = tmp_path / str(seq)
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"R_cb": R.tolist(), "t_cb": t.tolist(),
            "width": 640, "height": 480}), encoding="utf-8")
        records.append({"frame": str(seq), "_seq": seq, "_k": 1, "pnp_ok": True,
            "_K": K.tolist(), "_p3": [[0., 0., .1]] * 4, "_p2": [[320., 240.]] * 4,
            "_Rc": R.ravel().tolist(), "_tc": t.tolist(), "_calib_status": "accepted"})
    monkeypatch.setattr(cl, "load_predictions", lambda *a: ({}, {}))
    monkeypatch.setattr(cl, "evaluate_frames", lambda *a: (records, {}))
    monkeypatch.setattr(cl, "apply_calibrate_once", lambda *a:
                        {"n_seq": 3, "n_skipped": 0, "n_rejected": 0})
    results = []
    for i in range(2):
        if i:
            records[0].pop("_Rc")
            records[0]["_calib_status"] = "rejected"
        out = tmp_path / f"result{i}.json"
        monkeypatch.setattr(sys, "argv", ["test", "--data", str(tmp_path), "--pred", "stub",
            "--frames", "1,2", "--det-px", "0.5", "--no-sim", "--out", str(out)])
        cl.main()
        results.append(json.loads(out.read_text(encoding="utf-8")))
    a, b = results
    assert a["task_summary"]["n_planned"] == b["task_summary"]["n_planned"] == 6
    assert b["task_summary"]["status_counts"]["missing_frame"] == 3
    assert b["task_summary"]["status_counts"]["calibration_rejected"] == 1
    for seq in (1, 2):
        first = next(r for r in a["rows"] if r["seq"] == seq and r["k"] == 1)
        second = next(r for r in b["rows"] if r["seq"] == seq and r["k"] == 1)
        assert first["radius"] == second["radius"] and first["az_rel"] == second["az_rel"]
        np.testing.assert_allclose(first["uv"], second["uv"])


@pytest.mark.parametrize("no_sim", [False, True])
@pytest.mark.parametrize("fail_execution", [False, True])
def test_memory_tasks_keep_missing_b_and_rejected_a(tmp_path, monkeypatch, no_sim, fail_execution):
    from scripts import memory_closed_loop as mc
    K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1.]])
    R, t = np.diag([1., -1., -1.]), np.array([0., 0., .8])
    rec = {"_k": 1, "_K": K.tolist(), "_Rc": R.ravel().tolist(), "_tc": t.tolist(),
           "_p3": [[0., 0., .1]] * 4, "_p2": [[320., 240.]] * 4}
    a_by = {0: {1: dict(rec)}, 1: {1: {"_calib_status": "rejected"}}, 2: {1: dict(rec)}}
    b_by = {0: {1: dict(rec)}}
    st = {"n_seq": 2, "n_skipped": 0, "n_rejected": 1, "trans_cm": [0.], "rot_deg": [0.]}
    a_data, b_data = tmp_path / "a", tmp_path / "b"
    loaded, metadata = [], []
    def load(data, *args):
        loaded.append(data)
        return (a_by if data == a_data else b_by), st
    def meta(data, rec):
        metadata.append(data)
        return {"R_cb": R.tolist(), "t_cb": t.tolist(),
                "camera": {"yaw": 0.}, "box_visible_px": 500}
    monkeypatch.setattr(mc, "load_set", load)
    monkeypatch.setattr(mc, "meta_of", meta)
    commands = []
    def fake_pick(*a, **kw):
        assert "jitter" not in kw
        commands.append(np.asarray(kw["command_xy"]).copy())
        if fail_execution and len(commands) > 2:
            raise ValueError("测试执行失败")
        return {"ok": True, "ok_grasp": True, "align": 0., "in_hand": 0., "dz": .08, "slip": 0.}
    monkeypatch.setattr(mc.mp, "pick", fake_pick)
    out = tmp_path / "memory.json"
    argv = ["test", "--conds", "mem,stale,reobs", "--out", str(out),
            "--a-data", str(a_data), "--b-data", str(b_data)]
    monkeypatch.setattr(sys, "argv", argv + (["--no-sim"] if no_sim else []))
    mc.main()
    result = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == [a_data, b_data]
    assert metadata == [a_data, b_data, a_data]
    assert result["a_data"] == str(a_data.resolve())
    assert result["b_data"] == str(b_data.resolve())
    for c in ("mem", "stale", "reobs"):
        assert result["cond_summary"][c]["n_planned"] == 3
    for c in ("mem", "stale"):
        expected = None if no_sim else (0.5 if fail_execution else 1.0)
        assert result["cond_summary"][c]["grasp_rate"] == expected
        assert result["cond_summary"][c]["strict_rate"] == expected
    assert result["cond_summary"]["mem"]["n_attempted"] == (0 if no_sim else 2)
    assert result["cond_summary"]["reobs"]["n_attempted"] == 0
    tasks = [r for r in result["rows"] if r["cond"] != "memory"]
    assert len(tasks) == 9
    assert next(r for r in tasks if r["seq"] == 0 and r["cond"] == "reobs")["status"] == "visibility_unverified"
    if not no_sim:
        np.testing.assert_allclose(commands[0], commands[1])
        np.testing.assert_allclose(commands[2], commands[3])
