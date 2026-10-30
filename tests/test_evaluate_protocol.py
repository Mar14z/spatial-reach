"""口径守卫：2D 误差的两级统计单元。

同一份预测曾报出 1.79px 与 2.03px 两个数，差别只在统计单元。这两个测试把
「两级来自同一批样本」「两级会给出不同分位数」写成断言，防止以后又各算一遍。
"""

import numpy as np
import pytest

from touch.learning.evaluate import PROTOCOL, summarize


def _records(errs):
    """errs 形状 (帧数, 点数)，构造最小可评测 record（位姿字段填占位值）。"""
    return [
        {
            "frame": f"{i:03d}",
            "mean_px": float(e.mean()),
            "px_vec": [float(v) for v in e],
            "dv_vec": [0.0] * len(e),
            "nme": 0.0,
            "max_bias_v": 0.0,
            "spread2d": 1.0,
            "pnp_ok": True,
            "trans_cm": 1.0,
            "rot_deg": 1.0,
            "table_cm": 1.0,
        }
        for i, e in enumerate(errs)
    ]


def test_two_levels_share_the_same_samples():
    errs = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 9.0]])
    s = summarize(_records(errs), PROTOCOL)
    assert s["n_kp_samples"] == errs.size and s["n_frames"] == len(errs)
    # 每帧点数相同时，两级均值是同一批数的均值，必须相等
    assert s["kp_px_mean"] == pytest.approx(s["px_mean"])
    assert s["kp_px_mean"] == pytest.approx(errs.mean())


def test_levels_differ_when_within_frame_spread_is_large():
    # 每帧一个 10px 野点、两个 0px：帧均值是 3.33，逐点中位数却是 0
    errs = np.tile(np.array([0.0, 0.0, 10.0]), (10, 1))
    s = summarize(_records(errs), PROTOCOL)
    assert s["px_median"] == pytest.approx(10 / 3)
    assert s["kp_px_median"] == pytest.approx(0.0)
    # 帧级 PCK 问「整帧平均达标吗」，关键点级问「逐点达标吗」
    assert s["pck3"] == pytest.approx(0.0)
    assert s["kp_pck3"] == pytest.approx(2 / 3)


def test_failure_stays_in_usable_denominator_and_empty_is_safe():
    records = _records(np.zeros((2, 3)))
    records[1]["pnp_ok"] = False
    s = summarize(records)
    assert s["usable"]["3cm/5°"] == 0.5
    assert s["p_within_grip"] == 0.5
    assert summarize([])["n_pnp_ok"] == 0
    assert np.isnan(summarize([])["trans_p95"])


def test_bootstrap_preserves_whole_sequences(monkeypatch):
    from touch.learning import evaluate
    records = [{"_seq": s, "_k": k} for s in range(3) for k in range(4)]
    def checked(rs):
        for start in range(0, len(rs), 4):
            assert len({r["_seq"] for r in rs[start:start + 4]}) == 1
            assert [r["_k"] for r in rs[start:start + 4]] == [0, 1, 2, 3]
        return {"metric": 1.0}
    monkeypatch.setattr(evaluate, "summarize", checked)
    assert evaluate.bootstrap_ci(records, keys=("metric",), n=10)["metric"] == (1., 1.)


def test_task_counts_and_strict_json(tmp_path):
    import json
    from touch.learning.evaluate import trial_summary, write_result
    rows = [{"status": "executed", "ok_grasp": True, "ok": False},
            {"status": "calibration_rejected"}, {"status": "missing_frame"},
            {"status": "execution_failed"}, {"cond": "memory"}]
    summary = trial_summary(rows)
    assert summary["n_planned"] == 4 and summary["n_attempted"] == 2
    assert summary["attempt_success_rate"] == 0.5
    assert summary["end_to_end_success_rate"] == 0.25
    path = tmp_path / "run" / "result.json"
    write_result(path, {"summary": summary, "values": np.array([np.nan, np.inf, 1.])})
    def reject(value):
        raise AssertionError(value)
    saved = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    assert saved["values"] == [None, None, 1.0]
    assert list(path.parent.glob("*.tmp")) == []


@pytest.mark.parametrize("status", ["accepted", "rejected"])
def test_report_strata_preserves_frames_without_single_frame_pose(status):
    from scripts.eval_report import strata
    records = _records(np.zeros((2, 3)))
    for r in records:
        r.update(pnp_ok=False, _calib_status=status)
    groups = strata(records, PROTOCOL)
    group = groups["0.3-1.01"]
    assert group["n"] == 2
    assert group["usable"]["3cm/5°"] == (1.0 if status == "accepted" else 0.0)
