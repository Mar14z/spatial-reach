"""L2 记忆闭环：A 位置看并记住方块 → 相机换到 B 只标定臂 → 靠记忆抓。

与 mujoco_closed_loop 的区别只在「方块像素从哪来」：

    闭环 v2：方块像素来自当前相机的完美检测 → 用当前标定反投影
    本脚本： 方块像素来自 A 位置（记忆），相机换到 B 后只重标臂，
             抓取指令仍用 A 记下的基座系 3D 点 p_mem

为什么必须留对照组：仿真里抓取执行不需要相机，拿过期记忆去抓也会"走到那个点"，
盲抓成功本身不构成记忆证据。所以同时跑

    mem   ：方块就在 p_mem 指向的位置   → 应该抓到
    stale ：方块已经挪到另一个格子      → 应该抓空（记忆过期）

三件要分开看的事：

一、p_mem 只用 A 的标定反投影，B 的标定**不进入抓取链**（这是「存基座系」的结构
    性质，不是实验结果），所以 mem 的误差与「A 位置直接抓」逐位相同。这一点在报告
    里必须写清楚，否则会把结构性质当成实验发现。
二、px_match 混合 A 记忆误差与 B 外参误差；B 自定位要另看位姿真值误差和留出帧。
三、reobs 是显式启用的理想目标观测对照，像素来自真值投影，不是 B 的 RGB 检测。
    只接受逐姿态重算过的可见性记录；mem/stale 指令不使用 B 目标新观测。

口径提示：默认 A/B 归档预测来自不同解码版本，其中旧亚像素解码有已知缺陷。
这些输入仅供历史回放，不能作为当前精度验收；新成绩需用相同配方重新生成预测。

用法：
    python scripts/memory_closed_loop.py --limit 3
    python scripts/memory_closed_loop.py --conds mem,stale,reobs
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import mujoco_pick as mp  # noqa: E402
from touch.learning.evaluate import (  # noqa: E402
    apply_calibrate_once, evaluate_frames, load_predictions, trial_summary, write_result,
)
from touch.perception.geometry import backproject_to_plane  # noqa: E402
from experiment_config import AZ_FORWARD, CELLS, TOL_MM  # noqa: E402

A_DATA = ROOT / "data/synthetic/m0_seq/val"
A_PRED = ROOT / "outputs/eval/pred_m0_seq_val_ft320.json"
B_DATA = ROOT / "data/synthetic/m0_seq_b/val"
B_PRED = ROOT / "outputs/eval/pred_m0_seq_b_ft320.json"


def cell(seq: int, k: int):
    """(seq, k) 的方块位置：r/方位角相对正前方，返回桌面点（z=0）。"""
    r, az_rel = CELLS[(2 * seq + (k - 1)) % len(CELLS)]
    az = AZ_FORWARD + az_rel
    return np.array([r * np.cos(az), r * np.sin(az), 0.0]), r, az


def load_set(data: Path, pred: Path, calib):
    """评测 + 标一次 T，返回 {seq: {k: record}} 与标定统计。"""
    frames, _ = load_predictions(pred)
    records, cfg = evaluate_frames(data, frames)
    st = apply_calibrate_once(records, data, cfg, calib)
    by_seq: dict = {}
    for r in records:
        if r.get("_seq") is not None and r.get("_k") is not None:
            by_seq.setdefault(r["_seq"], {})[r["_k"]] = r
    return by_seq, st


def project(p3: np.ndarray, R, t, K) -> np.ndarray:
    uv, _ = cv2.projectPoints(np.asarray(p3, float).reshape(-1, 3),
                              cv2.Rodrigues(np.asarray(R, float))[0],
                              np.asarray(t, float), K, None)
    return uv.reshape(-1, 2)


def meta_of(data: Path, rec) -> dict:
    return json.loads((data / rec["frame"] / "meta.json").read_text(encoding="utf-8"))


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="记忆闭环 L2：A 记 → B 标臂 → 抓 + 过期对照")
    ap.add_argument("--conds", default="mem,stale",
                    help="逗号分隔：mem / stale / reobs（reobs 只跑 B 看得见方块的子集）")
    ap.add_argument("--vis-px", type=int, default=150,
                    help="B 视角方块可见像素阈值（低于它不跑 reobs）")
    ap.add_argument("--a-data", type=Path, default=A_DATA)
    ap.add_argument("--b-data", type=Path, default=B_DATA)
    ap.add_argument("--a-pred", default=str(A_PRED))
    ap.add_argument("--b-pred", default=str(B_PRED))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个序列")
    ap.add_argument("--no-sim", action="store_true", help="只算误差不进 MuJoCo")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/memory_closed_loop_v3.json"))
    a = ap.parse_args()

    conds = [c.strip() for c in a.conds.split(",") if c.strip()]
    if not conds or len(set(conds)) != len(conds) or set(conds) - {"mem", "stale", "reobs"}:
        ap.error("条件必须是 mem/stale/reobs 的非空、不重复组合")
    if a.limit < 0 or a.vis_px < 1:
        ap.error("limit 必须非负，vis-px 必须为正")
    a_by, st_a = load_set(a.a_data, Path(a.a_pred), (0, 3))
    b_by, st_b = load_set(a.b_data, Path(a.b_pred), (0, 3))
    print(f"[A] 标定帧 0,3：成功 {st_a['n_seq']} 个序列，跳过 {st_a['n_skipped']}，"
          f"物理拒识 {st_a['n_rejected']}；平移中位 {pct(st_a['trans_cm'], 50):.2f}cm "
          f"旋转中位 {pct(st_a['rot_deg'], 50):.2f}°，翻转 {sum(r > 15 for r in st_a['rot_deg'])}")
    print(f"[B] 标定帧 0,3：成功 {st_b['n_seq']} 个序列，跳过 {st_b['n_skipped']}，"
          f"物理拒识 {st_b['n_rejected']}；平移中位 {pct(st_b['trans_cm'], 50):.2f}cm "
          f"旋转中位 {pct(st_b['rot_deg'], 50):.2f}°，翻转 {sum(r > 15 for r in st_b['rot_deg'])}")

    seqs = sorted(a_by)  # A 的任务全集；B 缺帧不能删除已经定义的记忆任务
    if a.limit:
        seqs = seqs[: a.limit]
    rows: list = []

    n_b_ok = 0
    for seq in seqs:
        planned = {c: {"seq": int(seq), "cond": c, "err_mm": None,
                       "status": "a_memory_unavailable"} for c in conds}
        rows.extend(planned.values())
        ra = a_by[seq].get(1)          # A 的留出帧：这里扮演「第一次看一眼」
        rb = b_by.get(seq, {}).get(1) or b_by.get(seq, {}).get(0)
        if ra is None or not ra.get("_Rc"):
            status = "missing_frame" if ra is None else "calibration_" + ra.get("_calib_status", "failed")
            for row in planned.values():
                row["status"] = status
            continue                   # A 都没记住，无从谈起
        # B 标定失败（物理拒识）不该把整个序列丢掉：抓取链只用 p_mem，不用 B 的 T。
        # 失败的序列照跑 mem/stale，只是没有 px_match / reobs，单独计数。
        b_ok = rb is not None and bool(rb.get("_Rc"))
        n_b_ok += int(b_ok)
        K_a = np.asarray(ra["_K"], float)
        R_a = np.asarray(ra["_Rc"], float).reshape(3, 3)
        t_a = np.asarray(ra["_tc"], float)

        p_true1, r1, az1 = cell(seq, 1)
        p_true2, r2, az2 = cell(seq, 2)

        # ---- A：看一眼 + 记住（基座系 3D 点，与视角解耦）----
        ma = meta_of(a.a_data, ra)
        uv_a = project(p_true1, ma["R_cb"], ma["t_cb"], K_a)[0]
        p_mem = backproject_to_plane(K_a, uv_a.reshape(1, 2), R_a, t_a, 0.0)[0]
        if not np.isfinite(p_mem).all() or (R_a @ p_mem + t_a)[2] <= 0:
            for row in planned.values():
                row["status"] = "geometry_failed"
            continue
        err_mem = float(np.linalg.norm((p_mem - p_true1)[:2])) * 1000.0

        # ---- B：只标臂，能不能在新视角下指认记忆里的点 ----
        px_match, vis_px, p_b = None, 0, None
        yaw_gap = None
        reobs_status = "b_calibration_unavailable"
        if b_ok:
            K_b = np.asarray(rb["_K"], float)
            R_b = np.asarray(rb["_Rc"], float).reshape(3, 3)
            t_b = np.asarray(rb["_tc"], float)
            mb = meta_of(a.b_data, rb)
            uv_mem_b = project(p_mem, R_b, t_b, K_b)[0]
            uv_ref_b = project(p_true1, mb["R_cb"], mb["t_cb"], K_b)[0]
            px_match = float(np.linalg.norm(uv_mem_b - uv_ref_b))
            yaw_gap = abs(float(mb["camera"]["yaw"]) - float(ma["camera"]["yaw"])) % 360.0
            yaw_gap = min(yaw_gap, 360.0 - yaw_gap)
            vis_px = int(mb.get("box_visible_px", 0))
            visibility_verified = mb.get("visibility_version") == 2
            reobs_status = "target_not_visible" if visibility_verified else "visibility_unverified"
            # 仅显式 reobs 条件使用理想目标观测；mem/stale 指令不访问它。
            if "reobs" in conds and visibility_verified and vis_px >= a.vis_px:
                uv_b = project(p_true1, mb["R_cb"], mb["t_cb"], K_b)[0]
                p_b = backproject_to_plane(K_b, uv_b.reshape(1, 2), R_b, t_b, 0.0)[0]
                if not np.isfinite(p_b).all() or (R_b @ p_b + t_b)[2] <= 0:
                    p_b = None
                    reobs_status = "geometry_failed"

        base = {"seq": int(seq), "yaw_gap_deg": yaw_gap, "vis_px": vis_px,
                "b_eval_frame_role": "heldout" if rb and rb.get("_k") not in (0, 3) else "calibration_or_missing",
                "b_calib_ok": b_ok, "err_mem": err_mem, "px_match": px_match}
        if "reobs" in planned:
            planned["reobs"].update(base, status=reobs_status)
        # A 标定在留出帧上的重投影 RMS：部署时算得出来（不用 GT），当质量分/散点纵轴
        uv_fit = project(np.asarray(ra["_p3"], float), R_a, t_a, K_a)
        base["rms_px"] = float(np.sqrt(np.mean(np.sum(
            (uv_fit - np.asarray(ra["_p2"], float)) ** 2, axis=1))))
        # 记一行「记忆本身」的质量，cond=memory 只用于统计，不进抓取
        rows.append({**base, "cond": "memory", "err_mm": err_mem,
                     "p_mem": p_mem.tolist(), "p_true": p_true1.tolist(),
                     "err_b_mm": (None if p_b is None else
                                  float(np.linalg.norm((p_b - p_true1)[:2])) * 1000.0),
                     "p_b_minus_mem": (None if p_b is None else
                                       (float(np.linalg.norm((p_b - p_mem)[:2])) * 1000.0))})

        trials = []
        if "mem" in conds:
            trials.append(("mem", p_true1, r1, az1, p_mem, p_true1))
        if "stale" in conds:
            trials.append(("stale", p_true2, r2, az2, p_mem, p_true2))
        if "reobs" in conds and p_b is not None:
            trials.append(("reobs", p_true1, r1, az1, p_b, p_true1))

        for cond, p_cube, radius, az, p_cmd, p_ref in trials:
            err = float(np.linalg.norm((p_cmd - p_ref)[:2])) * 1000.0
            row = planned[cond]
            row.update(base, err_mm=err, err_mem=err_mem, radius=radius,
                       command_xy=p_cmd[:2].tolist(),
                       status="geometry_only" if a.no_sim else "executed")
            if a.no_sim:
                tag = "-"
            else:
                try:
                    res = mp.pick(0.03, radius, az, verbose=False, command_xy=p_cmd[:2])
                except (ValueError, RuntimeError) as exc:
                    row.update(status="execution_failed", error=str(exc))
                    continue
                tag = "OK" if res["ok"] else ("OK*" if res["ok_grasp"] else "FAIL")
                row.update({"ok": bool(res["ok"]), "ok_grasp": bool(res["ok_grasp"]),
                            "align_mm": res["align"] * 1000,
                            "in_hand_mm": res["in_hand"] * 1000,
                            "dz_mm": res["dz"] * 1000, "slip_mm": res["slip"] * 1000})
            pm = "-" if px_match is None else f"{px_match:5.1f}"
            yaw_text = "-" if yaw_gap is None else f"{yaw_gap:5.1f}"
            print(f"seq={seq:3d} {cond:5s} vis={vis_px:4d} yaw差={yaw_text}° "
                  f"err={err:5.1f}mm px_match={pm} {tag}", flush=True)

    # ---------------- 汇总 ----------------
    summary: dict = {}
    mem = [r for r in rows if r["cond"] == "memory"]
    em = np.array([r["err_mem"] for r in mem]) if mem else np.array([])
    pxm = np.array([r["px_match"] for r in mem if r["px_match"] is not None])
    yaws = [r["yaw_gap_deg"] for r in mem if r["yaw_gap_deg"] is not None]
    if len(em):
        summary["memory"] = {
            "n": int(len(em)), "med_mm": float(np.median(em)),
            "p95_mm": float(np.percentile(em, 95)), "max_mm": float(em.max()),
            "within_tol": float(np.mean(em <= TOL_MM)),
            "n_b_calib_ok": int(n_b_ok),
            "px_match_med": float(np.median(pxm)) if len(pxm) else None,
            "px_match_p90": float(np.percentile(pxm, 90)) if len(pxm) else None,
            "yaw_gap_med": float(np.median(yaws)) if yaws else None,
        }
        vis = [r for r in mem if r["err_b_mm"] is not None]
        if vis:
            summary["reobs_consistency"] = {
                "n": len(vis),
                "err_b_med_mm": float(np.median([r["err_b_mm"] for r in vis])),
                "p_b_vs_mem_med_mm": float(np.median([r["p_b_minus_mem"] for r in vis])),
            }
    print(f"\n=== 汇总（{len(seqs)} 个序列，B 标定成功 {n_b_ok}）===")
    if len(em):
        print(f"记忆精度 p_mem：中位 {np.median(em):.1f}mm  p95 {np.percentile(em, 95):.1f}mm  "
              f"P(≤{TOL_MM:.0f}mm) {np.mean(em <= TOL_MM):.1%}  max {em.max():.1f}mm")
    if len(pxm):
        print(f"B 视角指认 p_mem：像素偏差中位 {np.median(pxm):.1f}px  "
              f"p90 {np.percentile(pxm, 90):.1f}px（B 与 A 视角差中位 "
              f"{np.median(yaws):.0f}°）")
    vis = [r for r in mem if r["err_b_mm"] is not None]
    if vis:
        print(f"B 重观测（{len(vis)} 个可见序列）：p_B 误差中位 "
              f"{np.median([r['err_b_mm'] for r in vis]):.1f}mm，"
              f"|p_B - p_mem| 中位 {np.median([r['p_b_minus_mem'] for r in vis]):.1f}mm")

    conds_sum = {}
    for c in ("mem", "stale", "reobs"):
        rs = [r for r in rows if r["cond"] == c and "ok_grasp" in r]
        if not rs:
            continue
        e = np.array([r["err_mm"] for r in rs])
        ok = float(np.mean([r["ok_grasp"] for r in rs]))
        conds_sum[c] = {"n": len(rs), "grasp_rate": ok,
                        "strict_rate": float(np.mean([r["ok"] for r in rs])),
                        "err_med_mm": float(np.median(e)),
                        "err_max_mm": float(e.max()),
                        "align_med_mm": float(np.median([r["align_mm"] for r in rs])),
                        "in_hand_med_mm": float(np.median([r["in_hand_mm"] for r in rs])),
                        "dz_med_mm": float(np.median([r["dz_mm"] for r in rs]))}
        print(f"[{c:5s}] 已返回结果子集（不含执行异常）n={len(rs):3d} 抓到 {ok:6.1%}"
              f"（严格对齐 {conds_sum[c]['strict_rate']:.1%}）"
              f" 误差中位 {conds_sum[c]['err_med_mm']:5.1f}mm  "
              f"align 中位 {conds_sum[c]['align_med_mm']:5.1f}mm  "
              f"in_hand 中位 {conds_sum[c]['in_hand_med_mm']:5.1f}mm  "
              f"dz 中位 {conds_sum[c]['dz_med_mm']:5.1f}mm")

    for c in conds:
        counts = trial_summary([r for r in rows if r["cond"] == c], no_sim=a.no_sim)
        conds_sum.setdefault(c, {}).update(counts)
        conds_sum[c].update(grasp_rate=counts["attempt_success_rate"],
                            strict_rate=(counts["n_strict"] / counts["n_attempted"]
                                         if counts["n_attempted"] else None))
        print(f"[{c}] 计划 {counts['n_planned']}，执行 {counts['n_attempted']}，"
              f"执行后抓到率 {counts['attempt_success_rate']}，"
              f"全部任务抓到率 {counts['end_to_end_success_rate']}，状态 {counts['status_counts']}")
    write_result(a.out,
        {"script": "memory_closed_loop", "protocol_version": 3,
         "target_source": "gt_projection_proxy", "b_target_used_for_memory_command": False,
         "a_data": str(a.a_data.resolve()), "b_data": str(a.b_data.resolve()),
         "a_pred": str(Path(a.a_pred).resolve()),
         "b_pred": str(Path(a.b_pred).resolve()), "calib_a": "0,3", "calib_b": "0,3",
         "conds": conds, "vis_px": a.vis_px, "tol_mm": TOL_MM,
         "no_sim": a.no_sim, "calib_stats": {"a": {k: v for k, v in st_a.items()
                                                   if not isinstance(v, list)},
                                             "b": {k: v for k, v in st_b.items()
                                                   if not isinstance(v, list)}},
         "summary": summary, "cond_summary": conds_sum, "rows": rows})
    print("saved", a.out)


if __name__ == "__main__":
    main()
