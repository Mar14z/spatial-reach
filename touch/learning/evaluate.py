"""统一评测入口：预测 JSON + 数据目录 → 四级指标（2D / 位姿 / 桌面 / 可靠性）。

只依赖 numpy + OpenCV，不碰 torch —— 任何检测器（CNN、HSV、ArUco）都只需产出
同样格式的预测 JSON 即可复用这套口径。

2D 误差有两级统计单元，数值不可混用，输出时必须显式标注：
  - 关键点级 `kp_*`：每帧每点算一个样本，n = 帧数 × 点数。这是 PCK 的常规口径。
  - 帧级 `px_*`：先对一帧内的点求均值再统计，n = 帧数，反映「这一帧整体准不准」。
帧内求均值会削平尾部，所以帧级的 median / p95 系统性小于关键点级（同一份预测
实测 1.79/5.90 vs 2.03/4.60），两者不能互相引用。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from touch.perception.geometry import backproject_to_plane, pose_error
from touch.perception.pose import (
    camera_center, degeneracy_gap, solve_pnp, solve_pnp_sequence,
)

PROTOCOL = {
    "solver": cv2.SOLVEPNP_SQPNP,
    "reproj_threshold_px": 5.0,
    "flip_deg": 15.0,
    "pck_px": (3.0, 5.0),
    "plane_z": 0.0,
    "grid_x": (0.05, 0.45, 6),
    "grid_y": (-0.2, 0.2, 5),
    # 可用率的分级容差 (平移 cm, 旋转 deg)：两者同时满足才算「这一帧可用」
    "tolerance": ((3.0, 5.0), (5.0, 10.0)),
    # 夹爪横向容差（cm）：注入偏移实测 15mm 起开始失败，是悬崖不是软边界。
    # 抓取成功率该用 P(桌面误差 ≤ 此值) 预测，中位数和 3cm 可用率都看不出来。
    "grasp_tol_cm": 1.5,
}


def table_points(cfg=None):
    cfg = PROTOCOL if cfg is None else cfg
    x0, x1, nx = cfg["grid_x"]
    y0, y1, ny = cfg["grid_y"]
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, cfg["plane_z"])], axis=1)


def load_predictions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["frames"], data.get("summary", {})


def json_safe(value):
    """非有限数显式写为 null，兼容浏览器和严格 JSON 解析器。"""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_result(path, data):
    """完整序列化后原子替换，读取者不会读到半份结果。"""
    import os
    import tempfile
    path = Path(path)
    text = json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
        f.write(text)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def trial_summary(rows, no_sim=False):
    """每行一个计划任务；诊断行不计入任务分母，纯几何不计作执行。"""
    tasks = [r for r in rows if r.get("cond") != "memory"]
    attempted = [r for r in tasks if "ok_grasp" in r or r.get("status") == "execution_failed"]
    counts = {}
    for r in tasks:
        status = r.get("status", "executed" if "ok_grasp" in r else "geometry_only")
        counts[status] = counts.get(status, 0) + 1
    success = sum(bool(r.get("ok_grasp")) for r in attempted)
    strict = sum(bool(r.get("ok")) for r in attempted)
    return {"n_planned": len(tasks), "n_attempted": len(attempted),
            "n_success": success, "n_strict": strict, "status_counts": counts,
            "attempt_success_rate": success / len(attempted) if attempted else None,
            "end_to_end_success_rate": (success / len(tasks) if tasks and not no_sim and
                not all(r.get("status") == "geometry_only" for r in tasks) else None)}


def _table_error(K, R, t, R_gt, t_gt, cfg):
    pts = table_points(cfg)
    uv, _ = cv2.projectPoints(pts, cv2.Rodrigues(np.asarray(R_gt))[0], t_gt, K, None)
    uv = uv.reshape(-1, 2)
    h, w = cfg["image_hw"]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if inside.sum() < 3:
        return np.nan
    est = backproject_to_plane(K, uv[inside], R, t, cfg["plane_z"])
    err = np.linalg.norm(est - pts[inside], axis=1)
    return float(err.mean() * 100.0)


def evaluate_frames(data_dir, pred_frames, extra=None, cfg=None):
    """逐帧算指标；extra 为 {frame: (uv_extra, xyz_extra)}（可选）。"""
    cfg = dict(PROTOCOL if cfg is None else cfg)
    frames = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    if not frames:
        raise ValueError(f"数据目录没有帧：{data_dir}")
    meta0 = json.loads((frames[0] / "meta.json").read_text(encoding="utf-8"))
    cfg["image_hw"] = (meta0["height"], meta0["width"])
    records = []
    for frame in frames:
        meta = json.loads((frame / "meta.json").read_text(encoding="utf-8"))
        gt = np.asarray(meta["keypoints_2d"], dtype=np.float64)
        p3 = np.asarray(meta["keypoints_3d_base"], dtype=np.float64)
        if extra is not None:
            uv_x, xyz_x = extra[frame.name]
            gt = np.vstack([gt, uv_x])
            p3 = np.vstack([p3, xyz_x])
        raw = pred_frames.get(frame.name, {}).get("pred_2d")
        try:
            p2 = np.asarray(raw, dtype=np.float64)
            valid_pred = p2.shape == gt.shape and np.isfinite(p2).all()
        except (TypeError, ValueError):
            valid_pred = False
        if not valid_pred:
            records.append({
                "frame": frame.name, "mean_px": np.nan, "nme": np.nan,
                "px_vec": [np.nan] * len(gt), "dv_vec": [np.nan] * len(gt),
                "pnp_ok": False, "error": "missing_or_invalid_prediction",
                "_seq": meta.get("sequence"), "_k": meta.get("frame_in_seq"),
                "_K": meta["K"], "_p3": p3.tolist(),
                "_p2": np.full_like(gt, np.nan).tolist(),
            })
            continue
        K = np.asarray(meta["K"], dtype=np.float64)
        R_gt = np.asarray(meta["R_cb"], dtype=np.float64)
        t_gt = np.asarray(meta["t_cb"], dtype=np.float64)
        e = np.linalg.norm(p2 - gt, axis=1)
        ref = np.linalg.norm(gt[min(8, len(gt) - 1)] - gt[0])
        s2 = np.linalg.svd(gt - gt.mean(0), compute_uv=False)
        rec = {
            "frame": frame.name,
            "mean_px": float(e.mean()),
            "px_vec": [float(v) for v in e],
            "dv_vec": [float(v) for v in (p2 - gt)[:, 1]],
            "nme": float(e.mean() / max(ref, 1.0)),
            "max_bias_v": float(np.abs((p2 - gt)[:, 1].mean())),
            "spread2d": float(s2[1] / s2[0]) if len(gt) > 2 else np.nan,
            "pnp_ok": False,
            "_seq": meta.get("sequence"),
            "_k": meta.get("frame_in_seq"),
        }
        # 这三项与 PnP 成败无关，不放进 try：在线帧重投影 RMS 恰恰最需要「单帧解
        # 失败」的那些帧，放进 try 会把最难的一批样本悄悄排除。
        rec.update({"_K": K.tolist(), "_p3": p3.tolist(), "_p2": p2.tolist()})
        try:
            R, t, _ = solve_pnp(K, p3, p2, cfg["reproj_threshold_px"], False, cfg["solver"])
            if not (np.isfinite(R).all() and np.isfinite(t).all()):
                raise ValueError("PnP 返回非有限位姿")
            trans, rot = pose_error(R, t, R_gt, t_gt)
            rec.update({
                "pnp_ok": True,
                "trans_cm": trans,
                "rot_deg": rot,
                "flip": bool(rot > cfg["flip_deg"]),
                "table_cm": _table_error(K, R, t, R_gt, t_gt, cfg),
                "_R": [float(v) for v in np.asarray(R).ravel()],
                "_t": [float(v) for v in np.asarray(t).ravel()],
            })
        except Exception as exc:
            rec["error"] = str(exc)
        records.append(rec)
    return records, cfg


def _stack(records, key):
    """把每帧的逐点向量堆成 (帧数, 点数)；字段缺失时返回 None。"""
    rows = [np.asarray(r[key], dtype=np.float64) for r in records if key in r]
    return np.stack(rows) if rows else None


def apply_sequence_solve(records, data_dir, cfg=None):
    """用同一序列的多帧联合重解相机位姿，覆写单帧结果（原地修改）。

    前提是序列内相机不动（生成端 k==0 时一次性定下 view）。返回被改写的帧数、
    联合解与单帧解不一致的帧数、以及按序列统计的翻转变化。
    """
    cfg = PROTOCOL if cfg is None else cfg
    groups: dict = {}
    for i, r in enumerate(records):
        if r["pnp_ok"] and r.get("_seq") is not None:
            groups.setdefault(r["_seq"], []).append(i)
    changed = 0
    for _, idx in groups.items():
        if len(idx) < 2:
            continue
        obs = [(records[i]["_p3"], np.asarray(records[i]["_p2"], np.float64)) for i in idx]
        seeds = [(np.asarray(records[i]["_R"]).reshape(3, 3), np.asarray(records[i]["_t"])) for i in idx]
        K = np.asarray(records[idx[0]]["_K"], np.float64).reshape(3, 3)
        try:
            R, t, _, _ = solve_pnp_sequence(K, obs, seeds)
        except Exception:
            continue
        for i in idx:
            meta = json.loads((Path(data_dir) / records[i]["frame"] / "meta.json").read_text(encoding="utf-8"))
            R_gt = np.asarray(meta["R_cb"], np.float64)
            t_gt = np.asarray(meta["t_cb"], np.float64)
            trans, rot = pose_error(R, t, R_gt, t_gt)
            if bool(rot > cfg["flip_deg"]) != records[i]["flip"]:
                changed += 1
            records[i].update({
                "trans_cm": trans,
                "rot_deg": rot,
                "flip": bool(rot > cfg["flip_deg"]),
                "table_cm": _table_error(K, R, t, R_gt, t_gt, cfg),
                "_seq_solved": True,
            })
    return changed


def apply_calibrate_once(records, data_dir, cfg=None, calib_frames=(0, 3)):
    """标一次 T：每序列只用 frame_in_seq 为 calib_frames 的两帧联合解一次位姿，其余帧**不再解 PnP**。

    与 apply_sequence_solve 的区别是那个要先把所有帧都解一遍再覆写；这里在线帧
    完全不解，直接用标定出来的 T。相机不动时序列内各帧的 GT 位姿相同，所以
    「按帧统计」是重复采样同一个样本，报告里按序列计。

    返回 dict：序列数、标定帧/在线帧的重投影 RMS、以及按序列统计的容差达标数。
    """
    cfg = PROTOCOL if cfg is None else cfg
    by_seq: dict = {}
    for i, r in enumerate(records):
        if r.get("_seq") is not None and r.get("_k") is not None:
            by_seq.setdefault(r["_seq"], {})[r["_k"]] = i
    st = {"n_seq": 0, "calib_rms": [], "online_rms": [], "trans_cm": [], "rot_deg": [],
          "n_within": 0, "n_online": 0, "n_skipped": 0, "n_rejected": 0,
          "n_planned": len(by_seq), "n_failed": 0}
    tol_t, tol_r = cfg["tolerance"][0]
    for _, by_k in sorted(by_seq.items()):
        for i in by_k.values():
            for key in ("_Rc", "_tc", "_calib", "_gap_px"):
                records[i].pop(key, None)
            records[i]["_calib_status"] = "skipped"
        # 标定帧按 frame_in_seq 取，不能按「单帧解成功的帧」里的位次取：后者在首帧
        # 失败时会把 30° 基线悄悄换成 20°，而统计仍记在「两帧 30°」名下。取不到就
        # 显式计入跳过数，不静默缩分母。
        if any(k not in by_k or not records[by_k[k]]["pnp_ok"] for k in calib_frames):
            st["n_skipped"] += 1
            continue
        calib = [by_k[k] for k in calib_frames]
        order = [by_k[k] for k in sorted(by_k)]
        metas = [json.loads((Path(data_dir) / records[i]["frame"] / "meta.json").read_text(encoding="utf-8"))
                 for i in order]
        K = np.asarray(records[calib[0]]["_K"], np.float64).reshape(3, 3)
        obs = [(records[i]["_p3"], np.asarray(records[i]["_p2"], np.float64)) for i in calib]
        seeds = [(np.asarray(records[i]["_R"]).reshape(3, 3), np.asarray(records[i]["_t"])) for i in calib]
        try:
            R, t, _, _, sols = solve_pnp_sequence(K, obs, seeds, return_solutions=True)
        except (ValueError, cv2.error):
            st["n_failed"] += 1
            for i in order:
                records[i]["_calib_status"] = "failed"
            continue
        gap_px = degeneracy_gap(sols)  # 退化解边距：双峰歧义探测器，越小越可疑
        # 拒识第一层：物理合法性。错谷解的光心可能在桌面下（val seq30 实测
        # z=-0.017m，真值 +0.416m）。检查不改变解本身，只拒用整个标定：
        # 不写 _Rc/_tc，闭环自动跳过（= 拒抓）；n_rejected 显式计数，不静默缩分母。
        C = camera_center(R, t)
        if C[2] < 0.0 or not 0.15 <= float(np.linalg.norm(C)) <= 2.5:
            st["n_rejected"] += 1
            for i in order:
                records[i]["_calib_status"] = "rejected"
            continue
        for i in order:
            records[i]["_calib_status"] = "accepted"
        rvec = cv2.Rodrigues(R)[0]
        # 相机不动 → 序列内 GT 位姿相同，误差只算一次
        meta0 = metas[order.index(calib[0])]
        R_gt, t_gt = np.asarray(meta0["R_cb"], np.float64), np.asarray(meta0["t_cb"], np.float64)
        trans, rot = pose_error(R, t, R_gt, t_gt)
        table = _table_error(K, R, t, R_gt, t_gt, cfg)
        within = trans <= tol_t and rot <= tol_r
        st["n_seq"] += 1
        st["trans_cm"].append(float(trans))
        st["rot_deg"].append(float(rot))
        st["n_within"] += int(within)
        for i in order:
            p3, p2 = np.asarray(records[i]["_p3"], np.float64), np.asarray(records[i]["_p2"], np.float64)
            uv, _ = cv2.projectPoints(p3, rvec, t, K, None)
            rms = float(np.sqrt(np.mean(np.sum((uv.reshape(-1, 2) - p2) ** 2, axis=1))))
            online = i not in calib
            st["online_rms" if online else "calib_rms"].append(rms)
            st["n_online"] += int(online)
            records[i].update({
                "trans_cm": trans, "rot_deg": rot,
                "flip": bool(rot > cfg["flip_deg"]), "table_cm": table,
                "_calib": "online" if online else "calib",
                # 标出来的 T 本身也要能取用：闭环要拿它把方块像素反投影回桌面
                "_Rc": np.asarray(R).ravel().tolist(),
                "_tc": np.asarray(t).ravel().tolist(),
                "_gap_px": gap_px,
            })
    return st


def _stat(values, percentile=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(values.mean() if percentile is None else np.percentile(values, percentile))


def summarize(records, cfg=None):
    cfg = PROTOCOL if cfg is None else cfg
    ok = [r for r in records if (r.get("_calib_status") == "accepted"
          if "_calib_status" in r else r["pnp_ok"])]
    px = np.array([r["mean_px"] for r in records])
    mat = _stack(records, "px_vec")
    dv = _stack(records, "dv_vec")
    kp = mat.ravel() if mat is not None else px
    trans = np.array([r["trans_cm"] for r in ok])
    rot = np.array([r["rot_deg"] for r in ok])
    table = np.array([r["table_cm"] for r in ok if np.isfinite(r["table_cm"])])
    out = {
        "n_frames": len(records),
        "n_pnp_ok": len(ok),
        # —— 帧级：每帧一个样本 ——
        "n_2d_valid": int(np.isfinite(px).sum()),
        "px_mean": _stat(px),
        "px_median": _stat(px, 50),
        "px_p95": _stat(px, 95),
        "pck3": _stat(px <= cfg["pck_px"][0]),
        "pck5": _stat(px <= cfg["pck_px"][1]),
        # —— 关键点级：每帧每点一个样本 ——
        "n_kp_samples": int(kp.size),
        "kp_px_mean": _stat(kp),
        "kp_px_median": _stat(kp, 50),
        "kp_px_p95": _stat(kp, 95),
        "kp_pck3": _stat(kp <= cfg["pck_px"][0]),
        "kp_pck5": _stat(kp <= cfg["pck_px"][1]),
        "per_kp_mean": [_stat(v) for v in mat.T] if mat is not None else [],
        "bias_v_max": _stat([abs(_stat(v)) for v in dv.T], 100) if dv is not None else float("nan"),
        "nme": _stat([r["nme"] for r in records]),
        "trans_median": _stat(trans, 50),
        "trans_p95": _stat(trans, 95),
        "rot_median": _stat(rot, 50),
        "rot_p95": _stat(rot, 95),
        "flips": int(np.sum(rot > cfg["flip_deg"])),
        "flip_rate": _stat(rot > cfg["flip_deg"]),
        "usable": {f"{tc:g}cm/{rd:g}°": (float(((trans <= tc) & (rot <= rd)).sum())
                   / len(records) if records else float("nan")) for tc, rd in cfg["tolerance"]},
        "n_table_valid": len(table),
        "table_median": _stat(table, 50),
        "table_p95": _stat(table, 95),
        "table_over5": int((table > 5).sum()),
        # 抓取看的是悬崖不是中位数：夹爪横向容差之外直接失败，这个概率才是抓取
        # 成功率的预测值，中位数和 3cm 可用率都反映不出来。
        "p_within_grip": (float((table <= cfg["grasp_tol_cm"]).sum()) / len(records)
                          if records else float("nan")),
        "metric_scope": "误差分位数仅含有限观测；可用率分母含全部输入帧；翻转率仅含有效位姿",
    }
    return out


def bootstrap_ci(records, keys=("px_median", "px_p95", "trans_p95", "rot_p95", "table_p95"), n=1000, seed=0):
    """有序列标识时按整序列重采样，否则按帧；避免把同相机多帧当独立实验。"""
    if not records or n <= 0:
        return {}
    groups = {}
    for i, r in enumerate(records):
        key = ("seq", r["_seq"]) if r.get("_seq") is not None else ("frame", i)
        groups.setdefault(key, []).append(r)
    units = list(groups.values())
    rng = np.random.default_rng(seed)
    values = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(units), len(units))
        s = summarize([r for i in idx for r in units[i]])
        for k in keys:
            values[k].append(s[k])
    return {k: (_stat(v, 2.5), _stat(v, 97.5)) for k, v in values.items()}
