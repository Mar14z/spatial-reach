"""统一评测入口：预测 JSON + 数据目录 → 四级指标（2D / 位姿 / 桌面 / 可靠性）。

只依赖 numpy + OpenCV，不碰 torch —— 任何检测器（CNN、HSV、ArUco）都只需产出
同样格式的预测 JSON 即可复用这套口径。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from touch.perception.geometry import backproject_to_plane, pose_error
from touch.perception.pose import solve_pnp

PROTOCOL = {
    "solver": cv2.SOLVEPNP_SQPNP,
    "reproj_threshold_px": 5.0,
    "flip_deg": 15.0,
    "pck_px": (3.0, 5.0),
    "plane_z": 0.0,
    "grid_x": (0.05, 0.45, 6),
    "grid_y": (-0.2, 0.2, 5),
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
        p2 = np.asarray(pred_frames[frame.name]["pred_2d"], dtype=np.float64)
        K = np.asarray(meta["K"], dtype=np.float64)
        R_gt = np.asarray(meta["R_cb"], dtype=np.float64)
        t_gt = np.asarray(meta["t_cb"], dtype=np.float64)
        e = np.linalg.norm(p2 - gt, axis=1)
        ref = np.linalg.norm(gt[min(8, len(gt) - 1)] - gt[0])
        s2 = np.linalg.svd(gt - gt.mean(0), compute_uv=False)
        rec = {
            "frame": frame.name,
            "mean_px": float(e.mean()),
            "nme": float(e.mean() / max(ref, 1.0)),
            "max_bias_v": float(np.abs((p2 - gt)[:, 1].mean())),
            "spread2d": float(s2[1] / s2[0]) if len(gt) > 2 else np.nan,
            "pnp_ok": False,
        }
        try:
            R, t, _ = solve_pnp(K, p3, p2, cfg["reproj_threshold_px"], False, cfg["solver"])
            trans, rot = pose_error(R, t, R_gt, t_gt)
            rec.update({
                "pnp_ok": True,
                "trans_cm": trans,
                "rot_deg": rot,
                "flip": bool(rot > cfg["flip_deg"]),
                "table_cm": _table_error(K, R, t, R_gt, t_gt, cfg),
            })
        except Exception as exc:
            rec["error"] = str(exc)
        records.append(rec)
    return records, cfg


def summarize(records, cfg=None):
    cfg = PROTOCOL if cfg is None else cfg
    ok = [r for r in records if r["pnp_ok"]]
    px = np.array([r["mean_px"] for r in records])
    trans = np.array([r["trans_cm"] for r in ok])
    rot = np.array([r["rot_deg"] for r in ok])
    table = np.array([r["table_cm"] for r in ok if np.isfinite(r["table_cm"])])
    out = {
        "n_frames": len(records),
        "n_pnp_ok": len(ok),
        "px_mean": float(px.mean()),
        "px_median": float(np.median(px)),
        "px_p95": float(np.percentile(px, 95)),
        "pck3": float((px <= cfg["pck_px"][0]).mean()),
        "pck5": float((px <= cfg["pck_px"][1]).mean()),
        "nme": float(np.mean([r["nme"] for r in records])),
        "trans_median": float(np.median(trans)),
        "trans_p95": float(np.percentile(trans, 95)),
        "rot_median": float(np.median(rot)),
        "rot_p95": float(np.percentile(rot, 95)),
        "flips": int(np.sum(rot > cfg["flip_deg"])),
        "table_median": float(np.median(table)),
        "table_p95": float(np.percentile(table, 95)),
    }
    return out


def bootstrap_ci(records, keys=("px_median", "px_p95", "trans_p95", "rot_p95", "table_p95"), n=1000, seed=0):
    """按“帧”有放回重采样，返回每个指标的 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    values = {k: [] for k in keys}
    for _ in range(n):
        idx = rng.integers(0, len(records), len(records))
        s = summarize([records[i] for i in idx])
        for k in keys:
            values[k].append(s[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in values.items()}
