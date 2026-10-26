"""多帧联解需要几帧、动多大角：真实渲染 + 真实预测的扫描。

数据用 m0_seq/val：同序列相机不动（实测光心漂移 0、姿态漂移 0.04°），pan 每帧
+10°，4 帧共 30°。与 multiframe_select.py 的区别是那里第 2 帧是合成噪声，这里
每一帧都是渲染图 + 网络预测，测的是真实施工条件下的收益。

用法：
    python scripts/multiframe_pairs.py [--pred outputs/eval/pred_cos_seq_val.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from touch.learning.evaluate import PROTOCOL  # noqa: E402
from touch.perception.geometry import pose_error  # noqa: E402
from touch.perception.pose import (  # noqa: E402
    solve_pnp,
    solve_pnp_candidates,
    solve_pnp_sequence,
)

VARIANTS = [
    ("单帧", [(0,), (1,), (2,), (3,)]),
    ("双帧 10°", [(0, 1), (1, 2), (2, 3)]),
    ("双帧 20°", [(0, 2), (1, 3)]),
    ("双帧 30°", [(0, 3)]),
    ("三帧 20°", [(0, 1, 2), (1, 2, 3)]),
    ("四帧 30°", [(0, 1, 2, 3)]),
]


def load_groups(split: Path, pred_path: Path):
    pred = json.loads(pred_path.read_text(encoding="utf-8"))["frames"]
    groups: dict[int, list] = {}
    for frame in sorted(p for p in split.iterdir() if p.is_dir()):
        meta = json.loads((frame / "meta.json").read_text(encoding="utf-8"))
        groups.setdefault(meta["sequence"], []).append((
            meta["frame_in_seq"],
            np.asarray(meta["K"], float),
            np.asarray(meta["keypoints_3d_base"], float),
            np.asarray(pred[frame.name]["pred_2d"], float),
            np.asarray(meta["R_cb"], float),
            np.asarray(meta["t_cb"], float),
        ))
    return [sorted(g, key=lambda x: x[0]) for g in groups.values()]


def sweep(groups, rows) -> None:
    print(f"\n判据：翻转 = 旋转 > {PROTOCOL['flip_deg']:g}°；可用 = 平移 ≤3cm 且旋转 ≤5°")
    print("| 口径 | 组数 | 单帧翻转 | 单帧可用 | 联解翻转 | 联解可用 | 联解 rot p95 | 联解 trans p95 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for name, combos in VARIANTS:
        single_rot, joint_rot = [], []
        for frames in groups:
            by_k = {f[0]: f for f in frames}
            for combo in combos:
                if any(k not in by_k for k in combo):
                    continue
                K = by_k[combo[0]][1]
                obs = [(by_k[k][2], by_k[k][3]) for k in combo]
                R_gt, t_gt = by_k[combo[0]][4], by_k[combo[0]][5]
                seeds = []
                for _, _, p3, p2, _, _ in (by_k[k] for k in combo):
                    R, t, _ = solve_pnp(K, p3, p2, PROTOCOL["reproj_threshold_px"],
                                      False, PROTOCOL["solver"])
                    single_rot.append(pose_error(R, t, R_gt, t_gt))
                    seeds.append((R, t))
                R, t, _, _ = solve_pnp_sequence(K, obs, seeds)
                joint_rot.append(pose_error(R, t, R_gt, t_gt))
        _report(name, single_rot, joint_rot, rows)


def _report(name, single, joint, rows) -> None:
    flip = PROTOCOL["flip_deg"]
    tol = PROTOCOL["tolerance"][0]
    s_rot = np.array([e[1] for e in single])
    s_cm = np.array([e[0] for e in single])
    j_rot = np.array([e[1] for e in joint])
    j_cm = np.array([e[0] for e in joint])
    s_ok = (s_cm <= tol[0]) & (s_rot <= tol[1])
    j_ok = (j_cm <= tol[0]) & (j_rot <= tol[1])
    rows.append({
        "name": name, "n_single": int(len(s_rot)), "n_joint": int(len(j_rot)),
        "single_flip": float((s_rot > flip).mean()), "single_usable": float(s_ok.mean()),
        "joint_flip": float((j_rot > flip).mean()), "joint_usable": float(j_ok.mean()),
        "joint_rot_p95": float(np.percentile(j_rot, 95)),
        "joint_trans_p95": float(np.percentile(j_cm, 95)),
        "single_rot_p95": float(np.percentile(s_rot, 95)),
        "single_trans_p95": float(np.percentile(s_cm, 95)),
    })
    print(f"| {name} | {len(s_rot)}/{len(j_rot)} | {(s_rot > flip).mean():.1%} | "
          f"{s_ok.mean():.1%} | {(j_rot > flip).mean():.1%} | {j_ok.mean():.1%} | "
          f"{np.percentile(j_rot, 95):.1f}° | {np.percentile(j_cm, 95):.2f}cm |")


def rescue(groups) -> None:
    """四帧全错时联解救不救得回来——决定「值不值得多拍 3 帧」。"""
    flip = PROTOCOL["flip_deg"]
    all_flip = rescued = any_flip = fixed = 0
    for frames in groups:
        K = frames[0][1]
        R_gt, t_gt = frames[0][4], frames[0][5]
        seeds, bad = [], 0
        for _, _, p3, p2, _, _ in frames:
            R, t, _ = solve_pnp(K, p3, p2, PROTOCOL["reproj_threshold_px"],
                                False, PROTOCOL["solver"])
            seeds.append((R, t))
            bad += pose_error(R, t, R_gt, t_gt)[1] > flip
        R, t, _, _ = solve_pnp_sequence(
            K, [(f[2], f[3]) for f in frames], seeds)
        joint_bad = pose_error(R, t, R_gt, t_gt)[1] > flip
        any_flip += bad > 0
        all_flip += bad == len(frames)
        rescued += (bad == len(frames)) and not joint_bad
        fixed += (bad > 0) and not joint_bad
    print(f"\n有任一帧翻转的序列 {any_flip} 组，联解修好 {fixed} 组；"
          f"四帧全翻的 {all_flip} 组，联解救回 {rescued} 组")


def policy(groups) -> None:
    """运行时判据：单帧候选出现两个相距 >5° 的解才值得再拍一帧。

    只看当前帧能算出来的量（候选分歧），不看 GT、不看翻转。
    """
    flip, tol = PROTOCOL["flip_deg"], PROTOCOL["tolerance"][0]
    n_trig = ok_pol = ok_single = 0
    joint_used = 0
    for frames in groups:
        K, R_gt, t_gt = frames[0][1], frames[0][4], frames[0][5]
        cands = solve_pnp_candidates(K, frames[0][2], frames[0][3],
                                     PROTOCOL["reproj_threshold_px"])
        gap = max((_gap(cands[0], c) for c in cands[1:]), default=0.0)
        R, t, _ = solve_pnp(K, frames[0][2], frames[0][3],
                            PROTOCOL["reproj_threshold_px"], False, PROTOCOL["solver"])
        if gap > 5.0:
            n_trig += 1
            seeds = [(c[0], c[1]) for c in cands]
            R, t, _, _ = solve_pnp_sequence(
                K, [(f[2], f[3]) for f in frames], seeds)
            joint_used += 1
        trans, rot = pose_error(R, t, R_gt, t_gt)
        good = trans <= tol[0] and rot <= tol[1]
        ok_pol += good
        R0, t0, _ = solve_pnp(K, frames[0][2], frames[0][3],
                              PROTOCOL["reproj_threshold_px"], False, PROTOCOL["solver"])
        s_trans, s_rot = pose_error(R0, t0, R_gt, t_gt)
        ok_single += s_trans <= tol[0] and s_rot <= tol[1]
    print(f"\n触发策略（首帧候选分歧 >5°）：{n_trig}/{len(groups)} 组需补拍，"
          f"多拍 {joint_used} 帧；可用率 {ok_single}/{len(groups)} → {ok_pol}/{len(groups)}")


def _gap(a, b) -> float:
    ang = np.degrees(np.arccos(np.clip((np.trace(a[0].T @ b[0]) - 1) / 2, -1, 1)))
    cm = np.linalg.norm(-a[0].T @ a[1] - (-b[0].T @ b[1])) * 100
    return max(float(ang), cm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(ROOT / "data/synthetic/m0_seq/val"))
    ap.add_argument("--pred", default=str(ROOT / "outputs/eval/pred_cos_seq_val.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/multiframe_pairs.json"))
    args = ap.parse_args()
    groups = load_groups(Path(args.split), Path(args.pred))
    print(f"序列 {len(groups)} 组，每组 {len(groups[0])} 帧")
    rows: list = []
    sweep(groups, rows)
    rescue(groups)
    policy(groups)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
