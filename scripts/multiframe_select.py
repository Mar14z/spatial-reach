"""多帧联合能不能破共面歧义：单帧 vs 跨帧仲裁 vs 联合优化。

设定：相机不动，机械臂转 pan 角 θ（关节角已知 → 3D 点新位置已知）。

两种数据口径，结论不能混：
  synth  两帧都是「真值投影 + 高斯噪声」，自洽，测的是算法上限
  real   第 1 帧用真实网络预测（含系统误差），第 2 帧只能合成

用法：
    python scripts/multiframe_select.py [--mode synth|real|both] [--sigma 2.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from touch.perception.pose import (  # noqa: E402
    select_candidate,
    select_candidate_multiframe,
    solve_pnp_candidates,
    solve_pnp_multiframe,
)

ANGLES = (0, 2, 5, 10, 15, 30)
FLIP_DEG = 15.0


def load_frames(pred_path: Path, split: Path, limit: int):
    pred = json.loads(pred_path.read_text(encoding="utf-8"))["frames"]
    out = []
    for frame in sorted(p for p in split.iterdir() if p.is_dir())[:limit]:
        meta = json.loads((frame / "meta.json").read_text(encoding="utf-8"))
        out.append((
            np.asarray(meta["K"], float),
            np.asarray(meta["keypoints_3d_base"], float),
            np.asarray(pred[frame.name]["pred_2d"], float),
            np.asarray(meta["R_cb"], float),
            np.asarray(meta["t_cb"], float),
        ))
    return out


def rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0],
                     [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def project(P, K, R, t):
    uv, _ = cv2.projectPoints(P, cv2.Rodrigues(R)[0], t, K, None)
    return uv.reshape(-1, 2)


def rot_err(R, Rg):
    return float(np.degrees(np.arccos(np.clip((np.trace(R.T @ Rg) - 1) / 2, -1, 1))))


def sweep(mode: str, frames, sigma: float) -> None:
    rng = np.random.default_rng(0)   # 每个 mode 重置种子，单独跑和 --mode both 结果一致
    print(f"\n[{mode}] 噪声 σ={sigma}px | 判据 rot<{FLIP_DEG:g}° | 样本=有双解的帧")
    print(f"{'pan':>6} {'样本':>6} {'单帧':>8} {'跨帧仲裁':>9} {'联合优化':>9} {'对照真值':>9}")

    for deg in ANGLES:
        n = s_ok = j_ok = o_ok = g_ok = 0
        for K, p3, p2_real, R_gt, t_gt in frames:
            P1, P2 = p3, p3 @ rz(deg).T
            if mode == "synth":
                p2_1 = project(P1, K, R_gt, t_gt) + rng.normal(0, sigma, (len(P1), 2))
            else:
                p2_1 = p2_real
            p2_2 = project(P2, K, R_gt, t_gt) + rng.normal(0, sigma, (len(P2), 2))

            cands = solve_pnp_candidates(K, P1, p2_1)
            if len(cands) < 2:
                continue
            n += 1
            s_ok += rot_err(select_candidate(cands)[0], R_gt) < FLIP_DEG
            j_ok += rot_err(select_candidate_multiframe(K, cands, [(P2, p2_2)])[0][0],
                            R_gt) < FLIP_DEG
            o_ok += rot_err(solve_pnp_multiframe(K, [(P1, p2_1), (P2, p2_2)],
                                                 seeds=[c[:2] for c in cands])[0],
                            R_gt) < FLIP_DEG
            g_ok += rot_err(solve_pnp_multiframe(K, [(P1, p2_1), (P2, p2_2)],
                                                 seeds=[(R_gt, t_gt)])[0],
                            R_gt) < FLIP_DEG
        print(f"{deg:>5.0f}° {n:>6d} {s_ok/n:>8.1%} {j_ok/n:>9.1%} {o_ok/n:>9.1%} {g_ok/n:>9.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--sigma", type=float, default=2.0, help="2D 噪声标准差(px)")
    ap.add_argument("--mode", default="both", choices=("synth", "real", "both"))
    ap.add_argument("--pred", default=str(ROOT / "outputs/eval/pred_test_gamma.json"))
    ap.add_argument("--split", default=str(ROOT / "data/synthetic/m0_kp9b_test/test"))
    args = ap.parse_args()

    frames = load_frames(Path(args.pred), Path(args.split), args.frames)
    modes = ("synth", "real") if args.mode == "both" else (args.mode,)
    for m in modes:
        sweep(m, frames, args.sigma)


if __name__ == "__main__":
    main()
