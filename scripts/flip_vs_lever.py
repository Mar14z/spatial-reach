"""侧向凸耳高度 δ 对 PnP 翻转率的影响（纯几何扫描，不重新渲染）。

原理：9 个标记几乎严格共面（平面度≈2e-4），镜像解无法靠重投影残差排除。
把 2 个标记沿臂平面法向移出 δ，点集变为非共面，镜像解被几何唯一性排除。
本脚本在每帧真实的臂平面法向上施加 δ，给出 δ → 翻转率 曲线，用于选凸耳高度。

只有凸耳这 2 个点用「重投影 + 高斯噪声」模拟（因为几何变了，旧预测像素不再适用）；
其余点沿用真实网络预测出的 2D，保留真实偏差。

用法：
    python scripts/flip_vs_lever.py [--frames 500] [--sigma 2.0]
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
from touch.perception.pose import camera_center, planarity_ratio, select_candidate, solve_pnp_candidates

DELTAS_CM = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)


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


def lug_indices(p3: np.ndarray) -> list[int]:
    """沿主方向取两端的点当凸耳，几何条件最好。"""
    _, _, vt = np.linalg.svd(p3 - p3.mean(axis=0))
    proj = (p3 - p3.mean(axis=0)) @ vt[0]
    return [int(np.argmin(proj)), int(np.argmax(proj))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--sigma", type=float, default=2.0, help="凸耳点的 2D 噪声标准差(px)")
    ap.add_argument("--pred", default=str(ROOT / "outputs/eval/pred_test_gamma.json"))
    ap.add_argument("--split", default=str(ROOT / "data/synthetic/m0_kp9b_test/test"))
    args = ap.parse_args()

    frames = load_frames(Path(args.pred), Path(args.split), args.frames)
    rng = np.random.default_rng(0)
    print(f"{len(frames)} frames | lens σ={args.sigma}px | 指标为 (median/p95)")
    print(f"{'δ(cm)':>6} {'平面度':>8} {'翻转':>10} {'trans cm':>14} {'rot deg':>13} {'可用@3cm5°':>10}")

    for delta_cm in DELTAS_CM:
        delta = delta_cm / 100.0
        flips, trans, rot, pl = 0, [], [], []
        for K, p3, p2, R_gt, t_gt in frames:
            p3s = p3.copy()
            idx = lug_indices(p3)
            _, _, vt = np.linalg.svd(p3 - p3.mean(axis=0))
            p3s[idx] += delta * vt[2]
            p2s = p2.copy()
            if delta > 0:
                uv, _ = cv2.projectPoints(p3s[idx], cv2.Rodrigues(R_gt)[0], t_gt, K, None)
                p2s[idx] = uv.reshape(-1, 2) + rng.normal(0, args.sigma, (len(idx), 2))
            R, t, _, _ = select_candidate(solve_pnp_candidates(K, p3s, p2s))
            tr = float(np.linalg.norm(camera_center(R, t) - camera_center(R_gt, t_gt)) * 100)
            ro = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1, 1))))
            trans.append(tr)
            rot.append(ro)
            flips += ro > 15
            pl.append(planarity_ratio(p3s))
        tr, ro = np.asarray(trans), np.asarray(rot)
        usable = ((tr <= 3) & (ro <= 5)).mean() * 100
        print(f"{delta_cm:6.1f} {np.median(pl):8.4f} {flips:4d}/{len(tr):<5d} "
              f"{np.median(tr):6.2f}/{np.percentile(tr, 95):6.2f} "
              f"{np.median(ro):5.2f}/{np.percentile(ro, 95):5.2f} {usable:9.1f}%")


if __name__ == "__main__":
    main()
