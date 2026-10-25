"""数据集覆盖面体检：参数分布、关键点边距、与参考集的分布差异。

用途：每生成一个新数据集就跑一遍。只看指标不看采样分布，是没法判断
「这个数据集到底覆盖了什么」的——尤其是当 train 和 test 同分布时，
测试集上的好成绩只说明模型记住了见过的世界。

用法：
    python scripts/check_dataset_coverage.py data/synthetic/m0_kp9b/train
    python scripts/check_dataset_coverage.py data/synthetic/m0_ood --ref data/synthetic/m0_kp9b/train

KS = 两样本 Kolmogorov-Smirnov 统计量（两个分布 CDF 的最大差），
取值 0-1；< 0.2 基本同分布，> 0.5 才算明显不同。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    ("yaw", lambda m: m["camera"]["yaw"]),
    ("pitch", lambda m: m["camera"]["pitch"]),
    ("dist", lambda m: m["camera"]["distance"]),
    ("pan", lambda m: m["joints"]["arm_shoulder_pan_joint"]),
    ("light.x", lambda m: m["light"]["direction"][0]),
    ("light.y", lambda m: m["light"]["direction"][1]),
    ("ambient", lambda m: m["light"]["ambient"]),
]


def load(path):
    metas = [
        json.loads((d / "meta.json").read_text(encoding="utf-8"))
        for d in sorted(p for p in Path(path).iterdir() if p.is_dir())
    ]
    if not metas:
        raise SystemExit(f"no frames under {path}")
    return metas


def margin(m):
    """关键点离画面边界的最小距离(px)。越小越接近「贴边/出画」。"""
    return min(min(u, v, m["width"] - u, m["height"] - v) for u, v in m["keypoints_2d"])


def ks_2samp(a, b):
    grid = np.concatenate([a, b])
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--ref", type=Path, default=None, help="参考集，用于算 KS")
    args = ap.parse_args()

    metas = load(args.data)
    ref = load(args.ref) if args.ref else None
    print(f"\n=== {args.data}  n={len(metas)} ===")
    print(f"{'field':>10} {'min':>8} {'p05':>8} {'med':>8} {'p95':>8} {'max':>8}"
          + (f" {'KS':>6}" if ref else ""))

    rows = [(name, np.array([fn(m) for m in metas])) for name, fn in FIELDS]
    rows.append(("kp_margin", np.array([margin(m) for m in metas])))
    rows.append(("kp_valid", np.array([np.mean(m["valid"]) for m in metas])))

    for name, arr in rows:
        line = (f"{name:>10} {arr.min():8.2f} {np.percentile(arr, 5):8.2f}"
                f" {np.median(arr):8.2f} {np.percentile(arr, 95):8.2f} {arr.max():8.2f}")
        if ref:
            ref_fn = dict(FIELDS + [("kp_margin", margin),
                                    ("kp_valid", lambda m: np.mean(m["valid"]))])[name]
            line += f" {ks_2samp(arr, np.array([ref_fn(m) for m in ref])):6.2f}"
        print(line)

    yaw = np.array([m["camera"]["yaw"] for m in metas]) % 360
    print(f"{'yaw/45deg':>10} {np.histogram(yaw, bins=8, range=(0, 360))[0].tolist()}")
    mg = np.array([margin(m) for m in metas])
    for thr in (10, 30, 60):
        print(f"{'margin<' + str(thr):>10} {(mg < thr).mean() * 100:5.1f}%")
    print(f"{'shadow=1':>10} {np.mean([m.get('shadow', 0) for m in metas]) * 100:5.1f}%")


if __name__ == "__main__":
    main()
