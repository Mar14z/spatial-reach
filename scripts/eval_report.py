"""统一评测 CLI：吃预测 JSON，输出 markdown + JSON（含按帧 bootstrap CI）。

用法：
    python scripts/eval_report.py --data data/synthetic/m0_kp9b_test/test \
        --pred outputs/eval/pred_test_kp9b_gamma.json --label "9点外抱环+gamma"

任何检测器只要产出 {"frames": {帧名: {"pred_2d": [[u,v],...]}}} 就能复用本脚本。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.learning.evaluate import (
    bootstrap_ci,
    evaluate_frames,
    load_predictions,
    summarize,
)


def fmt(v, ci=None, unit=""):
    s = f"{v:.2f}{unit}"
    if ci is not None:
        s += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return s


# GT 点集 PCA 的 minor/major 比：越接近 0，9 个点在图上越接近一条直线。
# 视线几乎平行臂平面时点集退化，PnP 必然镜像双解——这类帧再准也解不出唯一位姿，
# 混在总数里会把可解帧的精度淹没，必须分档看。
STRATA = [(0.0, 0.15, "近共线"), (0.15, 0.30, "边缘"), (0.30, 1.01, "可解")]


def strata(records, cfg):
    out = {}
    for lo, hi, name in STRATA:
        sub = [r for r in records if lo <= r.get("spread2d", np.nan) < hi]
        if not sub or not any(r["pnp_ok"] for r in sub):
            continue
        out[f"{lo:g}-{hi:g}"] = {"name": name, "n": len(sub), **summarize(sub, cfg)}
    return out


def strata_md(st, n_total):
    head = ("| GT 共线性档 | 帧数 | 占比 | kp PCK@3 | kp median | trans median | "
            "usable@3cm/5° | 翻转率 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = "".join(
        f"| {v['name']} {k} | {v['n']} | {v['n']/n_total*100:.1f}% | {v['kp_pck3']*100:.1f}% | "
        f"{v['kp_px_median']:.2f} px | {v['trans_median']:.2f} cm | "
        f"{v['usable'].get('3cm/5°', float('nan'))*100:.1f}% | {v['flip_rate']*100:.1f}% |\n"
        for k, v in st.items())
    return head + rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--extra", type=Path, default=None, help="额外点标签 npz")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    frames, _ = load_predictions(args.pred)
    extra = None
    if args.extra:
        z = np.load(args.extra)
        extra = {str(n): (u, x) for n, u, x in zip(z["frames"], z["uv"], z["xyz"])}
    records, cfg = evaluate_frames(args.data, frames, extra)
    s = summarize(records, cfg)
    ci = bootstrap_ci(records, n=args.bootstrap) if args.bootstrap else {}

    print(f"\n=== {args.label} | {len(records)} 帧 ===")
    # 两级统计单元：数值不可混用，必须标清
    print(f"2D kp  mean {s['kp_px_mean']:.2f}  median {s['kp_px_median']:.2f}  "
          f"p95 {s['kp_px_p95']:.2f} px  n={s['n_kp_samples']} 每帧每点")
    print(f"2D fr  mean {s['px_mean']:.2f}  median {fmt(s['px_median'], ci.get('px_median'), 'px')}  "
          f"p95 {fmt(s['px_p95'], ci.get('px_p95'), 'px')}  n={s['n_frames']} 帧内先求均值")
    print(f"     PCK kp @3 {s['kp_pck3']*100:.1f}%  @5 {s['kp_pck5']*100:.1f}%  "
          f"frame @3 {s['pck3']*100:.1f}%   NME {s['nme']:.4f}")
    print(f"PnP  trans {s['trans_median']:.2f} / {fmt(s['trans_p95'], ci.get('trans_p95'), 'cm')}   "
          f"rot {s['rot_median']:.2f} / {fmt(s['rot_p95'], ci.get('rot_p95'), '°')}   "
          f"flips {s['flips']} ({s['flip_rate']*100:.1f}%)")
    print("use  " + "   ".join(f"@{k} {v*100:.1f}%" for k, v in s["usable"].items()))
    print(f"tbl  median {s['table_median']:.2f}  p95 {fmt(s['table_p95'], ci.get('table_p95'), 'cm')}")

    st = strata(records, cfg)
    if st:
        print("\n按 GT 共线性分档（PCA minor/major）")
        print("  档位           帧数  占比   kpPCK@3  kp med   trans med  usable@3/5  flip")
        for k, v in st.items():
            print(f"  {v['name']:<6} {k:<9} {v['n']:>4}  {v['n']/len(records)*100:>4.1f}%  "
                  f"{v['kp_pck3']*100:>6.1f}%  {v['kp_px_median']:>6.2f}px  "
                  f"{v['trans_median']:>7.2f}cm  {v['usable'].get('3cm/5°', float('nan'))*100:>8.1f}%  "
                  f"{v['flip_rate']*100:>5.1f}%")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "result.json").write_text(
            json.dumps({"label": args.label, "summary": s, "ci95": ci,
                        "strata": st, "per_frame": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.out / "report.md").write_text(
            f"# {args.label}\n\n帧数 {len(records)}\n\n"
            f"| 指标 | 值 | 95% CI |\n|---|---|---|\n"
            f"| 2D median 关键点级 (px) | {s['kp_px_median']:.2f} | — |\n"
            f"| 2D p95 关键点级 (px) | {s['kp_px_p95']:.2f} | — |\n"
            f"| 2D median 帧级 (px) | {s['px_median']:.2f} | {ci.get('px_median')} |\n"
            f"| 2D p95 帧级 (px) | {s['px_p95']:.2f} | {ci.get('px_p95')} |\n"
            f"| PCK@3 关键点级 | {s['kp_pck3']*100:.1f}% | — |\n"
            f"| PCK@3 帧级 | {s['pck3']*100:.1f}% | — |\n"
            f"| trans median (cm) | {s['trans_median']:.2f} | — |\n"
            f"| trans p95 (cm) | {s['trans_p95']:.2f} | {ci.get('trans_p95')} |\n"
            f"| rot median (°) | {s['rot_median']:.2f} | — |\n"
            f"| rot p95 (°) | {s['rot_p95']:.2f} | {ci.get('rot_p95')} |\n"
            f"| flips | {s['flips']} ({s['flip_rate']*100:.1f}%) | — |\n"
            + "".join(f"| usable @{k} | {v*100:.1f}% | — |\n" for k, v in s["usable"].items())
            + f"| table median (cm) | {s['table_median']:.2f} | — |\n"
            f"| table p95 (cm) | {s['table_p95']:.2f} | {ci.get('table_p95')} |\n",
            encoding="utf-8",
        )
        (args.out / "report.md").write_text(
            (args.out / "report.md").read_text(encoding="utf-8")
            + "\n## 按 GT 共线性分档\n\n" + strata_md(st, len(records)),
            encoding="utf-8",
        )
        print("saved", args.out)


if __name__ == "__main__":
    main()
