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
    print(f"2D   mean {s['px_mean']:.2f}  median {fmt(s['px_median'], ci.get('px_median'), 'px')}  "
          f"p95 {fmt(s['px_p95'], ci.get('px_p95'), 'px')}")
    print(f"     PCK@3 {s['pck3']*100:.1f}%  PCK@5 {s['pck5']*100:.1f}%  NME {s['nme']:.4f}")
    print(f"PnP  trans {s['trans_median']:.2f} / {fmt(s['trans_p95'], ci.get('trans_p95'), 'cm')}   "
          f"rot {s['rot_median']:.2f} / {fmt(s['rot_p95'], ci.get('rot_p95'), '°')}   flips {s['flips']}")
    print(f"tbl  median {s['table_median']:.2f}  p95 {fmt(s['table_p95'], ci.get('table_p95'), 'cm')}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "result.json").write_text(
            json.dumps({"label": args.label, "summary": s, "ci95": ci,
                        "per_frame": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.out / "report.md").write_text(
            f"# {args.label}\n\n帧数 {len(records)}\n\n"
            f"| 指标 | 值 | 95% CI |\n|---|---|---|\n"
            f"| 2D median (px) | {s['px_median']:.2f} | {ci.get('px_median')} |\n"
            f"| 2D p95 (px) | {s['px_p95']:.2f} | {ci.get('px_p95')} |\n"
            f"| PCK@3 | {s['pck3']*100:.1f}% | — |\n"
            f"| trans median (cm) | {s['trans_median']:.2f} | — |\n"
            f"| trans p95 (cm) | {s['trans_p95']:.2f} | {ci.get('trans_p95')} |\n"
            f"| rot p95 (°) | {s['rot_p95']:.2f} | {ci.get('rot_p95')} |\n"
            f"| flips | {s['flips']} | — |\n"
            f"| table median (cm) | {s['table_median']:.2f} | — |\n"
            f"| table p95 (cm) | {s['table_p95']:.2f} | {ci.get('table_p95')} |\n",
            encoding="utf-8",
        )
        print("saved", args.out)


if __name__ == "__main__":
    main()
