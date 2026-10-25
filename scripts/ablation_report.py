"""同规格消融评测：输入 val 目录 + 预测 JSON，输出一行完整指标。

指标实现全部来自 touch.learning.evaluate —— 本脚本只做格式化，不再自己算一遍，
避免与 eval_report.py 分叉出第二套口径（历史教训：同一份预测曾报出 1.79px 与
2.03px，差别只在统计单元）。

2D 误差同时报两级，数值不可混用：
  - 2Dkp 关键点级：每帧每点一个样本，n = 帧数 × 点数，PCK 的常规口径
  - 2Dfr 帧级：帧内先求均值，n = 帧数，尾部被削平，数值偏小

用法：
    python scripts/ablation_report.py <val_dir> <pred_json> <label>
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from touch.learning.evaluate import evaluate_frames, load_predictions, summarize


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    val_dir, pred_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
    frames, _ = load_predictions(pred_path)
    extra = None
    if len(sys.argv) > 4:
        z = np.load(sys.argv[4])
        extra = {str(n): (u, x) for n, u, x in zip(z["frames"], z["uv"], z["xyz"])}

    records, cfg = evaluate_frames(val_dir, frames, extra)
    s = summarize(records, cfg)
    per_kp = " ".join(f"{v:.2f}" for v in s["per_kp_mean"])
    use = "   ".join(f"@{k} {v * 100:5.1f}%" for k, v in s["usable"].items())

    print(f"=== {label} ({s['n_frames']} val frames, {len(s['per_kp_mean'])} keypoints) ===")
    # 同一种误差、两种统计单元，数值不可混用
    print(f"2D kp  mean {s['kp_px_mean']:5.2f}  median {s['kp_px_median']:5.2f}  "
          f"p95 {s['kp_px_p95']:5.2f} px  n={s['n_kp_samples']:4d} 每帧每点")
    print(f"2D fr  mean {s['px_mean']:5.2f}  median {s['px_median']:5.2f}  "
          f"p95 {s['px_p95']:5.2f} px  n={s['n_frames']:4d} 帧内先求均值")
    print(f"PCK  kp @3 {s['kp_pck3'] * 100:5.1f}%  @5 {s['kp_pck5'] * 100:5.1f}%   "
          f"frame @3 {s['pck3'] * 100:5.1f}%   NME {s['nme']:.4f}")
    print(f"bias |dv| max {s['bias_v_max']:4.2f} px  "
          f"worst kp{int(np.argmax(s['per_kp_mean']))} {max(s['per_kp_mean']):.2f}px")
    print(f"per-kp px: {per_kp}")
    print(f"PnP  trans {s['trans_median']:5.2f}/{s['trans_p95']:6.2f} cm   "
          f"rot {s['rot_median']:5.2f}/{s['rot_p95']:6.2f} deg   "
          f"flips {s['flips']:2d} ({s['flip_rate'] * 100:4.1f}%)")
    print(f"use  {use}")
    print(f"tbl  median {s['table_median']:5.2f}  p95 {s['table_p95']:6.2f} cm   "
          f">5cm {s['table_over5']:2d}")


if __name__ == "__main__":
    main()
