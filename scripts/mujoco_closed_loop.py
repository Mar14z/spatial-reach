"""闭环抓取：网络预测 2D → 标一次 T → 反投影方块像素 → MuJoCo 抓取，报成功率。

和 mujoco_pick 原有的 24/24、20/20 区别在链路的入口：

    之前的代理实验：方块位置 = 建场景时给的 (radius, azimuth)，解析已知；
                    jitter 是事后注入的横向偏移，方向各向同性。

    本脚本：       方块真实位置 p_true → 用**真相机**投影得像素（完美检测器）
                   → 该像素 + **网络预测的关键点** 解出的 T 反投影回桌面 → p_est
                   → 方块摆在 p_true、指令点给 p_est，让它真的去抓。

于是误差从网络出来、穿过 PnP 和反投影，随相机姿态与方块在图上的位置而变，
不是事后注入的。p_est 的反投影用的就是评测里算 table_cm 的同一个估计器。

臂的 keypoints_3d_base 取自 meta：真机上它来自关节编码器 / FK，本来就不是
感知量，所以这里用真值不算作弊。相机位姿 T 每序列只用两帧标一次，与部署口径一致。

用法：
    python scripts/mujoco_closed_loop.py --limit 3     # 先看 3 个序列
    python scripts/mujoco_closed_loop.py               # 全部 80 个序列
    python scripts/mujoco_closed_loop.py --det-px 1.5  # 叠加检测器像素噪声
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
    apply_calibrate_once, evaluate_frames, load_predictions,
)
from touch.perception.geometry import backproject_to_plane  # noqa: E402

# 方块真实位置：落在 mujoco_pick_sweep 验证过可达的区间内（r 0.09–0.15、
# 方位角相对臂正前方 ±0.4），但不要正好踩在扫描格点上。
CELLS = ((0.10, 0.0), (0.12, -0.3), (0.12, 0.3), (0.14, 0.0))
# 只改方位角、保留每个格子的半径和出现次数：CELLS 里 r=0.12 只配了 ±0.3，
# 一旦失败集中在 r=0.12 就分不清是半径还是偏离光轴。这组用来拆开两者。
CELLS_AZ0 = ((0.10, 0.0), (0.12, 0.0), (0.12, 0.0), (0.14, 0.0))
TOL_MM = 15.0  # 夹爪横向容差，注入实验里开始失败的量级


def main() -> None:
    ap = argparse.ArgumentParser(description="闭环抓取成功率（感知 → PnP → 反投影 → MuJoCo）")
    ap.add_argument("--data", default=str(ROOT / "data/synthetic/m0_seq/val"))
    ap.add_argument("--pred", default=str(ROOT / "outputs/eval/pred_m0_seq_val_ft320.json"))
    ap.add_argument("--det-px", type=float, default=0.0, help="检测器像素噪声标准差")
    ap.add_argument("--frame-k", type=int, default=1,
                    help="每个序列用第几帧当抓取时刻（0/3 参与了标定，用 1/2 才是留出）")
    ap.add_argument("--az-mode", choices=("spread", "zero"), default="spread",
                    help="spread=方位角散开（默认）; zero=全摆正前方，用于拆半径/偏轴")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个序列（快速自查）")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/closed_loop.json"))
    a = ap.parse_args()

    data = Path(a.data)
    frames, _ = load_predictions(Path(a.pred))
    records, cfg = evaluate_frames(data, frames)
    st = apply_calibrate_once(records, data, cfg, (0, 3))
    print(f"标一次 T：{st['n_seq']} 个序列，跳过 {st['n_skipped']} 个"
          f"（检测器噪声 σ={a.det_px:g}px）")

    by_seq: dict = {}
    for r in records:
        if r.get("_seq") is not None and "_Rc" in r and r.get("_k") is not None:
            by_seq.setdefault(r["_seq"], {})[r["_k"]] = r

    rng = np.random.default_rng(0)
    cells = CELLS if a.az_mode == "spread" else CELLS_AZ0
    rows: list = []
    order = sorted(by_seq.items())
    if a.limit:
        order = order[:a.limit]
    for i, (seq, by_k) in enumerate(order):
        r = by_k.get(a.frame_k) or next(iter(by_k.values()))  # 默认在线帧：它没参与标定
        R_est = np.asarray(r["_Rc"], float).reshape(3, 3)
        t_est = np.asarray(r["_tc"], float).reshape(3)
        K = np.asarray(r["_K"], float)
        meta = json.loads((data / r["frame"] / "meta.json").read_text(encoding="utf-8"))
        R_gt = np.asarray(meta["R_cb"], float)
        t_gt = np.asarray(meta["t_cb"], float)

        radius, az_rel = cells[i % len(cells)]
        az = mp.AZ_FORWARD + az_rel
        p_true = np.array([radius * np.cos(az), radius * np.sin(az), 0.0])
        uv, _ = cv2.projectPoints(p_true.reshape(1, 3), cv2.Rodrigues(R_gt)[0], t_gt, K, None)
        uv = uv.reshape(2)
        inside = 0 <= uv[0] < meta["width"] and 0 <= uv[1] < meta["height"]
        if a.det_px > 0:
            uv = uv + rng.normal(0.0, a.det_px, 2)
        p_est = backproject_to_plane(K, uv, R_est, t_est, 0.0)[0]
        err_mm = float(np.linalg.norm((p_est - p_true)[:2])) * 1000.0

        res = mp.pick(0.03, radius, az, verbose=False, jitter=(p_est - p_true)[:2])
        # ok_grasp 才是夹爪容差判据（注入实验 20/20 用的就是它）；ok 额外要求
        # align<10mm，比夹爪实际能力严，标成 OK* 区分开。
        tag = "OK" if res["ok"] else ("OK*" if res["ok_grasp"] else "FAIL")
        rows.append({"seq": int(seq), "radius": radius, "az_rel": az_rel, "uv": uv.tolist(),
                     "visible": bool(inside), "err_mm": err_mm, "ok": bool(res["ok"]),
                     "ok_grasp": bool(res["ok_grasp"]), "align_mm": res["align"] * 1000,
                     "in_hand_mm": res["in_hand"] * 1000, "dz_mm": res["dz"] * 1000,
                     "slip_mm": res["slip"] * 1000, "gap_mm": res["gap"]})
        print(f"{i + 1:3d}/{len(order)} seq={seq:3d} r={radius:.2f} az{az_rel:+.1f} "
              f"err={err_mm:5.1f}mm {tag:5s} align={res['align'] * 1000:5.1f} "
              f"in_hand={res['in_hand'] * 1000:5.1f}" + ("" if inside else "  [方块出画]"),
              flush=True)

    n = len(rows)
    ok = np.array([r["ok"] for r in rows])
    grasp = np.array([r["ok_grasp"] for r in rows])
    e = np.array([r["err_mm"] for r in rows])
    vis = np.array([r["visible"] for r in rows])
    print(f"\n闭环 n={n}   OK=严格对齐  OK*=抓到但对齐≥10mm（仍在夹爪容差内）  FAIL=没抓到")
    print(f"  严格（抬升>5cm 且 对齐<10mm 且 夹持内偏移<20mm）{ok.sum()}/{n} = {ok.mean():.1%}")
    print(f"  抓到（抬升>2cm 且 夹持内偏移<20mm）            {grasp.sum()}/{n} = {grasp.mean():.1%}")
    print(f"  桌面误差 median {np.median(e):.1f}mm  p95 {np.percentile(e, 95):.1f}mm  "
          f"P(≤{TOL_MM:.0f}mm) {np.mean(e <= TOL_MM):.1%}  max {e.max():.1f}mm")
    for name, mask in ((f"误差≤{TOL_MM:.0f}mm", e <= TOL_MM), (f"误差>{TOL_MM:.0f}mm", e > TOL_MM)):
        if mask.sum():
            print(f"    {name}：{mask.sum():3d} 次，严格成功 {ok[mask].mean():.1%}，"
                  f"抓到 {grasp[mask].mean():.1%}")
    if not vis.all():
        print(f"  警告：{int((~vis).sum())} 个格子方块投到画外（相机不同），这些不是真实场景")
    Path(a.out).write_text(json.dumps({"det_px": a.det_px, "tol_mm": TOL_MM,
                                       "rows": rows}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
