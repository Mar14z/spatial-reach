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

# 可扩样本集：val 是同分布验证集，mild/hard 是分布外序列（翻转率更高，
# 正好拿来压失败样本）。三个集都是 4 帧/序列，帧 0/3 标定、1/2 留出。
SETS = {
    "val": ("data/synthetic/m0_seq/val", "outputs/eval/pred_m0_seq_val_ft320.json"),
    "mild": ("data/synthetic/m0_ood_seq/mild", "outputs/eval/pred_m0_ood_seq_mild_ft320.json"),
    "hard": ("data/synthetic/m0_ood_seq/hard", "outputs/eval/pred_m0_ood_seq_hard_ft320.json"),
}


def classif(r):
    """失败归因：先看误差大小，再看是哪种位姿病。"""
    if r["err_mm"] > 20:
        band = "位置偏>20mm"
    elif r["err_mm"] > TOL_MM:
        band = "15-20mm 临界"
    else:
        band = "容差内"
    if r.get("flip"):
        pose = "翻转"
    elif (r.get("rot_deg") or 0) >= 5:
        pose = "旋转误差≥5°"
    elif (r.get("trans_cm") or 0) >= 10:
        pose = "大平移小旋转"
    else:
        pose = "位姿指标看着正常"
    return band, pose


def main() -> None:
    ap = argparse.ArgumentParser(description="闭环抓取成功率（感知 → PnP → 反投影 → MuJoCo）")
    ap.add_argument("--sets", default="val", help="逗号分隔：val / mild / hard")
    ap.add_argument("--data", default="", help="覆盖 --sets，直接指定数据目录")
    ap.add_argument("--pred", default="", help="配合 --data")
    ap.add_argument("--calib", default="0,3",
                    help="联合解 T 用哪几帧（逗号分隔）；none = 每帧单独解（逐帧基线）")
    ap.add_argument("--frames", default="1", help="抓取时刻用哪几帧，逗号分隔；0/3 参与标定则用 1,2")
    ap.add_argument("--det-px", type=float, default=0.0, help="检测器像素噪声标准差")
    ap.add_argument("--az-mode", choices=("spread", "zero"), default="spread",
                    help="spread=方位角散开（默认）; zero=全摆正前方，用于拆半径/偏轴")
    ap.add_argument("--limit", type=int, default=0, help="每个集只跑前 N 个序列")
    ap.add_argument("--no-sim", action="store_true", help="只算误差不仿真（秒级，用于口径对比）")
    ap.add_argument("--out", default=str(ROOT / "outputs/eval/closed_loop.json"))
    a = ap.parse_args()

    calib = None if a.calib == "none" else tuple(int(x) for x in a.calib.split(","))
    ks = [int(x) for x in a.frames.split(",")]
    cells = CELLS if a.az_mode == "spread" else CELLS_AZ0
    rng = np.random.default_rng(0)
    rows: list = []

    for name in a.sets.split(","):
        data, pred = ((Path(a.data), Path(a.pred)) if a.data else
                      (ROOT / SETS[name][0], ROOT / SETS[name][1]))
        frames, _ = load_predictions(pred)
        records, cfg = evaluate_frames(data, frames)
        if calib is not None:
            st = apply_calibrate_once(records, data, cfg, calib)
            print(f"[{name}] 标定帧 {a.calib} 联合：{st['n_seq']} 个序列，"
                  f"跳过 {st['n_skipped']} 个；抓取帧 {ks}，仿真={not a.no_sim}")
        else:
            print(f"[{name}] 逐帧单解基线；抓取帧 {ks}，仿真={not a.no_sim}")

        by_seq: dict = {}
        for r in records:
            if r.get("_seq") is not None and r.get("_k") is not None:
                by_seq.setdefault(r["_seq"], {})[r["_k"]] = r
        order = sorted(by_seq.items())
        if a.limit:
            order = order[:a.limit]

        for seq, by_k in order:
            for k in ks:
                r = by_k.get(k)
                if r is None or not r.get("pnp_ok"):
                    continue
                if calib is None:
                    R_est = np.asarray(r["_R"], float).reshape(3, 3)
                    t_est = np.asarray(r["_t"], float).reshape(3)
                elif "_Rc" in r:
                    R_est = np.asarray(r["_Rc"], float).reshape(3, 3)
                    t_est = np.asarray(r["_tc"], float).reshape(3)
                else:
                    continue  # 该序列标定失败

                K = np.asarray(r["_K"], float)
                meta = json.loads((data / r["frame"] / "meta.json").read_text(encoding="utf-8"))
                radius, az_rel = cells[len(rows) % len(cells)]
                az = mp.AZ_FORWARD + az_rel
                p_true = np.array([radius * np.cos(az), radius * np.sin(az), 0.0])
                uv, _ = cv2.projectPoints(
                    p_true.reshape(1, 3), cv2.Rodrigues(np.asarray(meta["R_cb"], float))[0],
                    np.asarray(meta["t_cb"], float), K, None)
                uv = uv.reshape(2)
                inside = 0 <= uv[0] < meta["width"] and 0 <= uv[1] < meta["height"]
                if a.det_px > 0:
                    uv = uv + rng.normal(0.0, a.det_px, 2)
                p_est = backproject_to_plane(K, uv, R_est, t_est, 0.0)[0]
                err_mm = float(np.linalg.norm((p_est - p_true)[:2])) * 1000.0
                # 本帧在标定 T 下的重投影残差：部署时算得出来（不用 GT），可当质量分
                uv_fit, _ = cv2.projectPoints(
                    np.asarray(r["_p3"], float), cv2.Rodrigues(R_est)[0], t_est, K, None)
                rms_px = float(np.sqrt(np.mean(np.sum(
                    (uv_fit.reshape(-1, 2) - np.asarray(r["_p2"], float)) ** 2, axis=1))))
                row = {"set": name, "seq": int(seq), "k": int(k), "radius": radius,
                       "az_rel": az_rel, "uv": uv.tolist(), "visible": bool(inside),
                       "err_mm": err_mm, "rms_px": rms_px, "trans_cm": r.get("trans_cm"),
                       "rot_deg": r.get("rot_deg"), "flip": r.get("flip"),
                       "table_cm": r.get("table_cm"), "kp_px": r.get("mean_px")}
                if a.no_sim:
                    tag = "-"
                else:
                    res = mp.pick(0.03, radius, az, verbose=False, jitter=(p_est - p_true)[:2])
                    # ok_grasp 才是夹爪容差判据（注入实验 20/20 用的就是它）；ok 额外要求
                    # align<10mm，比夹爪实际能力严，标成 OK* 区分开。
                    tag = "OK" if res["ok"] else ("OK*" if res["ok_grasp"] else "FAIL")
                    row.update({"ok": bool(res["ok"]), "ok_grasp": bool(res["ok_grasp"]),
                                "align_mm": res["align"] * 1000,
                                "in_hand_mm": res["in_hand"] * 1000,
                                "dz_mm": res["dz"] * 1000, "slip_mm": res["slip"] * 1000,
                                "pad_rise_mm": res["pad_rise"] * 1000, "gap_mm": res["gap"]})
                rows.append(row)
                print(f"[{name}] {len(rows):4d} seq={seq:3d} k={k} r={radius:.2f} "
                      f"az{az_rel:+.1f} err={err_mm:5.1f}mm {tag:5s}"
                      + ("" if inside else "  [方块出画]"), flush=True)

    e = np.array([r["err_mm"] for r in rows])
    print(f"\n=== n={len(rows)} 次取点 ===")
    print(f"取点误差 median {np.median(e):.1f}mm  p95 {np.percentile(e, 95):.1f}mm  "
          f"P(≤{TOL_MM:.0f}mm) {np.mean(e <= TOL_MM):.1%}  max {e.max():.1f}mm")
    if not a.no_sim:
        grasp = np.array([r["ok_grasp"] for r in rows])
        ok = np.array([r["ok"] for r in rows])
        print(f"抓到 {grasp.sum()}/{len(rows)} = {grasp.mean():.1%}   "
              f"严格对齐 {ok.sum()}/{len(rows)} = {ok.mean():.1%}")
        for lo, hi in ((0, 15), (15, 20), (20, 30), (30, 1e9)):
            m = (e > lo) & (e <= hi)
            if m.sum():
                print(f"  误差 ({lo:2d},{hi if hi < 1e8 else 999:3d}]mm: {m.sum():3d} 次，"
                      f"抓到 {grasp[m].mean():5.1%}")
        bad = [r for r in rows if not r["ok_grasp"]]
        print(f"\n没抓到 {len(bad)} 次，归因（误差档 × 位姿病）：")
        tally: dict = {}
        for r in bad:
            band, pose = classif(r)
            tally[(band, pose)] = tally.get((band, pose), 0) + 1
            print(f"   [{r['set']}] seq={r['seq']:3d} k={r['k']} err={r['err_mm']:5.1f}mm "
                  f"trans={r['trans_cm']:5.1f}cm rot={r['rot_deg']:5.1f}° flip={r['flip']} "
                  f"align={r['align_mm']:5.1f} in_hand={r['in_hand_mm']:5.1f}"
                  f"  -> {band} / {pose}")
        for (band, pose), c in sorted(tally.items(), key=lambda x: -x[1]):
            print(f"     {band:12s} | {pose:14s} : {c}")
    for name in a.sets.split(","):
        m = np.array([r["set"] == name for r in rows])
        if m.sum():
            extra = ""
            if not a.no_sim:
                extra = f" 抓到 {np.mean([r['ok_grasp'] for r in rows if r['set'] == name]):.1%}"
            print(f"  [{name}] n={m.sum():3d} 误差中位 {np.median(e[m]):5.1f}mm "
                  f"P(≤{TOL_MM:.0f}mm) {np.mean(e[m] <= TOL_MM):.1%}{extra}")

    Path(a.out).write_text(json.dumps(
        {"sets": a.sets, "calib": a.calib, "frames": ks, "det_px": a.det_px,
         "az_mode": a.az_mode, "no_sim": a.no_sim, "tol_mm": TOL_MM, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
