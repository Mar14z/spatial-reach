"""PnP 对照评测：真值 vs 预测 keypoints，6 点 vs 5 点。

预测点由 scripts/predict_keypoints.py 生成，本脚本只依赖 OpenCV + numpy。

用法：
    conda activate arm_sim
    python scripts/eval_pnp.py
输出：
    outputs/eval/pnp_errors.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.perception.pose import solve_pnp

VAL = ROOT / "data/synthetic/m0_cad/val"
PRED = ROOT / "outputs/eval/pred_keypoints.json"
OUT = ROOT / "outputs/eval/pnp_errors.json"


def rotation_error_deg(R_est, R_gt):
    cos = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def camera_center(R_cb, t_cb):
    return -R_cb.T @ t_cb


def summarize(records):
    ok = [r for r in records if r["ok"]]
    if not ok:
        return {"n": len(records), "fail": len(records)}
    trans = np.array([r["trans_cm"] for r in ok])
    rot = np.array([r["rot_deg"] for r in ok])
    inliers = np.array([r["inliers"] for r in ok])
    return {
        "n": len(records),
        "ok": len(ok),
        "fail": len(records) - len(ok),
        "trans_cm_mean": float(trans.mean()),
        "trans_cm_median": float(np.median(trans)),
        "trans_cm_p95": float(np.percentile(trans, 95)),
        "trans_cm_max": float(trans.max()),
        "rot_deg_mean": float(rot.mean()),
        "rot_deg_median": float(np.median(rot)),
        "rot_deg_p95": float(np.percentile(rot, 95)),
        "rot_deg_max": float(rot.max()),
        "inliers_mean": float(inliers.mean()),
        "inliers_min": int(inliers.min()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, default=VAL)
    parser.add_argument("--pred", type=Path, default=PRED)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--drop-gripper", action="store_true")
    args = parser.parse_args()

    predictions = json.loads(
        args.pred.read_text(encoding="utf-8")
    )["frames"]
    variants = {
        "gt_all": [],
        "pred_all_ransac": [],
        "pred_all_iter": [],
        "pred_all_sqpnp": [],
    }

    frame_dirs = sorted(p for p in args.val.iterdir() if p.is_dir())
    for index, frame_dir in enumerate(frame_dirs):
        meta = json.loads(
            (frame_dir / "meta.json").read_text(encoding="utf-8")
        )
        points3d = np.asarray(meta["keypoints_3d_base"], dtype=np.float64)
        gt2d = np.asarray(meta["keypoints_2d"], dtype=np.float64)
        pred2d = np.asarray(
            predictions[frame_dir.name]["pred_2d"], dtype=np.float64
        )
        K = np.asarray(meta["K"], dtype=np.float64)
        R_gt = np.asarray(meta["R_cb"], dtype=np.float64)
        t_gt = np.asarray(meta["t_cb"], dtype=np.float64)

        n = len(pred2d)
        pred_err = np.linalg.norm(pred2d[:n] - gt2d[:n], axis=1).mean()
        print(f"{index:03d} {frame_dir.name} | pred {pred_err:.2f}px")

        specs = [
            ("gt_all", points3d[:n], gt2d[:n], False, cv2.SOLVEPNP_ITERATIVE),
            ("pred_all_ransac", points3d[:n], pred2d, True, cv2.SOLVEPNP_ITERATIVE),
            ("pred_all_iter", points3d[:n], pred2d, False, cv2.SOLVEPNP_ITERATIVE),
            ("pred_all_sqpnp", points3d[:n], pred2d, False, cv2.SOLVEPNP_SQPNP),
        ]
        for name, pts3, pts2, ransac, method in specs:
            record = {"frame": frame_dir.name, "ok": False}
            try:
                R_est, t_est, inliers = solve_pnp(
                    K,
                    pts3,
                    pts2,
                    args.threshold,
                    use_ransac=ransac,
                    method=method,
                )
                record.update(
                    {
                        "ok": True,
                        "trans_cm": float(
                            100.0
                            * np.linalg.norm(
                                camera_center(R_est, t_est)
                                - camera_center(R_gt, t_gt)
                            )
                        ),
                        "rot_deg": rotation_error_deg(R_est, R_gt),
                        "inliers": int(len(inliers)),
                    }
                )
            except Exception as exc:
                record["error"] = str(exc)
            variants[name].append(record)

    summary = {name: summarize(records) for name, records in variants.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"summary": summary, "records": variants},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== summary ===")
    for name, s in summary.items():
        if s.get("ok", 0) == 0:
            print(f"{name}: no successful PnP ({s['fail']} failures)")
            continue
        print(
            f"{name}: ok {s['ok']}/{s['n']} | "
            f"trans mean {s['trans_cm_mean']:.3f}cm "
            f"p95 {s['trans_cm_p95']:.3f} max {s['trans_cm_max']:.3f} | "
            f"rot mean {s['rot_deg_mean']:.3f}° "
            f"p95 {s['rot_deg_p95']:.3f} max {s['rot_deg_max']:.3f} | "
            f"inliers {s['inliers_mean']:.2f}/{s['inliers_min']}"
        )
    print("saved", args.out)


if __name__ == "__main__":
    main()
