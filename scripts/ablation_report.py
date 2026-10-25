"""同规格消融评测：输入 val 目录 + 预测 JSON，输出一行完整指标。

用法：
    python scripts/ablation_report.py <val_dir> <pred_json> <label>
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touch.perception.pose import solve_pnp

val_dir, pred_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
pred = json.loads(Path(pred_path).read_text(encoding="utf-8"))["frames"]
frames = sorted(p for p in Path(val_dir).iterdir() if p.is_dir())
extra = None
if len(sys.argv) > 4:
    z = np.load(sys.argv[4])
    extra = {str(n): (u, x) for n, u, x in zip(z["frames"], z["uv"], z["xyz"])}

gx, gy = np.meshgrid(np.linspace(0.05, 0.45, 6), np.linspace(-0.2, 0.2, 5))
TABLE = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

# 一级指标：翻转（镜像解）与可用率（位姿落在容差内的帧占比）。
# 中位误差会掩盖双峰分布，不能单独用作优化目标。
FLIP_DEG = 15.0
USABLE = ((3.0, 5.0), (5.0, 10.0))


def table_cm(K, R, t, R_gt, t_gt):
    uv, _ = cv2.projectPoints(TABLE, R_gt, t_gt, K, None)
    uv = uv.reshape(-1, 2)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
    if inside.sum() < 3:
        return np.nan
    C = -R.T @ t
    d = R.T @ (np.linalg.inv(K) @ np.hstack([uv[inside], np.ones((inside.sum(), 1))]).T)
    X = C[:, None] + d * (-C[2] / d[2])
    return float(np.linalg.norm(X.T - TABLE[inside], axis=1).mean() * 100)


rows, kp_errs, kp_bias, per_kp = [], [], [], []
for frame in frames:
    meta = json.loads((frame / "meta.json").read_text(encoding="utf-8"))
    gt = np.asarray(meta["keypoints_2d"], dtype=np.float64)
    p3 = np.asarray(meta["keypoints_3d_base"], dtype=np.float64)
    if extra is not None:
        uv_x, xyz_x = extra[frame.name]
        gt = np.vstack([gt, uv_x])
        p3 = np.vstack([p3, xyz_x])
    p2 = np.asarray(pred[frame.name]["pred_2d"], dtype=np.float64)
    K = np.asarray(meta["K"], dtype=np.float64)
    R_gt = np.asarray(meta["R_cb"], dtype=np.float64)
    t_gt = np.asarray(meta["t_cb"], dtype=np.float64)
    e = np.linalg.norm(p2 - gt, axis=1)
    kp_errs.append(e)
    kp_bias.append((p2 - gt)[:, 1])
    per_kp.append(e)
    R, t, _ = solve_pnp(K, p3, p2, 5.0, use_ransac=False, method=cv2.SOLVEPNP_SQPNP)
    rot = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1, 1)))
    rows.append((
        np.linalg.norm(-R.T @ t + R_gt.T @ t_gt) * 100, rot, table_cm(K, R, t, R_gt, t_gt),
        float(e.mean() / max(np.linalg.norm(gt[min(8, len(gt) - 1)] - gt[0]), 1.0)),
    ))

E, B, A = np.stack(kp_errs), np.stack(kp_bias), np.asarray(rows)
a = A[~np.isnan(A[:, 2])]
print(f"=== {label} ({len(frames)} val frames, {E.shape[1]} keypoints) ===")
print(f"2D   mean {E.mean():5.2f}  median {np.median(E):5.2f}  p95 {np.percentile(E,95):5.2f} px")
print(f"PCK  @3 {(E<=3).mean()*100:5.1f}%  @5 {(E<=5).mean()*100:5.1f}%   NME {A[:,3].mean():.4f}")
print(f"bias |dv| max {np.abs(B.mean(0)).max():4.2f} px  worst kp{int(np.argmax(E.mean(0)))} {E.mean(0).max():.2f}px")
print(f"per-kp px: " + " ".join(f"{v:.2f}" for v in E.mean(0)))
print(f"PnP  trans {np.median(a[:,0]):5.2f}/{np.percentile(a[:,0],95):6.2f} cm   "
      f"rot {np.median(a[:,1]):5.2f}/{np.percentile(a[:,1],95):6.2f} deg   "
      f"flips {(a[:,1]>FLIP_DEG).sum():2d} ({(a[:,1]>FLIP_DEG).mean()*100:4.1f}%)")
print("use  " + "   ".join(
    f"@{tc:g}cm/{rd:g}° {((a[:,0] <= tc) & (a[:,1] <= rd)).mean()*100:5.1f}%"
    for tc, rd in USABLE))
print(f"tbl  median {np.median(a[:,2]):5.2f}  p95 {np.percentile(a[:,2],95):6.2f} cm   >5cm {(a[:,2]>5).sum():2d}")
