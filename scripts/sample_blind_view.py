"""L1 记忆实验：给每个场景搜一个 B 相机——看得全臂（可标定）、看不到方块（失明）。

验证「相机换位置后靠记忆抓」的前提：B 位置必须真的看不到方块（出框或被臂挡），
同时 9 个臂关键点全在画面内（B 位置能独立标定）。输出相机参数供 L2 渲染配对数据。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import generate_frames as gf
from generate_frames import p
from touch.sim.renderer import (
    opencv_extrinsics_from_view,
    project_base_points,
    project_link_keypoints,
    render_frame,
)

URDF = ROOT / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints_v1.urdf"
# k=1 对应 2*seq（每序列两行轮转），与其它实验脚本共用同一套格子。
from experiment_config import AZ_FORWARD, CELLS  # noqa: E402


def box_point(seq: int) -> np.ndarray:
    r, az_rel = CELLS[(2 * seq) % len(CELLS)]
    az = AZ_FORWARD + az_rel
    return np.array([r * np.cos(az), r * np.sin(az), 0.015])


def render_with_cube(box: np.ndarray, view: list):
    """临时放一个 3cm 方块渲染一次，返回 (方块可见像素数, rgb, seg)。

    中心点被臂挡住不等于方块看不见——必须按体判定：seg 里方块像素数为 0
    才算真失明。方块只是判定用的道具，渲完立刻删。
    """
    cid = p.createMultiBody(
        0,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.015] * 3, rgbaColor=[1, 0, 0, 0.6]),
        basePosition=[float(box[0]), float(box[1]), 0.015],
    )
    try:
        rgb, _, seg = render_frame(
            gf.W, gf.H, 0.0, view, gf.NEAR, gf.FAR, K=gf.K)
        return int((seg == cid).sum()), rgb, seg
    finally:
        p.removeBody(cid)


def save_evidence(path: Path, rgb: np.ndarray, view: list, box: np.ndarray) -> None:
    """存一张 B 视角证据图：红框标出方块真值位置（画在臂的后面 = 被挡）。"""
    from PIL import Image, ImageDraw

    R, t = opencv_extrinsics_from_view(view)
    h = 0.015
    corners = np.array([[box[0] + sx * h, box[1] + sy * h, box[2] + sz * h]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    uv, _ = project_base_points(gf.K, R, t, corners)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    draw.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=2)
    draw.text((max(4.0, x0), max(4.0, y0 - 14)), "box GT: occluded",
              fill=(255, 40, 40))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="搜索看不到方块的 B 相机位姿（L1）")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/eval/blind_view.json")
    a = ap.parse_args()

    ev_dir = a.out.parent / "blind_views"
    ev_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    bid, joint_idx, link_idx, table_id = gf.setup(rng, URDF)

    data = ROOT / "data/synthetic/m0_seq/val"
    seqs: dict = {}
    for d in sorted(data.iterdir()):
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        seqs.setdefault(m["sequence"], {})[m["frame_in_seq"]] = m

    rows = []
    for seq in sorted(seqs)[: a.n]:
        frames = seqs[seq]
        poses = [frames[k]["joints"] for k in sorted(frames)]
        # 点集与顺序以 meta 为准（不同 URDF 的 kp_* 布局不同）
        kp = {n: link_idx[n] for n in frames[0]["link_names"]}
        cam_a = frames[0]["camera"]
        box = box_point(seq)
        found = None
        for _ in range(600):
            yaw = float(rng.uniform(0, 360))
            dy = abs(yaw - cam_a["yaw"]) % 360
            if min(dy, 360 - dy) < 45:  # 必须真的换了个位置
                continue
            pitch = float(rng.uniform(-22, -6))
            dist = float(rng.uniform(0.45, 0.65))
            target = [0.04, 0.0, float(rng.uniform(0.16, 0.21))]
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target, distance=dist, yaw=yaw,
                pitch=pitch, roll=0, upAxisIndex=2)
            # 1) 全臂可见（4 帧 × 9 点，10px 边距）——B 位置要能独立标定
            ok, margin = True, np.inf
            for pose in poses:
                gf.apply_pose(bid, joint_idx, pose)
                _, _, uv, valid, _ = project_link_keypoints(
                    bid, kp, gf.W, gf.H, 0.0, view, K=gf.K)
                d = np.minimum(
                    np.minimum(uv[:, 0], gf.W - uv[:, 0]),
                    np.minimum(uv[:, 1], gf.H - uv[:, 1]))
                margin = min(margin, float(d.min()))
                if not (valid.all() and margin >= 10):
                    ok = False
                    break
            if not ok:
                continue
            # 2) 快速初筛：方块中心投影出框，或视线被臂挡住
            R, t = opencv_extrinsics_from_view(view)
            uv_box, ok_box = project_base_points(gf.K, R, t, box.reshape(1, 3))
            u, v = float(uv_box[0, 0]), float(uv_box[0, 1])
            inside = bool(ok_box[0]) and 0 <= u < gf.W and 0 <= v < gf.H
            hit_id = p.rayTest((-R.T @ t).tolist(), box.tolist())[0][0]
            if inside and hit_id != bid:
                continue  # 视线畅通 = 看得见，不合格
            # 3) 体判定（硬标准）：渲染 seg，方块可见像素必须为 0
            vis_px, rgb, _ = render_with_cube(box, view)
            if vis_px:
                continue
            ev = ev_dir / f"seq_{seq:03d}.png"
            save_evidence(ev, rgb, view, box)
            found = {"seq": seq, "A": cam_a,
                     "B": {"yaw": yaw, "pitch": pitch, "dist": dist,
                           "target": target},
                     "box_uv": [u, v],
                     "mode": "out" if not inside else "occluded",
                     "visible_px": vis_px,
                     "kp_margin_px": margin,
                     "yaw_gap_deg": min(dy, 360 - dy),
                     "evidence": str(ev.relative_to(ROOT))}
            break
        rows.append(found or {"seq": seq, "failed": True})

    ok_rows = [r for r in rows if "failed" not in r]
    print(f"B 相机搜索: {len(ok_rows)}/{len(rows)} 成功")
    if ok_rows:
        out = sum(r["mode"] == "out" for r in ok_rows)
        print(f"  失明机制: 出框 {out} / 被臂挡 {len(ok_rows) - out}")
        print(f"  B-A yaw 差: 中位 {np.median([r['yaw_gap_deg'] for r in ok_rows]):.1f}°")
        print(f"  臂边距: 中位 {np.median([r['kp_margin_px'] for r in ok_rows]):.1f}px"
              f"  最小 {min(r['kp_margin_px'] for r in ok_rows):.1f}px")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"n": len(rows), "ok": len(ok_rows), "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
