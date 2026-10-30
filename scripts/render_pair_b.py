"""L2 记忆实验数据：B 相机下 0/3 标定帧、1/2 留出帧（RGB 只拍臂）。

「A 看物体 → B 只看臂」的设计来自用户：不要求 B 位置看不到物体，只要 B 能独立
标定。B 位姿优先取 outputs/eval/blind_view.json（L1 已证明全臂可见），没有的
现场采样（同样只要求全臂可见）。产出目录结构与 m0_seq 一致，可直接喂模型预测。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import generate_frames as gf
from generate_frames import p
from touch.sim.renderer import (
    opencv_extrinsics_from_view,
    project_link_keypoints,
    render_frame,
)

URDF = ROOT / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints_v1.urdf"
from experiment_config import AZ_FORWARD, CELLS  # noqa: E402


def box_point(seq: int) -> np.ndarray:
    r, az_rel = CELLS[(2 * seq) % len(CELLS)]
    az = AZ_FORWARD + az_rel
    return np.array([r * np.cos(az), r * np.sin(az), 0.015])


def arm_visible(bid, joint_idx, kp, poses, view, margin: float = 10.0) -> float:
    """所有姿态下 9 点都在画面内则返回最小边距，否则 -1。"""
    best = np.inf
    for pose in poses:
        gf.apply_pose(bid, joint_idx, pose)
        _, _, uv, valid, _ = project_link_keypoints(
            bid, kp, gf.W, gf.H, 0.0, view, K=gf.K)
        d = np.minimum(np.minimum(uv[:, 0], gf.W - uv[:, 0]),
                       np.minimum(uv[:, 1], gf.H - uv[:, 1]))
        best = min(best, float(d.min()))
        if not (valid.all() and best >= margin):
            return -1.0
    return best


def box_visible_px(box: np.ndarray, view: list) -> int:
    """临时放 3cm 方块渲一次，返回可见像素数（只记录，不作约束）。"""
    cid = p.createMultiBody(
        0,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.015] * 3, rgbaColor=[1, 0, 0, 0.6]),
        basePosition=[float(box[0]), float(box[1]), 0.015])
    try:
        _, _, seg = render_frame(gf.W, gf.H, 0.0, view, gf.NEAR, gf.FAR, K=gf.K)
        return int((seg == cid).sum())
    finally:
        p.removeBody(cid)


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染 B 相机配对帧（L2 记忆实验）")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--blind", type=Path,
                    default=ROOT / "outputs/eval/blind_view.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/synthetic/m0_seq_b_v2/val")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    blind = {}
    if a.blind.exists():
        d = json.loads(a.blind.read_text(encoding="utf-8"))
        blind = {r["seq"]: r for r in d["rows"] if "failed" not in r}
    rng = np.random.default_rng(a.seed)
    bid, joint_idx, link_idx, table_id = gf.setup(rng, URDF)

    data = ROOT / "data/synthetic/m0_seq/val"
    seqs: dict = {}
    for d0 in sorted(data.iterdir()):
        m = json.loads((d0 / "meta.json").read_text(encoding="utf-8"))
        seqs.setdefault(m["sequence"], {})[m["frame_in_seq"]] = (d0, m)

    a.out.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for seq in sorted(seqs)[: a.n]:
        frames = seqs[seq]
        if not {0, 1, 2, 3}.issubset(frames):
            raise ValueError(f"序列 {seq} 缺少 0/3 标定帧或 1/2 留出帧")
        fi_render = (0, 1, 2, 3)
        poses = [frames[f][1]["joints"] for f in fi_render]
        kp = {n: link_idx[n] for n in frames[0][1]["link_names"]}
        m0 = frames[0][1]
        cam_a = m0["camera"]
        box = box_point(seq)

        view = None
        if seq in blind:
            cb = blind[seq]["B"]
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cb["target"], distance=cb["dist"],
                yaw=cb["yaw"], pitch=cb["pitch"], roll=0, upAxisIndex=2)
            cam_b = dict(cb)
            if arm_visible(bid, joint_idx, kp, poses, view) < 10:
                view = None
        if view is None:  # 现场采一个全臂可见的 B（不要求看不见方块）
            for _ in range(600):
                yaw = float(rng.uniform(0, 360))
                dy = abs(yaw - cam_a["yaw"]) % 360
                if min(dy, 360 - dy) < 45:
                    continue
                pitch = float(rng.uniform(-22, -6))
                dist = float(rng.uniform(0.45, 0.65))
                target = [0.04, 0.0, float(rng.uniform(0.16, 0.21))]
                v = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=target, distance=dist, yaw=yaw,
                    pitch=pitch, roll=0, upAxisIndex=2)
                if arm_visible(bid, joint_idx, kp, poses, v) >= 10:
                    view = v
                    cam_b = {"yaw": yaw, "pitch": pitch, "dist": dist,
                             "target": target}
                    break
            if view is None:
                print(f"seq {seq}: B 采样失败，跳过")
                continue

        p.changeVisualShape(table_id, -1, rgbaColor=m0["table_color"])
        R_cb, t_cb = opencv_extrinsics_from_view(view)
        for f in fi_render:
            if f not in frames:
                continue
            src_dir, meta_a = frames[f]
            gf.apply_pose(bid, joint_idx, meta_a["joints"])
            vis_px = box_visible_px(box, view)
            lt = meta_a["light"]
            rgb, _, _ = render_frame(
                gf.W, gf.H, 0.0, view, gf.NEAR, gf.FAR, K=gf.K,
                shadow=meta_a.get("shadow", 1),
                light_direction=tuple(lt["direction"]),
                light_color=tuple(lt["color"]),
                ambient=lt["ambient"], diffuse=lt["diffuse"],
                specular=lt["specular"])
            names, p3, uv, valid, _ = project_link_keypoints(
                bid, kp, gf.W, gf.H, 0.0, view, K=gf.K)
            if not valid.all():
                print(f"seq {seq} f{f}: 有 keypoint 不可见（不应发生）")
            frame_id = f"s{seq:03d}_f{f}"
            out = a.out / frame_id
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(out / "rgb.png")
            meta = {
                "frame": frame_id, "src_frame": src_dir.name,
                "width": gf.W, "height": gf.H, "K": gf.K.tolist(),
                "R_cb": R_cb.tolist(), "t_cb": t_cb.tolist(),
                "camera": cam_b, "view_matrix": list(view),
                "sequence": seq, "frame_in_seq": f, "seq_len": len(fi_render),
                "frame_role": "calibration" if f in (0, 3) else "heldout",
                "visibility_version": 2, "visibility_frame_in_seq": f,
                "rgb_contains_target": False, "box_point_base": box.tolist(),
                "joints": meta_a["joints"], "link_names": names,
                "keypoints_3d_base": p3.tolist(),
                "keypoints_2d": uv.tolist(),
                "light": lt, "table_color": m0["table_color"],
                "near": gf.NEAR, "far": gf.FAR,
                "box_visible_px": vis_px,  # B 位置能否看到方块（仅记录）
            }
            (out / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            n_written += 1
        print(f"seq {seq:3d} | B yaw={cam_b['yaw']:.0f}° 方块可见 "
              f"{vis_px:4d}px", flush=True)

    print(f"\n写出 {n_written} 帧 →", a.out)


if __name__ == "__main__":
    main()
