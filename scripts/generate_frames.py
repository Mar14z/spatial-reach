"""生成 M0 合成帧：RGB、深度、分割、2D keypoints 真值。

相机与真实 Ace Pro 的**模型输入**对齐：
- 真实相机 1280×720 内参从 configs/camera_intrinsics.yaml 读取
- 中心裁 4:3 再缩放到 640×480
- 直接用裁切后的 K 在 640×480 上渲染，等价于真实模型输入

--urdf 必须显式给：不同 URDF 的 keypoint 布局不同，拿错 URDF 生成的数据
不会报错，但和 checkpoint 对不上。各数据集的配方记在 data/manifest.json。

用法：
    conda activate arm_sim
    python scripts/generate_frames.py --urdf assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints_v1.urdf --frames 10 --seed 0
    python scripts/generate_frames.py --urdf assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints_v1.urdf --frames 2000 --seed 500 \
        --out data/synthetic/m0_kp9/train --yaw-min 0 --yaw-max 360 \
        --dist-min 0.35 --dist-max 0.60 --pitch-min -25 --pitch-max -5 --no-depth
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.camera import load_intrinsics, model_intrinsics
from touch.sim.renderer import (
    opencv_extrinsics_from_view,
    post_process,
    project_link_keypoints,
    render_frame,
)

OUT = ROOT / "data/synthetic/m0"

INTR_FULL = load_intrinsics(ROOT / "configs/camera_intrinsics.yaml")
INTR = model_intrinsics(INTR_FULL, 640, 480)
W, H = INTR.width, INTR.height
K = INTR.K
NEAR, FAR = 0.01, 5.0

JOINT_RANGES = {
    "arm_shoulder_pan_joint": (-1.2, 1.2),
    "arm_bicep_joint": (-1.2, 1.0),
    "arm_forearm_joint": (-1.0, 0.5),
    "arm_wrist_flex_joint": (-1.0, 1.0),
    "gripper_joint": (0.005, 0.03),
}

KEYPOINT_LINKS = [
    "arm_shoulder_pan_link",
    "arm_bicep_link",
    "arm_forearm_link",
    "arm_wrist_flex_link",
    "gripper_active_link",
]

KP_LINK_NAMES = [
    "kp_base",
    "kp_shoulder_pan",
    "kp_shoulder_lift",
    "kp_upper_mid",
    "kp_elbow",
    "kp_forearm_mid",
    "kp_wrist",
    "kp_base_left",
    "kp_base_right",
]

COLORS = {
    "arm_base_link": [0.20, 0.20, 0.23, 1.0],
    "arm_shoulder_pan_link": [0.20, 0.20, 0.23, 1.0],
    "arm_bicep_link": [0.62, 0.62, 0.65, 1.0],
    "arm_forearm_link": [0.62, 0.62, 0.65, 1.0],
    "arm_wrist_flex_link": [0.20, 0.20, 0.23, 1.0],
    "gripper_active_link": [0.85, 0.15, 0.15, 1.0],
    "gripper_active2_link": [0.15, 0.70, 0.30, 1.0],
}

TABLE_COLORS = [
    [0.72, 0.74, 0.78, 1.0],
    [0.55, 0.42, 0.30, 1.0],
    [0.30, 0.42, 0.55, 1.0],
    [0.35, 0.50, 0.35, 1.0],
    [0.65, 0.60, 0.45, 1.0],
]


def keypoint_link_names(link_idx, override=None):
    """定下本次生成用哪几个 link 当 keypoint。

    带标记的 URDF 里有一堆 kp_* link，但布局不止一种（v1 是 9 点全在臂链上，
    新版是 7 链 + 2 底座横点）。名字对不上时必须报错：静默退回 6 点会让
    生成的数据点和 checkpoint 不匹配，而且不会抛任何异常。
    """
    if override:
        missing = [n for n in override if n not in link_idx]
        if missing:
            raise ValueError(f"URDF 缺少关键点 link: {missing}")
        return override
    if all(name in link_idx for name in KP_LINK_NAMES):
        return KP_LINK_NAMES
    found = sorted(n for n in link_idx if n.startswith("kp_"))
    if found:
        raise ValueError(
            "这个 URDF 的 keypoint 布局与默认不符，必须用 --kp-names 显式指定。"
            f"URDF 里有: {found}"
        )
    base = "base_link" if "base_link" in link_idx else "arm_base_link"
    return [base, *KEYPOINT_LINKS]


def setup(rng, urdf_path):
    p.connect(p.DIRECT)
    p.setGravity(0, 0, 0)
    bid = p.loadURDF(str(urdf_path), useFixedBase=True)
    joint_idx = {
        p.getJointInfo(bid, i)[1].decode(): i
        for i in range(p.getNumJoints(bid))
    }
    link_idx = {
        p.getJointInfo(bid, i)[12].decode(): i
        for i in range(p.getNumJoints(bid))
    }
    half = [0.55, 0.55, 0.01]
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half, rgbaColor=TABLE_COLORS[0]
    )
    table_id = p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=vis, basePosition=[0.12, 0.0, -0.01]
    )
    for name, rgba in COLORS.items():
        if name in link_idx:
            p.changeVisualShape(bid, link_idx[name], rgbaColor=rgba)
    if "base_link" in link_idx:
        p.changeVisualShape(
            bid, link_idx["base_link"], rgbaColor=COLORS["arm_base_link"]
        )
    return bid, joint_idx, link_idx, table_id


def set_random_pose(bid, joint_idx, rng, exclude=()):
    pose = {}
    for name, (low, high) in JOINT_RANGES.items():
        if name in exclude:
            continue
        q = float(rng.uniform(low, high))
        p.resetJointState(bid, joint_idx[name], q)
        pose[name] = q
    return pose


def apply_pose(bid, joint_idx, pose):
    """把一整套关节角设进仿真（序列集里每帧切一次姿态用）。"""
    for name, q in pose.items():
        p.resetJointState(bid, joint_idx[name], q)


def randomize_light(rng, x_range=(-0.5, 0.8), y_range=(-1.0, -0.2)):
    """光源方向/颜色/强度。y_range 默认恒为负（光总是从前方来）；
    改成正值就得到与训练集相反的光照方向，用于构造 OOD 集。"""
    return {
        "direction": [
            float(rng.uniform(*x_range)),
            float(rng.uniform(*y_range)),
            1.0,
        ],
        "color": [float(rng.uniform(0.85, 1.0)) for _ in range(3)],
        "ambient": float(rng.uniform(0.55, 0.85)),
        "diffuse": float(rng.uniform(0.55, 0.80)),
        "specular": float(rng.uniform(0.15, 0.35)),
    }


def sample_view(bid, joint_idx, link_idx, rng, args, poses=None, max_tries=200):
    """采一个相机位姿。

    poses 给了多套关节角时，要求每一套下所有 keypoint 都落在画面内才接受。
    序列集靠这个保证「同一台相机能拍完整个序列」，否则后面的帧会出画。
    """
    for _ in range(max_tries):
        yaw = float(rng.uniform(args.yaw_min, args.yaw_max))
        pitch = float(rng.uniform(args.pitch_min, args.pitch_max))
        dist = float(rng.uniform(args.dist_min, args.dist_max))
        target = [0.04, 0.0, float(rng.uniform(0.16, 0.21))]
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=dist,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
        )
        ok = True
        for pose in poses or [None]:
            if pose is not None:
                apply_pose(bid, joint_idx, pose)
            _, _, uv, valid, _ = project_link_keypoints(
                bid, link_idx, W, H, 0.0, view, K=K
            )
            inside = (
                (uv[:, 0] >= 10)
                & (uv[:, 0] < W - 10)
                & (uv[:, 1] >= 10)
                & (uv[:, 1] < H - 10)
            )
            if not (valid.all() and inside.all()):
                ok = False
                break
        if ok:
            return view, yaw, pitch, dist, target
    raise RuntimeError("no view keeps all keypoints inside the image")


def save_depth_png(path, depth):
    finite = depth[np.isfinite(depth)]
    low, high = float(finite.min()), float(finite.max())
    vis = ((depth - low) / (high - low + 1e-9) * 255).clip(0, 255)
    Image.fromarray(vis.astype(np.uint8)).save(path)


def save_seg_png(path, seg):
    ids = np.unique(seg)
    rng = np.random.default_rng(0)
    palette = {
        int(i): rng.integers(0, 255, 3, dtype=np.uint8) for i in ids
    }
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for seg_id, color in palette.items():
        out[seg == seg_id] = color
    Image.fromarray(out).save(path)


def save_overlay(path, rgb, uv):
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for i, (u, v) in enumerate(uv):
        u, v = int(round(u)), int(round(v))
        draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=(255, 60, 30), width=2)
        draw.text((u + 5, v - 5), str(i), font=font, fill=(255, 255, 255))
    img.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--yaw-min", type=float, default=0.0)
    parser.add_argument("--yaw-max", type=float, default=360.0)
    parser.add_argument("--pitch-min", type=float, default=-22.0)
    parser.add_argument("--pitch-max", type=float, default=-6.0)
    parser.add_argument("--dist-min", type=float, default=0.45)
    parser.add_argument("--dist-max", type=float, default=0.65)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument(
        "--kp-names",
        type=str,
        default=None,
        help="逗号分隔的 keypoint link 名单，默认用内置 9 点",
    )
    parser.add_argument(
        "--finger-colors",
        choices=["black", "split"],
        default="split",
        help="夹爪配色：split=红绿（新数据默认），black=与旧 checkpoint 一致",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        required=True,
        help="必填：keypoint 布局由 URDF 决定，配方见 data/manifest.json",
    )
    parser.add_argument("--light-x-min", type=float, default=-0.5)
    parser.add_argument("--light-x-max", type=float, default=0.8)
    parser.add_argument("--light-y-min", type=float, default=-1.0)
    parser.add_argument("--light-y-max", type=float, default=-0.2)
    parser.add_argument("--shadow", type=int, choices=[0, 1], default=0)
    parser.add_argument("--noise-sigma", type=float, default=0.0,
                        help="高斯噪声标准差（0-255 尺度），0 表示不加")
    parser.add_argument("--blur", action="store_true", help="3x3 高斯模糊")
    parser.add_argument("--jpeg-q", type=int, default=0,
                        help="JPEG 质量，0 表示不做 JPEG 压缩")
    parser.add_argument("--seq-len", type=int, default=1,
                        help=">1 时生成序列：同序列相机/桌面/光照固定，只转 pan")
    parser.add_argument("--pan-span", type=float, default=30.0,
                        help="序列内 pan 总跨度（度）")
    parser.add_argument("--shadow-p", type=float, default=0.0,
                        help="每条序列开阴影的概率，>0 时覆盖 --shadow")
    parser.add_argument("--noise-max", type=float, default=0.0,
                        help=">0 时每帧在 [0, noise-max] 采样，覆盖 --noise-sigma")
    parser.add_argument("--jpeg-min", type=int, default=0)
    parser.add_argument("--jpeg-max", type=int, default=0,
                        help=">0 时每帧在 [jpeg-min, jpeg-max] 采样，覆盖 --jpeg-q")
    parser.add_argument("--blur-p", type=float, default=0.0,
                        help="每帧加模糊的概率")
    args = parser.parse_args()
    if args.finger_colors == "black":
        COLORS["gripper_active_link"] = [0.22, 0.22, 0.22, 1.0]
        COLORS["gripper_active2_link"] = [0.22, 0.22, 0.22, 1.0]

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    bid, joint_idx, link_idx, table_id = setup(rng, args.urdf)
    link_subset = {
        name: link_idx[name]
        for name in keypoint_link_names(
            link_idx,
            args.kp_names.split(",") if args.kp_names else None,
        )
    }

    seq_len = max(1, args.seq_len)
    seq = None
    for frame_id in range(args.frames):
        k = frame_id % seq_len
        if k == 0:
            # 新序列：一次性定下桌面、光照、阴影开关、基础姿态和相机
            table_color = TABLE_COLORS[int(rng.integers(len(TABLE_COLORS)))]
            p.changeVisualShape(table_id, -1, rgbaColor=table_color)
            light = randomize_light(
                rng,
                (args.light_x_min, args.light_x_max),
                (args.light_y_min, args.light_y_max),
            )
            shadow = 1 if rng.random() < args.shadow_p else args.shadow
            base = set_random_pose(bid, joint_idx, rng)
            if seq_len > 1:
                lo, hi = JOINT_RANGES["arm_shoulder_pan_joint"]
                # --pan-span 是角度，JOINT_RANGES 是弧度，必须转
                span = min(float(np.deg2rad(args.pan_span)), hi - lo)
                pan0 = float(rng.uniform(lo, hi - span))
                poses = []
                for j in range(seq_len):
                    q = dict(base)
                    q["arm_shoulder_pan_joint"] = pan0 + span * j / (seq_len - 1)
                    poses.append(q)
            else:
                poses = [base]
            view, yaw, pitch, dist, target = sample_view(
                bid, joint_idx, link_subset, rng, args, poses=poses
            )
            seq = dict(
                table_color=table_color, light=light, shadow=shadow, poses=poses
            )
        else:
            table_color = seq["table_color"]
            light = seq["light"]
            shadow = seq["shadow"]

        pose = seq["poses"][k]
        apply_pose(bid, joint_idx, pose)
        rgb, depth, seg = render_frame(
            W,
            H,
            0.0,
            view,
            NEAR,
            FAR,
            K=K,
            shadow=shadow,
            light_direction=light["direction"],
            light_color=light["color"],
            ambient=light["ambient"],
            diffuse=light["diffuse"],
            specular=light["specular"],
        )
        noise = (
            float(rng.uniform(0.0, args.noise_max))
            if args.noise_max > 0
            else args.noise_sigma
        )
        jq = (
            int(rng.integers(args.jpeg_min, args.jpeg_max + 1))
            if args.jpeg_max > 0
            else args.jpeg_q
        )
        blur = bool(rng.random() < args.blur_p) if args.blur_p > 0 else args.blur
        rgb = post_process(rgb, rng, noise, blur, jq)
        names, points3d, uv, valid, _ = project_link_keypoints(
            bid, link_subset, W, H, 0.0, view, K=K
        )
        assert valid.all()

        frame_dir = args.out / f"{frame_id:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(frame_dir / "rgb.png")
        if not args.no_depth:
            np.save(frame_dir / "depth.npy", depth)
            np.save(frame_dir / "seg.npy", seg)
            save_depth_png(frame_dir / "depth.png", depth)
            save_seg_png(frame_dir / "seg.png", seg)
        save_overlay(frame_dir / "overlay.png", rgb, uv)

        r_cb, t_cb = opencv_extrinsics_from_view(view)
        meta = {
            "frame": frame_id,
            "seed": args.seed,
            "width": W,
            "height": H,
            "K": K.tolist(),
            "K_full": INTR_FULL.K.tolist(),
            "full_width": INTR_FULL.width,
            "full_height": INTR_FULL.height,
            "camera_model": "pinhole",
            "distortion_applied": False,
            "intrinsics_source": "Ace Pro 1280x720 center-crop 4:3 -> 640x480",
            "near": NEAR,
            "far": FAR,
            "camera": {
                "yaw": yaw,
                "pitch": pitch,
                "distance": dist,
                "target": target,
            },
            "light": light,
            "shadow": shadow,
            "degrade": {"noise_sigma": noise, "blur": blur, "jpeg_q": jq},
            "sequence": frame_id // seq_len,
            "frame_in_seq": k,
            "seq_len": seq_len,
            "table_color": table_color,
            "view_matrix": [float(x) for x in view],
            "R_cb": r_cb.tolist(),
            "t_cb": t_cb.tolist(),
            "joints": pose,
            "link_names": names,
            "keypoints_3d_base": points3d.tolist(),
            "keypoints_2d": uv.tolist(),
            "valid": [bool(x) for x in valid],
        }
        (frame_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"frame {frame_id:03d}: {frame_dir}")

    p.disconnect()


if __name__ == "__main__":
    main()
