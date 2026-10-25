"""离屏渲染 PhantomX Pincher URDF（PyBullet TinyRenderer）。

用法：
    conda activate arm_sim
    python scripts/render_urdf.py
输出：
    outputs/renders/pincher_three_quarter.png
    outputs/renders/pincher_front.png
    outputs/renders/pincher_gripper.png
    outputs/renders/pincher_views.png
"""

import sys
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.sim.camera import intrinsics_from_fov, project_base_points

URDF = ROOT / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints.urdf"
OUT = ROOT / "outputs/renders"

POSE = {
    "arm_shoulder_pan_joint": 0.5,
    "arm_bicep_joint": -0.8,
    "arm_forearm_joint": 0.9,
    "arm_wrist_flex_joint": -0.4,
    "gripper_joint": 0.025,
}

COLORS = {
    "arm_base_link": [0.20, 0.20, 0.23, 1.0],
    "arm_shoulder_pan_link": [0.20, 0.20, 0.23, 1.0],
    "arm_bicep_link": [0.62, 0.62, 0.65, 1.0],
    "arm_forearm_link": [0.62, 0.62, 0.65, 1.0],
    "arm_wrist_flex_link": [0.20, 0.20, 0.23, 1.0],
    "gripper_active_link": [0.22, 0.22, 0.22, 1.0],
    "gripper_active2_link": [0.22, 0.22, 0.22, 1.0],
}

# name, yaw, pitch, distance, target（None = 自动对准夹爪）
VIEWS = [
    ("three_quarter", 50, -12, 0.52, [0.02, 0.0, 0.12]),
    ("front", 0, -8, 0.52, [0.02, 0.0, 0.12]),
    ("gripper", 20, -15, 0.26, None),
]

W, H = 1024, 768
FOV_DEG = 50


def camera_matrices(yaw: float, pitch: float, dist: float, target):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=dist,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=FOV_DEG, aspect=W / H, nearVal=0.01, farVal=5.0
    )
    return view, proj


def render_view(yaw: float, pitch: float, dist: float, target) -> Image.Image:
    view, proj = camera_matrices(yaw, pitch, dist, target)
    _, _, rgb, _, _ = p.getCameraImage(
        W,
        H,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=[0.4, -0.6, 1.0],
        lightColor=[1.0, 1.0, 1.0],
        lightDistance=2.0,
        shadow=1,
        lightAmbientCoeff=0.75,
        lightDiffuseCoeff=0.7,
        lightSpecularCoeff=0.25,
    )
    rgba = np.reshape(rgb, (H, W, 4)).astype(np.uint8)
    return Image.fromarray(rgba[:, :, :3])


def opencv_extrinsics(view):
    """PyBullet 的 OpenGL view → OpenCV base→camera 外参。"""
    v_gl = np.array(view, dtype=float).reshape(4, 4).T
    t_cv = np.diag([1.0, -1.0, -1.0, 1.0]) @ v_gl
    return t_cv[:3, :3], t_cv[:3, 3]


def annotate(img: Image.Image, bid: int, links: dict) -> Image.Image:
    _, yaw, pitch, dist, target = VIEWS[0]
    view, _ = camera_matrices(yaw, pitch, dist, target)
    r_cb, t_cb = opencv_extrinsics(view)
    kp_names = [
        "kp_base",
        "kp_shoulder_pan",
        "kp_shoulder_lift",
        "kp_elbow",
        "kp_wrist",
        "kp_gripper_base",
    ]
    if all(name in links for name in kp_names):
        labels = ["基座", "肩 pan", "肩 lift", "肘", "腕 flex", "夹爪基座"]
        points = [(label, links[name]) for label, name in zip(labels, kp_names)]
    else:
        base_name = "base_link" if "base_link" in links else "arm_base_link"
        points = [
            ("基座", links[base_name]),
            ("肩 pan", links["arm_shoulder_pan_link"]),
            ("肩 lift", links["arm_bicep_link"]),
            ("肘", links["arm_forearm_link"]),
            ("腕 flex", links["arm_wrist_flex_link"]),
            ("夹爪", links["gripper_active_link"]),
        ]
    positions = np.array(
        [p.getLinkState(bid, link_idx)[4] for _, link_idx in points]
    )
    k = intrinsics_from_fov(W, H, FOV_DEG)
    uv, valid = project_base_points(k, r_cb, t_cb, positions)
    assert valid.all()

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    for (label, _), (u, v) in zip(points, uv):
        u, v = int(u), int(v)
        draw.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(255, 90, 40), outline="white", width=2)
        draw.text((u + 9, v - 26), label, font=font, fill=(20, 20, 20),
                  stroke_width=3, stroke_fill=(255, 255, 255))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p.connect(p.DIRECT)
    p.setGravity(0, 0, 0)

    bid = p.loadURDF(str(URDF), useFixedBase=True)
    names = {
        p.getJointInfo(bid, i)[1].decode(): i for i in range(p.getNumJoints(bid))
    }
    links = {
        p.getJointInfo(bid, i)[12].decode(): i for i in range(p.getNumJoints(bid))
    }

    # 桌面
    half = [0.55, 0.55, 0.01]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half, rgbaColor=[0.72, 0.74, 0.78, 1.0]
    )
    p.createMultiBody(0, col, vis, basePosition=[0.12, 0.0, -0.01])

    for link_name, rgba in COLORS.items():
        if link_name in links:
            p.changeVisualShape(bid, links[link_name], rgbaColor=rgba)
    if "base_link" in links:
        p.changeVisualShape(
            bid, links["base_link"], rgbaColor=COLORS["arm_base_link"]
        )
    for joint_name, q in POSE.items():
        p.resetJointState(bid, names[joint_name], q)
    p.stepSimulation()

    gripper_pos = p.getLinkState(bid, links["gripper_active_link"])[4]
    print("gripper position:", tuple(round(v, 4) for v in gripper_pos))

    images = []
    for name, yaw, pitch, dist, target in VIEWS:
        if target is None:
            target = [gripper_pos[0], gripper_pos[1], gripper_pos[2] + 0.01]
        img = render_view(yaw, pitch, dist, target)
        path = OUT / f"pincher_{name}.png"
        img.save(path)
        images.append(img)
        print("saved", path)

    sheet = Image.new("RGB", (W * len(images), H))
    for i, img in enumerate(images):
        sheet.paste(img, (i * W, 0))
    sheet_path = OUT / "pincher_views.png"
    sheet.save(sheet_path)
    print("saved", sheet_path)

    labeled = annotate(images[0].copy(), bid, links)
    labeled_path = OUT / "pincher_labeled.png"
    labeled.save(labeled_path)
    print("saved", labeled_path)

    p.disconnect()


if __name__ == "__main__":
    main()
