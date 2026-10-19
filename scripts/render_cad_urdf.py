"""渲染 CAD 版 PhantomX Pincher URDF（第一版，对照用）。

用法：
    conda activate arm_sim
    python scripts/render_cad_urdf.py

输出：
    outputs/renders/pincher_cad_three_quarter.png
    outputs/renders/pincher_cad_front.png
    outputs/renders/pincher_cad_side.png
    outputs/renders/pincher_cad_views.png
"""

from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf"
OUT = ROOT / "outputs/renders"

POSE = {
    # 直立位（在 URDF 关节限位内）：腕部约在 (0, 0.04, 0.32) m
    "arm_shoulder_pan_joint": 0.002,
    "arm_bicep_joint": 0.077,
    "arm_forearm_joint": -1.366,
    "arm_wrist_flex_joint": 0.0,
    "gripper_joint": 0.0,
}

COLORS = {
    "arm_base_link": [0.20, 0.20, 0.23, 1.0],
    "arm_shoulder_pan_link": [0.20, 0.20, 0.23, 1.0],
    "arm_bicep_link": [0.55, 0.55, 0.58, 1.0],
    "arm_forearm_link": [0.65, 0.65, 0.68, 1.0],
    "arm_wrist_flex_link": [0.30, 0.30, 0.33, 1.0],
    "gripper_active_link": [0.15, 0.15, 0.15, 1.0],
    "gripper_active2_link": [0.15, 0.15, 0.15, 1.0],
}

VIEWS = [
    ("three_quarter", 45, -10, 0.72, [0.0, 0.02, 0.20]),
    ("front", 0, -5, 0.72, [0.0, 0.02, 0.20]),
    ("side", 90, -5, 0.72, [0.0, 0.02, 0.20]),
]

W, H = 1024, 768


def render_view(yaw: float, pitch: float, dist: float, target) -> Image.Image:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=dist,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=50, aspect=W / H, nearVal=0.01, farVal=5.0
    )
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p.connect(p.DIRECT)
    p.setGravity(0, 0, 0)
    bid = p.loadURDF(str(URDF), useFixedBase=True)
    print("joints:", p.getNumJoints(bid))

    names = {
        p.getJointInfo(bid, i)[1].decode(): i for i in range(p.getNumJoints(bid))
    }
    links = {
        p.getJointInfo(bid, i)[12].decode(): i for i in range(p.getNumJoints(bid))
    }
    for name, rgba in COLORS.items():
        if name in links:
            p.changeVisualShape(bid, links[name], rgbaColor=rgba)
    for joint, q in POSE.items():
        p.resetJointState(bid, names[joint], q)
    p.stepSimulation()

    images = []
    for name, yaw, pitch, dist, target in VIEWS:
        img = render_view(yaw, pitch, dist, target)
        path = OUT / f"pincher_cad_{name}.png"
        img.save(path)
        images.append(img)
        print("saved", path)

    sheet = Image.new("RGB", (W * len(images), H))
    for i, img in enumerate(images):
        sheet.paste(img, (i * W, 0))
    sheet_path = OUT / "pincher_cad_views.png"
    sheet.save(sheet_path)
    print("saved", sheet_path)
    p.disconnect()


if __name__ == "__main__":
    main()
