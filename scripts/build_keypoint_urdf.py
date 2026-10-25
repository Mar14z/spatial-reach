"""在 CAD URDF 上追加 11 个彩色 keypoint 标记。

生成：
    assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints.urdf

标记是固定 link + 无碰撞 visual：
- kp_base / kp_gripper_base / kp_base_left / kp_base_right：小球
- 其余：垂直于连杆轴线的圆环，半径大于连杆截面（约 0.026m），从外侧包住连杆
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf"
DST = ROOT / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints.urdf"

KEYPOINTS = [
    dict(
        name="kp_base",
        parent="base_link",
        xyz=(0.0, 0.0, 0.02),
        rpy=(0.0, 0.0, 0.0),
        shape="sphere",
        radius=0.018,
        color=(0.90, 0.15, 0.15, 1.0),
    ),
    dict(
        name="kp_shoulder_pan",
        parent="arm_shoulder_pan_link",
        xyz=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        shape="cylinder",
        radius=0.018,
        length=0.006,
        color=(0.15, 0.75, 0.25, 1.0),
    ),
    dict(
        name="kp_shoulder_lift",
        parent="arm_bicep_link",
        xyz=(0.0, 0.0, 0.015),
        rpy=(0.0, 0.0, 0.0),
        shape="cylinder",
        radius=0.030,
        length=0.008,
        color=(0.15, 0.35, 0.90, 1.0),
    ),
    dict(
        name="kp_upper_mid",
        parent="arm_bicep_link",
        xyz=(0.0, 0.0, 0.0535),
        rpy=(0.0, 0.0, 0.0),
        shape="cylinder",
        radius=0.028,
        length=0.008,
        color=(1.00, 0.50, 0.10, 1.0),
    ),
    dict(
        name="kp_elbow",
        parent="arm_forearm_link",
        xyz=(0.0, 0.0, 0.015),
        rpy=(0.0, 0.0, 0.0),
        shape="cylinder",
        radius=0.030,
        length=0.008,
        color=(0.90, 0.80, 0.15, 1.0),
    ),
    dict(
        name="kp_forearm_mid",
        parent="arm_forearm_link",
        xyz=(0.0, 0.0, 0.0535),
        rpy=(0.0, 0.0, 0.0),
        shape="cylinder",
        radius=0.028,
        length=0.008,
        color=(0.55, 0.20, 0.85, 1.0),
    ),
    dict(
        name="kp_wrist",
        parent="arm_wrist_flex_link",
        xyz=(0.0, 0.0, 0.012),
        rpy=(0.0, 0.0, 0.0),
        shape="cylinder",
        radius=0.028,
        length=0.008,
        color=(0.80, 0.15, 0.80, 1.0),
    ),
    dict(
        name="kp_base_left",
        parent="base_link",
        xyz=(-0.09, 0.06, 0.008),
        rpy=(0.0, 0.0, 0.0),
        shape="sphere",
        radius=0.010,
        color=(0.55, 1.00, 0.10, 1.0),
    ),
    dict(
        name="kp_base_right",
        parent="base_link",
        xyz=(-0.09, -0.06, 0.008),
        rpy=(0.0, 0.0, 0.0),
        shape="sphere",
        radius=0.010,
        color=(1.00, 0.20, 0.60, 1.0),
    ),
]


def marker_xml(kp):
    name = kp["name"]
    parent = kp["parent"]
    color_name = f"kp_{name}_color"
    rgba = " ".join(f"{v:.2f}" for v in kp["color"])
    rpy = " ".join(str(v) for v in kp["rpy"])
    xyz = " ".join(str(v) for v in kp["xyz"])
    if kp["shape"] == "sphere":
        geometry = f'<sphere radius="{kp["radius"]}"/>'
    else:
        geometry = (
            f'<cylinder radius="{kp["radius"]}" length="{kp["length"]}"/>'
        )
    return f"""
  <!-- keypoint marker: {name} -->
  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="{rpy}"/>
      <geometry>{geometry}</geometry>
      <material name="{color_name}">
        <color rgba="{rgba}"/>
      </material>
    </visual>
  </link>
  <joint name="{name}_joint" type="fixed">
    <origin xyz="{xyz}" rpy="0 0 0"/>
    <parent link="{parent}"/>
    <child link="{name}"/>
  </joint>
"""


def main():
    text = SRC.read_text(encoding="utf-8")
    if "</robot>" not in text:
        raise RuntimeError("source URDF has no </robot>")
    markers = "\n".join(marker_xml(kp) for kp in KEYPOINTS)
    text = text.replace("</robot>", markers + "\n</robot>")
    DST.write_text(text, encoding="utf-8")
    print("wrote", DST)
    print("keypoints:", [kp["name"] for kp in KEYPOINTS])


if __name__ == "__main__":
    main()
