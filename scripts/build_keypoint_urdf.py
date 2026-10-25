"""在 CAD URDF 上追加 9 个彩色 keypoint 标记（可选加 2 个侧向凸耳）。

生成：
    assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints.urdf
    assets/robots/phantomx_pincher_cad/phantomx_pincher_cad_keypoints_lug.urdf  (--lug-cm > 0)

标记是固定 link + 无碰撞 visual：
- kp_base / kp_base_left / kp_base_right：小球
- 其余：垂直于连杆轴线的圆环，半径大于连杆截面（约 0.026m），从外侧包住连杆

--lug-cm N：再追加 kp_lug_left / kp_lug_right 两个小球，分别挂在 bicep 与 forearm 的
 **局部 y 方向两端**，离臂平面 ±N cm。9 个链上关键点严格共面（不共面度≈1e-4），
镜像双解无法靠重投影排除；凸耳提供离面杠杆。实测 δ=5cm 时可用率 71.6%→80.0%，
但 δ≤1cm 反而变差，**不要用小值**。真机上必须保证至少 1 个凸耳可见。
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


LUGS = [
    dict(
        name="kp_lug_left",
        parent="arm_bicep_link",
        xyz=(0.0, 0.0, 0.0535),
        rpy=(0.0, 0.0, 0.0),
        shape="sphere",
        radius=0.010,
        color=(0.20, 0.85, 0.95, 1.0),
        lug_axis=1,
    ),
    dict(
        name="kp_lug_right",
        parent="arm_forearm_link",
        xyz=(0.0, 0.0, 0.0535),
        rpy=(0.0, 0.0, 0.0),
        shape="sphere",
        radius=0.010,
        color=(0.95, 0.75, 0.20, 1.0),
        lug_axis=-1,
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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--lug-cm", type=float, default=0.0,
                    help="侧向凸耳离臂平面的距离(cm)；0 表示不加凸耳")
    args = ap.parse_args()

    keypoints = list(KEYPOINTS)
    dst = DST
    if args.lug_cm > 0:
        for lug in LUGS:
            item = dict(lug)
            axis = item.pop("lug_axis")
            x, y, z = item["xyz"]
            item["xyz"] = (x, y + axis * args.lug_cm / 100.0, z)
            keypoints.append(item)
        dst = DST.with_name(DST.stem + "_lug.urdf")

    text = SRC.read_text(encoding="utf-8")
    if "</robot>" not in text:
        raise RuntimeError("source URDF has no </robot>")
    markers = "\n".join(marker_xml(kp) for kp in keypoints)
    text = text.replace("</robot>", markers + "\n</robot>")
    dst.write_text(text, encoding="utf-8")
    print("wrote", dst)
    print("keypoints:", [kp["name"] for kp in keypoints])


if __name__ == "__main__":
    main()
