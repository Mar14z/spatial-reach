"""从 PhantomX Pincher Mark II STEP/GLB 生成 CAD 运动学 URDF。

做法：
- 读取 GLB 中关键零件的装配变换（PincherBase / Component1 / LowerArm / UpperArm / GripperBody / LeftFinger）
- 把零件局部坐标系直接作为 link frame
- joint origin = 父 link 到子 link 的相对变换
- 关节轴从装配世界轴推出来（pan = Y，pitch = X）
- 底座作为 base_link 的一部分，整体从 CAD 的 Y-up 转到 Z-up

用法：
    conda activate arm_sim
    python scripts/build_cad_urdf.py
输出：
    assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from trimesh.transformations import euler_from_matrix, rotation_matrix

ROOT = Path(__file__).resolve().parents[1]
CAD_DIR = ROOT / "assets/robots/phantomx_pincher_cad"
GLB = CAD_DIR / "pincher.step.glb"
OUT = CAD_DIR / "phantomx_pincher_cad.urdf"

# 零件 → link 名
PART_LINK = {
    "PincherBase": "base_link",
    "Component1": "arm_shoulder_pan_link",
    "LowerArm v2": "arm_bicep_link",
    "UpperArm v2": "arm_forearm_link",
    "GripperBody v4": "arm_wrist_flex_link",
    "LeftFinger v2": "gripper_active_link",
    "LeftFinger v2_1": "gripper_active2_link",
}

NODE = {
    "base": "PincherBase v8:1",
    "shoulder": "Component1:1",
    "bicep": "LowerArm v2:1",
    "forearm": "UpperArm v2:1",
    "wrist": "GripperBody v4:1",
    "finger1": "LeftFinger v2:1",
    "finger2": "LeftFinger v2:2",
}

MASS = {
    "base_link": 0.30,
    "arm_shoulder_pan_link": 0.08,
    "arm_bicep_link": 0.10,
    "arm_forearm_link": 0.08,
    "arm_wrist_flex_link": 0.06,
    "gripper_active_link": 0.01,
    "gripper_active2_link": 0.01,
}

LIMITS = {
    "pan": (-1.74, 1.74),
    "lift": (-1.98, 1.88),
    "elbow": (-1.58, 0.59),
    "wrist": (-1.79, 1.67),
    # 夹持面间隙 gap(q) = 29.6mm + q（实测网格）。旧值 (0.002, 0.032) 使"闭合"时两指
    # 仍隔 31.6mm，3cm 物体夹不住；按闭合间隙 2mm 反推下限 = -27.6mm。
    "gripper": (-0.0276, 0.032),
}


def fmt(v: float) -> str:
    return f"{float(v):.8f}".rstrip("0").rstrip(".")


def mat_to_origin(T: np.ndarray) -> str:
    xyz = T[:3, 3]
    rpy = euler_from_matrix(T[:3, :3], "sxyz")  # URDF rpy 单位是弧度
    return f'xyz="{fmt(xyz[0])} {fmt(xyz[1])} {fmt(xyz[2])}" rpy="{fmt(rpy[0])} {fmt(rpy[1])} {fmt(rpy[2])}"'


def axis_in_child(T_child: np.ndarray, world_axis: np.ndarray) -> np.ndarray:
    a = T_child[:3, :3].T @ world_axis
    return a / np.linalg.norm(a)


def axis_str(a: np.ndarray) -> str:
    return f'xyz="{fmt(a[0])} {fmt(a[1])} {fmt(a[2])}"'


def inertial(mass: float) -> str:
    # 近似值；CAD URDF 第一版不追求精确动力学
    return f"""    <inertial>
      <mass value="{fmt(mass)}"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>"""


def visual(mesh: str, rgba: str = "0.6 0.6 0.62 1") -> str:
    return f"""    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="parts/{mesh}" scale="1.0 1.0 1.0"/></geometry>
      <material name="grey"><color rgba="{rgba}"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="parts/{mesh}" scale="1.0 1.0 1.0"/></geometry>
    </collision>"""


def main() -> None:
    scene = trimesh.load(str(GLB), process=False)
    g = scene.graph
    T = {k: g.get(v)[0].copy() for k, v in NODE.items()}

    # 底座从 CAD 的 Y-up 转到 Z-up：base_link 仍是 CAD 零件坐标系，
    # base_footprint 通过一个固定关节把它立起来。
    Rx90 = rotation_matrix(np.radians(90), [1, 0, 0])
    T_base_footprint = Rx90 @ T["base"]

    pan_axis = axis_in_child(T["shoulder"], np.array([0.0, 1.0, 0.0]))
    lift_axis = axis_in_child(T["bicep"], np.array([1.0, 0.0, 0.0]))
    elbow_axis = axis_in_child(T["forearm"], np.array([1.0, 0.0, 0.0]))
    wrist_axis = axis_in_child(T["wrist"], np.array([1.0, 0.0, 0.0]))
    finger_axis = axis_in_child(T["finger1"], np.array([1.0, 0.0, 0.0]))

    def rel(parent: str, child: str) -> np.ndarray:
        return np.linalg.inv(T[parent]) @ T[child]

    xml = f"""<?xml version="1.0" ?>
<!-- 由 scripts/build_cad_urdf.py 从 PhantomX Pincher Mark II STEP/GLB 生成。
     运动学来自 CAD 装配：link frame = 零件局部坐标系，joint origin = 父→子相对变换。
     base_link 包含底座；base_footprint 把 CAD 的 Y-up 转成 Z-up。
-->
<robot name="phantomx_pincher_cad">
  <link name="base_footprint"/>
  <joint name="arm_base_joint" type="fixed">
    <origin {mat_to_origin(T_base_footprint)}/>
    <parent link="base_footprint"/>
    <child link="base_link"/>
  </joint>

  <link name="base_link">
{inertial(MASS["base_link"])}
{visual("PincherBase.stl", "0.25 0.25 0.28 1")}
{visual("PincherBase_1.stl", "0.25 0.25 0.28 1")}
  </link>

  <joint name="arm_shoulder_pan_joint" type="revolute">
    <origin {mat_to_origin(rel("base", "shoulder"))}/>
    <axis {axis_str(pan_axis)}/>
    <limit effort="6" lower="{LIMITS['pan'][0]}" upper="{LIMITS['pan'][1]}" velocity="1.571"/>
    <dynamics friction="0.13"/>
    <parent link="base_link"/>
    <child link="arm_shoulder_pan_link"/>
  </joint>
  <link name="arm_shoulder_pan_link">
{inertial(MASS["arm_shoulder_pan_link"])}
{visual("Component1.stl", "0.22 0.22 0.25 1")}
  </link>

  <joint name="arm_bicep_joint" type="revolute">
    <origin {mat_to_origin(rel("shoulder", "bicep"))}/>
    <axis {axis_str(lift_axis)}/>
    <limit effort="6" lower="{LIMITS['lift'][0]}" upper="{LIMITS['lift'][1]}" velocity="1.571"/>
    <dynamics friction="0.13"/>
    <parent link="arm_shoulder_pan_link"/>
    <child link="arm_bicep_link"/>
  </joint>
  <link name="arm_bicep_link">
{inertial(MASS["arm_bicep_link"])}
{visual("LowerArm_v2.stl", "0.6 0.6 0.62 1")}
  </link>

  <joint name="arm_forearm_joint" type="revolute">
    <origin {mat_to_origin(rel("bicep", "forearm"))}/>
    <axis {axis_str(elbow_axis)}/>
    <limit effort="6" lower="{LIMITS['elbow'][0]}" upper="{LIMITS['elbow'][1]}" velocity="1.571"/>
    <dynamics friction="0.13"/>
    <parent link="arm_bicep_link"/>
    <child link="arm_forearm_link"/>
  </joint>
  <link name="arm_forearm_link">
{inertial(MASS["arm_forearm_link"])}
{visual("UpperArm_v2.stl", "0.68 0.68 0.7 1")}
  </link>

  <joint name="arm_wrist_flex_joint" type="revolute">
    <origin {mat_to_origin(rel("forearm", "wrist"))}/>
    <axis {axis_str(wrist_axis)}/>
    <limit effort="6" lower="{LIMITS['wrist'][0]}" upper="{LIMITS['wrist'][1]}" velocity="1.571"/>
    <dynamics friction="0.13"/>
    <parent link="arm_forearm_link"/>
    <child link="arm_wrist_flex_link"/>
  </joint>
  <link name="arm_wrist_flex_link">
{inertial(MASS["arm_wrist_flex_link"])}
{visual("GripperBody_v4.stl", "0.3 0.3 0.33 1")}
  </link>

  <joint name="gripper_joint" type="prismatic">
    <origin {mat_to_origin(rel("wrist", "finger1"))}/>
    <axis {axis_str(finger_axis)}/>
    <limit effort="30" lower="{LIMITS['gripper'][0]}" upper="{LIMITS['gripper'][1]}" velocity="0.5"/>
    <parent link="arm_wrist_flex_link"/>
    <child link="gripper_active_link"/>
  </joint>
  <link name="gripper_active_link">
{inertial(MASS["gripper_active_link"])}
{visual("LeftFinger_v2.stl", "0.15 0.15 0.15 1")}
  </link>

  <joint name="gripper2_joint" type="fixed">
    <origin {mat_to_origin(rel("wrist", "finger2"))}/>
    <parent link="arm_wrist_flex_link"/>
    <child link="gripper_active2_link"/>
  </joint>
  <link name="gripper_active2_link">
{inertial(MASS["gripper_active2_link"])}
{visual("LeftFinger_v2_1.stl", "0.15 0.15 0.15 1")}
  </link>
</robot>
"""
    OUT.write_text(xml, encoding="utf-8")
    print("wrote", OUT)
    print("base_footprint->base_link:", mat_to_origin(T_base_footprint))
    for name, a in [("pan", pan_axis), ("lift", lift_axis), ("elbow", elbow_axis), ("wrist", wrist_axis), ("finger", finger_axis)]:
        print(name, "axis", np.round(a, 4).tolist())


if __name__ == "__main__":
    main()
