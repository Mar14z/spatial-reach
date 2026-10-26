"""MuJoCo 场景：CAD URDF + 桌面 + 3cm 方块 + 位置执行器。

URDF 里没有 <actuator>，MuJoCo 导入后 nu=0，必须在 MjSpec 里补。
基座是固定的：MuJoCo 把 base_footprint→base_link 的固定关节焊进了 world，
所以不需要额外的 weld 或 freejoint 处理。

夹爪：URDF 把第二指写成 fixed，但真机是齿条联动的对称双指（CAD 把两指装在
腕轴 ±28.4mm 处、装配姿态是“半开”）。这里补一个 slide 关节并用等式约束做成
q2 = -q1，于是指腹中点永远在腕部系 y=0 平面上。
"""
import mujoco
from pathlib import Path

URDF = Path(__file__).resolve().parents[1] / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf"
ARM_JOINTS = ["arm_shoulder_pan_joint", "arm_bicep_joint", "arm_forearm_joint", "arm_wrist_flex_joint"]
GRIPPER_JOINT = "gripper_joint"
GRIPPER2_JOINT = "gripper2_joint"
# 每指行程：CAD 姿态下两指指腹相距 29.7mm，各外移 16mm 到全开 61.7mm，
# 各内移 14.85mm 时指腹贴合。URDF 里 (0.002,0.032) / -0.0276 是单指动的旧模型。
GRIP_OPEN = 0.016
GRIP_HALF = 0.01485
# URDF 没有 rotor inertia / joint damping，MuJoCo 里位置伺服会自激（实测稳态速度 37 rad/s）。
# armature 补电机转子惯量，damping 补摩擦，kv 由 2*sqrt(kp*armature) 给临界阻尼。
ARM_ARMATURE = 0.02
ARM_DAMPING = 0.2
KP = 50.0
GRIP_ARMATURE = 0.02
GRIP_KP = 1000.0
GRIP_FORCE = 8.0  # 真实小夹爪量级的夹持力；30N 硬夹会把物体楔出去


def build_spec(cube_xy=(0.22, 0.0), cube_size=0.03):
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.option.timestep = 0.002
    spec.option.gravity = [0.0, 0.0, -9.81]
    world = spec.worldbody
    table = world.add_geom()
    table.name = "table"
    table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.size = [0.55, 0.55, 0.01]
    table.pos = [0.12, 0.0, -0.01]
    cube = world.add_body()
    cube.name = "cube"
    cube.pos = [cube_xy[0], cube_xy[1], cube_size / 2]
    geom = cube.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [cube_size / 2] * 3
    geom.friction = [1.0, 0.005, 0.0001]
    cube.add_freejoint()
    cube.joints[-1].name = "cube_free"
    f2 = spec.body("gripper_active2_link")
    j2 = f2.add_joint()
    j2.name = GRIPPER2_JOINT
    j2.type = mujoco.mjtJoint.mjJNT_SLIDE
    j2.axis = [0.0, -1.0, 0.0]  # 与第一指反向；等式约束 q2 = q1 即两指对开对合
    j2.pos = [0.0, 0.0, 0.0]
    j2.range = [-GRIP_OPEN, GRIP_HALF]
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_JOINT
    eq.name1, eq.name2 = GRIPPER2_JOINT, GRIPPER_JOINT
    eq.solref = [0.002, 1.0]
    for j in spec.joints:
        if j.name == GRIPPER_JOINT:
            j.range = [-GRIP_HALF, GRIP_OPEN]
        elif j.name == GRIPPER2_JOINT:
            j.armature = GRIP_ARMATURE  # 从动指：只补惯量/阻尼，由等式约束带动
            j.damping = [ARM_DAMPING, 0.0, 0.0]
        if j.name not in ARM_JOINTS + [GRIPPER_JOINT]:
            continue
        grip = j.name == GRIPPER_JOINT
        armature = GRIP_ARMATURE if grip else ARM_ARMATURE
        j.armature = armature
        j.damping = [ARM_DAMPING, 0.0, 0.0]
        kp = GRIP_KP if grip else KP
        act = spec.add_actuator()
        act.name = j.name.replace("_joint", "_motor")
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = j.name
        act.set_to_position(kp, kv=2.0 * (kp * armature) ** 0.5, inheritrange=True)
        act.forcerange = [-(GRIP_FORCE if grip else 30.0), GRIP_FORCE if grip else 30.0]
        act.forcelimited = True
        assert j.range[1] > j.range[0]
    return spec


def build(**kw):
    spec = build_spec(**kw)
    return spec.compile(), spec


def main():
    model, _ = build()
    data = mujoco.MjData(model)
    print(f"nu={model.nu} nq={model.nq} nbody={model.nbody} ngeom={model.ngeom}")
    print("actuators:", [model.actuator(i).name for i in range(model.nu)])
    ctrl_id = {model.actuator(i).name: i for i in range(model.nu)}
    targets = [
        {"arm_shoulder_pan_joint": 0.0, "arm_bicep_joint": 0.6, "arm_forearm_joint": -1.2, "arm_wrist_flex_joint": 0.4},
        {"arm_shoulder_pan_joint": 0.8, "arm_bicep_joint": 0.3, "arm_forearm_joint": -0.8, "arm_wrist_flex_joint": 0.5},
        {"arm_shoulder_pan_joint": -0.6, "arm_bicep_joint": 0.9, "arm_forearm_joint": -1.5, "arm_wrist_flex_joint": -0.2},
    ]
    for t in targets:
        data.qpos[:] = 0.0
        data.qpos[4] = 0.002
        mujoco.mj_forward(model, data)
        for name, val in t.items():
            data.ctrl[ctrl_id[name.replace("_joint", "_motor")]] = val
        data.ctrl[ctrl_id["gripper_motor"]] = 0.01
        for _ in range(1000):
            mujoco.mj_step(model, data)
        err = {n: abs(data.qpos[model.joint(n).qposadr[0]] - v) for n, v in t.items()}
        print("target", {k: round(v, 2) for k, v in t.items()})
        print("  q   ", {n: round(float(data.qpos[model.joint(n).qposadr[0]]), 3) for n in t})
        print("  max err %.3f rad" % max(err.values()))
        tip = data.body("gripper_active_link").xpos
        print("  gripper tip", [round(float(v), 4) for v in tip])


if __name__ == "__main__":
    main()
