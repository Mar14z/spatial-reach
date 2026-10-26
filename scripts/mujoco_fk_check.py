"""MuJoCo 迁移第一步：同一份 URDF 分别用 PyBullet / MuJoCo 加载，对拍 FK。

固定关节的子 body 会被 MuJoCo 焊进父 body（base_footprint/base_link 焊进 world，
gripper_active2_link 焊进 wrist），所以只在两边都存在的 body 上比较。
"""
import sys
from pathlib import Path

import numpy as np
import pybullet as pb
import mujoco

URDF = Path(__file__).resolve().parents[1] / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf"


def tf(pos, quat):
    x, y, z, w = quat
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = pos
    return T


def pybullet_fk(q):
    pb.connect(pb.DIRECT)
    bid = pb.loadURDF(str(URDF), useFixedBase=True)
    for j in range(pb.getNumJoints(bid)):
        name = pb.getJointInfo(bid, j)[1].decode()
        if name in q:
            pb.resetJointState(bid, j, float(q[name]))
    p, r = pb.getBasePositionAndOrientation(bid)
    out = {"base": tf(p, r)}
    for j in range(pb.getNumJoints(bid)):
        p, qq = pb.getLinkState(bid, j, computeForwardKinematics=True)[4:6]
        out[pb.getJointInfo(bid, j)[12].decode()] = tf(p, qq)
    pb.disconnect()
    return out


def mujoco_model():
    m = mujoco.MjModel.from_xml_path(str(URDF))
    return m, mujoco.MjData(m)


def mujoco_fk(m, d, q):
    d.qpos[:] = 0.0
    for j in range(m.njnt):
        d.qpos[m.jnt_qposadr[j]] = q.get(m.joint(j).name, 0.0)
    mujoco.mj_forward(m, d)
    # mujoco xquat 是 (w, x, y, z)，这里转成 tf() 约定的 (x, y, z, w)
    return {m.body(b).name: tf(d.xpos[b], np.roll(d.xquat[b], -1)) for b in range(m.nbody)}


def main():
    m, d = mujoco_model()
    names = [m.joint(j).name for j in range(m.njnt)]
    print(f"mujoco: nbody={m.nbody} njnt={m.njnt} nu={m.nu} ngeom={m.ngeom}")
    rng = np.random.default_rng(0)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    cases = [dict.fromkeys(names, 0.0)]
    for _ in range(5):
        cases.append({n: float(v) for n, v in zip(names, rng.uniform(lo, hi))})
    worst = 0.0
    for case, q in enumerate(cases):
        pb_fk, mj_fk = pybullet_fk(q), mujoco_fk(m, d, q)
        for name in mj_fk:
            if name not in pb_fk:
                continue
            A, B = np.linalg.inv(pb_fk["base"]) @ pb_fk[name], mj_fk[name]
            dp = np.linalg.norm(A[:3, 3] - B[:3, 3])
            da = np.degrees(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1)))
            worst = max(worst, dp)
            if case == 0:
                print(f"  {name:26s} pos_err={dp:.2e} m rot_err={da:.2e} deg")
            elif dp > 1e-6:
                print(f"  case{case} {name:26s} pos_err={dp:.2e} rot_err={da:.2e}")
        print(f"case {case}: " + " ".join(f"{k}={v:.2f}" for k, v in q.items()))
    print(f"max pos err = {worst:.3e} m")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
