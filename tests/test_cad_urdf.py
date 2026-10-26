from pathlib import Path
import sys

import mujoco
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf"
sys.path.insert(0, str(ROOT / "scripts"))
import mujoco_pick as mp  # noqa: E402
import mujoco_scene as ms  # noqa: E402


def test_cad_urdf_loads():
    cid = p.connect(p.DIRECT)
    try:
        bid = p.loadURDF(str(URDF), useFixedBase=True)
        assert p.getNumJoints(bid) == 7
        names = [
            p.getJointInfo(bid, i)[1].decode()
            for i in range(p.getNumJoints(bid))
        ]
        assert "arm_shoulder_pan_joint" in names
        assert "arm_bicep_joint" in names
        assert "arm_forearm_joint" in names
        assert "arm_wrist_flex_joint" in names
        assert "gripper_joint" in names
    finally:
        p.disconnect()


def _gripper(model, q):
    """合爪量为 q 时，两指在腕坐标系里的 y 坐标：返回 (指腹间距, 指腹中点 y)。

    锁的是「对称双指、抓取点在 y=0」这个结论。上一轮把它当成非对称的
    「固定指面 + 半块」，抓取点偏了 31mm——比夹爪横向容差还大，必然失败。
    """
    data = mujoco.MjData(model)
    data.qpos[4] = data.qpos[5] = q  # 等式约束只在积分时生效，第二指要手动摆
    mujoco.mj_forward(model, data)
    wid = model.body("arm_wrist_flex_link").id
    R = data.xmat[wid].reshape(3, 3)
    y = [float((R.T @ (data.xpos[model.body(n).id] - data.xpos[wid]))[1])
         for n in ("gripper_active_link", "gripper_active2_link")]
    return abs(y[0] - y[1]), 0.5 * (y[0] + y[1])


def test_gripper_geometry_is_symmetric():
    model = ms.build_spec().compile()
    g_open, mid_open = _gripper(model, ms.GRIP_OPEN)
    g_closed, mid_closed = _gripper(model, mp.GRIP_CLOSED)
    # 齿条联动：两指各走一半，行程是单指的两倍。退回「一指动、另一指固定」时
    # 这个斜率变成 1.0 —— 上一轮那套非对称模型就会在这里挂掉。
    slope = (g_open - g_closed) / (ms.GRIP_OPEN - mp.GRIP_CLOSED)
    assert abs(slope - 2.0) < 1e-6, f"两指行程比 {slope:.4f}，应为 2.0"
    # 指腹中点恒在腕部 y=0 附近：实测偏 -0.076mm（URDF 两指原点的对称误差），
    # 只占开口的 0.2%，相对 15mm 夹爪容差可忽略，但不能再往大放——
    # 非对称模型的偏量是 ~31mm，这两者差三个量级。
    assert abs(mid_open) < 0.0002 and abs(mid_closed) < 0.0002, \
        f"指腹中点偏离 y=0：{mid_open*1000:.3f} / {mid_closed*1000:.3f}mm"
    # CAD 的装配姿态（两指本体在腕轴 ±28.42mm）就是 q=0
    assert abs(0.5 * (g_open - 2 * ms.GRIP_OPEN) - 0.02842) < 5e-4
    # 合到底留 ~2mm：GRIP_CLOSED 比「指腹刚好贴合」的行程浅 1mm
    assert abs(ms.GRIP_HALF + mp.GRIP_CLOSED - 0.001) < 1e-5
    assert mp.GRASP_OFFSET[1] == 0.0, "抓取点在两指中点上，y 分量必须为 0"


def test_gripper_command_inside_joint_range():
    model = ms.build_spec().compile()
    lo, hi = model.jnt_range[model.joint(ms.GRIPPER_JOINT).id]
    assert lo <= mp.GRIP_CLOSED < ms.GRIP_OPEN <= hi
