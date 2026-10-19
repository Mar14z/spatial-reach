from pathlib import Path

import numpy as np
import pybullet as p

from touch.sim.camera import intrinsics_from_fov, project_base_points

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf"


def test_projection_matches_pybullet():
    p.connect(p.DIRECT)
    p.setGravity(0, 0, 0)
    bid = p.loadURDF(str(URDF), useFixedBase=True)
    links = {
        p.getJointInfo(bid, i)[12].decode(): i
        for i in range(p.getNumJoints(bid))
    }
    names = [
        "base_link",
        "arm_shoulder_pan_link",
        "arm_bicep_link",
        "arm_forearm_link",
        "arm_wrist_flex_link",
        "gripper_active_link",
    ]
    points = np.array([p.getLinkState(bid, links[n])[4] for n in names])

    width, height, fov = 1024, 768, 50
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.02, 0.0, 0.20],
        distance=0.66,
        yaw=50,
        pitch=-12,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=fov, aspect=width / height, nearVal=0.01, farVal=5.0
    )

    # PyBullet 的 view 是 OpenGL 约定：x 右、y 上、z 后。
    # 转换到 OpenCV：x 右、y 下、z 前。
    v_gl = np.array(view, dtype=float).reshape(4, 4).T
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    t_cv = flip @ v_gl
    r_cb, t_cb = t_cv[:3, :3], t_cv[:3, 3]

    k = intrinsics_from_fov(width, height, fov)
    uv, valid = project_base_points(k, r_cb, t_cb, points)
    assert valid.all()

    p_proj = np.array(proj, dtype=float).reshape(4, 4).T
    expected = []
    for point in points:
        clip = p_proj @ v_gl @ np.array([*point, 1.0])
        ndc = clip[:3] / clip[3]
        expected.append(
            [(ndc[0] * 0.5 + 0.5) * width, (1.0 - (ndc[1] * 0.5 + 0.5)) * height]
        )

    error = np.abs(uv - np.array(expected)).max()
    p.disconnect()
    assert error < 1e-4, f"max pixel error: {error}"
