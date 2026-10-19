from pathlib import Path

import pybullet as p

URDF = (
    Path(__file__).resolve().parents[1]
    / "assets/robots/phantomx_pincher_cad/phantomx_pincher_cad.urdf"
)


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
