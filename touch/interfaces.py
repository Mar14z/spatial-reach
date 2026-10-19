"""仿真与真机共用的后端契约。

上层（perception / spatial / grasp / pipeline）只依赖这两个协议，
不感知底层是 PyBullet 还是真实串口与 USB 相机。
"""

from typing import Protocol

import numpy as np


class ArmBackend(Protocol):
    """机械臂后端：状态读取、指令下发、运动学。"""

    def get_q(self) -> np.ndarray:
        """返回当前关节角（rad），形状 (n_joints,)。"""
        ...

    def set_q(self, q: np.ndarray) -> None:
        """下发目标关节角并等待到位。"""
        ...

    def fk(self, q: np.ndarray | None = None) -> np.ndarray:
        """正运动学：关节角 → 末端 4x4 位姿（基座系）。"""
        ...

    def ik(self, pose: np.ndarray) -> np.ndarray | None:
        """逆运动学：末端 4x4 位姿 → 关节角；无解返回 None。"""
        ...


class CameraSource(Protocol):
    """相机后端：内参与图像采集。"""

    def intrinsics(self) -> np.ndarray:
        """返回 3x3 相机内参矩阵。"""
        ...

    def capture(self) -> np.ndarray:
        """返回一帧 RGB 图像，形状 (H, W, 3)，uint8。"""
        ...
