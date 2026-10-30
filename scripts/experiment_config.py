"""闭环实验共享常量：方块布置、夹爪容差、臂正前方方位角。

mujoco_closed_loop / memory_closed_loop / render_pair_b / sample_blind_view
必须使用同一套格子，否则 A/B 序列与抓取任务对不上。改这里等于改实验配方，
必须重跑对应回归。本模块保持纯数值，不导入 MuJoCo / PyBullet / torch。
"""

from __future__ import annotations

import math

# 臂正前方在世界系的方位角：arm_base_joint 绕 z 转了 -90°，pan≈0 时夹爪朝世界 -y。
AZ_FORWARD = -math.pi / 2

# 方块真实位置：落在 mujoco_pick_sweep 验证过可达的区间内，且不踩扫描格点。
CELLS = ((0.10, 0.0), (0.12, -0.3), (0.12, 0.3), (0.14, 0.0))

# 只改方位角、保留每个格子的半径与出现次数，用于拆开半径/偏轴两类失败。
CELLS_AZ0 = ((0.10, 0.0), (0.12, 0.0), (0.12, 0.0), (0.14, 0.0))

# 夹爪横向容差：注入实验里开始失败的量级，是悬崖不是软边界。
TOL_MM = 15.0


def cell_xy(seq: int, k: int = 1) -> tuple[float, float]:
    """(seq, k) 的方块中心 (x, y)；k=1 对应每序列两行的 2*seq。"""
    r, az_rel = CELLS[(2 * seq + (k - 1)) % len(CELLS)]
    az = AZ_FORWARD + az_rel
    return r * math.cos(az), r * math.sin(az)
