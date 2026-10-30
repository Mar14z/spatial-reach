"""可抓/不可抓判定：把「臂够不够得着」和「这次感知敢不敢信」合成一个三档决策。

M2 空间认知五问的第三问（什么关系 → 可抓/不可抓）的第一块砖。
判据全部来自实测，改阈值前先看来源、重新实测：

- 工作空间：mujoco_pick_sweep 24/24（方块 3/4cm × r 0.09–0.15m × |az|≤0.4rad）。
  边界外是「未验证」，按 no 处理——宁可拒也不盲试。
- 定位悬崖：闭环 360 次（2026-10-27）——取点误差 ≤20mm 抓到 99.4%，>20mm 仅 15.8%。
- 安全线 15mm = 悬崖 20mm 留 5mm 余量；夹爪 in_hand 门限也是 20mm。
"""

from __future__ import annotations

# 工作空间（mujoco_pick_sweep 实测全过的范围）
R_MIN_M = 0.09
R_MAX_M = 0.15
AZ_MAX_RAD = 0.4

# 定位质量（闭环 360 次，2026-10-27）
SAFE_ERR_MM = 15.0
CLIFF_ERR_MM = 20.0


def grippable(radius_m: float, az_rel_rad: float, err_p95_mm: float | None) -> dict:
    """三档判定，返回 {"verdict": yes|maybe|no, "reason": str, "detail": str}。

    err_p95_mm 是该相机序列取点误差的 p95（在线可估：拿历史序列的
    反投影误差统计）。None = 还没有误差估计，只判工作空间。
    """
    if not R_MIN_M <= radius_m <= R_MAX_M:
        return {"verdict": "no", "reason": "out_of_workspace",
                "detail": f"r={radius_m:.3f}m 超出实测可达 [{R_MIN_M}, {R_MAX_M}]"}
    if abs(az_rel_rad) > AZ_MAX_RAD:
        return {"verdict": "no", "reason": "az_out_of_range",
                "detail": f"|az|={abs(az_rel_rad):.2f}rad 超出实测 {AZ_MAX_RAD}"}
    if err_p95_mm is None:
        return {"verdict": "maybe", "reason": "no_error_estimate",
                "detail": "工作空间内但无误差估计，降速试探"}
    if err_p95_mm <= SAFE_ERR_MM:
        return {"verdict": "yes", "reason": "ok",
                "detail": f"err_p95={err_p95_mm:.1f}mm ≤ 安全线 {SAFE_ERR_MM}"}
    if err_p95_mm <= CLIFF_ERR_MM:
        return {"verdict": "maybe", "reason": "err_near_cliff",
                "detail": f"err_p95={err_p95_mm:.1f}mm 落在 {SAFE_ERR_MM}–{CLIFF_ERR_MM}mm 余量带"}
    return {"verdict": "no", "reason": "err_over_cliff",
            "detail": f"err_p95={err_p95_mm:.1f}mm > 悬崖 {CLIFF_ERR_MM}mm"}
