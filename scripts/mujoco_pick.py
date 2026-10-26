"""MuJoCo 开环抓取：IK 到方块正上方 -> 下探 -> 合爪 -> 抬起。

夹爪是齿条联动的对称双指（见 mujoco_scene）：两指各走一半，所以**指腹中点恒在
腕部臂系 y=0 平面**，抓取点就是它。指腹中心在腕部系 z=88.5mm（指尖沿 +z 伸出
68~109mm），开口间隙 61.7mm、合到底留 ~2mm。

坐标都在 arm_wrist_flex_link 系：x 径向、y 合爪轴、z 指尖方向。

用法：
    python scripts/mujoco_pick.py --view                 # 开窗口看一次抓取
    python scripts/mujoco_pick.py --size 0.04 --az 0.4   # 无头（批量扫描必须这样）
"""
import argparse
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import mujoco_scene as ms

# 臂正前方在世界系里的方位角：`arm_base_joint` 绕 z 转了 -90°，pan≈0 时夹爪朝
# 世界 -y。命令行/扫描一律用「相对正前方」的方位角，否则 +0.2 就撞 pan 限位。
AZ_FORWARD = -np.pi / 2
# mujoco.viewer 的 passive 句柄。None = 无头：批量扫描必须保持 None。
VIEW = None

GRASP_OFFSET = np.array([0.0, 0.0, 0.0885])
GRIP_OPEN = ms.GRIP_OPEN
GRIP_CLOSED = -0.01385  # 空夹时指腹间隙 1.96mm；方块自己会先挡住
W_YAW = 1.0    # 偏航必须严格：指腹与方块面不平行，合爪会变成推
W_TILT = 0.05  # 倾斜可以放宽，换低处可达
W_JOINT = 0.05


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _step(model, data):
    """推进一步物理；开着窗口时顺带同步画面，并按仿真时间放慢到真实速度。"""
    mujoco.mj_step(model, data)
    if VIEW is None:
        return
    if not VIEW.is_running():
        raise SystemExit("viewer 已关闭，中断仿真")
    VIEW.sync()
    time.sleep(model.opt.timestep)  # 不睡的话 0.002s 一步会瞬间跑完，看不到过程


def _errs(model, data, body_id, p_tgt, R_tgt):
    """返回 (位置误差, 加权旋转误差向量, 权重)，旋转误差按偏航/倾斜拆开。"""
    mujoco.mj_forward(model, data)
    R = data.xmat[body_id].reshape(3, 3)
    p = data.xpos[body_id] + R @ GRASP_OFFSET
    e_rot = 0.5 * sum(np.cross(R_tgt[:, i], R[:, i]) for i in range(3))
    w = np.array([W_YAW, W_YAW, W_TILT])
    return p_tgt - p, w * e_rot, w


def _refine(model, data, body_id, p_tgt, R_tgt, iters, lo, hi):
    """带回溯线搜索的 DLS：直接信任线性化会发散（实测从 7mm 冲到 170mm）。"""
    e_pos, e_rot, W = _errs(model, data, body_id, p_tgt, R_tgt)
    cost = np.linalg.norm(e_pos) + np.linalg.norm(e_rot)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(iters):
        R = data.xmat[body_id].reshape(3, 3)
        r = R @ GRASP_OFFSET
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        J = np.vstack([jacp - skew(r) @ jacr, jacr])[:6, :4]
        J[3:] = W[:, None] * J[3:]
        e = np.concatenate([e_pos, e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
        q0 = data.qpos[:4].copy()
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            data.qpos[:4] = np.clip(q0 + alpha * np.clip(dq, -0.2, 0.2), lo, hi)
            e_pos, e_rot, _ = _errs(model, data, body_id, p_tgt, R_tgt)
            c = np.linalg.norm(e_pos) + np.linalg.norm(e_rot)
            if c < cost - 1e-9:
                cost, improved = c, True
                break
        if not improved:
            data.qpos[:4] = q0
            break
    e_pos, e_rot, _ = _errs(model, data, body_id, p_tgt, R_tgt)
    return np.linalg.norm(e_pos), np.linalg.norm(e_rot), cost


def solve_ik(model, data, body_id, p_tgt, R_tgt, iters=120, refine_top=8,
             q_prev=None, pan=None, rng=None):
    """4 自由度 + 6 维目标：先在 pan 网格 × pitch 初值上粗筛，只精修最好的几个。
    单起点 DLS 实测会陷局部极小（差 57~70mm）。pan 锁死是连续性约束——否则 pan
    会从一侧翻到另一侧，机械臂横扫把方块推走。"""
    rng = np.random.default_rng(0) if rng is None else rng
    q_save = data.qpos[:4].copy()  # 内部搜索会改 qpos，结尾还原，否则等于瞬移机械臂
    lo, hi = model.jnt_range[:4, 0], model.jnt_range[:4, 1]
    if pan is not None:
        lo, hi = lo.copy(), hi.copy()
        lo[0] = hi[0] = pan
    cands = [[p, *q] for p in ([pan] if pan is not None else np.linspace(lo[0], hi[0], 48))
             for q in ((0.6, -1.2, 0.6), (1.5, -0.5, 1.0), (-1.5, 0.5, 0.0))]
    cands += list(rng.uniform(lo, hi, size=(1500, 4)))
    if pan is not None:
        for c in cands:
            c[0] = pan
    if q_prev is not None:
        # 枚举出来的到位姿常在一条极窄的流形上，随机采样碰不到；不把它当候选，
        # 精修会从别处出发，结果比初值还差（实测 4mm -> 21mm）。
        seed = np.array(q_prev, dtype=float)
        cands = [[*seed]] + [[*np.clip(seed + rng.normal(0, 0.15, 4), lo, hi)]
                             for _ in range(200)] + cands
        if pan is not None:
            for c in cands:
                c[0] = pan
    scored = []
    for q in cands:
        data.qpos[:4] = q
        e_pos, e_rot, _ = _errs(model, data, body_id, p_tgt, R_tgt)
        c = np.linalg.norm(e_pos) + np.linalg.norm(e_rot)
        if q_prev is not None:
            c += W_JOINT * np.linalg.norm(np.asarray(q) - q_prev)
        scored.append((c, q))
    scored.sort(key=lambda t: t[0])
    best = None
    for _, q in scored[:refine_top]:
        data.qpos[:4] = q
        pe, re_, cost = _refine(model, data, body_id, p_tgt, R_tgt, iters, lo, hi)
        if q_prev is not None:
            cost += W_JOINT * np.linalg.norm(data.qpos[:4] - q_prev)
        if best is None or cost < best[0]:
            best = (cost, pe, re_, data.qpos[:4].copy())
    _, pos_err, rot_err, q = best
    data.qpos[:4] = q_save
    return pos_err, rot_err, q


def move(model, data, q_from, q_to, grip_from, grip_to, seconds=1.5):
    """ctrl 一步跳变会让方块被甩脱，按关节空间线性插值。第二指由等式约束跟随。"""
    steps = max(1, int(seconds / model.opt.timestep))
    for i in range(1, steps + 1):
        a = i / steps
        for j in range(4):
            data.ctrl[j] = q_from[j] + a * (q_to[j] - q_from[j])
        data.ctrl[4] = grip_from + a * (grip_to - grip_from)
        _step(model, data)


def cartesian_move(model, data, wid, R, p_from, p_to, grip, pan, steps=8, per_step=0.3,
                   warn=True):
    """按笛卡尔直线分步走。关节空间直接插值会让手腕中途转十几度，夹住的方块
    被两指挤出去（实测抬升 0.15s 内就滑脱）。"""
    q_prev = data.qpos[:4].copy()
    for i in range(1, steps + 1):
        p = p_from + (p_to - p_from) * (i / steps)
        pe, re_, q = solve_ik(model, data, wid, p, R, q_prev=q_prev, pan=pan)
        if pe > 0.01 and warn:
            print(f"  warn: step {i}/{steps} pos_err {pe*1000:.1f} mm")
        move(model, data, q_prev, q, grip, grip, per_step)
        q_prev = q
    return q_prev


def pad_pos(model, data, body_id):
    mujoco.mj_forward(model, data)
    R = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + R @ GRASP_OFFSET, R


def find_grasp_pose(model, data, wid, p_goal, azimuth=0.0, n=200000, rng=None):
    """枚举「指尖朝下 + 合爪轴沿切向 + 抓取点高度可达」的姿态，取代调权重的 IK。

    这条 4 自由度臂在严格竖直 + 偏航对齐时只能落在很窄的流形上，逐点 IK 会在
    位置和姿态之间来回妥协（实测最优也要差 17mm）。
    """
    rng = np.random.default_rng(0) if rng is None else rng
    lo, hi = model.jnt_range[:4, 0], model.jnt_range[:4, 1]
    radial = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    best = None
    for _ in range(n // 5000):
        for q in rng.uniform(lo, hi, size=(5000, 4)):
            data.qpos[:4] = q
            mujoco.mj_forward(model, data)
            R = data.xmat[wid].reshape(3, 3)
            if R[2, 2] > -0.996 or abs(R[:, 1] @ radial) > 0.087:  # tilt<5°, 合爪轴切向
                continue
            p = data.xpos[wid] + R @ GRASP_OFFSET
            if not 0.010 < p[2] < 0.024:  # 方块中心高度，再低指尖就戳桌面
                continue
            d = np.linalg.norm(p - p_goal)
            margin = np.minimum(hi - q, q - lo).min()  # 贴限位的姿态下探时解不动
            d += 0.5 * max(0.0, 0.15 - margin)
            if best is None or d < best[0]:
                best = (d, q.copy(), p.copy(), R.copy())
    return best


def pick(cube_size=0.03, radius=0.12, azimuth=0.0, n=200000, verbose=True,
         jitter=0.0, seed=0, view=False):
    """jitter: 指令抓取点相对方块中心的横向偏移（米），用来注入感知误差。
    方块不动，动的是"我以为它在哪"。

    view: 打开 MuJoCo passive viewer 看过程。批量扫描不要开（每格一个窗口）。
    """
    global VIEW
    model = ms.build_spec(cube_xy=(radius * np.cos(azimuth), radius * np.sin(azimuth)),
                          cube_size=cube_size).compile()
    data = mujoco.MjData(model)
    if view:
        from mujoco import viewer  # 延迟引入：无头路径不碰 glfw
        VIEW = viewer.launch_passive(model, data)
        print("viewer 已打开；下面枚举抓取姿态要跑几十万次 mj_forward，"
              "窗口会短暂无响应，属正常")
    data.qpos[4] = GRIP_OPEN
    data.qpos[5] = GRIP_OPEN  # 第二指也要摆到开口位；等式约束只在积分时生效
    cube = model.body("cube").id
    wid = model.body("arm_wrist_flex_link").id
    goal = np.array([radius * np.cos(azimuth), radius * np.sin(azimuth), cube_size / 2])
    cmd = goal.copy()
    if np.ndim(jitter) > 0:
        # 直接给偏移向量：闭环用，误差由几何链算出来，方向和大小都是真的
        cmd[:2] += np.asarray(jitter, dtype=float).reshape(2)
    elif jitter > 0:
        # 标量：随机方向注入，用于容差扫描
        a = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        cmd[:2] += jitter * np.array([np.cos(a), np.sin(a)])
    dist, q_seed, p_seed, R_TARGET = find_grasp_pose(model, data, wid, cmd, azimuth, n)
    pe_g, re_g, q_grasp = solve_ik(model, data, wid, cmd, R_TARGET, q_prev=q_seed)
    adr = model.jnt_qposadr[model.joint("cube_free").id]
    data.qpos[adr:adr + 3] = goal
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    data.qpos[:4] = 0.0
    data.qpos[4] = data.qpos[5] = GRIP_OPEN
    mujoco.mj_forward(model, data)
    p_cube = data.xpos[cube].copy()
    # pregrasp 高度不能写死：同一位姿抬升 10cm 处勉强可达、12cm 完全不可达（实测差 47mm），
    # 从固定 9cm 起步会让机械臂在第一步就被拽到一个不可达点。逐档降高取第一个可解的。
    # pregrasp 阶段放开 pan（锁死时高处位姿常不可达，实测 az≠0 残差 50mm+），下探才锁。
    p_pre, q_pre, pe, re_ = None, None, np.inf, np.inf
    best_h = (np.inf, None, None, np.inf, np.inf)
    for h in (0.09, 0.06, 0.03, 0.0):
        pe_h, re_h, q_h = solve_ik(model, data, wid, cmd + [0, 0, h], R_TARGET,
                                   q_prev=q_grasp)
        if pe_h < best_h[0]:
            best_h = (pe_h, cmd + [0, 0, h], q_h, pe_h, re_h)
        if pe_h < 0.004:
            p_pre, q_pre, pe, re_ = cmd + [0, 0, h], q_h, pe_h, re_h
            break
    if q_pre is None:  # 这一格整个不可达：用最好的一档硬走，失败留给判据记录
        _, p_pre, q_pre, pe, re_ = best_h
    move(model, data, data.qpos[:4].copy(), q_pre, GRIP_OPEN, GRIP_OPEN, 1.5)
    cartesian_move(model, data, wid, R_TARGET, p_pre, cmd,
                   GRIP_OPEN, q_grasp[0], steps=6, per_step=0.25, warn=verbose)
    p_now, _ = pad_pos(model, data, wid)
    align = np.linalg.norm((p_now - data.xpos[cube])[:2])
    cube_before = data.xpos[cube].copy()
    move(model, data, data.qpos[:4].copy(), q_grasp, GRIP_OPEN, GRIP_CLOSED, 1.0)
    gap = 29.65 + 2 * float(data.qpos[4]) * 1000
    cartesian_move(model, data, wid, R_TARGET, p_now, p_now + [0, 0, 0.08],
                   GRIP_CLOSED, q_grasp[0], steps=8, per_step=0.25, warn=verbose)
    p_end = data.xpos[cube].copy()
    p_grip_end, _ = pad_pos(model, data, wid)
    in_hand = np.linalg.norm((p_end - p_grip_end)[:2])
    dz = p_end[2] - p_cube[2]
    # 方块没抬到 80mm 有两种原因，必须拆开记：夹爪自己没走到位（伺服重力下垂 +
    # IK 残差），还是方块相对指腹下滑（打滑）。混在一起会把可达性算成打滑。
    pad_rise = p_grip_end[2] - p_now[2]
    rise_err = 0.08 - pad_rise  # 夹爪相对指令少走了多少
    slip = pad_rise - dz        # 方块相对指腹下滑了多少
    res = dict(ok=dz > 0.05 and in_hand < 0.02 and align < 0.010, dz=dz, align=align,
               in_hand=in_hand, gap=gap, q_grasp=q_grasp, pose_err=dist,
               pre_err=pe, pre_rot=re_, jitter=jitter, pad_rise=pad_rise,
               rise_err=rise_err, slip=slip,
               # 夹住没夹住是抓取问题；能不能抬 8cm 是这条 4 自由度臂的可达性问题。
               # 两个判据分开记，混在一起会把「抓住了但抬不到那么高」算成抓取失败。
               ok_grasp=dz > 0.02 and in_hand < 0.02)
    if verbose:
        print("grasp pose: q=%s p=%s (goal dist %.1f mm, IK err %.1f mm)"
              % (np.round(q_grasp, 3), np.round(goal, 4), dist * 1000, pe_g * 1000))
        print("cube at", np.round(p_cube, 4))
        print(f"IK pregrasp pos_err={pe*1000:5.2f} mm rot_err={re_:.4f} q={np.round(q_pre, 3)}")
        print("grasp pt at", np.round(p_now, 4), "cube", np.round(cube_before, 4),
              "align %.1f mm" % (align * 1000))
        print("after close: gap %.1f mm" % gap)
        print("after lift : cube", np.round(p_end, 4), "dz = %.3f m" % dz,
              "in-hand offset %.1f mm" % (in_hand * 1000))
        print("             pad rose %.1f mm (缺 %.1f) / cube slipped %.1f mm"
              % (pad_rise * 1000, rise_err * 1000, slip * 1000))
        print("SUCCESS" if res["ok"] else "FAILED")
    if VIEW is not None:
        # 先把最终姿态停在窗口里 3 秒，否则函数一返回窗口就没了，白开
        t_end = time.time() + 3.0
        while VIEW.is_running() and time.time() < t_end:
            VIEW.sync()
            time.sleep(0.02)
        VIEW.close()
        VIEW = None
    return res


def main():
    ap = argparse.ArgumentParser(description="单次开环抓取；--view 打开 MuJoCo 窗口")
    ap.add_argument("--size", type=float, default=0.03, help="方块边长（m）")
    ap.add_argument("--radius", type=float, default=0.12, help="工作半径（m）")
    ap.add_argument("--az", type=float, default=0.0, help="相对臂正前方的方位角（弧度）")
    ap.add_argument("--view", action="store_true", help="打开 passive viewer 看过程")
    a = ap.parse_args()
    return pick(a.size, a.radius, AZ_FORWARD + a.az, view=a.view)


if __name__ == "__main__":
    main()
