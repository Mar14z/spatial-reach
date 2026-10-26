"""抓取成功率 vs 感知误差：指令抓取点相对方块中心的横向偏移。

方块不动，偏移量代表「反投影把物体位置算错了多少」。现有精度是桌面反投影
median 0.72cm / p90 2.07cm / p95 2.73cm，所以 0–15mm 这段才是真实工作点。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco_pick as mp

AZ_FORWARD = mp.AZ_FORWARD
SIZES = (0.03, 0.04)
RADII = (0.09, 0.13)
JITTERS = (0.000, 0.005, 0.007, 0.010, 0.015)
if len(sys.argv) > 1:  # 自定义偏移（毫米），例如 python mujoco_jitter_sweep.py 20 25 30
    JITTERS = tuple(float(x) / 1000 for x in sys.argv[1:])


def main():
    for s in SIZES:
        for r in RADII:
            cells = []
            for j in JITTERS:
                res = mp.pick(s, r, AZ_FORWARD, verbose=False, jitter=j,
                              seed=int(s * 1e4 + r * 1e3 + j * 1e5))
                ok = "✓" if res["ok_grasp"] else "✗"
                cells.append(f"{ok} {res['dz']*1000:5.1f}")
                print(f"cube {s*100:.0f}cm r={r:.2f} jitter={j*1000:4.1f}mm -> "
                      f"{'OK ' if res['ok_grasp'] else 'FAIL'} dz={res['dz']*1000:5.1f}mm "
                      f"align={res['align']*1000:5.1f}mm in-hand={res['in_hand']*1000:5.1f}mm "
                      f"gap={res['gap']:5.1f}mm", flush=True)
            print(f"  | {s*100:.0f}cm r={r:.2f} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
