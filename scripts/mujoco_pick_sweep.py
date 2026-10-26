"""开环抓取成功率扫描：方块尺寸 × 工作半径 × 方位角。

判据与 mujoco_pick.pick 一致：抬升 >5cm、抓取点横向对齐 <10mm、夹持内偏移 <20mm。

方位角是**相对臂正前方**的：URDF 里 arm_base_joint 绕 z 转了 -90°，所以 pan≈0 时
夹爪对着世界 -y；直接按世界 +x 取方位角，+0.2 就撞 pan 限位了，看起来像"抓不到"。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco_pick as mp

AZ_FORWARD = -np.pi / 2  # 臂正前方在世界系里的方位角
SIZES = (0.03, 0.04)
RADII = (0.09, 0.11, 0.13, 0.15)
AZIMUTHS = (-0.4, 0.0, 0.4)


def main():
    rows = []
    for s in SIZES:
        for r in RADII:
            cells = []
            for az_rel in AZIMUTHS:
                az = AZ_FORWARD + az_rel
                res = mp.pick(s, r, az, verbose=False)
                ok = "✓" if res["ok"] else ("g" if res["ok_grasp"] else "✗")
                cells.append(f"{ok} {res['align']*1000:4.1f}/{res['in_hand']*1000:4.1f}")
                rows.append((s, r, az_rel, res))
                print(f"cube {s*100:.0f}cm r={r:.2f} az{az_rel:+.1f} -> "
                      f"{'OK ' if res['ok'] else ('GRASP' if res['ok_grasp'] else 'FAIL')} "
                      f"dz={res['dz']*1000:5.1f}mm "
                      f"align={res['align']*1000:5.1f}mm in-hand={res['in_hand']*1000:5.1f}mm "
                      f"gap={res['gap']:5.1f}mm")
            print(f"  | {s*100:.0f}cm r={r:.2f} | " + " | ".join(cells) + " |", flush=True)
    n_ok = sum(r[3]["ok"] for r in rows)
    n_g = sum(r[3]["ok_grasp"] for r in rows)
    print(f"\n严格（抬升>5cm 且 对齐<10mm 且 夹持内偏移<20mm）：{n_ok}/{len(rows)} = {n_ok/len(rows):.0%}")
    print(f"抓取（抬升>2cm 且 夹持内偏移<20mm）：{n_g}/{len(rows)} = {n_g/len(rows):.0%}")
    print("失败明细：")
    for s, r, az_rel, res in rows:
        if not res["ok"]:
            print(f"  cube={s} r={r} az_rel={az_rel:+.1f} dz={res['dz']*1000:.1f}mm "
                  f"align={res['align']*1000:.1f}mm in-hand={res['in_hand']*1000:.1f}mm "
                  f"pre_err={res['pre_err']*1000:.1f}mm")


if __name__ == "__main__":
    main()
