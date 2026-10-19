"""颜色块 demo：识别颜色块，输出它在相机坐标系下的 3D 位置。

用法：
    conda activate cv_env
    python scripts/demo_color_pose.py --color red --distance 0.5
    python scripts/demo_color_pose.py --color green --distance 0.6 --once
    python scripts/demo_color_pose.py --selftest

坐标约定（OpenCV 相机系）：
    X 向右，Y 向下，Z 向前
    输出的 (X, Y, Z) 单位是米。

尺度假设：
    单目图像只能给出“像素射线”，不能直接给出深度。
    这个 demo 用 --distance 作为“物体所在平面到相机的垂直距离”：
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = distance
    如果相机是斜着看桌面的，constant-Z 不成立；下一步要用桌面平面做
    射线-平面求交（plane: n·X = d）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.camera import AceProCamera

COLOR_RANGES = {
    "red": [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (180, 255, 255))],
    "green": [((35, 60, 40), (85, 255, 255))],
    "blue": [((90, 60, 40), (130, 255, 255))],
    "yellow": [((20, 80, 60), (35, 255, 255))],
}


def detect_color(image_rgb: np.ndarray, color: str, min_area: int = 500):
    """返回最大色块的 (u, v, area) 或 None。"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]
    return u, v, area, contour


def pixel_to_camera(K: np.ndarray, u: float, v: float, z: float):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return (u - cx) * z / fx, (v - cy) * z / fy, z


def selftest() -> None:
    from touch.camera import load_intrinsics

    intr = load_intrinsics("configs/camera_intrinsics.yaml")
    image = np.zeros((intr.height, intr.width, 3), dtype=np.uint8)
    # 画一个红色方块，中心在 (800, 400)
    image[360:440, 760:840] = (255, 0, 0)
    result = detect_color(image, "red")
    assert result is not None
    u, v, area, _ = result
    x, y, z = pixel_to_camera(intr.K, u, v, 0.5)
    print(f"selftest: center=({u:.1f}, {v:.1f}) area={area:.0f}")
    print(f"camera xyz=({x:+.3f}, {y:+.3f}, {z:+.3f}) m")
    assert abs(u - 800) < 5 and abs(v - 400) < 5
    assert abs(x - (u - intr.K[0, 2]) * 0.5 / intr.K[0, 0]) < 1e-9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", choices=sorted(COLOR_RANGES), default="red")
    parser.add_argument("--distance", type=float, default=0.5, help="object plane Z in meters")
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-undistort", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    with AceProCamera(undistort=not args.no_undistort) as cam:
        K = cam.intrinsics()
        print(f"camera {cam.width}x{cam.height}, undistorted={cam.is_undistorted}")
        print(f"assumed object plane Z = {args.distance:.3f} m")
        while True:
            image = cam.capture()
            result = detect_color(image, args.color, args.min_area)
            text = f"{args.color} not found"
            if result is not None:
                u, v, area, contour = result
                x, y, z = pixel_to_camera(K, u, v, args.distance)
                text = f"uv=({u:.1f},{v:.1f})  xyz=({x:+.3f},{y:+.3f},{z:+.3f}) m"
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                cv2.circle(image, (int(round(u)), int(round(v))), 5, (0, 255, 0), -1)
                cv2.drawMarker(image, (int(K[0, 2]), int(K[1, 2])), (255, 0, 0),
                               cv2.MARKER_CROSS, 20, 2)

            cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "Q/ESC = quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if args.once:
                print(text)
                break

            cv2.imshow("color pose (camera frame)",
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
