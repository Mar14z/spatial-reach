"""棋盘格位姿 demo：实时输出标定板在相机坐标系下的 3D 位置。

用法：
    conda activate cv_env
    python scripts/demo_board_pose.py --square-size 0.0366
    python scripts/demo_board_pose.py --once      # 只采一帧，打印结果后退出

坐标约定（OpenCV 相机系）：
    X 向右，Y 向下，Z 向前（相机看出去的方向）
    tvec 是标定板原点在相机系下的位置，单位米。

注意：
    --square-size 必须是你屏幕上实测的方格边长（米）。
    AceProCamera 默认去畸变，所以这里 PnP 使用零畸变；
    如果 --no-undistort，则使用标定文件里的 dist。
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


def find_corners(gray: np.ndarray, cols: int, rows: int):
    flags = 0
    if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
        flags |= cv2.CALIB_CB_EXHAUSTIVE
    if hasattr(cv2, "CALIB_CB_ACCURACY"):
        flags |= cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
    if ok:
        return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ok, corners


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=int, default=9, help="inner corners along x")
    parser.add_argument("--rows", type=int, default=6, help="inner corners along y")
    parser.add_argument(
        "--square-size", type=float, default=0.0366, help="measured square edge in meters"
    )
    parser.add_argument("--once", action="store_true", help="capture one frame and exit")
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="use raw frames and the calibrated distortion coefficients",
    )
    args = parser.parse_args()

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * args.square_size
    )
    center = np.array(
        [
            (args.cols - 1) * args.square_size / 2.0,
            (args.rows - 1) * args.square_size / 2.0,
            0.0,
        ],
        dtype=np.float64,
    ).reshape(3, 1)

    with AceProCamera(undistort=not args.no_undistort) as cam:
        K = cam.intrinsics()
        dist = np.zeros(5, dtype=np.float64) if cam.is_undistorted else cam.distortion()
        print(f"camera {cam.width}x{cam.height}, undistorted={cam.is_undistorted}")
        print("K:\n", K)
        print("press Q/ESC to quit" if not args.once else "once mode")

        while True:
            image = cam.capture()  # RGB
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ok, corners = find_corners(gray, args.cols, args.rows)

            text = "board not found"
            if ok:
                ok_pnp, rvec, tvec = cv2.solvePnP(
                    objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ok_pnp:
                    R, _ = cv2.Rodrigues(rvec)
                    center_cam = (R @ center + tvec).reshape(-1)
                    x, y, z = center_cam
                    dist_m = float(np.linalg.norm(center_cam))
                    text = (
                        f"center camera xyz = ({x:+.3f}, {y:+.3f}, {z:+.3f}) m  "
                        f"distance = {dist_m:.3f} m"
                    )
                    cv2.drawFrameAxes(image, K, dist, rvec, tvec, args.square_size * 2)

            cv2.putText(
                image,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "Q/ESC = quit" if not args.once else "",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if args.once:
                print(text)
                break

            cv2.imshow("board pose (camera frame)", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
