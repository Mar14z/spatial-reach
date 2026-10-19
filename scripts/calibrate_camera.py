"""相机内参标定：棋盘格 + OpenCV。

棋盘格图案：
    assets/calibration/checkerboard_9x6_1920x1080_130px.png
    （1920x1080 屏幕全屏显示；9x6 个内角点；130px 方格）

用法（Windows，需要带 OpenCV 的环境，例如 cv_env）：
    conda activate cv_env
    python scripts/calibrate_camera.py --probe          # 只看相机实际分辨率/帧率
    python scripts/calibrate_camera.py --selftest       # 用生成的图案自检角点
    python scripts/calibrate_camera.py --square-size 0.036

操作：
    SPACE  采集当前帧（必须检测到棋盘格）
    R      清空已采集样本
    ESC/Q  结束并开始标定

注意：
    --square-size 是你用尺子量到的“屏幕上单个方格的实际边长（米）”，
    不是图案生成器写的标称值，也不是相机分辨率。
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATTERN = ROOT / "assets/calibration/checkerboard_9x6_1920x1080_130px.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera intrinsic calibration with a checkerboard")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--width", type=int, default=1280, help="capture width")
    p.add_argument("--height", type=int, default=720, help="capture height")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cols", type=int, default=9, help="inner corners along x")
    p.add_argument("--rows", type=int, default=6, help="inner corners along y")
    p.add_argument(
        "--square-size",
        type=float,
        default=0.036,
        help="measured physical square edge in meters (default 0.036; measure it!)",
    )
    p.add_argument("--backend", choices=["msmf", "dshow"], default="dshow")
    p.add_argument("--min-samples", type=int, default=15)
    p.add_argument("--out", default="configs/camera_intrinsics.yaml")
    p.add_argument("--fisheye", action="store_true", help="use OpenCV fisheye model")
    p.add_argument("--probe", action="store_true", help="print camera info and exit")
    p.add_argument("--selftest", action="store_true", help="detect corners on the generated pattern")
    return p.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    backend = cv2.CAP_MSMF if args.backend == "msmf" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera} with {args.backend}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if args.backend == "dshow":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def camera_info(cap: cv2.VideoCapture) -> tuple[int, int, float, int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return w, h, fps, fourcc


def find_corners(gray: np.ndarray, cols: int, rows: int):
    flags_sb = 0
    if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
        flags_sb |= cv2.CALIB_CB_EXHAUSTIVE
    if hasattr(cv2, "CALIB_CB_ACCURACY"):
        flags_sb |= cv2.CALIB_CB_ACCURACY

    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags_sb)
    if ok:
        return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if ok:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ok, corners


def calibrate(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    size: tuple[int, int],
    fisheye: bool,
):
    if fisheye:
        objp = objpoints[0].reshape(-1, 1, 3)
        obj_f = [objp.copy() for _ in objpoints]
        img_f = [p.reshape(-1, 1, 2) for p in imgpoints]
        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_f, img_f, size, K, D, flags=flags, criteria=criteria
        )
        errors = []
        for i in range(len(obj_f)):
            projected, _ = cv2.fisheye.projectPoints(obj_f[i], rvecs[i], tvecs[i], K, D)
            errors.append(cv2.norm(img_f[i], projected, cv2.NORM_L2) / len(projected))
        return rms, K, D, errors

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None
    )
    errors = []
    for i, objp in enumerate(objpoints):
        projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
        errors.append(cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected))
    return rms, K, dist, errors


def save_result(path: Path, K, dist, rms: float, args, size, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("model", "fisheye" if args.fisheye else "pinhole")
    fs.write("image_width", size[0])
    fs.write("image_height", size[1])
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist)
    fs.write("rms_reprojection_error", float(rms))
    fs.write("square_size_m", float(args.square_size))
    fs.write("samples", n)
    fs.release()


def main() -> None:
    args = parse_args()

    if args.selftest:
        gray = cv2.imread(str(PATTERN), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"pattern not found: {PATTERN}")
        ok, corners = find_corners(gray, args.cols, args.rows)
        print("pattern:", PATTERN)
        print("detected:", ok, "corners:", None if corners is None else len(corners))
        return

    cap = open_camera(args)
    w, h, fps, fourcc = camera_info(cap)
    print(f"camera open: {w}x{h} @ {fps:.1f}fps fourcc={fourcc}")
    if args.probe:
        ok, frame = cap.read()
        print("read frame:", ok, None if not ok else frame.shape)
        cap.release()
        return

    out_dir = ROOT / "outputs/calibration" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * args.square_size

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    preview = None

    print("SPACE=capture, R=reset, ESC/Q=finish")
    print(f"samples dir: {out_dir}")
    print("IMPORTANT: measure the real square size on your screen and pass --square-size")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame read failed")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, args.cols, args.rows)
        preview = frame.copy()
        if found:
            cv2.drawChessboardCorners(preview, (args.cols, args.rows), corners, found)
        cv2.putText(
            preview,
            f"samples: {len(objpoints)}   SPACE=capture  R=reset  ESC=finish",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if found else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("camera calibration", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key in (ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            objpoints.clear()
            imgpoints.clear()
            print("reset")
            continue
        if key == 32:  # SPACE
            if not found:
                print("board not found; move the board into view")
                continue
            objpoints.append(objp.copy())
            imgpoints.append(corners.reshape(-1, 1, 2).copy())
            idx = len(objpoints) - 1
            cv2.imwrite(str(out_dir / f"sample_{idx:02d}.png"), preview)
            print(f"captured {len(objpoints)}")

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < args.min_samples:
        print(f"need at least {args.min_samples} samples, got {len(objpoints)}; abort")
        return

    print("calibrating...")
    rms, K, dist, errors = calibrate(objpoints, imgpoints, (w, h), args.fisheye)
    print(f"RMS reprojection error: {rms:.4f} px")
    print("K =\n", K)
    print("dist =\n", dist.reshape(-1))
    print("per-view errors (px):", [round(float(e), 3) for e in errors])

    out_path = ROOT / args.out
    save_result(out_path, K, dist, rms, args, (w, h), len(objpoints))
    print("saved:", out_path)

    report = out_dir / "report.txt"
    report.write_text(
        "\n".join(
            [
                f"camera: {w}x{h} @ {fps:.1f}fps",
                f"model: {'fisheye' if args.fisheye else 'pinhole'}",
                f"samples: {len(objpoints)}",
                f"square_size_m: {args.square_size}",
                f"rms_px: {rms:.4f}",
                "K:",
                str(K),
                "dist:",
                str(dist.reshape(-1)),
                "per_view_errors_px:",
                str([round(float(e), 3) for e in errors]),
            ]
        ),
        encoding="utf-8",
    )
    print("report:", report)


if __name__ == "__main__":
    main()
