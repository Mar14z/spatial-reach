"""用 keypoint 模型预测 val 帧的 2D 点，保存为 JSON。

用法：
    conda activate arm_sim
    python scripts/predict_keypoints.py
输出：
    outputs/eval/pred_keypoints.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.data.dataset import M0Dataset
from touch.learning.keypoint_model import (
    INPUT_H,
    INPUT_W,
    MobileNetHeatmapNet,
    ResNetHeatmapNet,
    TinyHeatmapNet,
    normalize,
    soft_argmax,
)

VAL = ROOT / "data/synthetic/m0_cad/val"
CKPT = ROOT / "outputs/checkpoints/m0_cad_keypoints.pth"
OUT = ROOT / "outputs/eval/pred_keypoints.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", type=Path, default=VAL)
    parser.add_argument("--ckpt", type=Path, default=CKPT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--model", choices=["tiny", "resnet18", "mobilenet_v3"], default="resnet18"
    )
    parser.add_argument("--drop-gripper", action="store_true")
    parser.add_argument(
        "--preprocess",
        choices=["none", "gray", "gamma", "blur", "norm", "equalize"],
        default="none",
    )
    parser.add_argument("--extra-labels", type=Path, default=None)
    parser.add_argument(
        "--input-size",
        type=str,
        default=f"{INPUT_W}x{INPUT_H}",
        help="必须与训练时一致；热图始终是 640x480/stride，本参数只改 encoder 看到的清晰度",
    )
    args = parser.parse_args()

    in_w, in_h = (int(v) for v in args.input_size.lower().split("x"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = M0Dataset(
        args.val,
        stride=4,
        sigma=3.0,
        image_size=(in_h, in_w),
        preprocess=args.preprocess,
        extra_labels=args.extra_labels,
    )
    n_keypoints = dataset.n_keypoints - (1 if args.drop_gripper else 0)
    if args.model == "resnet18":
        model = ResNetHeatmapNet(
            n_keypoints=n_keypoints, out_size=(120, 160)
        )
    elif args.model == "mobilenet_v3":
        model = MobileNetHeatmapNet(
            n_keypoints=n_keypoints, out_size=(120, 160)
        )
    else:
        model = TinyHeatmapNet(n_keypoints=n_keypoints, out_size=(120, 160))
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval().to(device)

    results = {}
    all_errors = []
    for index in range(len(dataset)):
        sample = dataset[index]
        # 与 collate 同理：permute 只改 strides，不补齐连续内存 cuDNN 会走慢路径
        image = torch.from_numpy(
            np.ascontiguousarray(sample["image"].transpose(2, 0, 1))
        )[None]
        image = F.interpolate(
            image, size=(in_h, in_w), mode="bilinear", align_corners=False
        )
        image = normalize(image).to(device)
        with torch.no_grad():
            heatmaps = model(image)[0].cpu()
        pred_2d = soft_argmax(heatmaps[None])[0].numpy() * 4.0
        gt = sample["keypoints"][:n_keypoints]
        errors = np.linalg.norm(pred_2d - gt, axis=1)
        all_errors.append(errors)
        results[sample["frame"]] = {
            "pred_2d": pred_2d.tolist(),
            "mean_err_px": float(errors.mean()),
        }
        print(f"{index:03d} {sample['frame']} | mean {errors.mean():.2f}px")

    all_errors = np.stack(all_errors)
    summary = {
        "frames": len(dataset),
        "mean_px": float(all_errors.mean()),
        "pck3": float((all_errors <= 3.0).mean()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"summary": summary, "frames": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"summary: mean {summary['mean_px']:.2f}px | "
        f"PCK@3 {summary['pck3']:.3f}"
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
