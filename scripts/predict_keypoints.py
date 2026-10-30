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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.data.dataset import M0Dataset
from touch.data.recipe import Recipe, sha256_of
from touch.learning import inference
from touch.learning.keypoint_model import INPUT_H, INPUT_W

VAL = ROOT / "data/synthetic/m0_cad/val"
CKPT = ROOT / "outputs/checkpoints/m0_cad_keypoints.pth"
OUT = ROOT / "outputs/eval/pred_keypoints.json"

# 热图 = 640x480 / STRIDE，必须与 M0Dataset 的 stride 一致
STRIDE = 4


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
        "--decode",
        choices=["soft", "argmax", "subpixel"],
        default="soft",
        help="热图解码：soft=全局期望（历史口径）; argmax=峰值; subpixel=峰值+抛物线",
    )
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
        stride=STRIDE,
        sigma=3.0,
        image_size=(in_h, in_w),
        preprocess=args.preprocess,
        extra_labels=args.extra_labels,
    )
    n_keypoints = dataset.n_keypoints - (1 if args.drop_gripper else 0)
    # 用命令行参数现场组一份配方，推理走与训练验证相同的契约；
    # 想用登记过的配方时，改从 configs/baseline.yaml 读，不要在这里另写一套。
    recipe = Recipe(
        name=args.ckpt.stem,
        mode="legacy_replay",
        checkpoint=args.ckpt,
        sha256=sha256_of(args.ckpt),
        model=args.model,
        n_keypoints=n_keypoints,
        input_width=in_w,
        input_height=in_h,
        out_width=inference.CANONICAL_W // STRIDE,
        out_height=inference.CANONICAL_H // STRIDE,
        preprocess=args.preprocess,
        normalize=True,
        decode=args.decode,
        keypoints=tuple(f"kp{i}" for i in range(n_keypoints)),
    )
    model = inference.load_model(recipe, device)

    results = {}
    all_errors = []
    for index in range(len(dataset)):
        sample = dataset[index]
        pred_2d = inference.predict(
            sample["image"], recipe, model=model, device=device
        )
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
