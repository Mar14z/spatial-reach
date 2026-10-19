"""M0 keypoint 训练：train/val 泛化验证。

用法：
    conda activate arm_sim
    python scripts/train_keypoints.py --steps 2000
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.data.dataset import M0Dataset
from touch.learning.keypoint_model import (
    INPUT_H,
    INPUT_W,
    ResNetHeatmapNet,
    TinyHeatmapNet,
    collate,
    heatmap_argmax,
    keypoint_loss,
    normalize,
)

CKPT = ROOT / "outputs/checkpoints/m0_keypoints.pth"


def evaluate(model, loader, device, stride):
    model.eval()
    errors = []
    with torch.no_grad():
        for images, _, keypoints in loader:
            images = F.interpolate(
                images, size=(INPUT_H, INPUT_W), mode="bilinear", align_corners=False
            )
            images = normalize(images).to(device)
            pred_xy = heatmap_argmax(model(images)).cpu() * stride
            errors.append(torch.linalg.norm(pred_xy - keypoints, dim=-1))
    errors = torch.cat(errors)
    pck = (errors <= 3.0).float().mean().item()
    return errors.mean().item(), errors.max().item(), pck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-root", type=Path, default=ROOT / "data/synthetic/m0/train"
    )
    parser.add_argument(
        "--val-root", type=Path, default=ROOT / "data/synthetic/m0/val"
    )
    parser.add_argument("--model", choices=["tiny", "resnet18"], default="resnet18")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument(
        "--ckpt", type=Path, default=CKPT
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = M0Dataset(args.train_root, stride=args.stride, sigma=args.sigma)
    val_ds = M0Dataset(args.val_root, stride=args.stride, sigma=args.sigma)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    out_size = (480 // args.stride, 640 // args.stride)
    model = (
        ResNetHeatmapNet(out_size=out_size)
        if args.model == "resnet18"
        else TinyHeatmapNet()
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    model.train()
    while step < args.steps:
        for images, targets, _ in train_loader:
            images = F.interpolate(
                images, size=(INPUT_H, INPUT_W), mode="bilinear", align_corners=False
            )
            images = normalize(images).to(device)
            targets = targets.to(device)
            loss = keypoint_loss(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 50 == 0 or step == 1:
                print(f"step {step:4d} | loss {loss.item():.6f}")
            if step % args.eval_every == 0:
                tr_mean, tr_max, tr_pck = evaluate(
                    model, train_loader, device, args.stride
                )
                va_mean, va_max, va_pck = evaluate(
                    model, val_loader, device, args.stride
                )
                print(
                    f"  train mean {tr_mean:.2f} px | pck@3 {tr_pck:.3f}\n"
                    f"  val   mean {va_mean:.2f} px | pck@3 {va_pck:.3f} "
                    f"| max {va_max:.2f}"
                )
                model.train()
            if step >= args.steps:
                break

    tr_mean, tr_max, tr_pck = evaluate(
        model, train_loader, device, args.stride
    )
    va_mean, va_max, va_pck = evaluate(
        model, val_loader, device, args.stride
    )
    print(
        f"final train mean {tr_mean:.2f} px | pck@3 {tr_pck:.3f}\n"
        f"final val   mean {va_mean:.2f} px | pck@3 {va_pck:.3f} | max {va_max:.2f}"
    )

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.ckpt)
    print("saved", args.ckpt)


if __name__ == "__main__":
    main()
