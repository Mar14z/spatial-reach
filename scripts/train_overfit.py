"""最小过拟合训练：确认 M0 数据管线和 heatmap loss 可用。

用法：
    conda activate arm_sim
    python scripts/train_overfit.py --steps 2000
输出：
    outputs/checkpoints/m0_overfit.pth
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
    HM_H,
    HM_W,
    TinyHeatmapNet,
    collate,
    heatmap_argmax,
    keypoint_loss,
)

DATA = ROOT / "data/synthetic/m0"
CKPT = ROOT / "outputs/checkpoints/m0_overfit.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = M0Dataset(DATA, sigma=args.sigma)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    model = TinyHeatmapNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    model.train()
    while step < args.steps:
        for images, targets, _ in loader:
            images = F.interpolate(
                images, size=(HM_H, HM_W), mode="bilinear", align_corners=False
            ).to(device)
            targets = targets.to(device)
            loss = keypoint_loss(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0 or step == 1:
                print(f"step {step:4d} | loss {loss.item():.6f}")
            if step >= args.steps:
                break

    model.eval()
    errors = []
    with torch.no_grad():
        for images, _, keypoints in loader:
            images = F.interpolate(
                images, size=(HM_H, HM_W), mode="bilinear", align_corners=False
            ).to(device)
            pred_xy = heatmap_argmax(model(images)).cpu() * 4.0
            errors.append(torch.linalg.norm(pred_xy - keypoints, dim=-1))
    errors = torch.cat(errors)
    print(
        f"keypoint error: mean {errors.mean():.2f} px | "
        f"max {errors.max():.2f} px"
    )

    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    print("saved", CKPT)


if __name__ == "__main__":
    main()
