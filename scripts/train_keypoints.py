"""M0 keypoint 训练：train/val 泛化验证（GPU 友好版）。

用法：
    conda activate arm_sim
    python scripts/train_keypoints.py --steps 12000 --batch-size 32 --workers 8
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.data.dataset import M0Dataset
from touch.learning.keypoint_model import (
    INPUT_H,
    INPUT_W,
    KEYPOINT_NAMES,
    MobileNetHeatmapNet,
    ResNetHeatmapNet,
    TinyHeatmapNet,
    collate,
    heatmap_argmax,
    keypoint_loss,
    normalize,
    soft_argmax,
)

CKPT = ROOT / "outputs/checkpoints/m0_keypoints.pth"


def evaluate(model, loader, device, stride, drop_gripper, use_amp):
    """返回一套指标：argmax/soft-argmax 双解码 + 逐点误差 + 带符号偏差 + NME。"""
    model.eval()
    e_argmax, e_soft, bias, nme = [], [], [], []
    with torch.no_grad():
        for images, _, keypoints in loader:
            images = images.to(device, non_blocking=True)
            if images.shape[-2:] != (INPUT_H, INPUT_W):
                images = F.interpolate(
                    images,
                    size=(INPUT_H, INPUT_W),
                    mode="bilinear",
                    align_corners=False,
                )
            images = normalize(images)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                pred = model(images).float().cpu()
            gt = keypoints[:, :-1] if drop_gripper else keypoints
            soft = soft_argmax(pred) * stride
            amax = heatmap_argmax(pred) * stride
            e_argmax.append(torch.linalg.norm(amax - gt, dim=-1))
            e_soft.append(torch.linalg.norm(soft - gt, dim=-1))
            bias.append(soft - gt)
            ref = torch.linalg.norm(gt[:, -1] - gt[:, 0], dim=-1).clamp_min(1.0)
            nme.append(e_soft[-1].mean(dim=1) / ref)
    e, s, b = torch.cat(e_argmax), torch.cat(e_soft), torch.cat(bias)
    return {
        "mean_px": e.mean().item(),
        "median_px": e.median().item(),
        "p95_px": torch.quantile(e, 0.95).item(),
        "pck@3": (e <= 3.0).float().mean().item(),
        "pck@5": (e <= 5.0).float().mean().item(),
        "soft_mean_px": s.mean().item(),
        "soft_pck@3": (s <= 3.0).float().mean().item(),
        "nme": torch.cat(nme).mean().item(),
        "per_kp_px": s.mean(dim=0).tolist(),
        "per_kp_pck@3": (s <= 3.0).float().mean(dim=0).tolist(),
        "bias_v_px": b[:, :, 1].mean(dim=0).tolist(),
    }


SUMMARY_LABELS = {
    "mean_px": "平均像素误差(px,argmax)",
    "median_px": "中位像素误差(px,argmax)",
    "p95_px": "95分位像素误差(px,argmax)",
    "pck@3": "PCK@3px(argmax)",
    "pck@5": "PCK@5px(argmax)",
    "soft_mean_px": "平均像素误差(px,软解码)",
    "soft_pck@3": "PCK@3px(软解码)",
    "nme": "NME归一化误差",
}

GROUPS = {
    "train": "1-主指标(训练集)",
    "val": "1-主指标(验证集)",
    "final/train": "1-主指标(训练集)",
    "final/val": "1-主指标(验证集)",
}


def log_metrics(writer, split, m, step, per_keypoint=False):
    group = GROUPS[split]
    for key, label in SUMMARY_LABELS.items():
        writer.add_scalar(f"{group}/{label}", m[key], step)
    if per_keypoint:
        n = len(m["per_kp_px"])
        names = KEYPOINT_NAMES if n in (9, 11) else [f"点{i}" for i in range(n)]
        for i in range(n):
            writer.add_scalar(
                f"2-逐点/{names[i]}/像素误差(px,软解码)", m["per_kp_px"][i], step
            )
            writer.add_scalar(
                f"2-逐点/{names[i]}/PCK@3px", m["per_kp_pck@3"][i], step
            )
            writer.add_scalar(
                f"2-逐点/{names[i]}/纵向偏差v(px)", m["bias_v_px"][i], step
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-root", type=Path, default=ROOT / "data/synthetic/m0/train"
    )
    parser.add_argument(
        "--val-root", type=Path, default=ROOT / "data/synthetic/m0/val"
    )
    parser.add_argument(
        "--model", choices=["tiny", "resnet18", "mobilenet_v3"], default="resnet18"
    )
    parser.add_argument("--drop-gripper", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--preprocess",
        choices=["none", "gray", "gamma", "blur", "norm", "equalize"],
        default="none",
    )
    parser.add_argument(
        "--rand-gamma",
        type=float,
        default=0.0,
        help="训练时每样本随机 gamma 强度（0.3 表示 γ~U(0.7,1.3)）",
    )
    parser.add_argument("--train-extra", type=Path, default=None)
    parser.add_argument("--val-extra", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--logdir", type=Path, default=ROOT / "outputs/tb_zh")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="TensorBoard 里显示的名字（可用 a/b 分组），默认取权重文件名",
    )
    parser.add_argument("--ckpt", type=Path, default=CKPT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    torch.backends.cudnn.benchmark = False

    train_ds = M0Dataset(
        args.train_root,
        stride=args.stride,
        sigma=args.sigma,
        image_size=(INPUT_H, INPUT_W),
        cache=args.cache,
        preprocess=args.preprocess,
        extra_labels=args.train_extra,
    )
    val_ds = M0Dataset(
        args.val_root,
        stride=args.stride,
        sigma=args.sigma,
        image_size=(INPUT_H, INPUT_W),
        cache=args.cache,
        preprocess=args.preprocess,
        extra_labels=args.val_extra,
    )
    if args.cache and args.workers > 0:
        print("cache=True: forcing workers=0 to avoid duplicating cache")
        args.workers = 0
    loader_kwargs = dict(
        batch_size=args.batch_size,
        pin_memory=(device.type == "cuda"),
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=True, collate_fn=collate, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, drop_last=False, collate_fn=collate, **loader_kwargs
    )

    out_size = (480 // args.stride, 640 // args.stride)
    n_keypoints = train_ds.n_keypoints - (1 if args.drop_gripper else 0)
    if args.model == "resnet18":
        model = ResNetHeatmapNet(
            n_keypoints=n_keypoints, out_size=out_size
        )
    elif args.model == "mobilenet_v3":
        model = MobileNetHeatmapNet(
            n_keypoints=n_keypoints, out_size=out_size
        )
    else:
        model = TinyHeatmapNet(n_keypoints=n_keypoints, out_size=out_size)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"device={device} | batch={args.batch_size} | workers={args.workers} | "
        f"amp={use_amp} | keypoints={n_keypoints} | train={len(train_ds)} "
        f"| val={len(val_ds)}"
    )
    run_name = args.run_name or args.ckpt.stem
    writer = SummaryWriter(log_dir=str(args.logdir / run_name), flush_secs=10)
    writer.add_text("配置", str(vars(args)), 0)

    step = 0
    started = time.time()
    model.train()
    while step < args.steps:
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()
            if args.drop_gripper:
                targets = targets[:, :-1]
            if images.shape[-2:] != (INPUT_H, INPUT_W):
                images = F.interpolate(
                    images,
                    size=(INPUT_H, INPUT_W),
                    mode="bilinear",
                    align_corners=False,
                )
            if args.rand_gamma > 0:
                g = 1.0 + (torch.rand(images.shape[0], device=device) * 2 - 1) * args.rand_gamma
                images = images.clamp(0, 1) ** g.view(-1, 1, 1, 1)
            images = normalize(images)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                pred = model(images)
            loss = keypoint_loss(pred.float(), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % 10 == 0:
                writer.add_scalar("3-训练过程/损失", loss.item(), step)

            if step % 100 == 0 or step == 1:
                elapsed = time.time() - started
                print(
                    f"step {step:5d} | loss {loss.item():.6f} | "
                    f"{step / max(elapsed, 1e-9):.1f} step/s"
                )
                writer.add_scalar(
                    "3-训练过程/吞吐(步每秒)", step / max(elapsed, 1e-9), step
                )
            if step % args.eval_every == 0:
                tr = evaluate(
                    model, train_loader, device, args.stride,
                    args.drop_gripper, use_amp,
                )
                va = evaluate(
                    model, val_loader, device, args.stride,
                    args.drop_gripper, use_amp,
                )
                print(
                    f"  train mean {tr['mean_px']:.2f}px pck@3 {tr['pck@3']:.3f}\n"
                    f"  val   mean {va['mean_px']:.2f}px med {va['median_px']:.2f}px "
                    f"p95 {va['p95_px']:.2f}px pck@3 {va['pck@3']:.3f} "
                    f"soft {va['soft_mean_px']:.2f}px NME {va['nme']:.4f}"
                )
                print("  val kp px:    " + " ".join(f"{v:5.1f}" for v in va["per_kp_px"]))
                print("  val kp bias_v:" + " ".join(f"{v:+5.1f}" for v in va["bias_v_px"]))
                log_metrics(writer, "train", tr, step)
                log_metrics(writer, "val", va, step, per_keypoint=True)
                writer.add_scalar("3-训练过程/学习率", args.lr, step)
                model.train()
            if step >= args.steps:
                break

    tr = evaluate(
        model, train_loader, device, args.stride, args.drop_gripper, use_amp
    )
    va = evaluate(
        model, val_loader, device, args.stride, args.drop_gripper, use_amp
    )
    print(
        f"final train mean {tr['mean_px']:.2f}px pck@3 {tr['pck@3']:.3f}\n"
        f"final val   mean {va['mean_px']:.2f}px med {va['median_px']:.2f}px "
        f"p95 {va['p95_px']:.2f}px pck@3 {va['pck@3']:.3f} "
        f"soft {va['soft_mean_px']:.2f}px NME {va['nme']:.4f}"
    )
    print("final val kp px:    " + " ".join(f"{v:5.1f}" for v in va["per_kp_px"]))
    print("final val kp bias_v:" + " ".join(f"{v:+5.1f}" for v in va["bias_v_px"]))
    log_metrics(writer, "final/train", tr, args.steps)
    log_metrics(writer, "final/val", va, args.steps, per_keypoint=True)

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.ckpt)
    writer.flush()
    writer.close()
    print("saved", args.ckpt)


if __name__ == "__main__":
    main()
