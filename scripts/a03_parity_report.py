"""A03 逐级一致性比对：训练验证路径 vs 独立预测路径。

比的是同一条算式有没有被写成两份，不是模型精度。四个级别分别比对：

  L0 规范图像    DataLoader 的 collate 输出 vs 直接从 dataset 取的数组
  L1 预处理输入  prepare_batch(collate) vs to_input(单帧)
  L2 热图        同一输入两次前向的输出
  L3 规范坐标    decode_tensor 之后的 640x480 像素坐标

基础比对固定为：同设备、fp32、batch=1。混合精度与批次变化另列对照，
不允许用它们解释尚未定位的基础路径差异。

用法：
    conda activate arm_sim
    python scripts/a03_parity_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.data.dataset import M0Dataset
from touch.data.recipe import load_recipe
from touch.learning import inference
from touch.learning.keypoint_model import collate

BASELINE = ROOT / "configs/baseline.yaml"
PROFILE = "m0_ft320_matched"
FRAMES = ROOT / "configs/a03_reference_frames.json"
TOLERANCE_PX = 0.01


def forward(images, model):
    with torch.no_grad():
        return model(images).float()


def level_max(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu() - b.detach().cpu()).abs().max())


def compare_frames(dataset, lookup, recipe, model, device, frames):
    """逐帧跑两条路径，返回每个级别的最大差与逐帧坐标差。"""
    rows = []
    for item in frames:
        index = lookup[item["frame_dir"]]
        sample = dataset[index]

        # 路径 A：训练验证走的 collate -> prepare_batch
        images, _, _ = collate([sample])
        canonical_a = images.clone()
        prepared_a = inference.prepare_batch(images.to(device), recipe)
        heat_a = forward(prepared_a, model)
        coord_a = inference.decode_tensor(heat_a, recipe, "soft")

        # 路径 B：独立预测走 to_input -> predict
        canonical_b = torch.from_numpy(
            np.ascontiguousarray(sample["image"].transpose(2, 0, 1))
        ).float()[None]
        prepared_b = inference.to_input(sample["image"], recipe, device=device)
        heat_b = forward(prepared_b, model)
        coord_b = inference.decode_tensor(heat_b, recipe, "soft")

        pred = inference.predict(sample["image"], recipe, model=model, device=device)
        gt = sample["keypoints"][: recipe.n_keypoints]
        rows.append(
            {
                "frame_dir": item["frame_dir"],
                "category": item["category"],
                "l0_image": level_max(canonical_a[0], canonical_b),
                "l1_input": level_max(prepared_a[0], prepared_b[0]),
                "l2_heatmap": level_max(heat_a[0], heat_b[0]),
                "l3_coord_px": level_max(coord_a[0], coord_b[0]),
                "predict_vs_decode_px": float(
                    np.abs(pred - coord_b[0].cpu().numpy()).max()
                ),
                "gt_mean_err_px": float(
                    np.linalg.norm(pred - gt, axis=1).mean()
                ),
            }
        )
    return rows


def batch_control(dataset, lookup, recipe, model, device, frames):
    """batch=1 与 batch=8 的逐帧坐标差；只作对照，不使用于基础结论。"""
    indices = [lookup[item["frame_dir"]] for item in frames]
    singles = []
    for index in indices:
        sample = dataset[index]
        tensor = inference.to_input(sample["image"], recipe, device=device)
        singles.append(inference.decode_tensor(forward(tensor, model), recipe, "soft")[0])
    singles = torch.stack(singles)

    batched = []
    for start in range(0, len(indices), 8):
        chunk = [dataset[i]["image"] for i in indices[start : start + 8]]
        prepared = inference.prepare_batch(
            torch.from_numpy(
                np.ascontiguousarray(np.stack(chunk).transpose(0, 3, 1, 2))
            ).float().to(device),
            recipe,
        )
        batched.append(inference.decode_tensor(forward(prepared, model), recipe, "soft"))
    batched = torch.cat(batched)
    return float((singles - batched).abs().max())


def amp_control(dataset, lookup, recipe, model, device, frames):
    """fp32 与 fp16 的逐帧坐标差；只在 cuda 上有意义。"""
    if torch.device(device).type != "cuda":
        return None
    index = lookup[frames[0]["frame_dir"]]
    tensor = inference.to_input(dataset[index]["image"], recipe, device=device)
    reference = inference.decode_tensor(forward(tensor, model), recipe, "soft")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        mixed = model(tensor).float()
    return float(
        (inference.decode_tensor(mixed, recipe, "soft") - reference).abs().max()
    )


def dev_mean_error(dataset, recipe, model, device, batch_size: int = 8):
    """全量 dev 集的平均像素误差，用来证明 20 帧抽样没有偏乐观。"""
    totals = []
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        images = torch.from_numpy(
            np.ascontiguousarray(
                np.stack([s["image"] for s in samples]).transpose(0, 3, 1, 2)
            )
        ).float().to(device)
        prepared = inference.prepare_batch(images, recipe)
        coords = inference.decode_tensor(forward(prepared, model), recipe, "soft")
        truth = torch.from_numpy(
            np.stack([s["keypoints"][: recipe.n_keypoints] for s in samples])
        )
        totals.append(torch.linalg.norm(coords.cpu() - truth, dim=-1))
    return len(dataset), float(torch.cat(totals).mean())


def write_report(path: Path, data: dict) -> None:
    rows = data["frames"]
    worst_l3 = max(rows, key=lambda r: r["l3_coord_px"])
    worst_err = max(rows, key=lambda r: r["gt_mean_err_px"])
    errors = [row["gt_mean_err_px"] for row in rows]
    lines = [
        "# A03 逐级一致性报告",
        "",
        f"- 权重：`{data['checkpoint']}`（sha256 前 12 位 `{data['sha256'][:12]}`）",
        f"- 配方：`configs/baseline.yaml` / `{data['profile']}`",
        f"- 参考帧：`configs/a03_reference_frames.json`，{len(rows)} 帧，"
        "5 类 × 4 帧，来自 20 个不同序列",
        f"- 设备：{data['device']}，fp32，batch=1",
        f"- 验收线：同配置坐标最大差 ≤ {TOLERANCE_PX}px",
        f"- 这 20 帧与真值的平均误差 {sum(errors) / len(errors):.2f}px"
        f"（{min(errors):.2f}–{max(errors):.2f}px），与全量 dev 集 {data['dev_reference_px']:.2f}px "
        "同量级，可用作代表性抽样",
        "",
        "## 逐级差异（路径 A 训练验证 vs 路径 B 独立预测）",
        "",
        "| 级别 | 内容 | 全帧最大差 |",
        "| --- | --- | --- |",
        f"| L0 | 规范图像（640x480，dataset 输出） | {data['levels']['l0_image']:.3e} |",
        f"| L1 | 预处理输入（320x240 归一化张量） | {data['levels']['l1_input']:.3e} |",
        f"| L2 | 热图（9x120x160） | {data['levels']['l2_heatmap']:.3e} |",
        f"| L3 | 规范坐标（640x480 像素） | {data['levels']['l3_coord_px']:.3e} |",
        "",
        f"结论：坐标级最大差 {data['levels']['l3_coord_px']:.3e}px，"
        f"验收线 {TOLERANCE_PX}px，"
        f"{'通过' if data['levels']['l3_coord_px'] <= TOLERANCE_PX else '未通过'}。",
        "",
        "## 逐帧明细",
        "",
        "| 类别 | 帧 | L3 坐标最大差(px) | predict 与 decode 差(px) | 与真值平均误差(px) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['category']} | {row['frame_dir']} | {row['l3_coord_px']:.2e} | "
            f"{row['predict_vs_decode_px']:.2e} | {row['gt_mean_err_px']:.2f} |"
        )
    lines += [
        "",
        f"路径差异最大的帧：`{worst_l3['frame_dir']}`（{worst_l3['category']}），"
        f"L3 差 {worst_l3['l3_coord_px']:.3e}px。",
        f"与真值误差最大的帧：`{worst_err['frame_dir']}`（{worst_err['category']}），"
        f"{worst_err['gt_mean_err_px']:.2f}px；这是模型精度，不是路径差异。",
        "",
        "## 对照（只作参考，不用于解释基础路径差异）",
        "",
    ]
    if data["controls"]["batch_1_vs_8_px"] is None:
        lines.append("- 批次变化：未执行")
    else:
        lines.append(
            f"- batch=1 vs batch=8 坐标最大差："
            f"{data['controls']['batch_1_vs_8_px']:.3e}px"
        )
    if data["controls"]["fp32_vs_fp16_px"] is None:
        lines.append("- 混合精度：当前设备非 cuda，未执行")
    else:
        mixed = data["controls"]["fp32_vs_fp16_px"]
        lines.append(
            f"- fp32 vs fp16 坐标最大差：{mixed:.3e}px（"
            + ("在验收线内" if mixed <= TOLERANCE_PX else "超出验收线")
            + "，属精度位宽效应，不作为基础路径结论）"
        )
    lines.append(
        "- 以上两项都非零：批次与精度会引入真实数值差异，"
        "所以基础一致性必须先在同一精度、同一批次下量完再谈它们。"
    )
    lines += [
        "",
        "## 已知缺口",
        "",
        "- 权重不含关键点名称与顺序，本报告的通道顺序以 dataset `link_names` 为准；"
        "换数据集必须重新核对，否则误差会以「模型很准但坐标错位」的形式出现。",
        "- 本报告只证明编码路径一致，不证明模型精度；精度结论见 A04/A06。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=FRAMES)
    parser.add_argument("--profile", type=str, default=PROFILE)
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument(
        "--skip-dev",
        action="store_true",
        help="跳过全量 dev 集平均误差（报告里的代表性对照会缺失）",
    )
    args = parser.parse_args()

    if not args.frames.is_file():
        print(
            f"缺少参考帧清单 {args.frames}，先跑 scripts/a03_reference_frames.py",
            file=sys.stderr,
        )
        return 2
    frame_data = json.loads(args.frames.read_text(encoding="utf-8"))
    frames = frame_data["frames"]
    if not frames:
        print("参考帧清单为空", file=sys.stderr)
        return 2

    recipe = load_recipe(BASELINE, args.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inference.load_model(recipe, device)
    dataset = M0Dataset(
        ROOT / frame_data["dataset"],
        stride=inference.stride_of(recipe),
        sigma=3.0,
        image_size=(recipe.input_height, recipe.input_width),
        preprocess=recipe.preprocess,
    )
    if dataset.n_keypoints != recipe.n_keypoints:
        print(
            f"数据集 {dataset.n_keypoints} 个关键点，配方 {recipe.n_keypoints} 个，拒绝比对",
            file=sys.stderr,
        )
        return 2

    # 帧目录名 -> 数据集下标；不假设目录名与下标数值相等
    lookup = {path.name: index for index, path in enumerate(dataset.dirs)}
    missing = [item["frame_dir"] for item in frames if item["frame_dir"] not in lookup]
    if missing:
        print(f"参考帧不在数据集中：{missing}", file=sys.stderr)
        return 2

    # 权重不含关键点顺序，只能拿数据集真值核对；对不上就不要出报告
    orders = {tuple(item["link_names"]) for item in frames}
    if len(orders) != 1:
        print(f"参考帧的关键点顺序不一致：{orders}", file=sys.stderr)
        return 2
    try:
        inference.check_keypoints(recipe, frames[0]["link_names"])
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    rows = compare_frames(dataset, lookup, recipe, model, device, frames)
    if args.skip_dev:
        dev_frames, dev_px = 0, float("nan")
    else:
        dev_frames, dev_px = dev_mean_error(dataset, recipe, model, device)
    levels = {
        key: max(row[key] for row in rows)
        for key in ("l0_image", "l1_input", "l2_heatmap", "l3_coord_px")
    }
    data = {
        "schema_version": 1,
        "kind": "a03_parity",
        "run_id": "a03-parity-" + uuid4().hex,
        "checkpoint": recipe.checkpoint.relative_to(ROOT).as_posix(),
        "sha256": recipe.sha256,
        "profile": recipe.name,
        "device": str(device),
        "tolerance_px": TOLERANCE_PX,
        "passed": levels["l3_coord_px"] <= TOLERANCE_PX,
        "levels": levels,
        "dev_frames": dev_frames,
        "dev_reference_px": dev_px,
        "frames": rows,
        "controls": {
            "batch_1_vs_8_px": (
                None
                if args.skip_controls
                else batch_control(dataset, lookup, recipe, model, device, frames)
            ),
            "fp32_vs_fp16_px": (
                None
                if args.skip_controls
                else amp_control(dataset, lookup, recipe, model, device, frames)
            ),
        },
        "keypoint_order_source": frame_data["keypoint_order"],
    }
    run = args.out_root / data["run_id"]
    run.mkdir(parents=True, exist_ok=False)
    (run / "report.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(run / "report.md", data)

    print(json.dumps({"run": str(run), **data["levels"], "passed": data["passed"]},
                     ensure_ascii=False, indent=2))
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
