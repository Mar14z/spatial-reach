"""A03 参考帧选择：只用 meta.json 的真值字段，不看任何模型输出。

名单一旦写入 configs/a03_reference_frames.json 就固定，不允许按比对结果
回头调整。五个类别各取 4 帧，共 20 帧，排序键全部可从 meta.json 复算：

  normal    全点 valid，且光照强度代理最接近全集中位数
  dim       光照强度代理 = light.ambient * light.diffuse 最小
  edge      任一点到画面边界的最小距离最小
  collinear 9 点 2D 的 PCA 主轴/末轴标准差比最小（越小越接近共线）
  hard      拍摄距离最小（目标尺度最大）。本 dev 集 valid 全为 True、
            且无出界点，所以不能用「遮挡 / 裁切」定义困难视角

同一序列只取一帧：m0_seq 每 4 帧共享同一姿态，只按指标排序会选出
4 张几乎相同的图，覆盖度是假的。

这些是几何与光照条件，不是模型表现；选帧过程不加载权重。

用法：
    conda activate arm_sim
    python scripts/a03_reference_frames.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data/synthetic/m0_seq/val"
OUT = ROOT / "configs/a03_reference_frames.json"
PER_CATEGORY = 4
CATEGORIES = ("normal", "dim", "edge", "collinear", "hard")


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def features_of(meta: dict) -> dict:
    """从单帧 meta.json 算四组与模型无关的几何/光照特征。"""
    points = np.asarray(meta["keypoints_2d"], dtype=float)
    valid = np.asarray(meta["valid"], dtype=bool)
    width, height = int(meta["width"]), int(meta["height"])
    used = points[valid] if valid.any() else points

    # 到画面四边的最小距离：越小越贴边，越容易踩到边界峰
    border = np.minimum.reduce(
        [used[:, 0], used[:, 1], width - 1 - used[:, 0], height - 1 - used[:, 1]]
    )
    # PCA：主轴/末轴标准差之比，越小越接近共线，PnP 越容易退化
    centered = used - used.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    light = meta["light"]
    return {
        "brightness_proxy": float(light["ambient"] * light["diffuse"]),
        "min_border_px": float(border.min()),
        "axis_ratio": float(singular[0] / max(singular[-1], 1e-9)),
        "hull_scale_px": float(np.hypot(*(used.max(axis=0) - used.min(axis=0)))),
        "distance_m": float(meta["camera"]["distance"]),
        "pitch_deg": float(meta["camera"]["pitch"]),
        "invalid_points": int((~valid).sum()),
    }


def rank(frames: list[dict], category: str) -> list[dict]:
    """按类别给出排序后的候选；同分用帧号兜底，保证结果可复现。"""
    if category == "normal":
        median = float(np.median([f["features"]["brightness_proxy"] for f in frames]))
        eligible = [f for f in frames if f["features"]["invalid_points"] == 0]
        key = lambda f: (abs(f["features"]["brightness_proxy"] - median), f["frame"])
        return sorted(eligible, key=key)
    if category == "dim":
        return sorted(
            frames,
            key=lambda f: (f["features"]["brightness_proxy"], f["frame"]),
        )
    if category == "edge":
        return sorted(
            frames, key=lambda f: (f["features"]["min_border_px"], f["frame"])
        )
    if category == "collinear":
        return sorted(frames, key=lambda f: (f["features"]["axis_ratio"], f["frame"]))
    if category == "hard":
        return sorted(
            frames,
            key=lambda f: (f["features"]["distance_m"], f["frame"]),
        )
    raise ValueError(f"未知类别 {category!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, default=VAL)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--per-category", type=int, default=PER_CATEGORY)
    args = parser.parse_args()

    if not args.val.is_dir():
        print(f"缺少开发集 {args.val}", file=sys.stderr)
        return 2

    frames = []
    for meta_path in sorted(args.val.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frames.append(
            {
                "frame_dir": meta_path.parent.name,
                "frame": int(meta["frame"]),
                "sequence": int(meta.get("sequence", -1)),
                "link_names": list(meta["link_names"]),
                "features": features_of(meta),
                "meta_sha256": sha256_of(meta_path),
            }
        )
    if not frames:
        print(f"{args.val} 下没有 meta.json", file=sys.stderr)
        return 2

    chosen, taken = [], set()
    for category in CATEGORIES:
        picked = 0
        for candidate in rank(frames, category):
            # 同序列只取一帧：连续帧共享姿态，选进来只是重复
            if candidate["sequence"] in taken:
                continue
            chosen.append(
                {
                    "frame_dir": candidate["frame_dir"],
                    "frame": candidate["frame"],
                    "sequence": candidate["sequence"],
                    "category": category,
                    "features": candidate["features"],
                    "link_names": candidate["link_names"],
                    "meta_sha256": candidate["meta_sha256"],
                    "rgb_sha256": sha256_of(args.val / candidate["frame_dir"] / "rgb.png"),
                }
            )
            taken.add(candidate["sequence"])
            picked += 1
            if picked == args.per_category:
                break
        if picked < args.per_category:
            print(
                f"警告：{category} 只选到 {picked} 帧（需要 {args.per_category}）",
                file=sys.stderr,
            )

    data = {
        "schema_version": 1,
        "kind": "a03_reference_frames",
        "note": "名单在跑比对之前固定；不允许按比对结果回头替换",
        "dataset": args.val.relative_to(ROOT).as_posix(),
        "selection": {
            "source": "仅 meta.json 的真值字段，不加载权重、不读模型输出",
            "categories": list(CATEGORIES),
            "per_category": args.per_category,
            "total": len(chosen),
            "brightness_proxy": "light.ambient * light.diffuse",
            "min_border_px": "任一点到画面四边的最小距离（越小越贴边）",
            "axis_ratio": "9 点 2D 的 PCA 主轴/末轴标准差之比（越小越共线）",
            "distance_m": "相机到目标的距离（越小目标尺度越大）",
            "hull_scale_px": "有效点包围盒对角线（越小越集中）",
            "sequence_dedup": "同一序列只取一帧",
        },
        "frames": chosen,
        "keypoint_order": {
            "value": chosen[0]["link_names"],
            "source": "dataset meta.json 的 link_names",
            "note": "权重本身不存关键点顺序；此处为数据集侧证据",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in chosen:
        f = item["features"]
        print(
            f"{item['category']:<10} {item['frame_dir']} "
            f"bright={f['brightness_proxy']:.3f} border={f['min_border_px']:.0f}px "
            f"axis={f['axis_ratio']:.2f} invalid={f['invalid_points']}"
        )
    print(f"\n{len(chosen)} 帧 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
