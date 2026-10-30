"""A01 权重身份盘点：只读权重与训练记录，不跑模型、不出预测。

对 outputs/checkpoints 下每份 .pth 分行核实，能核实的写值并记来源，
查不到的写 unknown，绝不用文件名猜。

判定手段只有两种：
  1) 结构族与关键点数 —— 用 build_network 建三种结构各自尝试 strict 加载，
     能 strict 通过的那个就是真结构；文件名不参与判定。
  2) 输入尺寸 / 预处理 / 精度模式 / stride —— 读训练时写进 TensorBoard 的
     配置文本（train_keypoints.py 的 writer.add_text("配置", vars(args))）。

所有 .pth 都只存参数，因此热图输出尺寸、关键点名称与顺序、训练数据来源
无法从权重反推；有登记配方的引用 baseline.yaml，没有的写 unknown。

用法：
    conda activate arm_sim
    python scripts/check_weight_identity.py --out configs/weight_identity.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touch.learning.inference import CANONICAL_H, CANONICAL_W, MODELS, build_network

# add_text 写成 "配置" 时 tag 是 "配置/text_summary"；早期英文版是 "config/text_summary"
CONFIG_TAGS = ("配置/text_summary", "config/text_summary")
CKPTS = ROOT / "outputs/checkpoints"
TB_DIRS = [ROOT / "outputs/tb_zh", ROOT / "outputs/tb"]
BASELINE = ROOT / "configs/baseline.yaml"

# vars(args) 的 repr 带 WindowsPath('...')，literal_eval 不认；剥成普通字符串
_PATH_REPR = re.compile(r"\w*Path\((['\"])(.*?)\1\)")


def parse_args_repr(text: str) -> dict:
    """解析 writer.add_text 里写入的 vars(args) 文本。"""
    return ast.literal_eval(_PATH_REPR.sub(r"\1\2\1", text))


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_tb_config(stem: str):
    """返回 (参数 dict, 事件文件相对路径)；找不到配置就返回 (None, None)。"""
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    for base in TB_DIRS:
        run_dir = base / stem
        if not run_dir.is_dir():
            continue
        for event in sorted(run_dir.glob("events.out.tfevents.*")):
            try:
                loader = EventFileLoader(str(event))
                for record in loader.Load():
                    for value in record.summary.value:
                        if value.tag not in CONFIG_TAGS:
                            continue
                        text = value.tensor.string_val[0].decode("utf-8")
                        rel = event.relative_to(ROOT).as_posix()
                        return parse_args_repr(text), rel
            except Exception as error:  # 事件文件损坏时如实记录，不中断盘点
                return None, f"unreadable:{error}"
    return None, None


def candidate_keypoint_counts(state) -> list[int]:
    """1x1 卷积的输出通道就是候选关键点数；只用来缩小搜索范围，不作结论。"""
    counts = set()
    for tensor in state.values():
        if tensor.ndim == 4 and tuple(tensor.shape[-2:]) == (1, 1):
            counts.add(int(tensor.shape[0]))
    return sorted(counts)


def resolve_structure(state):
    """建三种结构试 strict 加载，唯一通过的即真结构；都不通过则如实报失败。"""
    matches = []
    for family in sorted(MODELS):
        for n_keypoints in candidate_keypoint_counts(state):
            network = build_network(
                family, n_keypoints, CANONICAL_W // 4, CANONICAL_H // 4
            )
            try:
                network.load_state_dict(state, strict=True)
            except RuntimeError:
                continue
            matches.append(
                {
                    "family": family,
                    "n_keypoints": n_keypoints,
                    "params": sum(p.numel() for p in network.parameters()),
                }
            )
    if len(matches) == 1:
        return {**matches[0], "verified_by": "strict_load_state_dict"}
    return {
        "family": None,
        "n_keypoints": None,
        "verified_by": "no_unique_strict_match",
        "candidates": matches,
    }


def registered_profiles() -> dict[str, list[str]]:
    """登记的 checkpoint 文件名 -> 配方名列表；读取时会顺带校验权重哈希。"""
    from touch.data.recipe import list_profiles, load_recipe

    table: dict[str, list[str]] = {}
    if not BASELINE.is_file():
        return table
    for group, name in list_profiles(BASELINE):
        try:
            recipe = load_recipe(BASELINE, name)
        except (ValueError, OSError) as error:
            print(f"登记配方不可用 {group}/{name}: {error}", file=sys.stderr)
            continue
        table.setdefault(recipe.checkpoint.stem, []).append(f"{group}/{name}")
    return table


def describe(path: Path, profiles: dict) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    record = {
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "structure": resolve_structure(state),
    }
    args, source = read_tb_config(path.stem)
    if args is None:
        record["training_record"] = {"source": source or "missing"}
        record["input_size"] = {"value": None, "source": "unknown"}
        record["preprocess"] = {"value": None, "source": "unknown"}
        record["precision"] = {"value": None, "source": "unknown"}
        record["out_size"] = {"value": None, "source": "unknown"}
    else:
        raw_input = args.get("input_size")
        width, height = (
            (int(v) for v in str(raw_input).lower().split("x"))
            if raw_input
            else (None, None)
        )
        stride = int(args["stride"]) if args.get("stride") else None
        missing = "（该次训练的 args 里没有此字段）"
        record["training_record"] = {
            "source": source,
            "model": args.get("model"),
            "preprocess": args.get("preprocess"),
            "input_size": raw_input,
            "stride": stride,
            "drop_gripper": bool(args.get("drop_gripper")),
            "steps": args.get("steps"),
            "batch_size": args.get("batch_size"),
            "lr": args.get("lr"),
            "rand_gamma": args.get("rand_gamma"),
            "consistency": args.get("consistency"),
            "no_amp": args.get("no_amp"),
            "train_root": str(args.get("train_root")),
        }
        record["input_size"] = {
            "value": {"width": width, "height": height} if width else None,
            "source": source if width else f"unknown{missing}",
        }
        record["preprocess"] = {
            "value": args.get("preprocess"),
            "source": source if args.get("preprocess") else f"unknown{missing}",
        }
        record["precision"] = {
            "value": (
                {"amp": not bool(args["no_amp"])}
                if args.get("no_amp") is not None
                else None
            ),
            "source": (
                f"{source}#no_amp" if args.get("no_amp") is not None
                else f"unknown{missing}"
            ),
        }
        record["out_size"] = {
            "value": (
                {"width": CANONICAL_W // stride, "height": CANONICAL_H // stride}
                if stride
                else None
            ),
            "source": (
                f"{source}#stride（推导，权重本身不存输出尺寸）"
                if stride
                else f"unknown{missing}"
            ),
        }
        # 训练记录里的模型名只是当时传入的参数，与 strict 加载结果交叉核对
        recorded = args.get("model")
        record["training_record"]["model_agrees_with_structure"] = (
            None
            if record["structure"]["family"] is None or recorded is None
            else recorded == record["structure"]["family"]
        )
    # 权重只存参数：以下三项无论有无训练记录都不可从权重反推
    record["keypoints"] = {
        "value": None,
        "source": "unknown：权重不存关键点名称与顺序；有登记配方时见 registered_profiles",
    }
    record["origin"] = {
        "value": "unproven",
        "source": "训练记录未保存产物哈希，无法证明本文件就是那次训练的输出",
    }
    record["registered_profiles"] = profiles.get(path.stem, [])
    return record


def unknown_fields(record: dict) -> list[str]:
    fields = ["keypoints"]
    for key in ("input_size", "preprocess", "precision", "out_size"):
        if record[key]["value"] is None:
            fields.append(key)
    if record["structure"]["family"] is None:
        fields.append("structure")
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "configs/weight_identity.json")
    parser.add_argument("--ckpts", type=Path, default=CKPTS)
    args = parser.parse_args()

    if not BASELINE.is_file():
        print(f"缺少 {BASELINE}", file=sys.stderr)
        return 2
    profiles = registered_profiles()
    files = sorted(args.ckpts.glob("*.pth"))
    if not files:
        print(f"{args.ckpts} 下没有 .pth", file=sys.stderr)
        return 2

    weights, unresolved = {}, []
    for path in files:
        record = describe(path, profiles)
        record["unknown_fields"] = unknown_fields(record)
        weights[path.relative_to(ROOT).as_posix()] = record
        if record["unknown_fields"]:
            unresolved.append(
                {"checkpoint": path.name, "unknown": record["unknown_fields"]}
            )
        print(
            f"{path.name:<34} {str(record['structure']['family']):<12} "
            f"kp={record['structure']['n_keypoints']} "
            f"unknown={','.join(record['unknown_fields']) or '-'}"
        )

    # 不记录 git HEAD：本文件一提交 HEAD 就变，写进去反而立刻过期。
    # 可核验证据是逐文件 sha256；git 溯源看 outputs/runs/a01-inventory-*。
    data = {
        "schema_version": 1,
        "kind": "weight_identity",
        "conventions": {
            "structure": "三种结构逐一 strict 加载，唯一通过者为真结构",
            "input_size": "仅取训练时写入 TensorBoard 的 args，不从文件名推断",
            "out_size": "权重不存输出尺寸，只能由训练记录里的 stride 推导",
            "keypoints": "权重不存关键点名称与顺序，一律 unknown 或引用登记配方",
            "origin": "训练记录未保存产物哈希，一律 unproven",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "count": len(weights),
        "weights": weights,
        "unresolved": unresolved,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{len(weights)} 份权重 -> {args.out}")
    print(f"存在 unknown 字段的权重：{len(unresolved)} 份")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
