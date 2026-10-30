"""A01 基线配方加载与严格校验。

设计约束：
- 不导入 torch / cv2，几何与任务进程可直接使用。
- 缺字段、点名重复、权重缺失或哈希不符时在启动前失败，不回退默认值。
- out_size 无法从权重推断，只能由配方声明；结构校验只覆盖可验证的部分。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
MODES = ("legacy_replay", "matched_recipe")
DECODES = ("soft", "argmax", "subpixel")
PREPROCESS = ("none", "gray", "gamma", "blur", "norm", "equalize")
REQUIRED = (
    "mode", "checkpoint", "sha256", "model", "n_keypoints",
    "input_size", "out_size", "preprocess", "normalize", "decode",
    "keypoints",
)
HEAD_KEY = "head.weight"


@dataclass(frozen=True)
class Recipe:
    """一条可执行配方；字段单位与坐标约定见 configs/baseline.yaml。"""

    name: str
    mode: str
    checkpoint: Path
    sha256: str
    model: str
    n_keypoints: int
    input_width: int
    input_height: int
    out_width: int
    out_height: int
    preprocess: str
    normalize: bool
    decode: str
    keypoints: tuple[str, ...]


def sha256_of(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _size(entry: dict, field: str, name: str) -> tuple[int, int]:
    value = entry[field]
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise ValueError(f"{name}: {field} 必须是 {{width, height}} 字典")
    out = []
    for key in ("width", "height"):
        v = value[key]
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name}: {field}.{key} 必须是正整数")
        out.append(v)
    return out[0], out[1]


def _entry(profiles: dict, name: str) -> tuple[str, dict]:
    hits = [
        (mode, entry)
        for mode, group in profiles.items()
        if isinstance(group, dict)
        for key, entry in group.items()
        if key == name
    ]
    if not hits:
        raise ValueError(f"配方中没有条目：{name}")
    if len(hits) > 1:
        raise ValueError(f"条目名重复：{name}")
    return hits[0]


def load_recipe(path, name: str, root=ROOT) -> Recipe:
    """按条目名加载配方；任何不确定都抛 ValueError，不返回半成品。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("配方顶层必须是字典")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version 必须为 {SCHEMA_VERSION}")
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("配方缺少 profiles")
    for mode, group in profiles.items():
        if mode not in MODES:
            raise ValueError(f"未知 profile 分组：{mode}")
        if not isinstance(group, dict) or not group:
            raise ValueError(f"{mode}: 分组必须是非空字典")
    group, entry = _entry(profiles, name)
    missing = [f for f in REQUIRED if f not in entry]
    if missing:
        raise ValueError(f"{name}: 缺少字段 {', '.join(missing)}")
    if entry["mode"] != group:
        raise ValueError(f"{name}: mode {entry['mode']!r} 与分组 {group!r} 不一致")
    keypoints = entry["keypoints"]
    if not isinstance(keypoints, list) or not all(
        isinstance(k, str) and k for k in keypoints
    ):
        raise ValueError(f"{name}: keypoints 必须是非空字符串列表")
    dupes = sorted({k for k in keypoints if keypoints.count(k) > 1})
    if dupes:
        raise ValueError(f"{name}: keypoints 重复 {dupes}")
    count = entry["n_keypoints"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"{name}: n_keypoints 必须是整数")
    if count != len(keypoints):
        raise ValueError(
            f"{name}: n_keypoints={count} 与 keypoints 长度 {len(keypoints)} 不符"
        )
    if entry["decode"] not in DECODES:
        raise ValueError(f"{name}: decode 非法 {entry['decode']!r}")
    if entry["preprocess"] not in PREPROCESS:
        raise ValueError(f"{name}: preprocess 非法 {entry['preprocess']!r}")
    if not isinstance(entry["normalize"], bool):
        raise ValueError(f"{name}: normalize 必须是布尔值")
    checkpoint = Path(entry["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path(root) / checkpoint
    if not checkpoint.is_file():
        raise ValueError(f"{name}: 权重不存在 {checkpoint}")
    if sha256_of(checkpoint) != entry["sha256"]:
        raise ValueError(f"{name}: 权重哈希不符 {checkpoint}")
    iw, ih = _size(entry, "input_size", name)
    ow, oh = _size(entry, "out_size", name)
    return Recipe(
        name=name, mode=group, checkpoint=checkpoint, sha256=entry["sha256"],
        model=entry["model"], n_keypoints=count,
        input_width=iw, input_height=ih, out_width=ow, out_height=oh,
        preprocess=entry["preprocess"], normalize=entry["normalize"],
        decode=entry["decode"], keypoints=tuple(keypoints),
    )


def check_state_shapes(recipe: Recipe, shapes: dict) -> None:
    """注入权重形状做结构校验；本函数不加载模型，out_size 不在权重里。"""
    if HEAD_KEY not in shapes:
        raise ValueError(f"{recipe.name}: 权重缺少 {HEAD_KEY}")
    actual = tuple(int(v) for v in shapes[HEAD_KEY])
    if actual[0] != recipe.n_keypoints:
        raise ValueError(
            f"{recipe.name}: {HEAD_KEY} 通道 {actual[0]} 与 "
            f"n_keypoints {recipe.n_keypoints} 不符"
        )


def list_profiles(path) -> list[tuple[str, str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles") or {}
    return [
        (mode, name)
        for mode, group in profiles.items()
        if isinstance(group, dict)
        for name in group
    ]
