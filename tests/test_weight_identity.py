"""A01 权重身份清单的防漂移测试。

configs/weight_identity.json 是入库证据：只要权重文件被覆盖、结构被改，
这里就会失败。本文件导入 torch，必须与使用 cv2 的测试分进程执行。
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from touch.learning import inference

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "configs" / "weight_identity.json"
BASELINE = ROOT / "configs" / "baseline.yaml"


@pytest.fixture(scope="module")
def identity():
    if not IDENTITY.is_file():
        pytest.skip("尚未生成 configs/weight_identity.json")
    return json.loads(IDENTITY.read_text(encoding="utf-8"))


def test_every_hashed_weight_still_matches(identity):
    """清单里的哈希必须与磁盘一致；不等说明权重被替换，清单已失效。"""
    stale = []
    for relative, record in identity["weights"].items():
        path = ROOT / relative
        if not path.is_file():
            stale.append(f"{relative}: 文件不存在")
            continue
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        if digest != record["sha256"]:
            stale.append(f"{relative}: 哈希不符")
    assert not stale, "\n".join(stale)


def test_structure_claim_loads_strictly(identity):
    """清单声明结构与关键点数的权重，必须能被该结构 strict 加载。"""
    checked = 0
    for relative, record in identity["weights"].items():
        structure = record["structure"]
        if structure["family"] is None:
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        network = inference.build_network(
            structure["family"],
            structure["n_keypoints"],
            inference.CANONICAL_W // 4,
            inference.CANONICAL_H // 4,
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        network.load_state_dict(state, strict=True)
        checked += 1
    assert checked > 0


def test_registered_baseline_checkpoint_is_listed(identity):
    """baseline.yaml 引用的权重必须在本清单里，否则两处证据脱钩。"""
    from touch.data.recipe import list_profiles, load_recipe

    listed = set(identity["weights"])
    for _, name in list_profiles(BASELINE):
        recipe = load_recipe(BASELINE, name)
        key = recipe.checkpoint.relative_to(ROOT).as_posix()
        assert key in listed, f"{name} 引用的 {key} 不在权重清单里"
