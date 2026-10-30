import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from touch.data.recipe import (
    HEAD_KEY,
    check_state_shapes,
    list_profiles,
    load_recipe,
    sha256_of,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "baseline.yaml"
CKPT = ROOT / "outputs/checkpoints/m0_ft320.pth"


def _entry(ckpt, **over):
    base = {
        "mode": "legacy_replay",
        "checkpoint": ckpt.name,
        "sha256": sha256_of(ckpt),
        "model": "resnet18",
        "n_keypoints": 3,
        "input_size": {"width": 320, "height": 240},
        "out_size": {"width": 160, "height": 120},
        "preprocess": "gamma",
        "normalize": True,
        "decode": "soft",
        "keypoints": ["kp_base", "kp_elbow", "kp_wrist"],
    }
    base.update(over)
    return base


def _write(tmp_path, entry, group="legacy_replay", name="demo"):
    path = tmp_path / "baseline.yaml"
    path.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "profiles": {group: {name: entry}}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def ckpt(tmp_path):
    path = tmp_path / "w.pth"
    path.write_bytes(b"a01 recipe fixture")
    return path


def test_loads_recipe_and_unpacks_sizes(tmp_path, ckpt):
    recipe = load_recipe(_write(tmp_path, _entry(ckpt)), "demo", root=tmp_path)
    assert (recipe.input_width, recipe.input_height) == (320, 240)
    assert (recipe.out_width, recipe.out_height) == (160, 120)
    assert recipe.mode == "legacy_replay"
    assert recipe.keypoints == ("kp_base", "kp_elbow", "kp_wrist")
    again = load_recipe(_write(tmp_path, _entry(ckpt)), "demo", root=tmp_path)
    assert recipe == again


def test_missing_field_fails_before_use(tmp_path, ckpt):
    entry = _entry(ckpt)
    entry.pop("decode")
    with pytest.raises(ValueError, match="decode"):
        load_recipe(_write(tmp_path, entry), "demo", root=tmp_path)


def test_duplicate_keypoints_fail(tmp_path, ckpt):
    entry = _entry(ckpt, keypoints=["kp_base", "kp_elbow", "kp_elbow"])
    with pytest.raises(ValueError, match="重复"):
        load_recipe(_write(tmp_path, entry), "demo", root=tmp_path)


def test_missing_checkpoint_fails(tmp_path, ckpt):
    entry = _entry(ckpt, checkpoint="absent.pth")
    with pytest.raises(ValueError, match="权重不存在"):
        load_recipe(_write(tmp_path, entry), "demo", root=tmp_path)


def test_hash_mismatch_fails(tmp_path, ckpt):
    entry = _entry(ckpt, sha256="0" * 64)
    with pytest.raises(ValueError, match="哈希不符"):
        load_recipe(_write(tmp_path, entry), "demo", root=tmp_path)


def test_mode_must_match_group(tmp_path, ckpt):
    entry = _entry(ckpt, mode="matched_recipe")
    with pytest.raises(ValueError, match="不一致"):
        load_recipe(_write(tmp_path, entry), "demo", root=tmp_path)


def test_n_keypoints_must_match_names(tmp_path, ckpt):
    entry = _entry(ckpt, n_keypoints=4)
    with pytest.raises(ValueError, match="n_keypoints"):
        load_recipe(_write(tmp_path, entry), "demo", root=tmp_path)


def test_state_shapes_are_checked_by_injection(tmp_path, ckpt):
    recipe = load_recipe(_write(tmp_path, _entry(ckpt)), "demo", root=tmp_path)
    check_state_shapes(recipe, {HEAD_KEY: (3, 64, 1, 1)})
    with pytest.raises(ValueError, match="通道"):
        check_state_shapes(recipe, {HEAD_KEY: (9, 64, 1, 1)})
    with pytest.raises(ValueError, match=HEAD_KEY):
        check_state_shapes(recipe, {})


@pytest.mark.skipif(not CKPT.is_file(), reason="缺少归档权重")
def test_real_baseline_exposes_both_modes():
    names = list_profiles(BASELINE)
    assert ("legacy_replay", "m0_ft320_archive") in names
    assert ("matched_recipe", "m0_ft320_matched") in names
    recipe = load_recipe(BASELINE, "m0_ft320_matched")
    assert (recipe.input_width, recipe.input_height) == (320, 240)
    assert (recipe.out_width, recipe.out_height) == (160, 120)
    assert recipe.sha256 == sha256_of(CKPT)


def test_recipe_import_pulls_no_heavy_runtime():
    code = (
        "import sys, touch.data.recipe; "
        "print(int('torch' in sys.modules), int('cv2' in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert out == ["0", "0"]
