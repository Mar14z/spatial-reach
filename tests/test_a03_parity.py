"""A03 验收：训练验证路径与独立预测路径在参考帧上必须同数。

直接调用 scripts/a03_parity_report.py 的比对函数，避免测试里再写一份
比对逻辑。需要权重与 dev 集，缺任一项则跳过。导入 torch，需分进程。
"""

import json
from pathlib import Path

import pytest
import torch

from scripts.a03_parity_report import TOLERANCE_PX, compare_frames
from touch.data.dataset import M0Dataset
from touch.data.recipe import load_recipe
from touch.learning import inference

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "baseline.yaml"
PROFILE = "m0_ft320_matched"
FRAMES = ROOT / "configs/a03_reference_frames.json"


@pytest.fixture(scope="module")
def harness():
    if not FRAMES.is_file():
        pytest.skip("缺少 configs/a03_reference_frames.json")
    frame_data = json.loads(FRAMES.read_text(encoding="utf-8"))
    recipe = load_recipe(BASELINE, PROFILE)
    if not recipe.checkpoint.is_file():
        pytest.skip(f"缺少权重 {recipe.checkpoint}")
    dataset_root = ROOT / frame_data["dataset"]
    if not dataset_root.is_dir():
        pytest.skip(f"缺少 dev 集 {dataset_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inference.load_model(recipe, device)
    dataset = M0Dataset(
        dataset_root,
        stride=inference.stride_of(recipe),
        sigma=3.0,
        image_size=(recipe.input_height, recipe.input_width),
        preprocess=recipe.preprocess,
    )
    lookup = {path.name: index for index, path in enumerate(dataset.dirs)}
    return frame_data, recipe, model, device, dataset, lookup


def test_reference_frame_list_is_fixed_and_unique(harness):
    frame_data, *_ = harness
    frames = frame_data["frames"]
    assert len(frames) == 20
    assert len({item["frame_dir"] for item in frames}) == 20
    assert len({item["sequence"] for item in frames}) == 20
    assert {item["category"] for item in frames} == {
        "normal",
        "dim",
        "edge",
        "collinear",
        "hard",
    }


def test_keypoint_order_matches_recipe(harness):
    frame_data, recipe, *_ = harness
    inference.check_keypoints(recipe, frame_data["keypoint_order"]["value"])


def test_paths_agree_within_tolerance(harness):
    frame_data, recipe, model, device, dataset, lookup = harness
    rows = compare_frames(dataset, lookup, recipe, model, device, frame_data["frames"])
    worst = max(row["l3_coord_px"] for row in rows)
    assert worst <= TOLERANCE_PX, f"坐标最大差 {worst:.3e}px 超出 {TOLERANCE_PX}px"


def test_earlier_levels_are_bit_identical(harness):
    frame_data, recipe, model, device, dataset, lookup = harness
    rows = compare_frames(dataset, lookup, recipe, model, device, frame_data["frames"])
    for level in ("l0_image", "l1_input", "l2_heatmap"):
        worst = max(row[level] for row in rows)
        assert worst == 0.0, f"{level} 最大差 {worst:.3e}，应逐位一致"
