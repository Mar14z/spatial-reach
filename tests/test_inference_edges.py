"""A03 第 4 条：边界与错误输入测试。

覆盖计划点名的六类非常规输入：不同输入尺寸、非方形热图、边界峰、
错误关键点顺序、空输入、非有限数。本文件导入 torch，需与 cv2 测试分进程。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from touch.data.recipe import Recipe
from touch.learning import inference

ROOT = Path(__file__).resolve().parents[1]


def _recipe(**over):
    base = dict(
        name="edge",
        mode="legacy_replay",
        checkpoint=ROOT / "unused.pth",
        sha256="0" * 64,
        model="tiny",
        n_keypoints=2,
        input_width=320,
        input_height=240,
        out_width=160,
        out_height=120,
        preprocess="gamma",
        normalize=True,
        decode="soft",
        keypoints=("a", "b"),
    )
    base.update(over)
    return Recipe(**base)


def test_input_size_different_from_heatmap_size():
    """输入 320x240 与热图 160x120 是两个量，解码只按热图尺寸算 stride。"""
    recipe = _recipe()
    assert (recipe.input_width, recipe.input_height) == (320, 240)
    assert inference.stride_of(recipe) == 4
    image = np.full((240, 320, 3), 0.5, dtype=np.float32)
    tensor = inference.to_input(image, recipe)
    assert tuple(tensor.shape) == (1, 3, 240, 320)


def test_smaller_input_resizes_but_keeps_canonical_decode():
    """输入尺寸变小不改规范坐标：热图仍按 640/160 换算。"""
    recipe = _recipe(input_width=160, input_height=120)
    image = np.full((120, 160, 3), 0.5, dtype=np.float32)
    tensor = inference.to_input(image, recipe)
    assert tuple(tensor.shape) == (1, 3, 120, 160)
    heatmaps = torch.zeros(1, 2, 120, 160)
    heatmaps[0, 0, 0, 0] = 1.0
    assert np.allclose(inference.decode(heatmaps, recipe, mode="argmax")[0, 0], [0, 0])


def test_non_square_heatmap_is_supported():
    """120x160 本身就是非方形；只要宽高比与 640x480 一致就能换算。"""
    recipe = _recipe()
    heatmaps = torch.zeros(1, 2, 120, 160)
    heatmaps[0, 1, 119, 159] = 1.0
    point = inference.decode(heatmaps, recipe, mode="argmax")[0, 1]
    assert np.allclose(point, [159 * 4, 119 * 4])


def test_edge_peak_stays_inside_canonical_frame():
    """峰值落在四角时不得越界，也不得被推成负数。"""
    recipe = _recipe()
    heatmaps = torch.zeros(1, 2, 120, 160)
    heatmaps[0, 0, 0, 0] = 1.0
    heatmaps[0, 1, 0, 159] = 1.0
    points = inference.decode(heatmaps, recipe, mode="argmax")
    assert points.min() >= 0
    assert points.max() <= max(inference.CANONICAL_W, inference.CANONICAL_H)


def test_empty_image_is_rejected():
    recipe = _recipe()
    with pytest.raises(ValueError):
        inference.to_input(np.zeros((0, 0, 3), dtype=np.float32), recipe)


def test_empty_heatmap_is_rejected():
    with pytest.raises(ValueError, match="热图实际"):
        inference.decode(torch.zeros(1, 2, 0, 0), _recipe())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_heatmap_is_rejected(bad):
    """NaN/inf 若不拦住，会被 soft-argmax 静默传播成 NaN 坐标。"""
    recipe = _recipe()
    heatmaps = torch.zeros(1, 2, 120, 160)
    heatmaps[0, 0, 10, 10] = bad
    with pytest.raises(ValueError, match="非有限值"):
        inference.decode(heatmaps, recipe)


def test_keypoint_order_match_passes():
    recipe = _recipe(keypoints=("base", "grip"))
    inference.check_keypoints(recipe, ["base", "grip"])


def test_keypoint_order_swap_is_rejected():
    """顺序互换在参数层面完全合法，只能靠数据集真值发现。"""
    recipe = _recipe(keypoints=("base", "grip"))
    with pytest.raises(ValueError, match="关键点顺序不一致"):
        inference.check_keypoints(recipe, ["grip", "base"])


def test_keypoint_count_mismatch_is_rejected():
    recipe = _recipe(keypoints=("base", "grip"))
    with pytest.raises(ValueError, match="关键点数不符"):
        inference.check_keypoints(recipe, ["base"])
