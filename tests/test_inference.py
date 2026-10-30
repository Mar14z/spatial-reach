"""A02 推理契约测试。

本文件会导入 torch / torchvision，必须与使用 cv2 的测试分进程执行，
否则可能触发 OMP Error #15（libiomp5md.dll already initialized）。
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from touch.data.recipe import Recipe
from touch.learning import inference
from touch.learning.keypoint_model import normalize, soft_argmax

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "baseline.yaml"
WEIGHTS = ROOT / "outputs/checkpoints/m0_ft320.pth"
VAL = ROOT / "data/synthetic/m0_seq/val"
PROFILE = "m0_ft320_matched"


def _recipe(**over):
    base = dict(
        name="demo",
        mode="legacy_replay",
        checkpoint=ROOT / "unused.pth",
        sha256="0" * 64,
        model="tiny",
        n_keypoints=9,
        input_width=320,
        input_height=240,
        out_width=160,
        out_height=120,
        preprocess="gamma",
        normalize=True,
        decode="soft",
        keypoints=tuple(f"kp{i}" for i in range(9)),
    )
    base.update(over)
    return Recipe(**base)


def test_stride_is_canonical_over_heatmap():
    assert inference.stride_of(_recipe()) == 4


def test_heatmap_must_divide_canonical_size():
    with pytest.raises(ValueError, match="不能整除"):
        inference.stride_of(_recipe(out_width=100))


def test_stride_must_agree_between_axes():
    with pytest.raises(ValueError, match="不同 stride"):
        inference.stride_of(_recipe(out_width=160, out_height=80))


def test_build_model_uses_recipe_out_size():
    model = inference.build_model(_recipe())
    assert tuple(model.out_size) == (120, 160)
    assert model.head.weight.shape[0] == 9


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="未知模型"):
        inference.build_model(_recipe(model="vgg16"))


def test_to_input_rejects_unscaled_uint8():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="0~1 浮点"):
        inference.to_input(image, _recipe())


def test_to_input_rejects_wrong_size_when_resize_disabled():
    image = np.zeros((480, 640, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="不符"):
        inference.to_input(image, _recipe(), resize=False)


def test_to_input_resizes_and_normalizes():
    image = np.full((480, 640, 3), 0.5, dtype=np.float32)
    tensor = inference.to_input(image, _recipe())
    assert tuple(tensor.shape) == (1, 3, 240, 320)
    assert torch.allclose(tensor, torch.zeros_like(tensor))


def test_decode_maps_heatmap_cell_to_canonical_pixel():
    recipe = _recipe(n_keypoints=2, keypoints=("a", "b"))
    heatmaps = torch.zeros(1, 2, 120, 160)
    heatmaps[0, 0, 30, 40] = 1.0
    heatmaps[0, 1, 10, 100] = 1.0
    points = inference.decode(heatmaps, recipe, mode="argmax")
    assert points.shape == (1, 2, 2)
    assert np.allclose(points[0, 0], [160.0, 120.0])
    assert np.allclose(points[0, 1], [400.0, 40.0])


def test_decode_rejects_bad_mode():
    with pytest.raises(ValueError, match="未知解码"):
        inference.decode(torch.zeros(1, 2, 120, 160), _recipe(), mode="median")


def test_decode_rejects_heatmap_shape_mismatch():
    """热图尺寸与配方不符时必须报错，否则会静默乘错 stride。"""
    with pytest.raises(ValueError, match="热图实际"):
        inference.decode(torch.zeros(1, 2, 60, 80), _recipe())


def test_decode_rejects_channel_mismatch():
    """热图通道数与配方的 n_keypoints 不符时同样不能静默通过。"""
    with pytest.raises(ValueError, match="n_keypoints"):
        inference.decode(torch.zeros(1, 3, 120, 160), _recipe())


@pytest.mark.skipif(
    not WEIGHTS.is_file() or not VAL.is_dir(),
    reason="需要 m0_ft320.pth 与 m0_seq/val",
)
def test_predict_matches_legacy_prediction_path():
    """A02 验收：同一张图、同一权重、同一精度，新旧两条路必须同数。"""
    from touch.data.dataset import M0Dataset
    from touch.data.recipe import load_recipe

    recipe = load_recipe(BASELINE, PROFILE)
    assert recipe.normalize is True
    model = inference.load_model(recipe, "cpu")
    dataset = M0Dataset(
        VAL, stride=4, sigma=3.0, image_size=(240, 320), preprocess="gamma"
    )
    sample = dataset[0]

    # 旧路径：scripts/predict_keypoints.py 第 98-112 行的逐行复现
    image = torch.from_numpy(
        np.ascontiguousarray(sample["image"].transpose(2, 0, 1))
    )[None]
    image = F.interpolate(
        image, size=(240, 320), mode="bilinear", align_corners=False
    )
    image = normalize(image)
    with torch.no_grad():
        heatmaps = model(image)[0].cpu()
    legacy = soft_argmax(heatmaps[None])[0].numpy() * 4.0

    current = inference.predict(sample["image"], recipe, model=model, device="cpu")
    diff = float(np.abs(legacy - current).max())
    assert diff <= 0.01, f"新旧路径最大差 {diff:.4f}px 超过 0.01px"


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="需要 m0_ft320.pth")
def test_load_model_rejects_wrong_hash():
    from touch.data.recipe import load_recipe

    recipe = load_recipe(BASELINE, PROFILE)
    with pytest.raises(ValueError, match="哈希不符"):
        inference.load_model(replace(recipe, sha256="0" * 64), "cpu")
