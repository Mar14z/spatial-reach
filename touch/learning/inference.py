"""确定性推理契约：模型构建、权重加载、输入归一化、解码与坐标还原。

训练侧 evaluate 与独立预测都必须经过本模块；任何一边私自改预处理或解码，
A03 的逐级一致性测试就会失败。本模块不含随机增强（训练侧）、几何评估
（touch/learning/evaluate.py）与数据读取（touch/data/dataset.py）。

输入契约：to_input / predict 收到的是**已完成 /255 与 recipe.preprocess 的
(H, W, 3) 浮点图像**，即 M0Dataset 的输出。传原始 uint8 会直接报错。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from touch.data.recipe import Recipe, check_state_shapes, sha256_of
from touch.learning.keypoint_model import (
    MobileNetHeatmapNet,
    ResNetHeatmapNet,
    TinyHeatmapNet,
    argmax_subpixel,
    heatmap_argmax,
    normalize,
    soft_argmax,
)

# 标签与评估用的规范坐标系（width, height）。热图必须是它按整数 stride 下采样的结果。
CANONICAL_W, CANONICAL_H = 640, 480

MODELS = {
    "tiny": TinyHeatmapNet,
    "resnet18": ResNetHeatmapNet,
    "mobilenet_v3": MobileNetHeatmapNet,
}

DECODERS = {
    "soft": soft_argmax,
    "argmax": heatmap_argmax,
    "subpixel": argmax_subpixel,
}


def stride_of(recipe: Recipe) -> int:
    """热图格点到规范像素的倍数；宽高必须反推出同一个整数。"""
    if recipe.out_width <= 0 or recipe.out_height <= 0:
        raise ValueError(
            f"{recipe.name}: 热图尺寸必须为正，实际 "
            f"{recipe.out_width}x{recipe.out_height}"
        )
    if CANONICAL_W % recipe.out_width or CANONICAL_H % recipe.out_height:
        raise ValueError(
            f"{recipe.name}: 热图 {recipe.out_width}x{recipe.out_height} "
            f"不能整除规范坐标 {CANONICAL_W}x{CANONICAL_H}"
        )
    sx = CANONICAL_W // recipe.out_width
    sy = CANONICAL_H // recipe.out_height
    if sx != sy:
        raise ValueError(
            f"{recipe.name}: 宽高推出不同 stride（{sx} vs {sy}），"
            "热图宽高比与规范坐标不一致"
        )
    return sx


def build_network(
    model_name: str, n_keypoints: int, out_width: int, out_height: int
) -> torch.nn.Module:
    """按名称建网络；训练期没有已冻结权重时也用它，保证结构只有一份定义。"""
    if model_name not in MODELS:
        raise ValueError(f"未知模型 {model_name!r}，可选 {sorted(MODELS)}")
    return MODELS[model_name](
        n_keypoints=n_keypoints,
        out_size=(out_height, out_width),
    )


def build_model(recipe: Recipe) -> torch.nn.Module:
    """按配方建网络；out_size 只能是 (h, w)，与模型类的签名一致。"""
    return build_network(
        recipe.model,
        recipe.n_keypoints,
        recipe.out_width,
        recipe.out_height,
    )


def load_model(
    recipe: Recipe, device: str | torch.device = "cpu"
) -> torch.nn.Module:
    """按配方建模型并严格加载权重；哈希、通道数、键名任一不符都直接失败。"""
    digest = sha256_of(recipe.checkpoint)
    if digest != recipe.sha256:
        raise ValueError(
            f"{recipe.name}: 权重哈希不符（期望 {recipe.sha256[:12]}…，"
            f"实际 {digest[:12]}…）"
        )
    state = torch.load(recipe.checkpoint, map_location="cpu", weights_only=True)
    check_state_shapes(recipe, {k: tuple(v.shape) for k, v in state.items()})
    model = build_model(recipe)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def resize_batch(images: torch.Tensor, recipe: Recipe) -> torch.Tensor:
    """(B, 3, h, w) -> 配方输入尺寸，保持 0~1 值域、不归一化。

    训练侧随机光增强必须作用在 0~1 像素上，所以把 resize 与 normalize
    拆成两步；两条路径仍共用同一份插值实现。
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"输入必须是 (B, 3, h, w)，实际 {tuple(images.shape)}")
    size = (recipe.input_height, recipe.input_width)
    if images.shape[-2:] != size:
        images = F.interpolate(
            images, size=size, mode="bilinear", align_corners=False
        )
    return images


def prepare_batch(images: torch.Tensor, recipe: Recipe) -> torch.Tensor:
    """(B, 3, h, w) 的 0~1 张量 -> 配方输入尺寸并归一化。

    训练验证的 DataLoader 与单帧预测都走这里，尺寸与归一化只此一份。
    """
    images = resize_batch(images, recipe)
    return normalize(images) if recipe.normalize else images


def to_input(
    image,
    recipe: Recipe,
    device=None,
    *,
    resize: bool = True,
) -> torch.Tensor:
    """(H, W, 3) 已预处理图像 -> (1, 3, in_h, in_w) 张量。

    resize=False 时尺寸不符直接报错，用来区分「数据侧没 resize 对」和
    「推理侧兜底」。PIL 与 torch 的 bilinear 实现不完全等价，正常流程应由
    dataset 统一 resize，这里只对尺寸不符兜底。
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"输入必须是 (H, W, 3)，实际 {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"输入必须是 0~1 浮点（已 /255），实际 dtype {array.dtype}")
    if float(array.max(initial=0.0)) > 2.0:
        raise ValueError("输入值域看起来是 0~255 未归一化，请先 /255")
    tensor = torch.as_tensor(
        np.ascontiguousarray(array.transpose(2, 0, 1)), dtype=torch.float32
    )[None]
    size = (recipe.input_height, recipe.input_width)
    if tensor.shape[-2:] != size and not resize:
        raise ValueError(
            f"输入尺寸 {tuple(tensor.shape[-2:])} 与配方 {size} 不符"
        )
    prepared = prepare_batch(tensor, recipe)
    return prepared if device is None else prepared.to(device)


def decode_tensor(
    heatmaps: torch.Tensor, recipe: Recipe, mode: str | None = None
) -> torch.Tensor:
    """(N, C, h, w) 热图 -> (N, C, 2) 规范坐标张量；训练验证用它算误差。"""
    if heatmaps.ndim != 4:
        raise ValueError(f"热图必须是 (N, C, h, w)，实际 {tuple(heatmaps.shape)}")
    key = mode or recipe.decode
    if key not in DECODERS:
        raise ValueError(
            f"{recipe.name}: 未知解码 {key!r}，可选 {sorted(DECODERS)}"
        )
    # 尺寸不符时乘 stride 会静默给出错坐标，必须在这里拦住
    expected = (recipe.out_height, recipe.out_width)
    if tuple(heatmaps.shape[-2:]) != expected:
        raise ValueError(
            f"{recipe.name}: 热图实际 {tuple(heatmaps.shape[-2:])} "
            f"与配方 {expected} 不符"
        )
    if heatmaps.shape[1] != recipe.n_keypoints:
        raise ValueError(
            f"{recipe.name}: 热图通道 {heatmaps.shape[1]} "
            f"与配方 n_keypoints {recipe.n_keypoints} 不符"
        )
    return DECODERS[key](heatmaps) * stride_of(recipe)


def decode(
    heatmaps: torch.Tensor, recipe: Recipe, mode: str | None = None
) -> np.ndarray:
    """(N, C, h, w) 热图 -> (N, C, 2) 规范坐标（640x480 像素）。

    格点索引乘 stride 即得规范坐标。历史实现没有加半格偏移，A04 冻结前
    不得顺手改，否则所有历史数字失去可比性。
    """
    return decode_tensor(heatmaps, recipe, mode).detach().cpu().numpy()


@torch.no_grad()
def predict(
    image,
    recipe: Recipe,
    model: torch.nn.Module | None = None,
    device=None,
    *,
    use_amp: bool = False,
) -> np.ndarray:
    """单帧确定性推理：已预处理图像 -> (C, 2) 规范坐标。不读真值。"""
    if model is None:
        model = load_model(recipe, device or "cpu")
    target = device or next(model.parameters()).device
    tensor = to_input(image, recipe, device=target)
    enabled = bool(use_amp) and torch.device(target).type == "cuda"
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=enabled
    ):
        heatmaps = model(tensor).float()
    return decode(heatmaps, recipe)[0]
