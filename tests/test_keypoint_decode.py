"""亚像素解码的解析峰值、边界与退化输入回归。"""
import pytest
import torch

from touch.learning.keypoint_model import argmax_subpixel


@pytest.mark.parametrize("cx,cy", [(3., 2.), (3.25, 2.3), (2.7, 3.2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_parabolic_peak(cx, cy, dtype):
    y, x = torch.meshgrid(torch.arange(7.), torch.arange(8.), indexing="ij")
    hm = (10 - (x - cx) ** 2 - (y - cy) ** 2).to(dtype)[None, None]
    actual = argmax_subpixel(hm)
    torch.testing.assert_close(actual, torch.tensor([[[cx, cy]]]), atol=0.01, rtol=0)


@pytest.mark.parametrize("h,w,x,y", [(5, 6, 0, 0), (5, 6, 5, 4), (1, 1, 0, 0),
                                      (1, 5, 2, 0), (5, 1, 0, 2), (2, 2, 1, 1)])
def test_boundary_and_small_maps(h, w, x, y):
    hm = torch.zeros(1, 1, h, w)
    hm[0, 0, y, x] = 1
    torch.testing.assert_close(argmax_subpixel(hm), torch.tensor([[[float(x), float(y)]]]))


def test_flat_map_has_no_offset():
    torch.testing.assert_close(argmax_subpixel(torch.ones(2, 3, 4, 5)), torch.zeros(2, 3, 2))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_map_is_rejected(bad):
    hm = torch.zeros(1, 1, 3, 3)
    hm[0, 0, 1, 1] = bad
    with pytest.raises(ValueError):
        argmax_subpixel(hm)
