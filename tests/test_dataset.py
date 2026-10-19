import json

import numpy as np
from PIL import Image

from touch.data.dataset import M0Dataset, make_heatmaps


def test_make_heatmaps_peak():
    heatmaps = make_heatmaps(
        np.array([[4.0, 4.0]]), width=8, height=8, stride=4, sigma=2.0
    )
    assert heatmaps.shape == (1, 2, 2)
    peak = np.unravel_index(np.argmax(heatmaps[0]), heatmaps[0].shape)
    assert peak == (1, 1)
    assert abs(float(heatmaps[0, 1, 1]) - 1.0) < 1e-6


def test_m0_dataset_reads_frame(tmp_path):
    frame_dir = tmp_path / "000000"
    frame_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        frame_dir / "rgb.png"
    )
    meta = {
        "width": 8,
        "height": 8,
        "keypoints_2d": [[4.0, 4.0]],
        "K": [[1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
        "R_cb": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "t_cb": [0.0, 0.0, 0.0],
    }
    (frame_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    dataset = M0Dataset(tmp_path)
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["image"].shape == (8, 8, 3)
    assert sample["keypoints"].shape == (1, 2)
    assert sample["heatmaps"].shape == (1, 2, 2)
    assert sample["K"].shape == (3, 3)
