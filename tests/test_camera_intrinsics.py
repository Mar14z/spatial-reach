import numpy as np

from touch.camera import (
    CameraIntrinsics,
    load_camera_config,
    load_intrinsics,
    prepare_model_input,
)


def test_load_intrinsics():
    intr = load_intrinsics("configs/camera_intrinsics.yaml")
    assert intr.width == 1280
    assert intr.height == 720
    assert intr.model == "pinhole"
    assert intr.K.shape == (3, 3)
    assert intr.dist.shape == (5,)
    assert np.isclose(intr.K[0, 0], 629.46817987545239)
    assert np.isclose(intr.K[1, 1], 622.59078435106574)
    assert np.isclose(intr.K[0, 2], 641.94571260559371)
    assert np.isclose(intr.K[1, 2], 361.10754175194074)
    assert intr.rms is not None and intr.rms < 0.5


def test_load_camera_config():
    cfg = load_camera_config("configs/camera_acepro.yaml")
    assert cfg["device_name"] == "Insta360 Ace Pro"
    assert cfg["width"] == 1280
    assert cfg["height"] == 720
    assert cfg["fps"] == 30
    assert cfg["calibration"] == "configs/camera_intrinsics.yaml"
    assert cfg["undistort"] is True


def test_prepare_model_input_crops_16_9_to_4_3():
    intr = load_intrinsics("configs/camera_intrinsics.yaml")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    image, intr2 = prepare_model_input(frame, intr, 640, 480)
    assert image.shape == (480, 640, 3)
    assert (intr2.width, intr2.height) == (640, 480)
    scale = 2.0 / 3.0
    crop_x0 = 160
    assert np.isclose(intr2.K[0, 0], intr.K[0, 0] * scale)
    assert np.isclose(intr2.K[1, 1], intr.K[1, 1] * scale)
    assert np.isclose(intr2.K[0, 2], (intr.K[0, 2] - crop_x0) * scale)
    assert np.isclose(intr2.K[1, 2], intr.K[1, 2] * scale)


def test_scaled_and_cropped():
    intr = load_intrinsics("configs/camera_intrinsics.yaml")
    half = intr.scaled(0.5)
    assert (half.width, half.height) == (640, 360)
    assert np.isclose(half.K[0, 0], intr.K[0, 0] * 0.5)
    assert np.isclose(half.K[0, 2], intr.K[0, 2] * 0.5)

    crop = intr.cropped(x0=160, y0=0, width=960, height=720)
    assert (crop.width, crop.height) == (960, 720)
    assert np.isclose(crop.K[0, 2], intr.K[0, 2] - 160)
    assert np.isclose(crop.K[1, 2], intr.K[1, 2])
