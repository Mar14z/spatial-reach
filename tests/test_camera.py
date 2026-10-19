import math

import numpy as np
import pytest

from touch.sim.camera import (
    intrinsics_from_fov,
    project_base_points,
    project_camera_point,
)

F = 240.0 / math.tan(math.radians(30.0))


def test_intrinsics_from_fov():
    K = intrinsics_from_fov(640, 480, 60)
    expected = np.array([[F, 0.0, 320.0], [0.0, F, 240.0], [0.0, 0.0, 1.0]])
    assert np.allclose(K, expected)


def test_project_camera_point():
    K = intrinsics_from_fov(640, 480, 60)
    u, v = project_camera_point(K, np.array([0.2, 0.1, 1.0]))
    assert abs(u - 403.1384387633061) < 1e-9
    assert abs(v - 281.56921938165306) < 1e-9


def test_project_camera_point_behind():
    K = intrinsics_from_fov(640, 480, 60)
    with pytest.raises(ValueError):
        project_camera_point(K, np.array([0.0, 0.0, -1.0]))


def test_project_base_points_identity_extrinsics():
    K = intrinsics_from_fov(640, 480, 60)
    points = np.array([[0.2, 0.1, 1.0], [0.0, 0.0, -1.0]])
    uv, valid = project_base_points(K, np.eye(3), np.zeros(3), points)
    assert valid.tolist() == [True, False]
    assert np.allclose(uv[0], [403.1384387633061, 281.56921938165306])
    assert np.isnan(uv[1]).all()
