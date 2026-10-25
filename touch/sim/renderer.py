"""PyBullet 离屏渲染：RGB、深度、分割与 2D keypoints。"""

from __future__ import annotations

import numpy as np
import pybullet as p

from touch.sim.camera import (
    intrinsics_from_fov,
    project_base_points,
    projection_from_intrinsics,
)


def opencv_extrinsics_from_view(
    view: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """PyBullet 的 OpenGL view → OpenCV base→camera 外参。"""
    v_gl = np.array(view, dtype=float).reshape(4, 4).T
    t_cv = np.diag([1.0, -1.0, -1.0, 1.0]) @ v_gl
    return t_cv[:3, :3], t_cv[:3, 3]


def render_frame(
    width: int,
    height: int,
    fov_deg: float,
    view: list[float],
    near: float = 0.01,
    far: float = 5.0,
    K: np.ndarray | None = None,
    shadow: int = 1,
    light_direction: tuple[float, float, float] = (0.4, -0.6, 1.0),
    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ambient: float = 0.75,
    diffuse: float = 0.7,
    specular: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (rgb, depth_m, seg)。"""
    if K is None:
        proj = p.computeProjectionMatrixFOV(
            fov=fov_deg, aspect=width / height, nearVal=near, farVal=far
        )
    else:
        proj = projection_from_intrinsics(K, width, height, near, far)
    _, _, rgb, depth, seg = p.getCameraImage(
        width,
        height,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=light_direction,
        lightColor=light_color,
        lightDistance=2.0,
        shadow=shadow,
        lightAmbientCoeff=ambient,
        lightDiffuseCoeff=diffuse,
        lightSpecularCoeff=specular,
    )
    rgb = np.reshape(rgb, (height, width, 4)).astype(np.uint8)[:, :, :3]
    depth_buffer = np.reshape(depth, (height, width)).astype(np.float32)
    depth_m = far * near / (far - (far - near) * depth_buffer)
    seg = np.reshape(seg, (height, width)).astype(np.int32)
    return rgb, depth_m, seg


def project_link_keypoints(
    body_id: int,
    link_indices: dict[str, int],
    width: int,
    height: int,
    fov_deg: float,
    view: list[float],
    K: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (link 名, 3D base 点, 2D 像素, valid, K)。"""
    names = list(link_indices)
    points_base = np.array(
        [p.getLinkState(body_id, link_indices[n])[4] for n in names],
        dtype=np.float64,
    )
    r_cb, t_cb = opencv_extrinsics_from_view(view)
    k = (
        intrinsics_from_fov(width, height, fov_deg)
        if K is None
        else np.asarray(K, dtype=np.float64)
    )
    uv, valid = project_base_points(k, r_cb, t_cb, points_base)
    return names, points_base, uv, valid, k
