"""Camera-frame and GraspNet/TCD grasp-frame conversion utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def look_at_rotation_world_camera(
    eye_world: Tensor, target_world: Tensor, up_world: Tensor
) -> Tensor:
    """Return ``R_world_camera`` for x-right, y-down, z-forward cameras."""

    forward = torch.nn.functional.normalize(target_world - eye_world, dim=-1)
    right = torch.nn.functional.normalize(
        torch.linalg.cross(forward, up_world, dim=-1), dim=-1
    )
    camera_up = torch.nn.functional.normalize(
        torch.linalg.cross(right, forward, dim=-1), dim=-1
    )
    return torch.stack((right, -camera_up, forward), dim=-1)


def world_to_camera_points(
    points_world: Tensor, rotation_world_camera: Tensor, eye_world: Tensor
) -> Tensor:
    """Transform row-vector points from world into camera coordinates."""

    return torch.einsum(
        "bni,bij->bnj", points_world - eye_world[:, None], rotation_world_camera
    )


def camera_to_world_points(
    points_camera: Tensor, rotation_world_camera: Tensor, eye_world: Tensor
) -> Tensor:
    """Transform row-vector points from camera into world coordinates."""

    return torch.einsum(
        "bij,bnj->bni", rotation_world_camera, points_camera
    ) + eye_world[:, None]


def graspnet_to_tcd_rotation(rotation_graspnet: Tensor) -> Tensor:
    """Map GraspNet axes (x=approach,y=closing,z=height) to TCD/AG axes.

    TCD/AG uses x=closing, y=height and z=approach.
    """

    return rotation_graspnet[..., :, (1, 2, 0)]


def camera_to_world_rotations(
    rotation_camera: Tensor, rotation_world_camera: Tensor
) -> Tensor:
    """Left-compose camera-frame rotations into the world frame."""

    return torch.einsum("bij,bkjl->bkil", rotation_world_camera, rotation_camera)
