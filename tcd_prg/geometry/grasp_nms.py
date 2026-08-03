"""SE(3)-aware non-maximum suppression for parallel-jaw grasps."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .se3 import quaternion_xyzw_to_matrix


def _rotation_angle(rotation_a: Tensor, rotation_b: Tensor) -> Tensor:
    """Return the geodesic angle, respecting 180-degree jaw-swap symmetry."""

    relative = rotation_a.transpose(-1, -2) @ rotation_b
    jaw_swap = torch.diag(
        torch.tensor([-1.0, -1.0, 1.0], dtype=rotation_a.dtype, device=rotation_a.device)
    )
    relative_swapped = rotation_a.transpose(-1, -2) @ (rotation_b @ jaw_swap)

    def angle(matrix: Tensor) -> Tensor:
        cosine = ((torch.trace(matrix) - 1.0) * 0.5).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    return torch.minimum(angle(relative), angle(relative_swapped))


def task_grasp_nms(
    pose_world: Tensor,
    width_m: Tensor,
    score: Tensor,
    object_index: Tensor,
    valid: Tensor,
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    width_threshold_m: float,
    approach_threshold_deg: float,
) -> Tensor:
    """Keep unique grasps in ``[B,K]`` candidate tensors.

    Candidates are clustered only within the same object. A candidate is a
    duplicate when translation, symmetric SO(3) rotation, gripper opening, and
    approach-axis differences are all below their configured thresholds.
    Higher-scoring candidates are retained deterministically.
    """

    expected = pose_world.shape[:2]
    if pose_world.ndim != 3 or pose_world.shape[-1] != 7:
        raise ValueError("pose_world must be [B,K,7] in metres and xyzw order")
    inputs = (
        ("width_m", width_m),
        ("score", score),
        ("object_index", object_index),
        ("valid", valid),
    )
    for name, value in inputs:
        if value.shape != expected:
            raise ValueError(f"{name} must be [B,K], got {tuple(value.shape)}")
    if min(
        translation_threshold_m,
        rotation_threshold_deg,
        width_threshold_m,
        approach_threshold_deg,
    ) <= 0:
        raise ValueError("All grasp NMS thresholds must be positive")

    # 只有同一物体且平移、对称旋转、接近轴和开口均接近时才视为重复抓取。
    finite = (
        valid.bool()
        & torch.isfinite(pose_world).all(-1)
        & torch.isfinite(width_m)
        & torch.isfinite(score)
        & (object_index >= 0)
    )
    identity_quaternion = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=pose_world.dtype, device=pose_world.device
    )
    safe_quaternion = torch.where(
        finite.unsqueeze(-1), pose_world[..., 3:], identity_quaternion
    )
    rotation = quaternion_xyzw_to_matrix(safe_quaternion)
    rotation_threshold = math.radians(rotation_threshold_deg)
    approach_threshold = math.radians(approach_threshold_deg)
    keep = torch.zeros_like(finite)

    for row in range(pose_world.shape[0]):
        indices = torch.nonzero(finite[row], as_tuple=False).flatten()
        if not len(indices):
            continue
        order = indices[torch.argsort(score[row, indices], descending=True, stable=True)]
        selected: list[int] = []
        for index_tensor in order:
            index = int(index_tensor)
            duplicate = False
            for previous in selected:
                if int(object_index[row, index]) != int(object_index[row, previous]):
                    continue
                translation_close = torch.linalg.vector_norm(
                    pose_world[row, index, :3] - pose_world[row, previous, :3]
                ) <= translation_threshold_m
                rotation_close = _rotation_angle(
                    rotation[row, index], rotation[row, previous]
                ) <= rotation_threshold
                approach_cosine = (
                    rotation[row, index, :, 2] * rotation[row, previous, :, 2]
                ).sum().clamp(-1.0, 1.0)
                approach_close = torch.acos(approach_cosine) <= approach_threshold
                width_close = (
                    width_m[row, index] - width_m[row, previous]
                ).abs() <= width_threshold_m
                if bool(translation_close & rotation_close & approach_close & width_close):
                    duplicate = True
                    break
            if not duplicate:
                keep[row, index] = True
                selected.append(index)
    return keep
