"""Shared parallel-jaw grasp NMS for internal convergence diagnostics.

This module is intentionally independent from the public GraspNet evaluator.
The official GraspNet protocol continues to use graspnetAPI's own NMS.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tcd_prg.geometry.se3 import (
    parallel_jaw_rotation_distance,
    quaternion_xyzw_to_matrix,
)


@torch.no_grad()
def grasp_nms(
    translation_world: Tensor,
    rotation_matrix: Tensor,
    width_m: Tensor,
    score: Tensor,
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    width_threshold_m: float,
    object_index: Tensor | None = None,
) -> Tensor:
    """Return score-ranked query indices after SE(3)+width NMS.

    For Global Grasp, pass ``object_index`` so suppression is restricted to
    predictions assigned to the same object.  Task Grasp omits it.
    """

    if translation_world.ndim != 2 or translation_world.shape[-1] != 3:
        raise ValueError("translation_world must be [Q,3]")
    queries = int(translation_world.shape[0])
    if rotation_matrix.shape != (queries, 3, 3):
        raise ValueError("rotation_matrix must be [Q,3,3]")
    if width_m.shape != (queries,) or score.shape != (queries,):
        raise ValueError("width_m and score must be [Q]")
    if object_index is not None and object_index.shape != (queries,):
        raise ValueError("object_index must be [Q]")
    if min(translation_threshold_m, rotation_threshold_deg, width_threshold_m) <= 0:
        raise ValueError("NMS thresholds must be positive")

    ranked = torch.argsort(score.detach().float(), descending=True, stable=True)
    selected: list[int] = []
    rotation_threshold = math.radians(float(rotation_threshold_deg))
    for value in ranked.tolist():
        value = int(value)
        if not selected:
            selected.append(value)
            continue
        prior = torch.as_tensor(selected, dtype=torch.long, device=translation_world.device)
        if object_index is not None:
            same_object = object_index[prior] == object_index[value]
            if not bool(same_object.any()):
                selected.append(value)
                continue
            prior = prior[same_object]
        translation_close = torch.linalg.vector_norm(
            translation_world[prior].float() - translation_world[value].float(), dim=-1
        ) <= float(translation_threshold_m)
        if not bool(translation_close.any()):
            selected.append(value)
            continue
        prior = prior[translation_close]
        rotation_close = (
            parallel_jaw_rotation_distance(
                rotation_matrix[prior],
                rotation_matrix[value].expand(len(prior), -1, -1),
            )
            <= rotation_threshold
        )
        width_close = (width_m[prior].float() - width_m[value].float()).abs() <= float(
            width_threshold_m
        )
        if not bool((rotation_close & width_close).any()):
            selected.append(value)
    return torch.as_tensor(selected, dtype=torch.long, device=translation_world.device)


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
    """Keep unique candidates in batched ``[B, K]`` task-grasp tensors."""

    expected = pose_world.shape[:2]
    if pose_world.ndim != 3 or pose_world.shape[-1] != 7:
        raise ValueError("pose_world must be [B,K,7] in metres and xyzw order")
    for name, value in (
        ("width_m", width_m),
        ("score", score),
        ("object_index", object_index),
        ("valid", valid),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must be [B,K], got {tuple(value.shape)}")
    if (
        min(
            translation_threshold_m,
            rotation_threshold_deg,
            width_threshold_m,
            approach_threshold_deg,
        )
        <= 0
    ):
        raise ValueError("All grasp NMS thresholds must be positive")

    finite = (
        valid.bool()
        & torch.isfinite(pose_world).all(-1)
        & torch.isfinite(width_m)
        & torch.isfinite(score)
        & (object_index >= 0)
    )
    identity_quaternion = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        dtype=pose_world.dtype,
        device=pose_world.device,
    )
    safe_quaternion = torch.where(finite.unsqueeze(-1), pose_world[..., 3:], identity_quaternion)
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
                translation_close = (
                    torch.linalg.vector_norm(
                        pose_world[row, index, :3] - pose_world[row, previous, :3]
                    )
                    <= translation_threshold_m
                )
                rotation_close = (
                    parallel_jaw_rotation_distance(rotation[row, index], rotation[row, previous])
                    <= rotation_threshold
                )
                approach_cosine = (
                    (rotation[row, index, :, 2] * rotation[row, previous, :, 2])
                    .sum()
                    .clamp(-1.0, 1.0)
                )
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
