"""Deterministic lightweight geometry protocol for Stage-B binary labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StageBGeometryResult:
    task_valid: bool
    reasons: tuple[str, ...]


def world_to_grasp_numpy(
    points: np.ndarray, translation: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    shifted = np.asarray(points) - np.asarray(translation)
    matrix = np.asarray(rotation)
    return np.stack(tuple((shifted * matrix[:, axis]).sum(-1) for axis in range(3)), axis=-1)


def evaluate_stageb_geometry(
    xyz_world: np.ndarray,
    point_valid: np.ndarray,
    target_mask: np.ndarray,
    region_target: np.ndarray,
    region_valid: np.ndarray,
    translation_world: np.ndarray,
    rotation_world: np.ndarray,
    width_m: float,
    gripper_points_tcp: np.ndarray,
    gripper_part_id: np.ndarray,
) -> StageBGeometryResult:
    """Apply fixed-open approach/close/contact/region/collision checks once."""
    xyz = np.asarray(xyz_world, np.float32)
    valid = np.asarray(point_valid, bool)
    target = valid & np.asarray(target_mask, bool)
    region_known = target & np.asarray(region_valid, bool)
    if target.sum() < 8 or region_known.sum() < 2:
        raise ValueError("insufficient target or functional-region observations")
    rotation = np.asarray(rotation_world, np.float32)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("invalid candidate rotation")
    gram = np.stack(
        [[float((rotation[:, i] * rotation[:, j]).sum()) for j in range(3)] for i in range(3)]
    )
    if not np.allclose(gram, np.eye(3), atol=2e-3):
        raise ValueError("candidate rotation is not orthonormal")
    local = world_to_grasp_numpy(xyz, translation_world, rotation)
    target_local = local[target]
    reasons: list[str] = []
    if not np.isfinite(width_m) or not 0.0 < float(width_m) <= 0.095:
        raise ValueError("candidate width is outside AG-160-95 limits")
    half_width = float(width_m) * 0.5

    closing = (
        (np.abs(target_local[:, 0]) <= half_width)
        & (np.abs(target_local[:, 1]) <= 0.026)
        & (target_local[:, 2] >= -0.060)
        & (target_local[:, 2] <= 0.012)
    )
    closing_count = int(closing.sum())
    if closing_count < 6:
        reasons.append("no_target_in_closing_volume")
    enclosed = target_local[closing]
    left_contact = len(enclosed) > 0 and float(enclosed[:, 0].max()) >= 0.006
    right_contact = len(enclosed) > 0 and float(enclosed[:, 0].min()) <= -0.006
    if not left_contact:
        reasons.append("missing_left_enclosure")
    if not right_contact:
        reasons.append("missing_right_enclosure")

    gripper = np.asarray(gripper_points_tcp, np.float32)
    part = np.asarray(gripper_part_id, np.int64)
    if gripper.ndim != 2 or gripper.shape[1:] != (3,) or part.shape != (len(gripper),):
        raise ValueError("invalid AG-160-95 label geometry")
    if len(gripper) < 1_000:
        raise ValueError("offline Stage-B labels require dense AG-160-95 geometry")
    non_target = valid & ~target
    other = local[non_target]
    approach_collision = (
        (np.abs(other[:, 0]) <= 0.082)
        & (np.abs(other[:, 1]) <= 0.036)
        & (other[:, 2] >= -0.225)
        & (other[:, 2] <= -0.065)
    ).any()
    palm_collision = (
        (np.abs(other[:, 0]) <= 0.080)
        & (np.abs(other[:, 1]) <= 0.034)
        & (other[:, 2] >= -0.195)
        & (other[:, 2] <= -0.075)
    ).any()
    finger_collision = (
        (np.abs(other[:, 0]) >= 0.008)
        & (np.abs(other[:, 0]) <= 0.084)
        & (np.abs(other[:, 1]) <= 0.030)
        & (other[:, 2] >= -0.080)
        & (other[:, 2] <= 0.005)
    ).any()
    if approach_collision:
        reasons.append("approach_corridor_collision")
    if palm_collision:
        reasons.append("palm_collision")
    if finger_collision:
        reasons.append("finger_collision")

    # Exact final-pose CAD surface proximity complements the conservative swept
    # boxes above. Target contact is allowed on fingers but never on the palm.
    scene_local = local[valid]
    scene_is_target = target[valid]
    squared = ((scene_local[:, None] - gripper[None]) ** 2).sum(-1)
    base_collision = bool((squared[:, part == 1] < 0.003**2).any())
    non_target_finger_collision = (
        bool((squared[~scene_is_target][:, part != 1] < 0.003**2).any())
        if (~scene_is_target).any()
        else False
    )
    if base_collision:
        reasons.append("cad_base_collision")
    if non_target_finger_collision:
        reasons.append("cad_finger_collision")

    target_indices = np.flatnonzero(target)
    contact_local_index = np.flatnonzero(closing)
    if len(contact_local_index):
        contact_world_index = target_indices[contact_local_index]
        contact_points = target_local[contact_local_index]
        left = contact_points[:, 0] >= contact_points[:, 0].max() - 0.004
        right = contact_points[:, 0] <= contact_points[:, 0].min() + 0.004
        functional = np.asarray(region_target, bool)[contact_world_index]
        known = np.asarray(region_valid, bool)[contact_world_index]
        left_known = known & left
        right_known = known & right
        if not left_known.any() or not right_known.any():
            raise ValueError("unknown functional region at contact")
        if not (functional & left_known).any() or not (functional & right_known).any():
            reasons.append("contact_outside_functional_region")
    else:
        reasons.append("contact_outside_functional_region")
    return StageBGeometryResult(not reasons, tuple(reasons))
