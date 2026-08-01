"""Build supervision tensors from unified state candidate groups."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType
from tcd_prg.geometry.se3 import quaternion_xyzw_to_matrix


def _nearest_point_indices(
    xyz: Tensor,
    point_mask: Tensor,
    instance_id: Tensor,
    contacts: Tensor,
    acted_object: Tensor,
    candidate_valid: Tensor,
    max_distance_m: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Associate candidates with visible object points in object-wise chunks."""

    batch_size, candidate_count = contacts.shape[:2]
    indices = torch.zeros((batch_size, candidate_count), dtype=torch.long, device=xyz.device)
    valid = candidate_valid.clone()
    for row in range(batch_size):
        objects = torch.unique(acted_object[row, candidate_valid[row]])
        for object_index in objects.tolist():
            candidates = torch.nonzero(
                candidate_valid[row] & (acted_object[row] == object_index), as_tuple=False
            ).flatten()
            points = torch.nonzero(
                point_mask[row] & (instance_id[row] == object_index), as_tuple=False
            ).flatten()
            finite = torch.isfinite(contacts[row, candidates]).all(-1)
            valid[row, candidates[~finite]] = False
            candidates = candidates[finite]
            if not len(points) or not len(candidates):
                valid[row, candidates] = False
                continue
            for start in range(0, len(candidates), 256):
                selected = candidates[start : start + 256]
                distance = torch.cdist(contacts[row, selected], xyz[row, points])
                nearest_distance, nearest = distance.min(-1)
                indices[row, selected] = points[nearest]
                if max_distance_m is not None:
                    valid[row, selected] &= nearest_distance <= max_distance_m
    return indices, valid


def _pack_grasp_set(
    pose: Tensor, width: Tensor, valid: Tensor, quality: Tensor, quality_valid: Tensor,
    queries: int,
) -> dict[str, Tensor]:
    """Pack an unordered positive grasp set without averaging valid modes."""

    b = pose.shape[0]
    translation = pose.new_full((b, queries, 3), float("nan"))
    rotation = pose.new_full((b, queries, 3, 3), float("nan"))
    width_target = pose.new_full((b, queries), float("nan"))
    quality_target = pose.new_zeros((b, queries))
    target_valid = torch.zeros((b, queries), dtype=torch.bool, device=pose.device)
    target_quality_valid = torch.zeros_like(target_valid)
    matrices = quaternion_xyzw_to_matrix(torch.nan_to_num(pose[..., 3:], nan=0.0))
    for row in range(b):
        selected = torch.nonzero(valid[row], as_tuple=False).flatten()
        if len(selected) > queries:
            selected = selected[quality[row, selected].argsort(descending=True, stable=True)[:queries]]
        count = len(selected)
        if not count:
            continue
        translation[row, :count] = pose[row, selected, :3]
        rotation[row, :count] = matrices[row, selected]
        width_target[row, :count] = width[row, selected]
        quality_target[row, :count] = quality[row, selected]
        target_valid[row, :count] = True
        target_quality_valid[row, :count] = quality_valid[row, selected]
    return {
        "translation_world": translation,
        "rotation_matrix": rotation,
        "width_m": width_target,
        "quality_target": quality_target,
        "target_valid": target_valid,
        "quality_valid": target_quality_valid,
    }


def build_grasp_proposal_labels(
    batch: dict[str, Tensor], config: ModelConfig
) -> dict[str, Tensor]:
    """Build the complete task-grasp set from final executable grasps."""

    parameters = batch["action_parameters"]
    candidate = batch["candidate_mask"] & (batch["action_type"] == int(ActionType.TASK_GRASP))
    pose = parameters["task_grasp_pose_world"]
    width = parameters["grasp_width_m"]
    overall_valid = parameters.get("verifier_overall_valid", torch.zeros_like(candidate)).bool()
    overall = parameters.get("verifier_overall_target", torch.zeros_like(width)).float()
    successful = torch.where(overall_valid, overall > 0.5, batch["action_improves_state"])
    valid = (
        candidate & successful & torch.isfinite(pose).all(-1) & torch.isfinite(width)
        & (width >= config.min_grasp_width_m) & (width <= config.max_grasp_width_m)
    )
    quality = torch.where(
        overall_valid, torch.nan_to_num(overall),
        torch.nan_to_num(parameters.get("grasp_confidence", torch.ones_like(width)), nan=1.0),
    ).clamp(0.0, 1.0)
    labels = _pack_grasp_set(
        pose, width, valid, quality, torch.ones_like(candidate), config.task_grasp_candidates
    )
    labels["sample_valid"] = candidate.any(-1)
    return labels


def build_global_grasp_labels(
    batch: dict[str, Tensor], config: ModelConfig
) -> dict[str, Tensor] | None:
    """Build task-free sets using only scene-certified executable grasps."""

    source = batch.get("global_grasp_labels")
    if source is None:
        return None
    certified = source["scene_executable"] >= 0
    positive = source["scene_executable"] == 1
    pose = source["grasp_pose_world"]
    width = source["width_m"]
    valid = (
        source["valid_mask"] & positive & torch.isfinite(pose).all(-1) & torch.isfinite(width)
        & (width >= config.min_grasp_width_m) & (width <= config.max_grasp_width_m)
    )
    quality = positive.float()
    labels = _pack_grasp_set(
        pose, width, valid, quality, certified, config.global_grasp_candidates
    )
    labels["sample_valid"] = batch.get(
        "global_loss_sample_valid", torch.ones(pose.shape[0], dtype=torch.bool, device=pose.device)
    ) & (source["valid_mask"] & certified).any(-1)
    return labels


def build_graph_labels(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    object_mask = batch["object_mask"] & batch["object_active"]
    pair_valid = object_mask[:, :, None, None] & object_mask[:, None, :, None]
    return {
        "physical_edge_target": batch["relation_graph"],
        "physical_edge_valid": pair_valid.expand_as(batch["relation_graph"]),
        "task_edge_target": batch["task_block_graph"],
        "task_edge_valid": object_mask[:, :, None].expand_as(batch["task_block_graph"]),
    }


def build_push_supervision(
    output: dict[str, Tensor], batch: dict[str, Tensor], config: ModelConfig
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    action_type = batch["action_type"]
    parameters = batch["action_parameters"]
    candidate = batch["candidate_mask"] & (action_type == int(ActionType.PUSH))
    point_index, parameter_valid = _nearest_point_indices(
        batch["xyz"], batch["point_mask"], batch["instance_id"],
        parameters["push_contact_world"], batch["acted_object"], candidate,
    )
    row = torch.arange(action_type.shape[0], device=action_type.device)[:, None]
    gathered = {
        "object_logits": output["object_logits"],
        "contact_logits": output["contact_logits"],
        "direction_logits": output["direction_logits"][row, point_index],
        "direction_residual": output["direction_residual"][row, point_index],
        "utility_delta": output["utility_delta"][row, point_index],
    }
    direction = torch.nn.functional.normalize(
        torch.nan_to_num(parameters["push_direction_world"]), dim=-1
    )
    angle = torch.atan2(direction[..., 1], direction[..., 0]).remainder(2 * math.pi)
    bins = output["direction_logits"].shape[-1]
    direction_bin = torch.floor(angle * bins / (2 * math.pi)).long().remainder(bins)
    center_angle = (direction_bin.float() + 0.5) * 2 * math.pi / bins
    center = torch.stack((torch.cos(center_angle), torch.sin(center_angle)), -1)
    contact_target = torch.zeros_like(output["contact_logits"])
    contact_valid = torch.zeros_like(output["contact_logits"], dtype=torch.bool)
    sigma_sq = float(config.contact_heatmap_sigma_m) ** 2
    for batch_row in range(action_type.shape[0]):
        for candidate_index in torch.nonzero(parameter_valid[batch_row], as_tuple=False).flatten().tolist():
            object_index = int(batch["acted_object"][batch_row, candidate_index])
            domain = batch["point_mask"][batch_row] & (batch["instance_id"][batch_row] == object_index)
            delta = batch["xyz"][batch_row] - parameters["push_contact_world"][batch_row, candidate_index]
            distance_sq = (delta * delta).sum(-1)
            neighborhood = domain & (distance_sq <= 9.0 * sigma_sq)
            contact_valid[batch_row] |= neighborhood
            contact_target[batch_row] = torch.maximum(
                contact_target[batch_row], torch.exp(-0.5 * distance_sq / sigma_sq) * neighborhood
            )
    positive = batch["action_improves_state"] & candidate
    object_positive = torch.zeros_like(batch["object_mask"])
    for batch_row in range(action_type.shape[0]):
        objects = batch["acted_object"][batch_row, positive[batch_row]]
        object_positive[batch_row, objects[objects >= 0]] = True
    component_weights = torch.as_tensor(
        config.push_utility_component_weights,
        dtype=batch["potential_delta"].dtype,
        device=batch["potential_delta"].device,
    )
    utility = (torch.nan_to_num(batch["potential_delta"]) * component_weights).sum(-1)
    failures = torch.stack((
        parameters["risk_unstable"], parameters["risk_out_of_workspace"],
        parameters["risk_other_invalid"],
    ), -1)
    penalties = torch.as_tensor(
        config.push_failure_penalties, dtype=utility.dtype, device=utility.device
    )
    utility = utility - (torch.nan_to_num(failures) * penalties).sum(-1)
    return gathered, {
        "object_positive": object_positive,
        "object_valid_mask": batch["object_mask"] & batch["object_active"],
        "contact_target": contact_target,
        "contact_valid": contact_valid,
        "direction_bin": direction_bin,
        "direction_residual": direction[..., :2] - center,
        "direction_valid": parameter_valid,
        "utility_delta": utility,
        "utility_valid": candidate & batch["potential_after_valid"],
    }


def build_verifier_labels(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Supervise only final certifier executability."""

    candidate_valid = batch["verifier_inputs"]["candidate_valid"]
    target = batch["action_parameters"]["verifier_overall_target"]
    valid = batch["action_parameters"]["verifier_overall_valid"] & candidate_valid
    return {
        "overall_target": torch.nan_to_num(target),
        "overall_valid": valid & torch.isfinite(target),
    }


def build_region_labels(batch: dict[str, Tensor]) -> dict[str, Tensor] | None:
    if "region_target" not in batch:
        return None
    return {
        "region_target": batch["region_target"],
        "region_valid": batch["region_valid"],
        "visibility_target": batch["visibility_target"],
        "visibility_valid": batch["visibility_valid"],
    }
