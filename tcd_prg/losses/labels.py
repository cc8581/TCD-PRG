"""Build supervision tensors from unified state candidate groups."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType, CandidateStatus
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
    queries: int, object_index: Tensor | None = None,
) -> dict[str, Tensor]:
    """Pack an unordered positive grasp set without averaging valid modes."""

    # 固定 query 容量只用于张量打包；超过容量时按质量并兼顾物体均衡截断。
    b = pose.shape[0]
    translation = pose.new_full((b, queries, 3), float("nan"))
    rotation = pose.new_full((b, queries, 3, 3), float("nan"))
    width_target = pose.new_full((b, queries), float("nan"))
    quality_target = pose.new_zeros((b, queries))
    target_valid = torch.zeros((b, queries), dtype=torch.bool, device=pose.device)
    target_quality_valid = torch.zeros_like(target_valid)
    target_object = torch.full(
        (b, queries), -1, dtype=torch.long, device=pose.device
    )
    matrices = quaternion_xyzw_to_matrix(torch.nan_to_num(pose[..., 3:], nan=0.0))
    for row in range(b):
        selected = torch.nonzero(valid[row], as_tuple=False).flatten()
        if len(selected) > queries:
            if object_index is None:
                selected = selected[
                    quality[row, selected].argsort(descending=True, stable=True)[:queries]
                ]
            else:
                buckets = []
                for object_value in torch.unique(object_index[row, selected], sorted=True).tolist():
                    local = selected[object_index[row, selected] == object_value]
                    buckets.append(local[
                        quality[row, local].argsort(descending=True, stable=True)
                    ])
                balanced: list[Tensor] = []
                depth = 0
                while len(balanced) < queries:
                    added = False
                    for bucket in buckets:
                        if depth < len(bucket):
                            balanced.append(bucket[depth])
                            added = True
                            if len(balanced) == queries:
                                break
                    if not added:
                        break
                    depth += 1
                selected = torch.stack(balanced)
        count = len(selected)
        if not count:
            continue
        translation[row, :count] = pose[row, selected, :3]
        rotation[row, :count] = matrices[row, selected]
        width_target[row, :count] = width[row, selected]
        quality_target[row, :count] = quality[row, selected]
        target_valid[row, :count] = True
        target_quality_valid[row, :count] = quality_valid[row, selected]
        if object_index is not None:
            target_object[row, :count] = object_index[row, selected]
    packed = {
        "translation_world": translation,
        "rotation_matrix": rotation,
        "width_m": width_target,
        "quality_target": quality_target,
        "target_valid": target_valid,
        "quality_valid": target_quality_valid,
    }
    if object_index is not None:
        packed["object_index"] = target_object
    return packed


def _attach_negative_grasp_set(
    labels: dict[str, Tensor], pose: Tensor, width: Tensor, valid: Tensor,
    object_index: Tensor | None = None,
) -> None:
    """Attach explicit known negatives without turning other queries negative."""

    # 显式负抓取保留独立集合，后续仅监督几何邻近的预测 query。
    labels["negative_translation_world"] = torch.nan_to_num(pose[..., :3])
    labels["negative_rotation_matrix"] = quaternion_xyzw_to_matrix(
        torch.nan_to_num(pose[..., 3:], nan=0.0)
    )
    labels["negative_width_m"] = torch.nan_to_num(width)
    labels["negative_valid"] = valid
    if object_index is not None:
        labels["negative_object_index"] = object_index


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
    known_negative = (
        candidate
        & (batch["evaluation_status"] == int(CandidateStatus.NEGATIVE))
        & torch.isfinite(pose).all(-1)
        & torch.isfinite(width)
        & (width >= config.min_grasp_width_m)
        & (width <= config.max_grasp_width_m)
    )
    # 不再通过“已采样候选均已评价”推断完备性，只信任数据生产者的显式字段。
    label_set_complete = batch.get(
        "task_grasp_label_set_complete",
        torch.zeros(pose.shape[0], dtype=torch.bool, device=pose.device),
    ).bool()
    labels = _pack_grasp_set(
        pose, width, valid, quality, torch.ones_like(candidate), config.task_grasp_candidates
    )
    _attach_negative_grasp_set(labels, pose, width, known_negative)
    labels["label_set_complete"] = label_set_complete
    labels["sample_valid"] = valid.any(-1) | known_negative.any(-1) | label_set_complete
    labels["unmatched_quality_valid"] = label_set_complete
    return labels


def build_global_grasp_labels(
    batch: dict[str, Tensor], config: ModelConfig
) -> dict[str, Tensor] | None:
    """Build task-free sets from known native PICK_REMOVE execution outcomes."""

    source = batch.get("global_grasp_labels")
    if source is None:
        return None
    # scene_executable=-1 是 UNKNOWN；只有 0/1 才属于已认证监督。
    certified = source["scene_executable"] >= 0
    positive = source["scene_executable"] == 1
    pose = source["grasp_pose_world"]
    width = source["width_m"]
    valid = (
        source["valid_mask"] & positive & torch.isfinite(pose).all(-1) & torch.isfinite(width)
        & (width >= config.min_grasp_width_m) & (width <= config.max_grasp_width_m)
    )
    quality = positive.float()
    known_negative = (
        source["valid_mask"] & (source["scene_executable"] == 0)
        & torch.isfinite(pose).all(-1) & torch.isfinite(width)
        & (width >= config.min_grasp_width_m) & (width <= config.max_grasp_width_m)
    )
    labels = _pack_grasp_set(
        pose, width, valid, quality, certified, config.global_grasp_candidates,
        object_index=source["object_index"],
    )
    _attach_negative_grasp_set(
        labels, pose, width, known_negative, object_index=source["object_index"]
    )
    representative = batch.get(
        "global_loss_sample_valid", torch.ones(pose.shape[0], dtype=torch.bool, device=pose.device)
    )
    label_set_complete = source.get(
        "label_set_complete", torch.zeros_like(representative)
    ).bool() & ~(source["valid_mask"] & ~certified).any(-1)
    labels["label_set_complete"] = label_set_complete
    labels["sample_valid"] = representative & (
        valid.any(-1) | known_negative.any(-1) | label_set_complete
    )
    labels["unmatched_quality_valid"] = representative & label_set_complete
    return labels


def build_graph_labels(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    # 图损失只覆盖当前物理活跃物体之间的边，padding 和已移除物体全部 ignore。
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
    direction = torch.nn.functional.normalize(
        torch.nan_to_num(parameters["push_direction_world"]), dim=-1
    )
    angle = torch.atan2(direction[..., 1], direction[..., 0]).remainder(2 * math.pi)
    bins = output["direction_logits"].shape[-1]
    direction_bin = torch.floor(angle * bins / (2 * math.pi)).long().remainder(bins)
    gathered = {
        "object_logits": output["object_logits"],
        "contact_logits": output["contact_logits"],
        "point_index": point_index,
        "direction_logits": output["direction_logits"][row, point_index],
        "direction_residual": output["direction_residual"][row, point_index],
        "utility_delta": output["utility_delta"][row, point_index, direction_bin],
    }

    evaluated = batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED)
    evaluated_push = candidate & parameter_valid & evaluated
    positive = evaluated_push & batch["action_improves_state"]
    center_angle = (direction_bin.float() + 0.5) * 2 * math.pi / bins
    center = torch.stack((torch.cos(center_angle), torch.sin(center_angle)), -1)
    action_residual = direction[..., :2] - center

    # Every evaluated contact neighbourhood is supervised. Failed pushes are
    # explicit negatives; untested surface regions remain UNKNOWN.
    contact_target = torch.zeros_like(output["contact_logits"])
    contact_valid = torch.zeros_like(output["contact_logits"], dtype=torch.bool)
    sigma_sq = float(config.contact_heatmap_sigma_m) ** 2
    for batch_row in range(action_type.shape[0]):
        for candidate_index in torch.nonzero(
            evaluated_push[batch_row], as_tuple=False
        ).flatten().tolist():
            object_index = int(batch["acted_object"][batch_row, candidate_index])
            domain = batch["point_mask"][batch_row] & (
                batch["instance_id"][batch_row] == object_index
            )
            delta = (
                batch["xyz"][batch_row]
                - parameters["push_contact_world"][batch_row, candidate_index]
            )
            distance_sq = (delta * delta).sum(-1)
            neighborhood = domain & (distance_sq <= 9.0 * sigma_sq)
            contact_valid[batch_row] |= neighborhood
            if bool(positive[batch_row, candidate_index]):
                contact_target[batch_row] = torch.maximum(
                    contact_target[batch_row],
                    torch.exp(-0.5 * distance_sq / sigma_sq) * neighborhood,
                )

    # Untested active objects are UNKNOWN and never enter the listwise denominator.
    object_positive = torch.zeros_like(batch["object_mask"])
    object_evaluated = torch.zeros_like(batch["object_mask"])
    for batch_row in range(action_type.shape[0]):
        evaluated_objects = batch["acted_object"][batch_row, evaluated_push[batch_row]]
        evaluated_objects = evaluated_objects[evaluated_objects >= 0]
        object_evaluated[batch_row, evaluated_objects] = True
        positive_objects = batch["acted_object"][batch_row, positive[batch_row]]
        positive_objects = positive_objects[positive_objects >= 0]
        object_positive[batch_row, positive_objects] = True

    # Aggregate all evaluated bins at each unique contact point. This makes the
    # direction head a multi-positive ranking problem rather than contradictory CE.
    direction_positive = torch.zeros(
        (*candidate.shape, bins), dtype=torch.bool, device=candidate.device
    )
    direction_evaluated = torch.zeros_like(direction_positive)
    direction_residual_target = output["direction_residual"].new_zeros(
        (*candidate.shape, bins, 2)
    )
    direction_residual_count = output["direction_residual"].new_zeros(
        (*candidate.shape, bins)
    )
    for batch_row in range(action_type.shape[0]):
        canonical: dict[int, int] = {}
        for candidate_index in torch.nonzero(
            evaluated_push[batch_row], as_tuple=False
        ).flatten().tolist():
            point = int(point_index[batch_row, candidate_index])
            slot = canonical.setdefault(point, candidate_index)
            bin_index = int(direction_bin[batch_row, candidate_index])
            direction_evaluated[batch_row, slot, bin_index] = True
            if bool(positive[batch_row, candidate_index]):
                direction_positive[batch_row, slot, bin_index] = True
                direction_residual_target[batch_row, slot, bin_index] += (
                    action_residual[batch_row, candidate_index]
                )
                direction_residual_count[batch_row, slot, bin_index] += 1.0
    direction_residual_valid = direction_residual_count > 0
    direction_residual_target = direction_residual_target / (
        direction_residual_count.clamp_min(1.0).unsqueeze(-1)
    )

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
    has_failure = (torch.nan_to_num(failures) > 0.5).any(-1)
    return gathered, {
        "object_positive": object_positive,
        "object_valid_mask": (
            batch["object_mask"] & batch["object_active"] & object_evaluated
        ),
        "contact_target": contact_target,
        "contact_valid": contact_valid,
        "direction_positive": direction_positive,
        "direction_evaluated": direction_evaluated,
        "direction_residual_target": direction_residual_target,
        "direction_residual_valid": direction_residual_valid,
        "utility_delta": utility,
        "utility_valid": (
            parameter_valid & evaluated & (batch["potential_after_valid"] | has_failure)
        ),
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
