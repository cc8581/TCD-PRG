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
    """Build sparse task-positive and explicit wrong-region grasp sets."""

    parameters = batch["action_parameters"]
    candidate = batch["candidate_mask"] & (batch["action_type"] == int(ActionType.TASK_GRASP))
    pose = parameters["task_grasp_pose_world"]
    width = parameters["grasp_width_m"]
    # verifier 字段为硬契约：缺失时必须报错，而不是静默退化成
    # action_improves_state 语义或全 1 质量目标。
    overall_valid = parameters["verifier_overall_valid"].bool()
    overall = parameters["verifier_overall_target"].float()
    successful = torch.where(overall_valid, overall > 0.5, batch["action_improves_state"])
    quality = torch.where(
        overall_valid, torch.nan_to_num(overall),
        torch.nan_to_num(parameters["grasp_confidence"], nan=1.0),
    ).clamp(0.0, 1.0)
    task_compatibility_valid = parameters.get(
        "verifier_task_compatibility_valid", torch.zeros_like(candidate)
    ).bool()
    task_compatibility_target = parameters.get(
        "verifier_task_compatibility_target", torch.zeros_like(width)
    )
    collision_valid = parameters.get(
        "verifier_collision_valid", torch.zeros_like(candidate)
    ).bool()
    collision_target = parameters.get(
        "verifier_collision_target", torch.zeros_like(width)
    )
    approach_valid = parameters.get(
        "verifier_approach_valid", torch.zeros_like(candidate)
    ).bool()
    approach_target = parameters.get(
        "verifier_approach_target", torch.zeros_like(width)
    )
    collision_diverted = (
        candidate
        & collision_valid
        & (collision_target > 0.5)
    )
    approach_diverted = (
        candidate
        & approach_valid
        & (approach_target < 0.5)
    )
    wrong_region = (
        candidate
        & (batch["evaluation_status"] == int(CandidateStatus.NEGATIVE))
        & torch.isfinite(pose).all(-1)
        & task_compatibility_valid
        & (task_compatibility_target < 0.5)
        & ~collision_diverted
        & ~approach_diverted
    )
    # Physical success alone is insufficient when an explicit task-incompatible
    # annotation exists. Keep legacy positives without the optional task field,
    # but make all three routed label sets mutually exclusive.
    valid = (
        candidate
        & successful
        & torch.isfinite(pose).all(-1)
        & ~wrong_region
        & ~collision_diverted
        & ~approach_diverted
    )
    # 不再通过“已采样候选均已评价”推断完备性，只信任数据生产者的显式字段。
    label_set_complete = batch.get(
        "task_grasp_label_set_complete",
        torch.zeros(pose.shape[0], dtype=torch.bool, device=pose.device),
    ).bool()
    labels = _pack_grasp_set(
        pose, width, valid, quality, torch.ones_like(candidate), config.task_grasp_candidates
    )
    _attach_negative_grasp_set(labels, pose, width, wrong_region)
    labels["wrong_region_translation_world"] = labels.pop(
        "negative_translation_world"
    )
    labels["wrong_region_rotation_matrix"] = labels.pop(
        "negative_rotation_matrix"
    )
    labels["wrong_region_width_m"] = labels.pop("negative_width_m")
    labels["wrong_region_valid"] = labels.pop("negative_valid")
    labels["width_valid"] = (
        labels["target_valid"]
        & torch.isfinite(labels["width_m"])
        & (labels["width_m"] >= config.min_grasp_width_m)
        & (labels["width_m"] <= config.max_grasp_width_m)
    )
    # Physical infeasibility is handled by the exact execution certifier,
    # so these examples are excluded from task-suitability ranking.
    labels["collision_excluded_from_task_score"] = collision_diverted.sum(-1)
    labels["approach_excluded_from_task_score"] = approach_diverted.sum(-1)
    labels["label_set_complete"] = label_set_complete
    labels["sample_valid"] = valid.any(-1) | wrong_region.any(-1)
    labels["unmatched_quality_valid"] = torch.zeros_like(label_set_complete)
    return labels



@torch.no_grad()
def build_push_training_hints(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map only evaluated GT PUSH contacts to visible points once per batch.

    The model receives only the union mask used to compute sparse Direction tokens.
    The candidate->point index remains loss-side and prevents a second nearest-point
    search in ``build_object_query_push_supervision``.
    """
    xyz = batch["xyz"]
    point_mask = batch["point_mask"].bool()
    instance_id = batch["instance_id"].long()
    contact = batch["action_parameters"]["push_contact_world"]
    acted_object = batch["acted_object"].long()
    evaluated = (
        batch["candidate_mask"].bool()
        & (batch["action_type"] == int(ActionType.PUSH))
        & (
            batch["evaluation_status"]
            != int(CandidateStatus.UNKNOWN_UNTESTED)
        )
        & torch.isfinite(contact).all(-1)
        & (acted_object >= 0)
    )
    point_index = torch.zeros_like(acted_object)
    point_valid = torch.zeros_like(evaluated)
    forced = torch.zeros_like(point_mask)

    for row in range(xyz.shape[0]):
        candidate_index = torch.nonzero(
            evaluated[row], as_tuple=False
        ).flatten()
        if candidate_index.numel() == 0:
            continue
        contacts = contact[row, candidate_index]
        objects = acted_object[row, candidate_index]
        domain = (
            point_mask[row][None]
            & (instance_id[row][None] == objects[:, None])
        )
        distance_sq = (
            xyz[row][None] - contacts[:, None]
        ).square().sum(-1)
        distance_sq = distance_sq.masked_fill(~domain, float("inf"))
        nearest = distance_sq.argmin(-1)
        valid = domain.any(-1)
        selected_candidates = candidate_index[valid]
        selected_points = nearest[valid]
        point_index[row, selected_candidates] = selected_points
        point_valid[row, selected_candidates] = True
        forced[row, selected_points] = True

    return {
        "push_direction_point_mask": forced,
        "push_gt_point_index": point_index,
        "push_gt_point_valid": point_valid,
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
