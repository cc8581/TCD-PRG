"""Build supervision tensors from unified state candidate groups."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tcd_prg.constants import ActionType, CandidateStatus


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


@torch.no_grad()
def build_push_training_hints(batch: dict[str, Tensor], max_distance_m: float | None = None) -> dict[str, Tensor]:
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
        nearest_distance = distance_sq.gather(1, nearest[:, None]).sqrt().squeeze(1)
        valid = domain.any(-1)
        if max_distance_m is not None:
            valid &= nearest_distance <= float(max_distance_m)
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
