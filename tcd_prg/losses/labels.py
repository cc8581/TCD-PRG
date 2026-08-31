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


def build_region_labels(batch: dict[str, Tensor]) -> dict[str, Tensor] | None:
    if "region_target" not in batch:
        return None
    return {
        "region_target": batch["region_target"],
        "region_valid": batch["region_valid"],
        "visibility_target": batch["visibility_target"],
        "visibility_valid": batch["visibility_valid"],
    }
