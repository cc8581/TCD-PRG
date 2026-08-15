"""Object-local ACRONYM grasp priors and open-world proposal matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tcd_prg.constants import CandidateStatus
from tcd_prg.geometry.se3 import parallel_jaw_rotation_distance


ACRONYM_DATABASE_FORMAT = "tcd_prg_acronym_object_grasps_v1"


def load_object_grasps(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if str(source["format"]) != ACRONYM_DATABASE_FORMAT:
            raise ValueError("ACRONYM object grasp database format is incompatible")
        result = {key: np.asarray(source[key]) for key in source.files}
    count = len(result["status"])
    if result["translation_object"].shape != (count, 3):
        raise ValueError("translation_object must be [N,3]")
    if result["rotation_object"].shape != (count, 3, 3):
        raise ValueError("rotation_object must be [N,3,3]")
    if not np.isin(
        result["status"],
        (int(CandidateStatus.POSITIVE), int(CandidateStatus.NEGATIVE)),
    ).all():
        raise ValueError("Every ACRONYM database row must be explicitly evaluated")
    return result


@torch.no_grad()
def match_object_grasp_priors(
    proposal_translation: Tensor,
    proposal_rotation: Tensor,
    proposal_valid: Tensor,
    database_translation: Tensor,
    database_rotation: Tensor,
    database_status: Tensor,
    *,
    translation_m: float = 0.02,
    rotation_deg: float = 20.0,
    chunk_size: int = 512,
) -> dict[str, Tensor]:
    """Assign POSITIVE/NEGATIVE/UNKNOWN without inventing uncovered negatives.

    Explicit negative matches take precedence.  A proposal that lies inside
    both class neighborhoods is NEGATIVE and is also recorded by
    ``match_conflict`` for diagnostics. Width
    is deliberately excluded from identity because the ACRONYM gripper and
    AG-160-95 have different opening conventions.
    """

    count = len(proposal_translation)
    status = torch.full(
        (count,), int(CandidateStatus.UNKNOWN_UNTESTED), dtype=torch.int8,
        device=proposal_translation.device,
    )
    matched_index = torch.full((count,), -1, dtype=torch.long, device=status.device)
    diagnostic = dict(
        size=(count,), fill_value=float("inf"), dtype=torch.float32,
        device=proposal_translation.device,
    )
    nearest_translation = torch.full(**diagnostic)
    nearest_rotation = torch.full(**diagnostic)
    positive_min_cost = torch.full(**diagnostic)
    negative_min_cost = torch.full(**diagnostic)
    positive_translation = torch.full(**diagnostic)
    positive_rotation = torch.full(**diagnostic)
    negative_translation = torch.full(**diagnostic)
    negative_rotation = torch.full(**diagnostic)
    conflict = torch.zeros(count, dtype=torch.bool, device=status.device)
    proposals = torch.nonzero(proposal_valid.bool(), as_tuple=False).flatten()
    for begin in range(0, len(proposals), chunk_size):
        rows = proposals[begin : begin + chunk_size]
        translation = torch.cdist(
            proposal_translation[rows].float(), database_translation.float()
        )
        rotation = torch.rad2deg(
            parallel_jaw_rotation_distance(
                proposal_rotation[rows, None].float(), database_rotation[None].float()
            )
        )
        nearby = (translation <= translation_m) & (rotation <= rotation_deg)
        positive_class = database_status[None] == int(CandidateStatus.POSITIVE)
        negative_class = database_status[None] == int(CandidateStatus.NEGATIVE)
        normalized_cost = translation / translation_m + rotation / rotation_deg
        pos_cost, pos_index = normalized_cost.masked_fill(~positive_class, float("inf")).min(-1)
        neg_cost, neg_index = normalized_cost.masked_fill(~negative_class, float("inf")).min(-1)
        positive_min_cost[rows], negative_min_cost[rows] = pos_cost, neg_cost
        positive_translation[rows] = translation.gather(1, pos_index[:, None]).squeeze(1)
        positive_rotation[rows] = rotation.gather(1, pos_index[:, None]).squeeze(1)
        negative_translation[rows] = translation.gather(1, neg_index[:, None]).squeeze(1)
        negative_rotation[rows] = rotation.gather(1, neg_index[:, None]).squeeze(1)
        positive = nearby & positive_class
        negative = nearby & negative_class
        has_positive, has_negative = positive.any(-1), negative.any(-1)
        conflict[rows] = has_positive & has_negative
        known_negative = has_negative
        known_positive = has_positive & ~has_negative
        row_status = status[rows]
        row_status[known_positive] = int(CandidateStatus.POSITIVE)
        row_status[known_negative] = int(CandidateStatus.NEGATIVE)
        status[rows] = row_status
        known = known_positive | known_negative
        if known.any():
            selected_rows = torch.nonzero(known, as_tuple=False).flatten()
            selected_status = row_status[selected_rows]
            same_class = database_status[None] == selected_status[:, None]
            cost = (
                translation[selected_rows] / translation_m
                + rotation[selected_rows] / rotation_deg
            ).masked_fill(~(nearby[selected_rows] & same_class), float("inf"))
            best = cost.argmin(-1)
            output_rows = rows[selected_rows]
            matched_index[output_rows] = best
            nearest_translation[output_rows] = translation[selected_rows, best]
            nearest_rotation[output_rows] = rotation[selected_rows, best]
    return {
        "status": status,
        "matched_index": matched_index,
        "translation_error_m": nearest_translation,
        "rotation_error_deg": nearest_rotation,
        "match_conflict": conflict,
        "known_mask": status != int(CandidateStatus.UNKNOWN_UNTESTED),
        "positive_min_cost": positive_min_cost,
        "positive_translation_error_m": positive_translation,
        "positive_rotation_error_deg": positive_rotation,
        "negative_min_cost": negative_min_cost,
        "negative_translation_error_m": negative_translation,
        "negative_rotation_error_deg": negative_rotation,
    }
