from __future__ import annotations

import torch

from tcd_prg.constants import ActionType
from tcd_prg.models.tcd_prg import TCDPRGModel


def test_shared_push_point_cache_reconstructs_original_evidence_subset():
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    point_mask = torch.tensor([[True, True, True]])
    instance_id = torch.tensor([[0, 0, 1]])
    contacts = torch.tensor([[[0.9, 0.0, 0.0], [0.0, 1.1, 0.0], [0.1, 0.0, 0.0]]])
    directions = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [float("nan"), 0.0, 0.0]]])
    kind = torch.tensor([[int(ActionType.PUSH)] * 3])
    candidate_mask = torch.tensor([[True, True, True]])
    acted_object = torch.tensor([[0, 1, 0]])
    batch = {
        "xyz": xyz,
        "point_mask": point_mask,
        "instance_id": instance_id,
        "candidate_mask": candidate_mask,
        "action_type": kind,
        "acted_object": acted_object,
        "action_parameters": {
            "push_contact_world": contacts,
            "push_direction_world": directions,
        },
    }

    cache = TCDPRGModel._push_candidate_point_cache(batch)
    assert cache is not None
    cached_index, cached_found = cache
    original_valid = candidate_mask & torch.isfinite(directions).all(-1)
    original_index, original_found = TCDPRGModel._nearest_object_point_indices(
        xyz, point_mask, instance_id, acted_object, contacts, original_valid
    )
    reconstructed_found = cached_found & torch.isfinite(directions).all(-1)
    assert torch.equal(reconstructed_found, original_found)
    assert torch.equal(cached_index[original_found], original_index[original_found])
