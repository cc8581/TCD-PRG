"""Regression tests for the August 2026 training performance hotfix."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from tcd_prg.datasets.collate import grid_sample
from tcd_prg.models.common import MaskedAttentionPool
from tcd_prg.models.tcd_prg import TCDPRGModel


def test_grouped_attention_pool_matches_per_group_calls() -> None:
    torch.manual_seed(7)
    pool = MaskedAttentionPool(8, 8)
    tokens = torch.randn(2, 11, 8)
    mask = torch.rand(2, 4, 11) > 0.35
    query = torch.randn(2, 8)
    expected = torch.stack(
        [pool(tokens, mask[:, group], query) for group in range(mask.shape[1])], dim=1
    )
    actual = pool.forward_grouped(tokens, mask, query)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_grid_sample_returns_selected_original_coordinates() -> None:
    xyz = np.asarray(
        [[0.001, 0.001, 0.001], [0.004, 0.002, 0.003],
         [0.006, 0.001, 0.001], [0.011, 0.001, 0.001]],
        dtype=np.float32,
    )
    indices, grid = grid_sample(xyz, 0.005, training=False)
    expected_grid = np.floor((xyz - xyz.min(0)) / 0.005).astype(np.int64)[indices]
    np.testing.assert_array_equal(grid, expected_grid.astype(np.int32))
    assert len(np.unique(grid, axis=0)) == len(grid)


def test_vectorized_nearest_object_point_indices() -> None:
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]])
    point_mask = torch.ones((1, 4), dtype=torch.bool)
    instance_id = torch.tensor([[0, 0, 1, 1]])
    object_index = torch.tensor([[0, 1, 2]])
    query = torch.tensor([[[0.9, 0.0, 0.0], [0.0, 1.8, 0.0], [0.0, 0.0, 0.0]]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    index, found = TCDPRGModel._nearest_object_point_indices(
        xyz, point_mask, instance_id, object_index, query, valid, chunk_size=2
    )
    assert index.tolist() == [[1, 3, 0]]
    assert found.tolist() == [[True, True, False]]


def test_candidate_evidence_accepts_half_precision_push_outputs() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(num_direction_bins=2),
        _nearest_object_point_indices=TCDPRGModel._nearest_object_point_indices,
    )
    batch = {
        "xyz": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        "point_mask": torch.ones((1, 2), dtype=torch.bool),
        "instance_id": torch.zeros((1, 2), dtype=torch.long),
        "object_mask": torch.ones((1, 1), dtype=torch.bool),
        "candidate_mask": torch.ones((1, 1), dtype=torch.bool),
        "action_parameters": {
            "push_contact_world": torch.tensor([[[0.9, 0.0, 0.0]]]),
            "push_direction_world": torch.tensor([[[1.0, 0.0, 0.0]]]),
        },
    }
    result = {
        "push": {
            "utility_delta": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float16),
            "contact_logits": torch.ones((1, 2), dtype=torch.float16),
            "direction_logits": torch.ones((1, 2, 2), dtype=torch.float16),
        },
        "global_grasp": {
            "translation_world": torch.zeros((1, 1, 3)),
            "quality_logit": torch.zeros((1, 1)),
        },
        "task_grasp": {
            "translation_world": torch.zeros((1, 1, 3)),
            "quality_logit": torch.zeros((1, 1)),
        },
    }
    evidence = TCDPRGModel._candidate_evidence_from_batch(
        model,
        batch,
        result,
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1), dtype=torch.long),
        torch.zeros((1, 1, 7)),
    )
    assert evidence.dtype == torch.float32
    assert evidence[0, 0, 0].item() == 3.0
