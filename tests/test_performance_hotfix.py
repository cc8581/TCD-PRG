"""Regression tests for the August 2026 training performance hotfix."""

from __future__ import annotations

import numpy as np
import torch

from tcd_prg.datasets.collate import grid_sample
from tcd_prg.models.common import MaskedAttentionPool


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
