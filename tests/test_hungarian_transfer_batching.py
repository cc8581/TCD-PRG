from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from tcd_prg.losses.proposal import CompleteGraspSetLoss


def test_batched_hungarian_is_identical_to_independent_scipy():
    generator = torch.Generator().manual_seed(20260808)
    costs = [
        torch.rand((32, 3), generator=generator),
        torch.rand((17, 11), generator=generator),
        torch.rand((8, 8), generator=generator),
        torch.rand((31, 2), generator=generator),
    ]
    actual = CompleteGraspSetLoss._hungarian_many(costs, torch.device("cpu"))
    for cost, (row, column) in zip(costs, actual, strict=True):
        expected_row, expected_column = linear_sum_assignment(
            np.asarray(cost.detach().float().cpu())
        )
        assert np.array_equal(row.cpu().numpy(), expected_row)
        assert np.array_equal(column.cpu().numpy(), expected_column)


def test_single_hungarian_wrapper_preserves_api():
    cost = torch.tensor([[3.0, 1.0], [0.5, 4.0]])
    row, column = CompleteGraspSetLoss._hungarian(cost, torch.device("cpu"))
    expected_row, expected_column = linear_sum_assignment(cost.numpy())
    assert np.array_equal(row.numpy(), expected_row)
    assert np.array_equal(column.numpy(), expected_column)
