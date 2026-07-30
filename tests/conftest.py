from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


@pytest.fixture
def dataset_root() -> Path:
    value = os.environ.get("TCD_DATASET_ROOT")
    if not value or not Path(value).exists():
        pytest.skip("Set TCD_DATASET_ROOT for real-data contract tests")
    return Path(value)


@pytest.fixture
def tiny_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    b, n, o = 1, 24, 3
    instance = torch.arange(n)[None] % o
    relation = torch.zeros(b, o, o, 5)
    relation[:, 0, 1, 2] = 1
    return {
        "xyz": torch.randn(b, n, 3),
        "rgb": torch.rand(b, n, 3),
        "instance_id": instance,
        "point_mask": torch.ones(b, n, dtype=torch.bool),
        "target_mask": instance == 1,
        "target_object": torch.tensor([1]),
        "object_pose": torch.cat((torch.randn(b, o, 3), torch.tensor([0, 0, 0, 1.0]).repeat(b, o, 1)), -1),
        "object_mask": torch.ones(b, o, dtype=torch.bool),
        "object_active": torch.ones(b, o, dtype=torch.bool),
        "task_category_id": torch.tensor([1]),
        "task_region_id": torch.tensor([2]),
        "relation_graph": relation,
        "remaining_steps": torch.tensor([5]),
    }
