"""Public, checkpoint-independent boundary between perception and task grasping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class StageBCondition:
    target_probability: Tensor
    region_probability: Tensor
    target_valid: Tensor
    task_category_id: Tensor
    task_region_id: Tensor

    def validate(self, point_count: int | None = None) -> "StageBCondition":
        if self.target_probability.shape != self.region_probability.shape:
            raise ValueError("Stage-B target and region probabilities must have equal shape")
        if self.target_probability.ndim != 2:
            raise ValueError("Stage-B point probabilities must have shape [B,N]")
        batch, points = self.target_probability.shape
        if point_count is not None and points != point_count:
            raise ValueError("Stage-B condition point axis does not match the fused scene")
        for value, name in (
            (self.target_valid, "target_valid"),
            (self.task_category_id, "task_category_id"),
            (self.task_region_id, "task_region_id"),
        ):
            if value.shape != (batch,):
                raise ValueError(f"Stage-B {name} must have shape [B]")
        if self.target_valid.dtype != torch.bool:
            raise TypeError("Stage-B target_valid must be bool")
        for probability, name in (
            (self.target_probability, "target_probability"),
            (self.region_probability, "region_probability"),
        ):
            if not probability.is_floating_point():
                raise TypeError(f"Stage-B {name} must be floating point")
            if not bool(((probability >= 0) & (probability <= 1)).all()):
                raise ValueError(f"Stage-B {name} must be in [0,1]")
        return self


def stageb_condition_from_gt(batch: Mapping[str, Tensor]) -> StageBCondition:
    """Convert label-side masks to the same public probability contract."""
    point_mask = batch["point_mask"].bool()
    target = point_mask & batch["target_mask"].bool()
    region = (
        target
        & batch["region_valid"].bool()
        & batch["region_target"].bool()
    )
    return StageBCondition(
        target_probability=target.float(),
        region_probability=region.float(),
        target_valid=target.any(-1),
        task_category_id=batch["task_category_id"].long(),
        task_region_id=batch["task_region_id"].long(),
    ).validate(point_mask.shape[1])
