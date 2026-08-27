"""Stable public conditioning contract for the standalone PUSH module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PushCondition:
    object_probability: Tensor  # [B,Q,N]
    object_valid: Tensor        # [B,Q]
    target_probability: Tensor  # [B,N]
    region_probability: Tensor  # [B,N]
    target_valid: Tensor        # [B]
    task_category_id: Tensor    # [B]
    task_region_id: Tensor      # [B]

    def validate(self, point_count: int) -> "PushCondition":
        b, q, n = self.object_probability.shape
        if n != point_count or self.object_valid.shape != (b, q):
            raise ValueError("PushCondition object tensors have incompatible shapes")
        if self.target_probability.shape != (b, n) or self.region_probability.shape != (b, n):
            raise ValueError("PushCondition point tensors have incompatible shapes")
        if self.target_valid.shape != (b,) or self.task_category_id.shape != (b,) or self.task_region_id.shape != (b,):
            raise ValueError("PushCondition batch tensors have incompatible shapes")
        if not torch.isfinite(self.object_probability).all() or not torch.isfinite(self.target_probability).all() or not torch.isfinite(self.region_probability).all():
            raise ValueError("PushCondition probabilities must be finite")
        for name, value in (("object_probability", self.object_probability), ("target_probability", self.target_probability), ("region_probability", self.region_probability)):
            if bool(((value < -1e-5) | (value > 1.0 + 1e-5)).any()):
                raise ValueError(f"PushCondition {name} must lie in [0, 1]")
        if bool(self.object_probability.masked_select(~self.object_valid[:, :, None]).abs().gt(1e-5).any()):
            raise ValueError("PushCondition invalid object slots must have zero probability")
        if bool((self.region_probability > self.target_probability + 1e-5).any()):
            raise ValueError("PushCondition region_probability must be target constrained")
        return self


def push_condition_from_gt(batch: Mapping[str, Tensor], query_count: int) -> PushCondition:
    """Convert GT perception labels to the sole Stage-C training input contract."""
    point_mask = batch["point_mask"].bool()
    instance_id = batch["instance_id"].long()
    b, n = point_mask.shape
    probability = torch.zeros((b, query_count, n), dtype=batch["xyz"].dtype, device=point_mask.device)
    valid = torch.zeros((b, query_count), dtype=torch.bool, device=point_mask.device)
    object_mask = batch["object_mask"].bool()
    if object_mask.shape[1] > query_count:
        raise ValueError(f"GT object count {object_mask.shape[1]} exceeds PushCondition slot count {query_count}")
    for slot in range(object_mask.shape[1]):
        membership = point_mask & (instance_id == slot)
        probability[:, slot] = membership.to(probability.dtype)
        valid[:, slot] = object_mask[:, slot] & membership.any(-1)
    target = (point_mask & batch["target_mask"].bool()).to(probability.dtype)
    region = (point_mask & batch["target_mask"].bool() & batch["region_valid"].bool() & batch["region_target"].bool()).to(probability.dtype)
    return PushCondition(probability, valid, target, region, target.any(-1), batch["task_category_id"].long(), batch["task_region_id"].long()).validate(n)
