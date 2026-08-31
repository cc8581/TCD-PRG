"""Instance/target supervision and GT-object -> predicted-query matching.

GT instance ids, target masks and object categories enter only this loss-side module.
They are never inputs to the perception/model forward path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn



@dataclass(slots=True)
class InstanceMatch:
    gt_to_query: Tensor      # [B,O], -1 for invisible/unmatched GT object
    query_to_gt: Tensor      # [B,Q], -1 for unmatched query
    target_query: Tensor     # [B], -1 if target is not visible/matched
    gt_visible: Tensor       # [B,O]


def build_instance_targets(batch: dict[str, Tensor], query_count: int) -> dict[str, Tensor]:
    instance_id = batch["instance_id"].long()
    point_mask = batch["point_mask"].bool()
    object_category = batch["object_category_id"].long()
    object_mask = batch["object_mask"].bool()
    b, n = instance_id.shape
    o = object_mask.shape[1]
    object_index = torch.arange(o, device=instance_id.device)
    masks = point_mask[:, None] & (instance_id[:, None] == object_index[None, :, None])
    visible = object_mask & masks.any(-1) & (object_category >= 0)
    target_object = batch["target_object"].long()
    safe_target = target_object.clamp(0, max(0, o - 1))
    target_category = object_category.gather(1, safe_target[:, None]).squeeze(1)
    same_category_count = (
        visible & (object_category == target_category[:, None])
    ).sum(-1)
    return {
        "mask": masks,
        "visible": visible,
        "category": object_category,
        "target_object": target_object,
        "target_mask": batch["target_mask"].bool(),
        # Loss-side hard-case diagnostic: >1 means category alone cannot identify
        # the requested physical instance.
        "same_category_target_count": same_category_count,
    }


class InstanceSetLoss(nn.Module):
    """Hungarian set matching with one compact perception loss family."""

    def __init__(
        self,
        matching_points: int = 2048,
        objectness_weight: float = 1.0,
        mask_weight: float = 2.0,
        dice_weight: float = 2.0,
        category_weight: float = 1.0,
        target_weight: float = 1.0,
        same_category_target_weight: float = 2.0,
        auxiliary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.matching_points = int(matching_points)
        self.same_category_target_weight = float(same_category_target_weight)
        self.auxiliary_weight = float(auxiliary_weight)
        if self.same_category_target_weight < 1.0:
            raise ValueError("same_category_target_weight must be >= 1")
        self.weights = {
            "instance_objectness": float(objectness_weight),
            "instance_mask": float(mask_weight),
            "instance_dice": float(dice_weight),
            "instance_category": float(category_weight),
            "target_query": float(target_weight),
        }

    @staticmethod
    def _dice_cost(pred: Tensor, target: Tensor) -> Tensor:
        # pred [Q,P], target [O,P] -> [Q,O]
        intersection = torch.einsum("qp,op->qo", pred, target)
        denominator = pred.sum(-1)[:, None] + target.sum(-1)[None]
        return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

    @torch.no_grad()
    def match(
        self,
        output,
        targets: dict[str, Tensor],
        target_query_logits: Tensor | None = None,
    ) -> InstanceMatch:
        mask_prob = output.mask_probability
        category_logits = output.category_logits
        b, q, n = mask_prob.shape
        o = targets["visible"].shape[1]
        gt_to_query = torch.full((b, o), -1, dtype=torch.long, device=mask_prob.device)
        query_to_gt = torch.full((b, q), -1, dtype=torch.long, device=mask_prob.device)

        costs: list[Tensor] = []
        metadata: list[tuple[int, Tensor]] = []
        for row in range(b):
            gt = torch.nonzero(targets["visible"][row], as_tuple=False).flatten()
            if not len(gt):
                continue
            valid_points = torch.nonzero(
                targets["mask"][row].any(0) | (mask_prob[row].amax(0) > 0.05),
                as_tuple=False,
            ).flatten()
            if not len(valid_points):
                valid_points = torch.arange(n, device=mask_prob.device)
            if len(valid_points) > self.matching_points:
                # Deterministic subset, independent of GT instance identity.
                step = max(1, len(valid_points) // self.matching_points)
                valid_points = valid_points[::step][: self.matching_points]

            pred = mask_prob[row, :, valid_points].float()
            target = targets["mask"][row, gt][:, valid_points].float()
            eps = 1e-6
            # Pairwise BCE without materializing Q*O*P expanded tensors.
            pos = -(target @ torch.log(pred.clamp_min(eps)).T).T
            neg = -((1.0 - target) @ torch.log((1.0 - pred).clamp_min(eps)).T).T
            bce = (pos + neg) / max(1, len(valid_points))
            dice = self._dice_cost(pred, target)
            category_cost = -torch.log_softmax(category_logits[row].float(), -1)[
                :, targets["category"][row, gt]
            ]
            costs.append(
                self.weights["instance_mask"] * bce
                + self.weights["instance_dice"] * dice
                + self.weights["instance_category"] * category_cost
            )
            metadata.append((row, gt))

        # Coalesce all QxO cost transfers into one GPU->CPU synchronization,
        # matching the optimized grasp-set Hungarian implementation.
        if costs:
            shapes = [tuple(int(v) for v in cost.shape) for cost in costs]
            sizes = [int(cost.numel()) for cost in costs]
            flat = torch.cat(
                [cost.detach().float().reshape(-1) for cost in costs], 0
            ).cpu().numpy()
            offset = 0
            for (row, gt), shape, size in zip(metadata, shapes, sizes, strict=True):
                matrix = np.asarray(flat[offset : offset + size]).reshape(shape)
                qi_np, gi_np = linear_sum_assignment(matrix)
                qi = torch.as_tensor(qi_np, dtype=torch.long, device=mask_prob.device)
                gi_local = torch.as_tensor(
                    gi_np, dtype=torch.long, device=mask_prob.device
                )
                gi = gt[gi_local]
                gt_to_query[row, gi] = qi
                query_to_gt[row, qi] = gi
                offset += size

        target_object = targets["target_object"]
        safe_target = target_object.clamp(0, max(0, o - 1))
        target_query = gt_to_query.gather(1, safe_target[:, None]).squeeze(1)
        target_query = torch.where(
            (target_object >= 0) & (target_object < o),
            target_query,
            torch.full_like(target_query, -1),
        )
        return InstanceMatch(gt_to_query, query_to_gt, target_query, targets["visible"])

    def forward(
        self,
        output,
        targets: dict[str, Tensor],
        target_query_logits: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], InstanceMatch]:
        match = self.match(output, targets, target_query_logits)
        b, q, n = output.mask_logits.shape

        objectness_target = (match.query_to_gt >= 0).float()
        objectness = torch.nn.functional.binary_cross_entropy_with_logits(
            output.objectness_logits, objectness_target
        )

        matched = torch.nonzero(match.query_to_gt >= 0, as_tuple=False)
        if len(matched):
            rows, queries = matched[:, 0], matched[:, 1]
            gt = match.query_to_gt[rows, queries]
            pred_logits = output.mask_logits[rows, queries]
            target = targets["mask"][rows, gt].float()
            valid = targets["mask"].new_ones(target.shape, dtype=torch.bool)
            # Background sensor points are valid negatives for each matched instance.
            mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_logits, target
            )
            prob = torch.sigmoid(pred_logits)
            intersection = (prob * target).sum(-1)
            dice_loss = (
                1.0 - (2.0 * intersection + 1.0)
                / (prob.sum(-1) + target.sum(-1) + 1.0)
            ).mean()
            category_loss = torch.nn.functional.cross_entropy(
                output.category_logits[rows, queries],
                targets["category"][rows, gt],
            )
        else:
            zero = output.mask_logits.sum() * 0.0
            mask_loss = dice_loss = category_loss = zero

        ambiguous_rows = targets["same_category_target_count"] > 1
        if target_query_logits is not None:
            valid_target = match.target_query >= 0
            if valid_target.any():
                per_row = torch.nn.functional.cross_entropy(
                    target_query_logits[valid_target],
                    match.target_query[valid_target],
                    reduction="none",
                )
                row_weight = torch.where(
                    ambiguous_rows[valid_target],
                    per_row.new_full(per_row.shape, self.same_category_target_weight),
                    torch.ones_like(per_row),
                )
                target_loss = (per_row * row_weight).sum() / row_weight.sum().clamp_min(1.0)
            else:
                target_loss = target_query_logits.sum() * 0.0
        else:
            target_loss = output.mask_logits.sum() * 0.0

        values = {
            "instance_objectness": objectness,
            "instance_mask": mask_loss,
            "instance_dice": dice_loss,
            "instance_category": category_loss,
            "target_query": target_loss,
        }
        numerator = sum(self.weights[k] * v for k, v in values.items())
        denominator = sum(self.weights.values())
        final_loss = numerator / denominator

        # Deep supervision for the stronger query decoder. Hungarian assignment is
        # fixed from the final decoder output; auxiliary layers learn the same
        # instance decomposition instead of running additional CPU matchers.
        auxiliary_terms = []
        for auxiliary in getattr(output, "aux_outputs", ()):
            aux_objectness = torch.nn.functional.binary_cross_entropy_with_logits(
                auxiliary.objectness_logits, objectness_target
            )
            if len(matched):
                aux_logits = auxiliary.mask_logits[rows, queries]
                aux_mask = torch.nn.functional.binary_cross_entropy_with_logits(
                    aux_logits, target
                )
                aux_prob = torch.sigmoid(aux_logits)
                aux_intersection = (aux_prob * target).sum(-1)
                aux_dice = (
                    1.0 - (2.0 * aux_intersection + 1.0)
                    / (aux_prob.sum(-1) + target.sum(-1) + 1.0)
                ).mean()
                aux_category = torch.nn.functional.cross_entropy(
                    auxiliary.category_logits[rows, queries],
                    targets["category"][rows, gt],
                )
            else:
                zero = auxiliary.mask_logits.sum() * 0.0
                aux_mask = aux_dice = aux_category = zero
            auxiliary_terms.append(
                (
                    self.weights["instance_objectness"] * aux_objectness
                    + self.weights["instance_mask"] * aux_mask
                    + self.weights["instance_dice"] * aux_dice
                    + self.weights["instance_category"] * aux_category
                )
                / (
                    self.weights["instance_objectness"]
                    + self.weights["instance_mask"]
                    + self.weights["instance_dice"]
                    + self.weights["instance_category"]
                )
            )
        auxiliary_loss = (
            torch.stack(auxiliary_terms).mean()
            if auxiliary_terms
            else final_loss.detach() * 0.0
        )
        loss = final_loss + self.auxiliary_weight * auxiliary_loss
        return {
            "loss": loss,
            **values,
            "instance_auxiliary": auxiliary_loss,
            "target_ambiguous_rows": ambiguous_rows.float().sum(),
        }, match


def map_gt_object_indices(indices: Tensor, match: InstanceMatch) -> tuple[Tensor, Tensor]:
    """Map arbitrary [...]-shaped per-row GT object ids to predicted query ids."""
    b = indices.shape[0]
    o = match.gt_to_query.shape[1]
    valid = (indices >= 0) & (indices < o)
    safe = indices.clamp(0, max(0, o - 1))
    mapped = match.gt_to_query.gather(1, safe.reshape(b, -1)).reshape_as(indices)
    valid &= mapped >= 0
    return mapped, valid


def gather_query_logits_to_gt_objects(
    query_logits: Tensor,
    match: InstanceMatch,
) -> tuple[Tensor, Tensor]:
    """Gather [B,Q] prediction logits into the GT object axis [B,O]."""
    b, q = query_logits.shape
    mapping = match.gt_to_query
    valid = mapping >= 0
    safe = mapping.clamp(0, max(0, q - 1))
    gathered = query_logits.gather(1, safe)
    gathered = gathered.masked_fill(~valid, -30.0)
    return gathered, valid
