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


def remap_graph_targets(
    physical: Tensor,
    task: Tensor,
    match: InstanceMatch,
    query_count: int,
) -> dict[str, Tensor]:
    """Scatter GT O-space graph targets into predicted Q-space."""
    b = physical.shape[0]
    r_phys = physical.shape[-1]
    r_task = task.shape[-1]
    physical_target = physical.new_zeros((b, query_count, query_count, r_phys))
    physical_valid = torch.zeros_like(physical_target, dtype=torch.bool)
    task_target = task.new_zeros((b, query_count, r_task))
    task_valid = torch.zeros_like(task_target, dtype=torch.bool)
    for row in range(b):
        gt = torch.nonzero(match.gt_to_query[row] >= 0, as_tuple=False).flatten()
        if not len(gt):
            continue
        q = match.gt_to_query[row, gt]
        task_target[row, q] = task[row, gt]
        task_valid[row, q] = True
        # Small O/Q (<= ~32); explicit indexed assignment is clearer and avoids
        # duplicate scatter semantics.
        for gi, qi in zip(gt.tolist(), q.tolist(), strict=True):
            for gj, qj in zip(gt.tolist(), q.tolist(), strict=True):
                physical_target[row, qi, qj] = physical[row, gi, gj]
                physical_valid[row, qi, qj] = True
    return {
        "physical_edge_target": physical_target,
        "physical_edge_valid": physical_valid,
        "task_edge_target": task_target,
        "task_edge_valid": task_valid,
    }


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


def _nearest_selected_push_points(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    candidate: Tensor,
    *,
    max_distance_m: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Loss-side association from GT PUSH contacts to prediction-selected points.

    The Push head never receives GT contact points.  This function instead finds
    the closest direction-computed sensor point on the GT acted object.  If no
    such predicted point exists, direction/utility supervision is ignored for
    that candidate while object/contact supervision remains valid.
    """
    contacts = batch["action_parameters"]["push_contact_world"]
    acted = batch["acted_object"]
    xyz = batch["xyz"]
    point_mask = batch["point_mask"]
    gt_instance = batch["instance_id"]
    selected = output["direction_point_mask"].bool()
    b, k = candidate.shape
    index = torch.zeros((b, k), dtype=torch.long, device=xyz.device)
    valid = candidate.clone()
    for row in range(b):
        for ci in torch.nonzero(candidate[row], as_tuple=False).flatten().tolist():
            object_id = int(acted[row, ci])
            domain = (
                point_mask[row]
                & selected[row]
                & (gt_instance[row] == object_id)
            )
            points = torch.nonzero(domain, as_tuple=False).flatten()
            if not len(points) or not torch.isfinite(contacts[row, ci]).all():
                valid[row, ci] = False
                continue
            distance = torch.linalg.vector_norm(
                xyz[row, points] - contacts[row, ci], dim=-1
            )
            nearest_distance, local = distance.min(0)
            if (
                max_distance_m is not None
                and bool(nearest_distance > max_distance_m)
            ):
                valid[row, ci] = False
                continue
            index[row, ci] = points[local]
    return index, valid


def build_object_query_push_supervision(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    config,
    match: InstanceMatch,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build PUSH supervision without feeding GT object ids/contacts into PushHead."""
    import math
    from tcd_prg.constants import ActionType, CandidateStatus

    action_type = batch["action_type"]
    parameters = batch["action_parameters"]
    candidate = (
        batch["candidate_mask"]
        & (action_type == int(ActionType.PUSH))
    )

    # Contact/object heads remain dense and can use every evaluated GT action.
    # Direction/utility heads are supervised only where the prediction-only
    # top-k selection actually computed a direction token.
    point_index, direction_parameter_valid = _nearest_selected_push_points(
        output, batch, candidate
    )

    row = torch.arange(
        action_type.shape[0], device=action_type.device
    )[:, None]
    direction = torch.nn.functional.normalize(
        torch.nan_to_num(parameters["push_direction_world"]),
        dim=-1,
    )
    angle = torch.atan2(
        direction[..., 1], direction[..., 0]
    ).remainder(2 * math.pi)
    bins = output["direction_logits"].shape[-1]
    direction_bin = torch.floor(
        angle * bins / (2 * math.pi)
    ).long().remainder(bins)

    gt_object_logits, gt_object_predicted = (
        gather_query_logits_to_gt_objects(
            output["object_logits"], match
        )
    )
    gathered = {
        "object_logits": gt_object_logits,
        "contact_logits": output["contact_logits"],
        "point_index": point_index,
        "direction_logits": output["direction_logits"][
            row, point_index
        ],
        "direction_residual": output["direction_residual"][
            row, point_index
        ],
        "utility_delta": output["utility_delta"][
            row, point_index, direction_bin
        ],
    }

    evaluated = (
        batch["evaluation_status"]
        != int(CandidateStatus.UNKNOWN_UNTESTED)
    )
    # Contact parameters themselves only need finite geometry and a visible GT
    # object point; this is entirely loss-side supervision.
    contact_parameter_valid = (
        candidate
        & torch.isfinite(
            parameters["push_contact_world"]
        ).all(-1)
    )
    evaluated_push = (
        candidate & contact_parameter_valid & evaluated
    )
    positive = (
        evaluated_push & batch["action_improves_state"]
    )

    center_angle = (
        direction_bin.float() + 0.5
    ) * 2 * math.pi / bins
    center = torch.stack(
        (torch.cos(center_angle), torch.sin(center_angle)), -1
    )
    action_residual = direction[..., :2] - center

    contact_target = torch.zeros_like(
        output["contact_logits"]
    )
    contact_valid = torch.zeros_like(
        output["contact_logits"], dtype=torch.bool
    )
    sigma_sq = float(config.contact_heatmap_sigma_m) ** 2
    for batch_row in range(action_type.shape[0]):
        for candidate_index in torch.nonzero(
            evaluated_push[batch_row], as_tuple=False
        ).flatten().tolist():
            object_index = int(
                batch["acted_object"][
                    batch_row, candidate_index
                ]
            )
            domain = (
                batch["point_mask"][batch_row]
                & (
                    batch["instance_id"][batch_row]
                    == object_index
                )
            )
            delta = (
                batch["xyz"][batch_row]
                - parameters["push_contact_world"][
                    batch_row, candidate_index
                ]
            )
            distance_sq = (delta * delta).sum(-1)
            neighborhood = (
                domain & (distance_sq <= 9.0 * sigma_sq)
            )
            contact_valid[batch_row] |= neighborhood
            if bool(positive[batch_row, candidate_index]):
                contact_target[batch_row] = torch.maximum(
                    contact_target[batch_row],
                    torch.exp(
                        -0.5 * distance_sq / sigma_sq
                    ) * neighborhood,
                )

    object_positive = torch.zeros_like(
        gt_object_logits, dtype=torch.bool
    )
    object_evaluated = torch.zeros_like(
        gt_object_logits, dtype=torch.bool
    )
    # Map each evaluated GT acted-object to its matched query, then gather back
    # to GT axis by using the GT-space logits above.  Existing PushLoss therefore
    # keeps its stable listwise semantics.
    gt_object_count = match.gt_to_query.shape[1]
    for batch_row in range(action_type.shape[0]):
        eval_objects = batch["acted_object"][
            batch_row, evaluated_push[batch_row]
        ]
        eval_objects = eval_objects[
            (eval_objects >= 0)
            & (eval_objects < gt_object_count)
        ]
        if len(eval_objects):
            object_evaluated[
                batch_row, torch.unique(eval_objects)
            ] = True
        pos_objects = batch["acted_object"][
            batch_row, positive[batch_row]
        ]
        pos_objects = pos_objects[
            (pos_objects >= 0)
            & (pos_objects < gt_object_count)
        ]
        if len(pos_objects):
            object_positive[
                batch_row, torch.unique(pos_objects)
            ] = True

    direction_positive = torch.zeros(
        (*candidate.shape, bins),
        dtype=torch.bool,
        device=candidate.device,
    )
    direction_evaluated = torch.zeros_like(
        direction_positive
    )
    direction_residual_target = (
        output["direction_residual"].new_zeros(
            (*candidate.shape, bins, 2)
        )
    )
    direction_residual_count = (
        output["direction_residual"].new_zeros(
            (*candidate.shape, bins)
        )
    )
    direction_eval_push = (
        evaluated_push & direction_parameter_valid
    )
    for batch_row in range(action_type.shape[0]):
        canonical: dict[int, int] = {}
        for candidate_index in torch.nonzero(
            direction_eval_push[batch_row],
            as_tuple=False,
        ).flatten().tolist():
            point = int(
                point_index[batch_row, candidate_index]
            )
            slot = canonical.setdefault(
                point, candidate_index
            )
            bin_index = int(
                direction_bin[
                    batch_row, candidate_index
                ]
            )
            direction_evaluated[
                batch_row, slot, bin_index
            ] = True
            if bool(
                positive[batch_row, candidate_index]
            ):
                direction_positive[
                    batch_row, slot, bin_index
                ] = True
                direction_residual_target[
                    batch_row, slot, bin_index
                ] += action_residual[
                    batch_row, candidate_index
                ]
                direction_residual_count[
                    batch_row, slot, bin_index
                ] += 1.0

    direction_residual_valid = (
        direction_residual_count > 0
    )
    direction_residual_target = (
        direction_residual_target
        / direction_residual_count.clamp_min(
            1.0
        ).unsqueeze(-1)
    )

    component_weights = torch.as_tensor(
        config.push_utility_component_weights,
        dtype=batch["potential_delta"].dtype,
        device=batch["potential_delta"].device,
    )
    utility = (
        torch.nan_to_num(batch["potential_delta"])
        * component_weights
    ).sum(-1)
    failures = torch.stack(
        (
            parameters["risk_unstable"],
            parameters["risk_out_of_workspace"],
            parameters["risk_other_invalid"],
        ),
        -1,
    )
    penalties = torch.as_tensor(
        config.push_failure_penalties,
        dtype=utility.dtype,
        device=utility.device,
    )
    utility = utility - (
        torch.nan_to_num(failures) * penalties
    ).sum(-1)
    has_failure = (
        torch.nan_to_num(failures) > 0.5
    ).any(-1)

    # Only GT objects that have a matched predicted query participate in object
    # ranking. Active/present is supervision validity, not forward input.
    gt_active = (
        batch["object_mask"]
        & batch["object_active"]
        & gt_object_predicted
    )
    return gathered, {
        "object_positive": object_positive,
        "object_valid_mask": (
            gt_active & object_evaluated
        ),
        "contact_target": contact_target,
        "contact_valid": contact_valid,
        "direction_positive": direction_positive,
        "direction_evaluated": direction_evaluated,
        "direction_valid": direction_eval_push,
        "direction_bin": direction_bin,
        "direction_residual_target": (
            direction_residual_target
        ),
        "direction_residual_valid": (
            direction_residual_valid
        ),
        "utility_delta": utility,
        "utility_valid": (
            direction_parameter_valid
            & evaluated
            & (
                batch["potential_after_valid"]
                | has_failure
            )
        ),
    }
