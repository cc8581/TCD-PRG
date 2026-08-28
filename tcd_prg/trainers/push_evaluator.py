"""Independent training primitives for the complete-action PUSH evaluator."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tcd_prg.constants import PUSH_DISTANCE_M, ActionType, CandidateStatus
from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.models import push_condition_from_gt
from tcd_prg.models.push.evaluator import nearest_object_contact_point


def freeze_push_proposal(model: nn.Module) -> None:
    """Freeze Stage C while leaving only the action evaluator trainable."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    evaluator = model.push_evaluator
    evaluator.train()
    for parameter in evaluator.parameters():
        parameter.requires_grad_(True)


def push_effectiveness_eligibility(
    batch: dict,
    condition,
    *,
    max_contact_distance_m: float,
) -> tuple[Tensor, Tensor]:
    """Select evaluated PUSH actions Stage C can represent from the observed cloud."""
    if max_contact_distance_m <= 0:
        raise ValueError("max_contact_distance_m must be positive")
    if condition.object_valid.shape[1] == 0:
        acted = batch["acted_object"].long()
        return torch.zeros_like(acted, dtype=torch.bool), torch.full_like(acted, -1)

    parameters = batch["action_parameters"]
    contacts = parameters["push_contact_world"]
    directions = parameters["push_direction_world"]
    acted = batch["acted_object"].long()
    in_range = (acted >= 0) & (acted < condition.object_valid.shape[1])
    safe_object = acted.clamp(0, condition.object_valid.shape[1] - 1)
    represented_object = in_range & condition.object_valid.gather(1, safe_object)
    finite_action = torch.isfinite(contacts).all(-1) & torch.isfinite(directions).all(-1)
    planar_norm = torch.linalg.vector_norm(
        torch.nan_to_num(directions[..., :2], nan=0.0), dim=-1
    )
    eligible = (
        batch["candidate_mask"].bool()
        & (batch["action_type"] == int(ActionType.PUSH))
        & (batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED))
        & condition.target_valid[:, None]
        & represented_object
        & finite_action
        & (planar_norm > 1e-8)
    )
    anchor_index = torch.full_like(acted, -1)
    for row, action_index in torch.nonzero(eligible, as_tuple=False).tolist():
        object_index = int(acted[row, action_index])
        try:
            anchor = nearest_object_contact_point(
                batch["xyz"][row], batch["point_mask"][row],
                condition.object_probability[row], object_index,
                contacts[row, action_index],
            )
        except ValueError:
            eligible[row, action_index] = False
            continue
        distance = torch.linalg.vector_norm(
            batch["xyz"][row, anchor] - contacts[row, action_index]
        )
        if not bool(torch.isfinite(distance)) or float(distance) > max_contact_distance_m:
            eligible[row, action_index] = False
            continue
        anchor_index[row, action_index] = anchor
    return eligible, anchor_index


def push_effectiveness_batch_loss(
    model: nn.Module,
    batch: dict,
    *,
    instance_queries: int,
    loss_function: PushEffectivenessLoss,
    max_contact_distance_m: float = 0.024,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Supervise only deployment-representable logged PUSH actions."""
    condition = push_condition_from_gt(batch, instance_queries)
    valid, anchor_index = push_effectiveness_eligibility(
        batch, condition, max_contact_distance_m=max_contact_distance_m
    )
    selected = torch.nonzero(valid, as_tuple=False)
    if not len(selected):
        zero = next(model.push_evaluator.parameters()).sum() * 0.0
        return zero, {
            "push_effectiveness": zero,
            "effective_logit": zero.new_empty(0),
            "effective_target": torch.empty(0, dtype=torch.bool, device=zero.device),
            "effective_group_index": torch.empty(0, dtype=torch.long, device=zero.device),
        }

    sensor = batch.get("model_inputs", batch)
    contacts = batch["action_parameters"]["push_contact_world"][valid]
    acted_objects = batch["acted_object"][valid].long()
    rows = selected[:, 0]
    anchors = anchor_index[valid].long()
    forced = torch.zeros_like(sensor["point_mask"], dtype=torch.bool)
    forced[rows, anchors] = True

    model_batch = dict(batch)
    model_batch["push_condition"] = condition
    model_batch["training_hints"] = {"push_direction_point_mask": forced}
    with torch.no_grad():
        output = model(model_batch, forward_mode="push")
    logits = model.push_evaluator.score_exact_actions(
        output["sensor"],
        output["push"],
        batch_index=rows,
        acted_object=acted_objects,
        contact_world=contacts,
        direction_world=batch["action_parameters"]["push_direction_world"][valid],
        push_distance=contacts.new_full((len(contacts),), PUSH_DISTANCE_M),
        point_index=anchors,
    )
    losses = loss_function(
        logits,
        batch["evaluation_status"][valid],
        batch["action_improves_state"][valid],
    )
    return losses["push_effectiveness"], {
        **losses,
        "effective_logit": logits,
        "effective_target": batch["action_improves_state"][valid].bool(),
        "effective_group_index": rows.long(),
    }
