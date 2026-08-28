"""Independent training primitives for the complete-action PUSH evaluator."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tcd_prg.constants import PUSH_DISTANCE_M, ActionType, CandidateStatus
from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.models import push_condition_from_gt


def freeze_push_proposal(model: nn.Module) -> None:
    """Freeze Stage C while leaving only the action evaluator trainable."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    evaluator = model.push_evaluator
    evaluator.train()
    for parameter in evaluator.parameters():
        parameter.requires_grad_(True)


def push_effectiveness_batch_loss(
    model: nn.Module,
    batch: dict,
    *,
    instance_queries: int,
    loss_function: PushEffectivenessLoss,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Encode frozen Stage C and supervise logged exact PUSH actions only."""
    valid = (
        batch["candidate_mask"].bool()
        & (batch["action_type"] == int(ActionType.PUSH))
        & (batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED))
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
    rows = selected[:, 0]
    forced = torch.zeros_like(sensor["point_mask"], dtype=torch.bool)
    for action, row in enumerate(rows.tolist()):
        points = torch.nonzero(sensor["point_mask"][row], as_tuple=False).flatten()
        nearest = torch.linalg.vector_norm(
            sensor["xyz"][row, points] - contacts[action], dim=-1
        ).argmin()
        forced[row, points[nearest]] = True

    model_batch = dict(batch)
    model_batch["push_condition"] = push_condition_from_gt(batch, instance_queries)
    model_batch["training_hints"] = {"push_direction_point_mask": forced}
    with torch.no_grad():
        output = model(model_batch, forward_mode="push")
    logits = model.push_evaluator.score_exact_actions(
        output["sensor"],
        output["push"],
        batch_index=rows,
        acted_object=batch["acted_object"][valid],
        contact_world=contacts,
        direction_world=batch["action_parameters"]["push_direction_world"][valid],
        push_distance=contacts.new_full((len(contacts),), PUSH_DISTANCE_M),
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
