"""Logged-action-only training. No rule sampling, contact matching, or forcing."""
import torch
from tcd_prg.constants import PUSH_DISTANCE_M, ActionType, CandidateStatus
from tcd_prg.models import push_condition_from_gt
from tcd_prg.models.push import PushActions


def push_effectiveness_eligibility(batch, condition):
    p = batch["action_parameters"]
    obj = batch["acted_object"].long()
    q = condition.object_valid.shape[1]
    represented = (obj >= 0) & (obj < q)
    if q:
        represented &= condition.object_valid.gather(1, obj.clamp(0, q-1))
    else:
        represented &= False
    d = p["push_direction_world"]
    return (batch["candidate_mask"].bool()
            & (batch["action_type"] == int(ActionType.PUSH))
            & (batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED))
            & represented & condition.target_valid[:, None]
            & torch.isfinite(p["push_contact_world"]).all(-1)
            & torch.isfinite(d).all(-1)
            & (d[..., :2].norm(dim=-1) > 1e-8) & (d[..., 2].abs() < 1e-5))


def logged_push_actions(batch, condition):
    valid = push_effectiveness_eligibility(batch, condition)
    selected = torch.nonzero(valid, as_tuple=False)
    p = batch["action_parameters"]
    direction = torch.nn.functional.normalize(p["push_direction_world"][valid], dim=-1)
    contacts = p["push_contact_world"][valid]
    # This dataset's execution protocol fixes the stroke at 0.15 m.
    actions = PushActions(selected[:, 0], batch["acted_object"][valid].long(),
                          contacts, direction, contacts.new_full((len(contacts),), PUSH_DISTANCE_M))
    return actions, valid


def push_effectiveness_batch_loss(model, batch, *, instance_queries, loss_function):
    condition = push_condition_from_gt(batch, instance_queries)
    actions, valid = logged_push_actions(batch, condition)
    if len(actions.batch_index):
        logits = model.score_actions(batch, condition, actions)
        losses = loss_function(logits, batch["evaluation_status"][valid],
                               batch["action_improves_state"][valid])
        loss = losses["push_effectiveness"]
    else:
        loss = sum(p.sum() * 0. for p in model.push_evaluator.parameters())
        logits = loss.expand(0)
        losses = {"push_effectiveness": loss}
    return loss, {**losses, "effective_logit": logits,
                  "effective_target": batch["action_improves_state"][valid].bool(),
                  "effective_group_index": actions.batch_index}
