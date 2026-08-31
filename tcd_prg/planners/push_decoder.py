"""Rank rule-generated complete actions by the independent evaluator."""
import math
import torch
from torch import Tensor
from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType, CandidateStatus

def push_nms_mask(candidates: dict[str, Tensor], config: ModelConfig) -> Tensor:
    """Return the deployment NMS mask for one unpadded candidate row."""
    keep = torch.ones(
        len(candidates["object"]), dtype=torch.bool, device=candidates["object"].device
    )
    order = candidates["proposal_score"].argsort(descending=True, stable=True)
    cosine_threshold = math.cos(math.radians(config.push_nms_direction_deg))
    accepted: list[int] = []
    for index in order.tolist():
        duplicate = False
        for prior in accepted:
            same_object = bool(candidates["object"][index] == candidates["object"][prior])
            contact_distance = torch.linalg.vector_norm(
                candidates["contact_world"][index] - candidates["contact_world"][prior]
            )
            direction_similarity = (
                torch.nn.functional.normalize(candidates["direction_world"][index, :2], dim=-1)
                * torch.nn.functional.normalize(candidates["direction_world"][prior, :2], dim=-1)
            ).sum()
            if (
                same_object
                and bool(contact_distance < config.push_nms_contact_m)
                and bool(direction_similarity >= cosine_threshold)
            ):
                duplicate = True
                break
        if duplicate:
            keep[index] = False
        else:
            accepted.append(index)
    return keep



def decode_push_candidates(sensor, condition, push, config):
    actions, logits = push["actions"], push["effective_logit"]
    pre, final = [], []
    for b in range(len(sensor["xyz"])):
        ids = torch.nonzero(actions.batch_index == b, as_tuple=False).flatten()
        ids = ids[logits[ids].argsort(descending=True, stable=True)]
        ids = ids[:config.max_push_candidates]
        a = actions.select(ids)
        score = logits[ids].sigmoid()
        k = len(ids)
        angle = torch.atan2(a.direction_world[:, 1], a.direction_world[:, 0]).remainder(2*math.pi)
        bins = (angle * config.num_direction_bins / (2*math.pi)).long()
        # Point index is output metadata only, never used for action scoring.
        anchors = []
        for i in range(k):
            member = sensor["point_mask"][b].bool() & (condition.object_probability[b, a.object[i]] >= .5)
            points = torch.nonzero(member, as_tuple=False).flatten()
            anchors.append(points[(sensor["xyz"][b, points]-a.contact_world[i]).norm(dim=-1).argmin()])
        row = {"object": a.object, "contact_world": a.contact_world,
               "direction_world": a.direction_world, "push_distance": a.push_distance,
               "point_index": torch.stack(anchors) if anchors else a.object.new_empty(0),
               "direction_bin": bins, "direction_residual": score.new_zeros((k, 2)),
               "object_score": torch.ones_like(score), "contact_score": torch.ones_like(score),
               "direction_score": torch.ones_like(score), "utility": score,
               "proposal_score": score, "effective_logit": logits[ids], "effective_probability": score}
        pre.append(row)
        keep = push_nms_mask(row, config)
        final.append({key: value[keep] for key, value in row.items()})
    return pre, final

def proposal_recall_counts(
    rows: list[dict[str, Tensor]],
    batch: dict[str, Tensor],
    *,
    contact_threshold_m: float,
    direction_threshold_deg: float,
) -> tuple[Tensor, Tensor]:
    """Count matched and eligible GT-positive PUSH actions without labeling UNKNOWNs."""
    parameters = batch["action_parameters"]
    candidate = (
        batch["candidate_mask"].bool()
        & (batch["action_type"] == int(ActionType.PUSH))
        & (batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED))
        & batch["action_improves_state"].bool()
        & torch.isfinite(parameters["push_contact_world"]).all(-1)
        & torch.isfinite(parameters["push_direction_world"]).all(-1)
    )
    reference = batch["xyz"]
    matched_total = reference.new_zeros(())
    positive_total = candidate.sum().to(reference.dtype)
    cosine_threshold = math.cos(math.radians(direction_threshold_deg))
    for row, decoded in enumerate(rows):
        for action_index in torch.nonzero(candidate[row], as_tuple=False).flatten().tolist():
            if not len(decoded["object"]):
                continue
            same_object = decoded["object"] == batch["acted_object"][row, action_index]
            contact_distance = torch.linalg.vector_norm(
                decoded["contact_world"]
                - parameters["push_contact_world"][row, action_index],
                dim=-1,
            )
            gt_direction = torch.nn.functional.normalize(
                parameters["push_direction_world"][row, action_index, :2], dim=-1
            )
            predicted_direction = torch.nn.functional.normalize(
                decoded["direction_world"][:, :2], dim=-1
            )
            direction_match = (predicted_direction * gt_direction[None]).sum(-1) >= cosine_threshold
            if bool(
                (
                    same_object
                    & (contact_distance <= float(contact_threshold_m))
                    & direction_match
                ).any()
            ):
                matched_total += 1.0
    return matched_total, positive_total
