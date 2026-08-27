"""Single deployment-equivalent decoder for Stage-C PUSH proposals."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tcd_prg.config import ModelConfig
from tcd_prg.constants import PUSH_DISTANCE_M, ActionType, CandidateStatus
from tcd_prg.models.push.head import push_contact_joint_score
from tcd_prg.models.push_condition import PushCondition


def _empty(device: torch.device, dtype: torch.dtype) -> dict[str, Tensor]:
    return {
        "object": torch.empty(0, dtype=torch.long, device=device),
        "contact_world": torch.empty(0, 3, dtype=dtype, device=device),
        "direction_world": torch.empty(0, 3, dtype=dtype, device=device),
        "point_index": torch.empty(0, dtype=torch.long, device=device),
        "direction_bin": torch.empty(0, dtype=torch.long, device=device),
        "direction_residual": torch.empty(0, 2, dtype=dtype, device=device),
        "push_distance": torch.empty(0, dtype=dtype, device=device),
        "object_score": torch.empty(0, dtype=dtype, device=device),
        "contact_score": torch.empty(0, dtype=dtype, device=device),
        "direction_score": torch.empty(0, dtype=dtype, device=device),
        "utility": torch.empty(0, dtype=dtype, device=device),
        "proposal_score": torch.empty(0, dtype=dtype, device=device),
        "push_effective_logit": torch.empty(0, dtype=dtype, device=device),
        "push_robustness_logit": torch.empty(0, dtype=dtype, device=device),
    }


def _top_contacts(
    contact_logits: Tensor,
    object_probability: Tensor,
    object_ids: Tensor,
    point_mask: Tensor,
    total: int,
) -> Tensor:
    if not len(object_ids) or total <= 0:
        return torch.empty(0, dtype=torch.long, device=contact_logits.device)
    per_object = max(1, math.ceil(total / len(object_ids)))
    owner = object_probability.argmax(0)
    selected: list[Tensor] = []
    for object_id in object_ids.tolist():
        membership = object_probability[object_id]
        domain = point_mask & (owner == object_id) & (membership >= 0.5)
        if not bool(domain.any()):
            domain = point_mask & (owner == object_id)
        points = torch.nonzero(domain, as_tuple=False).flatten()
        count = min(per_object, len(points))
        if count:
            joint = push_contact_joint_score(contact_logits[points], membership[points])
            selected.append(points[joint.topk(count).indices])
    if not selected:
        return torch.empty(0, dtype=torch.long, device=contact_logits.device)
    candidates = torch.unique(torch.cat(selected), sorted=True)
    membership = object_probability[object_ids][:, candidates]
    joint = push_contact_joint_score(
        contact_logits[candidates][None], membership
    ).amax(0)
    return candidates[joint.topk(min(total, len(candidates))).indices]


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


def decode_push_candidates(
    sensor: dict[str, Tensor],
    condition: PushCondition,
    push: dict[str, Tensor],
    config: ModelConfig,
    *,
    use_push_potential: bool,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Decode pre-NMS and final candidates through the one formal protocol."""
    condition.validate(sensor["xyz"].shape[1])
    pre_nms_rows: list[dict[str, Tensor]] = []
    final_rows: list[dict[str, Tensor]] = []
    for row in range(sensor["xyz"].shape[0]):
        xyz = sensor["xyz"][row]
        if not bool(condition.target_valid[row]):
            empty = _empty(xyz.device, xyz.dtype)
            pre_nms_rows.append(empty)
            final_rows.append(_empty(xyz.device, xyz.dtype))
            continue
        object_probability = condition.object_probability[row]
        active = condition.object_valid[row]
        object_score_all = torch.sigmoid(push["object_logits"][row])
        objects = torch.nonzero(active, as_tuple=False).flatten()
        if len(objects):
            objects = objects[
                object_score_all[objects].argsort(descending=True, stable=True)[
                    : config.push_object_topk
                ]
            ]
        point_index = _top_contacts(
            push["contact_logits"][row],
            object_probability,
            objects,
            sensor["point_mask"][row].bool() & push["direction_point_mask"][row].bool(),
            config.push_candidates,
        )
        if not len(point_index):
            pre_nms_rows.append(_empty(xyz.device, xyz.dtype))
            final_rows.append(_empty(xyz.device, xyz.dtype))
            continue
        direction_probability = torch.sigmoid(push["direction_logits"][row, point_index])
        directions_per_contact = min(
            config.push_directions_per_contact, direction_probability.shape[-1]
        )
        direction_score, direction_bin = direction_probability.topk(
            directions_per_contact, dim=-1
        )
        expanded_point = point_index[:, None].expand(-1, directions_per_contact).reshape(-1)
        direction_bin = direction_bin.reshape(-1)
        direction_score = direction_score.reshape(-1)
        contact_score = torch.sigmoid(push["contact_logits"][row, expanded_point])
        membership = object_probability[objects[:, None], expanded_point[None]].T
        pushed_object = objects[membership.argmax(-1)]
        object_score = object_score_all[pushed_object]
        utility = push["utility_delta"][row, expanded_point, direction_bin]
        if use_push_potential:
            utility_factor = torch.sigmoid(utility / float(config.push_utility_temperature))
            eligible = utility > float(config.push_utility_threshold)
        else:
            utility_factor = torch.ones_like(utility)
            eligible = torch.ones_like(utility, dtype=torch.bool)
        proposal_score = object_score * contact_score * direction_score * utility_factor
        eligible &= proposal_score >= float(config.push_candidate_probability_threshold)
        expanded_point = expanded_point[eligible]
        direction_bin = direction_bin[eligible]
        direction_score = direction_score[eligible]
        contact_score = contact_score[eligible]
        pushed_object = pushed_object[eligible]
        object_score = object_score[eligible]
        utility = utility[eligible]
        proposal_score = proposal_score[eligible]
        if len(expanded_point) > config.max_push_candidates:
            keep = proposal_score.topk(config.max_push_candidates).indices
            expanded_point, direction_bin = expanded_point[keep], direction_bin[keep]
            direction_score, contact_score = direction_score[keep], contact_score[keep]
            pushed_object, object_score = pushed_object[keep], object_score[keep]
            utility, proposal_score = utility[keep], proposal_score[keep]
        angle = (direction_bin.float() + 0.5) * 2.0 * math.pi / config.num_direction_bins
        center = torch.stack((torch.cos(angle), torch.sin(angle)), -1)
        residual = push["direction_residual"][row, expanded_point, direction_bin]
        planar = torch.nn.functional.normalize(center + residual, dim=-1)
        direction_world = torch.cat(
            (planar, torch.zeros(len(planar), 1, dtype=xyz.dtype, device=xyz.device)), -1
        )
        effective_field = push.get("push_effective_logit")
        robustness_field = push.get("push_robustness_logit")
        decoded = {
            "object": pushed_object,
            "contact_world": xyz[expanded_point],
            "direction_world": direction_world,
            "point_index": expanded_point,
            "direction_bin": direction_bin,
            "direction_residual": residual,
            "push_distance": torch.full_like(proposal_score, PUSH_DISTANCE_M),
            "object_score": object_score,
            "contact_score": contact_score,
            "direction_score": direction_score,
            "utility": utility,
            "proposal_score": proposal_score,
            "push_effective_logit": (
                effective_field[row, expanded_point, direction_bin]
                if effective_field is not None
                else torch.full_like(proposal_score, -30.0)
            ),
            "push_robustness_logit": (
                robustness_field[row, expanded_point, direction_bin]
                if robustness_field is not None
                else torch.full_like(proposal_score, -30.0)
            ),
        }
        pre_nms_rows.append(decoded)
        nms = push_nms_mask(decoded, config)
        final_rows.append({key: value[nms] for key, value in decoded.items()})
    return pre_nms_rows, final_rows


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
