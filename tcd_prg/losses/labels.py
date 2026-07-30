"""Build supervision tensors from unified state candidate groups."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.geometry.se3 import quaternion_xyzw_to_matrix


def _nearest_point_indices(
    xyz: Tensor,
    point_mask: Tensor,
    instance_id: Tensor,
    contacts: Tensor,
    acted_object: Tensor,
    candidate_valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """Find candidate contact points without allocating a K-by-N-by-3 tensor."""

    batch_size, candidate_count = contacts.shape[:2]
    indices = torch.zeros((batch_size, candidate_count), dtype=torch.long, device=xyz.device)
    valid = candidate_valid.clone()
    for row in range(batch_size):
        for candidate in torch.nonzero(candidate_valid[row], as_tuple=False).flatten().tolist():
            mask = point_mask[row] & (instance_id[row] == acted_object[row, candidate])
            points = torch.nonzero(mask, as_tuple=False).flatten()
            if not len(points) or not torch.isfinite(contacts[row, candidate]).all():
                valid[row, candidate] = False
                continue
            delta = xyz[row, points] - contacts[row, candidate]
            indices[row, candidate] = points[(delta * delta).sum(-1).argmin()]
    return indices, valid


def _rotation_bins(rotation: Tensor, bins: int) -> Tensor:
    """Quantize the x-axis rotation around the canonical +Z approach axis."""

    approach = torch.nn.functional.normalize(rotation[..., :, 2], dim=-1)
    x_axis = torch.nn.functional.normalize(rotation[..., :, 0], dim=-1)
    reference_z = torch.tensor([0.0, 0.0, 1.0], device=rotation.device).expand_as(approach)
    reference_y = torch.tensor([0.0, 1.0, 0.0], device=rotation.device).expand_as(approach)
    reference = torch.where((approach[..., 2].abs() > 0.9).unsqueeze(-1), reference_y, reference_z)
    tangent_x = torch.nn.functional.normalize(torch.cross(reference, approach, dim=-1), dim=-1)
    tangent_y = torch.cross(approach, tangent_x, dim=-1)
    angle = torch.atan2((x_axis * tangent_y).sum(-1), (x_axis * tangent_x).sum(-1))
    return torch.floor((angle + math.pi) * bins / (2 * math.pi)).long().remainder(bins)


def build_grasp_proposal_labels(
    batch: dict[str, Tensor], config: ModelConfig, generic_remove: bool = False
) -> dict[str, Tensor]:
    xyz = batch["xyz"]
    if generic_remove:
        point_domain = batch["point_mask"] & batch["object_active"].gather(
            1, batch["instance_id"].clamp(0, batch["object_active"].shape[1] - 1)
        )
        action_kind = int(ActionType.PICK_REMOVE)
        pose_key = "removal_grasp_pose_world"
    else:
        point_domain = batch["target_mask"] & batch["point_mask"]
        action_kind = int(ActionType.TASK_GRASP)
        pose_key = "task_grasp_pose_world"
    batch_size, point_count = xyz.shape[:2]
    parameters = batch["action_parameters"]
    candidate = batch["candidate_mask"] & (batch["action_type"] == action_kind)
    geometry_positive = parameters.get(
        "proposal_geometry_valid", torch.ones_like(candidate)
    ).bool()
    pose = parameters[pose_key]
    rotation = parameters["grasp_rotation_matrix_world"]
    candidate &= (
        torch.isfinite(pose).all(-1)
        & torch.isfinite(rotation).all(-1).all(-1)
        & torch.isfinite(parameters["grasp_width_m"])
    )
    point_index, candidate = _nearest_point_indices(
        xyz, point_domain, batch["instance_id"], pose[..., :3], batch["acted_object"], candidate
    )
    contact_target = torch.zeros((batch_size, point_count), dtype=xyz.dtype, device=xyz.device)
    proposal_valid = torch.zeros((batch_size, point_count), dtype=torch.bool, device=xyz.device)
    mode_valid = torch.zeros_like(proposal_valid)
    approach_target = torch.full((batch_size, point_count, 3), float("nan"), device=xyz.device)
    rotation_bin = torch.full((batch_size, point_count), -1, dtype=torch.long, device=xyz.device)
    width_target = torch.full((batch_size, point_count), float("nan"), device=xyz.device)
    confidence_target = torch.zeros((batch_size, point_count), device=xyz.device)
    candidate_rotation_bin = _rotation_bins(torch.nan_to_num(rotation), config.num_grasp_rotation_bins)
    selected_confidence = torch.full(
        (batch_size, point_count), -float("inf"), dtype=xyz.dtype, device=xyz.device
    )
    sigma_sq = float(config.contact_heatmap_sigma_m) ** 2
    support_sq = 9.0 * sigma_sq
    for row in range(batch_size):
        for candidate_index in torch.nonzero(candidate[row], as_tuple=False).flatten().tolist():
            point = int(point_index[row, candidate_index])
            object_index = int(batch["acted_object"][row, candidate_index])
            domain = point_domain[row] & (batch["instance_id"][row] == object_index)
            delta = xyz[row] - pose[row, candidate_index, :3]
            distance_sq = (delta * delta).sum(-1)
            neighborhood = domain & (distance_sq <= support_sq)
            proposal_valid[row] |= neighborhood
            if not bool(geometry_positive[row, candidate_index]):
                continue
            heat = torch.exp(-0.5 * distance_sq / sigma_sq) * neighborhood
            contact_target[row] = torch.maximum(contact_target[row], heat)
            confidence = torch.nan_to_num(
                parameters["grasp_confidence"][row, candidate_index], nan=1.0
            )
            # A single dense point cannot represent two rotation/width modes.
            # Select the highest-confidence candidate deterministically instead
            # of silently overwriting it with iteration order.
            if confidence <= selected_confidence[row, point]:
                continue
            selected_confidence[row, point] = confidence
            mode_valid[row, point] = True
            approach_target[row, point] = parameters["grasp_approach_world"][row, candidate_index]
            rotation_bin[row, point] = candidate_rotation_bin[row, candidate_index]
            width_target[row, point] = parameters["grasp_width_m"][row, candidate_index]
            confidence_target[row, point] = confidence
    if generic_remove:
        compatibility_target = point_domain
        compatibility_valid = point_domain
    else:
        compatibility_target = batch.get("region_target", contact_target).bool()
        compatibility_valid = batch.get("region_valid", point_domain).bool() & point_domain
    return {
        "proposal_valid": proposal_valid,
        "contact_target": contact_target,
        "mode_valid": mode_valid,
        "approach_target": approach_target,
        "rotation_bin": rotation_bin,
        "width_target_m": width_target,
        "width_valid": torch.isfinite(width_target)
        & (width_target >= config.min_grasp_width_m)
        & (width_target <= config.max_grasp_width_m),
        "confidence_target": confidence_target,
        "compatibility_target": compatibility_target,
        "compatibility_valid": compatibility_valid,
    }


def build_global_grasp_labels(
    batch: dict[str, Tensor], config: ModelConfig
) -> dict[str, Tensor] | None:
    """Map the task-free candidate set to unordered per-point grasp modes."""

    source = batch.get("global_grasp_labels")
    if source is None:
        return None
    xyz = batch["xyz"]
    b, n = xyz.shape[:2]
    modes = config.global_grasp_modes_per_point
    contact_target = xyz.new_zeros((b, n))
    contact_valid = torch.zeros((b, n), dtype=torch.bool, device=xyz.device)
    matching_valid = torch.zeros((b, n, modes), dtype=torch.bool, device=xyz.device)
    geometry_valid = torch.zeros_like(matching_valid)
    approach = xyz.new_full((b, n, modes, 3), float("nan"))
    rotation_bin = torch.full((b, n, modes), -1, dtype=torch.long, device=xyz.device)
    width = xyz.new_full((b, n, modes), float("nan"))
    intrinsic = xyz.new_zeros((b, n, modes))
    intrinsic_valid = torch.zeros_like(matching_valid)
    scene = xyz.new_zeros((b, n, modes))
    scene_valid = torch.zeros_like(matching_valid)
    candidate = source["valid_mask"] & torch.isfinite(source["grasp_pose_world"]).all(-1)
    point_index, candidate = _nearest_point_indices(
        xyz, batch["point_mask"], batch["instance_id"], source["contact_point_world"],
        source["object_index"], candidate,
    )
    rotation = quaternion_xyzw_to_matrix(torch.nan_to_num(source["grasp_pose_world"][..., 3:]))
    candidate_rotation_bin = _rotation_bins(rotation, config.num_grasp_rotation_bins)
    candidate_approach = source["approach_direction_world"]
    dominant_axis = candidate_approach.abs().argmax(-1)
    dominant_sign = (candidate_approach.gather(-1, dominant_axis.unsqueeze(-1)).squeeze(-1) >= 0).long()
    approach_anchor = dominant_axis * 2 + dominant_sign
    width_anchor = torch.floor(torch.nan_to_num(source["width_m"], nan=0.0) / 0.005).long()
    mode_anchor = (
        (approach_anchor * config.num_grasp_rotation_bins + candidate_rotation_bin) * 32
        + width_anchor.clamp(0, 31)
    ).detach().cpu().tolist()
    intrinsic_cpu = source["intrinsic_stable"].detach().cpu().tolist()
    sigma_sq = float(config.contact_heatmap_sigma_m) ** 2
    support_sq = 9.0 * sigma_sq
    for row in range(b):
        grouped: dict[int, list[int]] = {}
        for candidate_index in torch.nonzero(candidate[row], as_tuple=False).flatten().tolist():
            grouped.setdefault(int(point_index[row, candidate_index]), []).append(candidate_index)
        for point, candidates_at_point in grouped.items():
            # Anchor-diverse mode coverage avoids collapsing a dense ACRONYM
            # contact neighbourhood to several nearly identical poses without
            # per-candidate GPU synchronisation in the label builder.
            stable = [index for index in candidates_at_point if bool(intrinsic_cpu[row][index])]
            remaining = stable + [index for index in candidates_at_point if index not in stable]
            selected: list[int] = []
            used_anchors: set[int] = set()
            for index in remaining:
                anchor = int(mode_anchor[row][index])
                if anchor not in used_anchors:
                    selected.append(index)
                    used_anchors.add(anchor)
                if len(selected) == modes:
                    break
            if len(selected) < modes:
                selected.extend(index for index in remaining if index not in selected)
                selected = selected[:modes]
            for slot, candidate_index in enumerate(selected):
                object_index = int(source["object_index"][row, candidate_index])
                domain = batch["point_mask"][row] & (batch["instance_id"][row] == object_index)
                distance_sq = ((xyz[row] - source["contact_point_world"][row, candidate_index]) ** 2).sum(-1)
                neighborhood = domain & (distance_sq <= support_sq)
                contact_valid[row] |= neighborhood
                if bool(intrinsic_cpu[row][candidate_index]):
                    contact_target[row] = torch.maximum(
                        contact_target[row], torch.exp(-0.5 * distance_sq / sigma_sq) * neighborhood
                    )
                matching_valid[row, point, slot] = True
                geometry_valid[row, point, slot] = bool(intrinsic_cpu[row][candidate_index])
                approach[row, point, slot] = source["approach_direction_world"][row, candidate_index]
                rotation_bin[row, point, slot] = candidate_rotation_bin[row, candidate_index]
                width[row, point, slot] = source["width_m"][row, candidate_index]
                intrinsic[row, point, slot] = source["intrinsic_stable"][row, candidate_index].float()
                intrinsic_valid[row, point, slot] = True
                state = int(source["scene_executable"][row, candidate_index])
                if state >= 0:
                    scene[row, point, slot] = float(state)
                    scene_valid[row, point, slot] = True
    return {
        "contact_target": contact_target,
        "contact_valid": contact_valid,
        "mode_valid": matching_valid,
        "geometry_valid": geometry_valid,
        "approach_target": approach,
        "rotation_bin": rotation_bin,
        "width_target_m": width,
        "width_valid": torch.isfinite(width) & (width >= config.min_grasp_width_m) & (width <= config.max_grasp_width_m),
        "intrinsic_target": intrinsic,
        "intrinsic_valid": intrinsic_valid,
        "scene_target": scene,
        "scene_valid": scene_valid,
    }


def build_graph_labels(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    object_mask = batch["object_mask"] & batch["object_active"]
    pair_valid = object_mask[:, :, None, None] & object_mask[:, None, :, None]
    return {
        "physical_edge_target": batch["relation_graph"],
        "physical_edge_valid": pair_valid.expand_as(batch["relation_graph"]),
        "task_edge_target": batch["task_block_graph"],
        "task_edge_valid": object_mask[:, :, None].expand_as(batch["task_block_graph"]),
        "direct_blocker_target": batch["direct_blocker_target"],
        "indirect_blocker_target": batch["indirect_blocker_target"],
        "actionable_target": batch["actionable_blocker_target"],
        "blocker_valid": object_mask,
        "topology_target": batch["topology_target"],
        "topology_edge_valid": batch["topology_edge_valid"],
        "sequence_topology_valid": batch["sequence_topology_valid"],
    }


def build_push_supervision(
    output: dict[str, Tensor], batch: dict[str, Tensor], use_potential: bool, use_risk: bool
) -> tuple[dict[str, Tensor], dict[str, Tensor | bool]]:
    action_type = batch["action_type"]
    parameters = batch["action_parameters"]
    candidate = batch["candidate_mask"] & (action_type == int(ActionType.PUSH))
    point_index, parameter_valid = _nearest_point_indices(
        batch["xyz"],
        batch["point_mask"],
        batch["instance_id"],
        parameters["push_contact_world"],
        batch["acted_object"],
        candidate,
    )
    row = torch.arange(action_type.shape[0], device=action_type.device)[:, None]
    gathered = {
        "object_logits": output["object_logits"],
        "contact_logits": output["contact_logits"],
        "direction_logits": output["direction_logits"][row, point_index],
        "direction_residual": output["direction_residual"][row, point_index],
        "potential_delta": output["potential_delta"][row, point_index],
        "risk_logits": output["risk_logits"][row, point_index],
    }
    direction = torch.nn.functional.normalize(
        torch.nan_to_num(parameters["push_direction_world"]), dim=-1
    )
    angle = torch.atan2(direction[..., 1], direction[..., 0]).remainder(2 * math.pi)
    bins = output["direction_logits"].shape[-1]
    direction_bin = torch.floor(angle * bins / (2 * math.pi)).long().remainder(bins)
    center_angle = (direction_bin.float() + 0.5) * 2 * math.pi / bins
    center = torch.stack((torch.cos(center_angle), torch.sin(center_angle)), -1)
    contact_target = torch.zeros_like(output["contact_logits"])
    contact_valid = torch.zeros_like(output["contact_logits"], dtype=torch.bool)
    sigma_sq = float(batch.get("contact_heatmap_sigma_m", 0.008)) ** 2
    support_sq = 9.0 * sigma_sq
    for batch_row in range(action_type.shape[0]):
        for candidate_index in torch.nonzero(
            parameter_valid[batch_row], as_tuple=False
        ).flatten().tolist():
            object_index = int(batch["acted_object"][batch_row, candidate_index])
            domain = batch["point_mask"][batch_row] & (
                batch["instance_id"][batch_row] == object_index
            )
            delta = batch["xyz"][batch_row] - parameters["push_contact_world"][
                batch_row, candidate_index
            ]
            distance_sq = (delta * delta).sum(-1)
            neighborhood = domain & (distance_sq <= support_sq)
            contact_valid[batch_row] |= neighborhood
            heat = torch.exp(-0.5 * distance_sq / sigma_sq) * neighborhood
            contact_target[batch_row] = torch.maximum(contact_target[batch_row], heat)
    evaluated = batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED)
    positive = batch["action_improves_state"] & candidate
    object_positive = torch.zeros_like(batch["object_mask"])
    for batch_row in range(action_type.shape[0]):
        objects = batch["acted_object"][batch_row, positive[batch_row]]
        object_positive[batch_row, objects[objects >= 0]] = True
    risk_target = torch.stack(
        (
            parameters["risk_unstable"],
            parameters["risk_out_of_workspace"],
            parameters["risk_other_invalid"],
        ),
        -1,
    )
    labels: dict[str, Tensor | bool] = {
        "object_positive": object_positive,
        "object_valid_mask": batch["object_mask"] & batch["object_active"],
        "contact_target": contact_target,
        "contact_valid": contact_valid,
        "direction_bin": direction_bin,
        "direction_residual": direction[..., :2] - center,
        "direction_valid": parameter_valid,
        "potential_delta": batch["potential_delta"],
        "potential_after_valid": candidate & batch["potential_after_valid"],
        "risk_target": risk_target,
        "risk_valid": (candidate & evaluated).unsqueeze(-1).expand_as(risk_target),
        "use_potential": use_potential,
        "use_risk": use_risk,
    }
    return gathered, labels


def build_remove_labels(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    candidate = batch["candidate_mask"] & (batch["action_type"] == int(ActionType.PICK_REMOVE))
    evaluated = batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED)
    positive = candidate & batch["action_improves_state"]
    object_positive = torch.zeros_like(batch["object_mask"])
    for row in range(candidate.shape[0]):
        objects = batch["acted_object"][row, positive[row]]
        object_positive[row, objects[objects >= 0]] = True
    return {
        "object_positive": object_positive,
        "object_valid_mask": batch["object_mask"] & batch["object_active"],
        "candidate_positive": positive,
        "candidate_valid": candidate & evaluated,
    }


def build_verifier_labels(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Use only verifier labels that are truly present in the action data.

    Executed actions supervise stability/overall. Initial-state Steps 1--6
    screens additionally supervise region compatibility, collision/clearance,
    and coarse FR5 approach feasibility. Every head keeps its own mask.
    """

    candidate_valid = batch["verifier_inputs"]["candidate_valid"]
    parameters = batch["action_parameters"]
    result = {}
    for head in ("stability", "task_compatibility", "collision", "clearance", "approach", "overall"):
        target = parameters[f"verifier_{head}_target"]
        valid = parameters[f"verifier_{head}_valid"] & candidate_valid
        result[f"{head}_target"] = torch.nan_to_num(target)
        result[f"{head}_valid"] = valid & torch.isfinite(target)
    return result


def build_region_labels(batch: dict[str, Tensor]) -> dict[str, Tensor] | None:
    if "region_target" not in batch:
        return None
    return {
        "region_target": batch["region_target"],
        "region_valid": batch["region_valid"],
        "visibility_target": batch["visibility_target"],
        "visibility_valid": batch["visibility_valid"],
    }
