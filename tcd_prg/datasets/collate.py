"""Variable-object/variable-candidate collation with explicit validity masks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .types import UnifiedSample


def _pad(arrays: list[np.ndarray], value: float | int | bool = 0) -> tuple[Tensor, Tensor]:
    max_length = max(len(x) for x in arrays)
    shape = (len(arrays), max_length) + arrays[0].shape[1:]
    output = np.full(shape, value, dtype=arrays[0].dtype)
    mask = np.zeros((len(arrays), max_length), dtype=bool)
    for row, array in enumerate(arrays):
        output[row, : len(array)] = array
        mask[row, : len(array)] = True
    return torch.from_numpy(output), torch.from_numpy(mask)


def _pad_square(arrays: list[np.ndarray], value: float | int = 0) -> Tensor:
    """Pad ``[O,O,...]`` arrays along both object axes."""

    max_objects = max(array.shape[0] for array in arrays)
    tail = arrays[0].shape[2:]
    output = np.full(
        (len(arrays), max_objects, max_objects) + tail,
        value,
        dtype=arrays[0].dtype,
    )
    for row, array in enumerate(arrays):
        objects = array.shape[0]
        if array.shape[1] != objects:
            raise ValueError("Expected square object relation matrix")
        output[row, :objects, :objects] = array
    return torch.from_numpy(output)


def collate_unified(samples: list[UnifiedSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    observations = [x.observation for x in samples]
    candidates = [x.candidates for x in samples]
    xyz, point_mask = _pad([x.xyz for x in observations])
    rgb, _ = _pad([x.rgb for x in observations])
    instance, _ = _pad([x.instance_id for x in observations], -1)
    target_mask, _ = _pad([x.target_mask for x in observations], False)
    have_region = all(x.region_target is not None and x.region_valid is not None for x in observations)
    object_pose, object_mask = _pad([x.object_pose for x in observations])
    object_active, _ = _pad([x.object_active for x in observations], False)
    object_category_id, _ = _pad([x.object_category_id for x in observations], -1)
    action_type, candidate_mask = _pad([x.action_type for x in candidates], -1)
    acted_object, _ = _pad([x.acted_object for x in candidates], -1)
    valid_mask, _ = _pad([x.valid_mask for x in candidates], False)
    status, _ = _pad([x.evaluation_status for x in candidates], -1)
    outcome, _ = _pad([x.outcome_code for x in candidates], -1)
    success, _ = _pad([x.action_improves_state for x in candidates], False)
    policy_success_arrays = []
    for sample in samples:
        successful_ids: list[np.ndarray] = []
        for sequence in sample.sequences:
            if len(sequence.policy_action_ids):
                successful_ids.append(sequence.policy_action_ids)
            if len(sequence.terminal_action_ids):
                successful_ids.append(sequence.terminal_action_ids)
        union = (
            np.unique(np.concatenate(successful_ids))
            if successful_ids
            else np.empty(0, dtype=np.int64)
        )
        policy_success_arrays.append(np.isin(sample.candidates.candidate_action_ids, union))
    policy_success, _ = _pad(policy_success_arrays, False)
    potential_delta, _ = _pad([x.potential_delta for x in candidates], np.nan)
    after_state_valid, _ = _pad([x.after_state_valid for x in candidates], False)
    after_pose_valid, _ = _pad([x.after_pose_valid for x in candidates], False)
    potential_after_valid, _ = _pad([x.potential_after_valid for x in candidates], False)
    acted_object_motion_valid, _ = _pad([x.acted_object_motion_valid for x in candidates], False)
    target_motion_valid, _ = _pad([x.target_motion_valid for x in candidates], False)
    action_parameters = {}
    for key in candidates[0].action_parameters:
        fill = -1 if np.issubdtype(candidates[0].action_parameters[key].dtype, np.integer) else np.nan
        action_parameters[key], _ = _pad([x.action_parameters[key] for x in candidates], fill)
    relation_graph = _pad_square([x.state_labels.relation_graph for x in samples])
    task_block_graph, _ = _pad([x.state_labels.task_block_graph for x in samples])
    direct_blocker, _ = _pad([x.state_labels.direct_blocker_mask for x in samples], False)
    indirect_blocker, _ = _pad([x.state_labels.indirect_blocker_mask for x in samples], False)
    actionable_blocker, _ = _pad([x.state_labels.actionable_blocker_mask for x in samples], False)
    object_count = object_pose.shape[1]
    topology_target = torch.zeros(
        (len(samples), object_count, object_count), dtype=torch.bool
    )
    topology_edge_valid = torch.zeros_like(topology_target)
    for row, sample in enumerate(samples):
        order = sample.state_labels.prerequisite_object_order.tolist()
        for earlier_index, earlier in enumerate(order):
            for later in order[earlier_index + 1 :]:
                topology_target[row, earlier, later] = True
                topology_edge_valid[row, earlier, later] = True
                topology_edge_valid[row, later, earlier] = True
    remaining_steps_target = []
    remaining_steps_valid = []
    for sample in samples:
        candidates_remaining = []
        current_state = sample.observation.state_id
        for sequence in sample.sequences:
            positions = np.flatnonzero(sequence.state_ids == current_state)
            for position in positions:
                candidates_remaining.append(max(0, len(sequence.policy_action_ids) - int(position)))
        remaining_steps_valid.append(bool(candidates_remaining))
        remaining_steps_target.append(min(candidates_remaining) if candidates_remaining else 0)
    result = {
        "xyz": xyz.float(),
        "rgb": rgb.float(),
        "point_mask": point_mask,
        "instance_id": instance.long(),
        "target_mask": target_mask.bool(),
        "target_object": torch.tensor([x.target_object for x in observations], dtype=torch.long),
        "task_region_id": torch.tensor([x.task_region_id for x in observations], dtype=torch.long),
        "object_pose": object_pose.float(),
        "object_mask": object_mask,
        "object_active": object_active.bool(),
        "object_category_id": object_category_id.long(),
        "task_category_id": torch.tensor(
            [x.object_category_id[x.target_object] for x in observations], dtype=torch.long
        ),
        "action_type": action_type.long(),
        "acted_object": acted_object.long(),
        "candidate_mask": candidate_mask & valid_mask.bool(),
        "evaluation_status": status.long(),
        "outcome_code": outcome.long(),
        # Local transition improvement supervises action-effect heads.  Policy
        # behavior cloning uses only actions that occur in a successful
        # sequence; the two concepts must never be conflated.
        "action_improves_state": success.bool(),
        "success_mask": success.bool(),  # compatibility alias for reporting
        "policy_success_mask": policy_success.bool(),
        "potential_delta": potential_delta.float(),
        "after_state_valid": after_state_valid.bool(),
        "after_pose_valid": after_pose_valid.bool(),
        "potential_after_valid": potential_after_valid.bool(),
        "acted_object_motion_valid": acted_object_motion_valid.bool(),
        "target_motion_valid": target_motion_valid.bool(),
        "action_parameters": {
            key: value.long() if not value.dtype.is_floating_point else value.float()
            for key, value in action_parameters.items()
        },
        "relation_graph": relation_graph.float(),
        "task_block_graph": task_block_graph.float(),
        "direct_blocker_target": direct_blocker.bool(),
        "indirect_blocker_target": indirect_blocker.bool(),
        "actionable_blocker_target": actionable_blocker.bool(),
        "topology_target": topology_target,
        "topology_edge_valid": topology_edge_valid,
        "sequence_topology_valid": torch.tensor(
            [x.state_labels.sequence_topology_valid for x in samples], dtype=torch.bool
        ),
        "verified_positive_grasp_count": torch.tensor(
            [x.state_labels.verified_positive_grasp_count for x in samples], dtype=torch.long
        ),
        "required_grasp_count": torch.tensor(
            [x.state_labels.required_grasp_count for x in samples], dtype=torch.long
        ),
        "remaining_steps": torch.tensor(
            [max(0, 5 - x.state_labels.sequence_depth) for x in samples], dtype=torch.long
        ),
        "remaining_steps_target": torch.tensor(remaining_steps_target, dtype=torch.float32),
        "remaining_steps_valid": torch.tensor(remaining_steps_valid, dtype=torch.bool),
        "samples": samples,
    }
    if have_region:
        region_target, _ = _pad([x.region_target for x in observations], False)  # type: ignore[arg-type]
        region_valid, _ = _pad([x.region_valid for x in observations], False)  # type: ignore[arg-type]
        result["region_target"] = region_target.bool()
        result["region_valid"] = region_valid.bool()
        result["visibility_target"] = torch.tensor(
            [float(x.task_region_visibility or 0.0) for x in observations], dtype=torch.float32
        )
        result["visibility_valid"] = torch.tensor(
            [x.task_region_visibility is not None for x in observations], dtype=torch.bool
        )
    return result
