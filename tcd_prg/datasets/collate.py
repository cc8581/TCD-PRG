"""Variable-object/variable-candidate collation with explicit validity masks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .types import GlobalGraspLabels, UnifiedSample


def _empty_global_grasps_like(label: GlobalGraspLabels) -> GlobalGraspLabels:
    """Create a contract-complete empty placeholder for mixed supervision batches."""

    return GlobalGraspLabels(
        object_index=np.empty((0,), dtype=label.object_index.dtype),
        source_grasp_index=np.empty((0,), dtype=label.source_grasp_index.dtype),
        contact_point_world=np.empty((0, 3), dtype=label.contact_point_world.dtype),
        grasp_pose_world=np.empty((0, 7), dtype=label.grasp_pose_world.dtype),
        approach_direction_world=np.empty((0, 3), dtype=label.approach_direction_world.dtype),
        width_m=np.empty((0,), dtype=label.width_m.dtype),
        intrinsic_stable=np.empty((0,), dtype=label.intrinsic_stable.dtype),
        scene_executable=np.empty((0,), dtype=label.scene_executable.dtype),
        valid_mask=np.empty((0,), dtype=label.valid_mask.dtype),
        anchor_visible_distance_m=np.empty((0,), dtype=label.anchor_visible_distance_m.dtype),
        conversion_version=label.conversion_version,
        label_set_complete=False,
    )


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


def grid_sample(
    xyz: np.ndarray, grid_size_m: float, training: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return representatives and their original Pointcept voxel coordinates."""

    if grid_size_m <= 0 or len(xyz) == 0:
        indices = np.arange(len(xyz), dtype=np.int64)
        return indices, np.zeros((len(indices), 3), dtype=np.int32)
    grid = np.floor((xyz - xyz.min(0)) / grid_size_m).astype(np.int64)
    _, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(np.r_[0, counts[:-1]])
    if training:
        # Exact vectorized rule used by Pointcept's train-mode GridSample.
        offsets = np.random.randint(0, counts.max(), len(counts)) % counts
    else:
        offsets = np.zeros(len(counts), dtype=np.int64)
    selected = order[starts + offsets]
    return selected, grid[selected].astype(np.int32, copy=False)


def grid_sample_indices(xyz: np.ndarray, grid_size_m: float, training: bool) -> np.ndarray:
    """Compatibility wrapper returning only selected representative indices."""

    return grid_sample(xyz, grid_size_m, training)[0]


def collate_unified(
    samples: list[UnifiedSample], *, grid_size_m: float | None = None, training: bool = False
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    observations = [x.observation for x in samples]
    candidates = [x.candidates for x in samples]
    if grid_size_m is not None:
        point_samples = [
            grid_sample(x.xyz, float(grid_size_m), training) for x in observations
        ]
        point_indices = [item[0] for item in point_samples]
        grid_coord, _ = _pad([item[1] for item in point_samples])
    else:
        point_indices = [np.arange(len(x.xyz), dtype=np.int64) for x in observations]
        grid_coord = None
    # 所有变长轴都同时生成显式 mask，后续网络不得把 padding 当作真实点或候选。
    xyz, point_mask = _pad([x.xyz[i] for x, i in zip(observations, point_indices, strict=True)])
    rgb, _ = _pad([x.rgb[i] for x, i in zip(observations, point_indices, strict=True)])
    instance, _ = _pad([x.instance_id[i] for x, i in zip(observations, point_indices, strict=True)], -1)
    target_mask, _ = _pad([x.target_mask[i] for x, i in zip(observations, point_indices, strict=True)], False)
    have_region = all(x.region_target is not None and x.region_valid is not None for x in observations)
    object_pose, object_mask = _pad([x.object_pose for x in observations])
    object_present, _ = _pad([x.object_present for x in observations], False)
    object_active, _ = _pad([x.object_active for x in observations], False)
    object_category_id, _ = _pad([x.object_category_id for x in observations], -1)
    action_type, candidate_mask = _pad([x.action_type for x in candidates], -1)
    acted_object, _ = _pad([x.acted_object for x in candidates], -1)
    valid_mask, _ = _pad([x.valid_mask for x in candidates], False)
    status, _ = _pad([x.evaluation_status for x in candidates], -1)
    outcome, _ = _pad([x.outcome_code for x in candidates], -1)
    success, _ = _pad([x.action_improves_state for x in candidates], False)
    # 一个动作只要出现在任一成功序列中即为已知正策略候选；其余动作仍需看评价状态。
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
        dtype = candidates[0].action_parameters[key].dtype
        if np.issubdtype(dtype, np.bool_):
            fill = False
        elif np.issubdtype(dtype, np.integer):
            fill = -1
        else:
            fill = np.nan
        action_parameters[key], _ = _pad([x.action_parameters[key] for x in candidates], fill)
    relation_graph = _pad_square([x.state_labels.relation_graph for x in samples])
    task_block_graph, _ = _pad([x.state_labels.task_block_graph for x in samples])
    direct_blocker, _ = _pad([x.state_labels.direct_blocker_mask for x in samples], False)
    indirect_blocker, _ = _pad([x.state_labels.indirect_blocker_mask for x in samples], False)
    actionable_blocker, _ = _pad([x.state_labels.actionable_blocker_mask for x in samples], False)
    object_count = object_pose.shape[1]
    # prerequisite_object_order 转成有向先后关系，仅对有明确顺序的物体对监督。
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
        "object_present": object_present.bool(),
        "object_active": object_active.bool(),
        "object_category_id": object_category_id.long(),
        "task_category_id": torch.tensor(
            [x.object_category_id[x.target_object] for x in observations], dtype=torch.long
        ),
        "action_type": action_type.long(),
        "acted_object": acted_object.long(),
        "candidate_mask": candidate_mask & valid_mask.bool(),
        "evaluation_status": status.long(),
        "task_grasp_label_set_complete": torch.tensor(
            [x.label_set_complete for x in candidates], dtype=torch.bool
        ),
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
    if grid_coord is not None:
        # These are the coordinates used for representative selection.  Passing
        # them through avoids a second GPU unique/sort pass in the PTv3 adapter.
        result["grid_coord"] = grid_coord.to(torch.int32)
    if have_region:
        region_target, _ = _pad([  # type: ignore[index]
            x.region_target[i] for x, i in zip(observations, point_indices, strict=True)
        ], False)
        region_valid, _ = _pad([  # type: ignore[index]
            x.region_valid[i] for x, i in zip(observations, point_indices, strict=True)
        ], False)
        result["region_target"] = region_target.bool()
        result["region_valid"] = region_valid.bool()
        result["visibility_target"] = torch.tensor(
            [float(x.task_region_visibility or 0.0) for x in observations], dtype=torch.float32
        )
        result["visibility_valid"] = torch.tensor(
            [x.task_region_visibility is not None for x in observations], dtype=torch.bool
        )
    result["global_loss_sample_valid"] = torch.tensor(
        [sample.global_loss_valid and sample.global_grasps is not None for sample in samples],
        dtype=torch.bool,
    )
    if any(sample.global_grasps is not None for sample in samples):
        template = next(sample.global_grasps for sample in samples if sample.global_grasps is not None)
        assert template is not None

        packed = [sample.global_grasps or _empty_global_grasps_like(template) for sample in samples]
        global_valid, _ = _pad([x.valid_mask for x in packed], False)
        scene_executable, _ = _pad([x.scene_executable for x in packed], -1)
        result["global_grasp_labels"] = {
            "object_index": _pad([x.object_index for x in packed], -1)[0].long(),
            "source_grasp_index": _pad([x.source_grasp_index for x in packed], -1)[0].long(),
            "contact_point_world": _pad([x.contact_point_world for x in packed], np.nan)[0].float(),
            "grasp_pose_world": _pad([x.grasp_pose_world for x in packed], np.nan)[0].float(),
            "approach_direction_world": _pad([x.approach_direction_world for x in packed], np.nan)[0].float(),
            "width_m": _pad([x.width_m for x in packed], np.nan)[0].float(),
            "intrinsic_stable": _pad([x.intrinsic_stable for x in packed], False)[0].bool(),
            "scene_executable": scene_executable.to(torch.int8),
            "anchor_visible_distance_m": _pad(
                [x.anchor_visible_distance_m for x in packed], np.nan
            )[0].float(),
            "valid_mask": global_valid.bool(),
            "label_set_complete": torch.tensor(
                [x.label_set_complete for x in packed], dtype=torch.bool
            ),
        }
    return result
