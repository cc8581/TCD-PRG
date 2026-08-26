"""Variable-object/variable-candidate collation with explicit validity masks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .types import GlobalGraspLabels, UnifiedSample


def collate_stageb_binary(
    samples: list[Any], *, grid_size_m: float | None = None,
    training: bool = False, point_count: int = 0,
) -> dict[str, Any]:
    """Collate scene/task inputs and the concrete candidates labelled offline."""
    batch = collate_unified(
        [item.sample for item in samples], grid_size_m=grid_size_m,
        training=training, point_count=point_count, include_graspnet=False,
    )
    translation, valid = _pad([item.translation_world for item in samples], np.nan)
    rotation, _ = _pad([item.rotation_matrix for item in samples], np.nan)
    label, _ = _pad([item.label for item in samples], False)
    batch["stageb_candidates"] = {
        "translation_world": translation.float(),
        "rotation_matrix": rotation.float(),
        "valid": valid.bool(),
    }
    batch["stageb_candidate_valid"] = valid.bool()
    batch["stageb_label"] = label.bool()
    return batch


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


def _limit_point_sample(
    indices: np.ndarray,
    grid_coord: np.ndarray | None,
    point_count: int,
    training: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply a model-side point budget without changing cache request keys."""

    if point_count <= 0 or len(indices) <= point_count:
        return indices, grid_coord
    if training:
        selected = np.random.choice(len(indices), point_count, replace=False)
    else:
        # Evenly cover the stable grid order for deterministic validation.
        selected = np.linspace(0, len(indices) - 1, point_count, dtype=np.int64)
    return indices[selected], None if grid_coord is None else grid_coord[selected]


def _camera_view_payload(
    observations: list[Any],
    *,
    view_index: int,
    grid_size_m: float | None,
    training: bool,
    point_count: int,
) -> dict[str, Tensor]:
    """Build an independently sampled real-camera cloud and calibration payload."""

    xyz_arrays: list[np.ndarray] = []
    instance_arrays: list[np.ndarray] = []
    eye, target, up, camera_valid = [], [], [], []
    for observation in observations:
        source_view = observation.source_view
        usable = (
            source_view is not None
            and view_index < len(observation.camera_parameters)
        )
        if usable:
            source_indices = np.flatnonzero(source_view == view_index)
        else:
            source_indices = np.empty((0,), dtype=np.int64)
        if len(source_indices) and grid_size_m is not None:
            selected, grid = grid_sample(
                observation.xyz[source_indices], float(grid_size_m), training
            )
            selected, _ = _limit_point_sample(
                selected, grid, point_count, training
            )
            source_indices = source_indices[selected]
        elif len(source_indices):
            local = np.arange(len(source_indices), dtype=np.int64)
            selected, _ = _limit_point_sample(local, None, point_count, training)
            source_indices = source_indices[selected]
        xyz_arrays.append(observation.xyz[source_indices])
        instance_arrays.append(observation.instance_id[source_indices])
        if usable:
            camera = observation.camera_parameters[view_index]
            eye.append(np.asarray(camera.eye_world, np.float32))
            target.append(np.asarray(camera.target_world, np.float32))
            up.append(np.asarray(camera.up_world, np.float32))
            camera_valid.append(bool(len(source_indices)))
        else:
            eye.append(np.zeros(3, np.float32))
            target.append(np.asarray([0.0, 0.0, 1.0], np.float32))
            up.append(np.asarray([0.0, -1.0, 0.0], np.float32))
            camera_valid.append(False)

    xyz, point_mask = _pad(xyz_arrays)
    instance_id, _ = _pad(instance_arrays, -1)
    if xyz.shape[1] == 0:
        xyz = torch.zeros((len(observations), 1, 3), dtype=torch.float32)
        point_mask = torch.zeros((len(observations), 1), dtype=torch.bool)
        instance_id = torch.full((len(observations), 1), -1, dtype=torch.long)
    return {
        "graspnet_xyz_world": xyz.float(),
        "graspnet_point_mask": point_mask.bool(),
        # Loss/metrics only. TCDPRGModel.SENSOR_KEYS intentionally excludes it.
        "graspnet_instance_id": instance_id.long(),
        "camera2_eye_world": torch.from_numpy(np.stack(eye)).float(),
        "camera2_target_world": torch.from_numpy(np.stack(target)).float(),
        "camera2_up_world": torch.from_numpy(np.stack(up)).float(),
        "camera2_valid": torch.tensor(camera_valid, dtype=torch.bool),
    }


def collate_unified(
    samples: list[UnifiedSample], *, grid_size_m: float | None = None,
    training: bool = False, point_count: int = 0,
    graspnet_point_count: int = 0, graspnet_view_index: int = 2,
    include_graspnet: bool = True,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    observations = [x.observation for x in samples]
    candidates = [x.candidates for x in samples]
    if grid_size_m is not None:
        point_samples = [
            grid_sample(x.xyz, float(grid_size_m), training) for x in observations
        ]
        limited = [
            _limit_point_sample(item[0], item[1], point_count, training)
            for item in point_samples
        ]
        point_indices = [item[0] for item in limited]
        grid_coord, _ = _pad([item[1] for item in limited])
    else:
        point_indices = [
            _limit_point_sample(
                np.arange(len(x.xyz), dtype=np.int64), None, point_count, training
            )[0]
            for x in observations
        ]
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
    object_count = object_pose.shape[1]
    push_object_known = torch.zeros(
        (len(samples), object_count), dtype=torch.bool
    )
    push_object_positive = torch.zeros_like(push_object_known)
    for row, sample in enumerate(samples):
        state_group = sample.push_object_state
        if state_group is None:
            continue
        known = torch.from_numpy(state_group.evaluated_object).long()
        positive_objects = torch.from_numpy(state_group.positive_object).long()
        known = known[(known >= 0) & (known < object_count)]
        positive_objects = positive_objects[
            (positive_objects >= 0) & (positive_objects < object_count)
        ]
        push_object_known[row, known] = True
        push_object_positive[row, positive_objects] = True
    camera_payload = (
        _camera_view_payload(
            observations,
            view_index=graspnet_view_index,
            grid_size_m=grid_size_m,
            training=training,
            point_count=graspnet_point_count,
        )
        if include_graspnet
        else {}
    )
    source_view, _ = _pad(
        [
            (
                x.source_view[i]
                if x.source_view is not None
                else np.full(len(i), -1, np.int16)
            )
            for x, i in zip(observations, point_indices, strict=True)
        ],
        -1,
    )
    result = {
        "xyz": xyz.float(),
        "rgb": rgb.float(),
        "point_mask": point_mask,
        "source_view": source_view.long(),
        "instance_id": instance.long(),
        "target_mask": target_mask.bool(),
        "target_object": torch.tensor([x.target_object for x in observations], dtype=torch.long),
        "task_region_id": torch.tensor([x.task_region_id for x in observations], dtype=torch.long),
        "object_pose": object_pose.float(),
        "object_mask": object_mask,
        "object_present": object_present.bool(),
        "object_active": object_active.bool(),
        "object_category_id": object_category_id.long(),
        "target_model_id": [
            str(x.metadata.get("object_model_id", ("",))[x.target_object])
            if x.target_object < len(x.metadata.get("object_model_id", ())) else ""
            for x in observations
        ],
        "target_object_scale": torch.tensor(
            [
                float(x.metadata.get("object_scale", (1.0,))[x.target_object])
                if x.target_object < len(x.metadata.get("object_scale", ())) else 1.0
                for x in observations
            ],
            dtype=torch.float32,
        ),
        "push_object_known": push_object_known,
        "push_object_positive": push_object_positive,
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
        "verified_positive_grasp_count": torch.tensor(
            [x.state_labels.verified_positive_grasp_count for x in samples], dtype=torch.long
        ),
        "required_grasp_count": torch.tensor(
            [x.state_labels.required_grasp_count for x in samples], dtype=torch.long
        ),
        "remaining_steps": torch.tensor(
            [max(0, 5 - x.state_labels.sequence_depth) for x in samples], dtype=torch.long
        ),
        "samples": samples,
        **camera_payload,
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


def collate_global_grasp(
    samples: list[Any], *, grid_size_m: float | None = None,
    training: bool = False, point_count: int = 0,
    graspnet_point_count: int = 0, graspnet_view_index: int = 2,
) -> dict[str, Any]:
    """Minimal collator for the independent Global Grasp stream.

    The tensor values consumed by the neutral scene backbone and
    ``build_global_grasp_labels`` are constructed with the same helpers and
    dtypes as ``collate_unified``.  Task-policy graph/verifier fields are intentionally not materialized.
    """

    if not samples:
        raise ValueError("Cannot collate an empty Global Grasp batch")
    observations = [sample.observation for sample in samples]
    packed = [sample.global_grasps for sample in samples]
    if any(label is None for label in packed):
        raise RuntimeError("Global-only batch contains a sample without Global labels")

    if grid_size_m is not None:
        point_samples = [
            grid_sample(obs.xyz, float(grid_size_m), training) for obs in observations
        ]
        limited = [
            _limit_point_sample(item[0], item[1], point_count, training)
            for item in point_samples
        ]
        point_indices = [item[0] for item in limited]
        grid_coord, _ = _pad([item[1] for item in limited])
    else:
        point_indices = [
            _limit_point_sample(
                np.arange(len(obs.xyz), dtype=np.int64), None, point_count, training
            )[0]
            for obs in observations
        ]
        grid_coord = None

    xyz, point_mask = _pad(
        [obs.xyz[index] for obs, index in zip(observations, point_indices, strict=True)]
    )
    rgb, _ = _pad(
        [obs.rgb[index] for obs, index in zip(observations, point_indices, strict=True)]
    )
    instance_id, _ = _pad(
        [obs.instance_id[index] for obs, index in zip(observations, point_indices, strict=True)],
        -1,
    )
    target_mask, _ = _pad(
        [obs.target_mask[index] for obs, index in zip(observations, point_indices, strict=True)],
        False,
    )
    object_present, object_mask = _pad(
        [obs.object_present for obs in observations], False
    )
    object_active, _ = _pad([obs.object_active for obs in observations], False)
    object_category_id, _ = _pad(
        [obs.object_category_id for obs in observations], -1
    )

    # ``packed`` was checked above; keep the local alias type-agnostic so this
    # remains compatible with both GlobalGraspSample and legacy UnifiedSample.
    labels = packed
    global_valid, _ = _pad([label.valid_mask for label in labels], False)
    scene_executable, _ = _pad([label.scene_executable for label in labels], -1)
    result: dict[str, Any] = {
        "xyz": xyz.float(),
        "rgb": rgb.float(),
        "point_mask": point_mask.bool(),
        "instance_id": instance_id.long(),
        "target_mask": target_mask.bool(),
        "target_object": torch.tensor(
            [obs.target_object for obs in observations], dtype=torch.long
        ),
        "task_region_id": torch.tensor(
            [obs.task_region_id for obs in observations], dtype=torch.long
        ),
        "task_category_id": torch.tensor(
            [obs.object_category_id[obs.target_object] for obs in observations],
            dtype=torch.long,
        ),
        "object_mask": object_mask.bool(),
        "object_present": object_present.bool(),
        "object_active": object_active.bool(),
        # Loss-side category GT is used only for Hungarian instance matching.
        "object_category_id": object_category_id.long(),
        "global_loss_sample_valid": torch.tensor(
            [
                bool(getattr(sample, "global_loss_valid", True))
                and sample.global_grasps is not None
                for sample in samples
            ],
            dtype=torch.bool,
        ),
        "global_grasp_labels": {
            "object_index": _pad([label.object_index for label in labels], -1)[0].long(),
            "source_grasp_index": _pad(
                [label.source_grasp_index for label in labels], -1
            )[0].long(),
            "contact_point_world": _pad(
                [label.contact_point_world for label in labels], np.nan
            )[0].float(),
            "grasp_pose_world": _pad(
                [label.grasp_pose_world for label in labels], np.nan
            )[0].float(),
            "approach_direction_world": _pad(
                [label.approach_direction_world for label in labels], np.nan
            )[0].float(),
            "width_m": _pad([label.width_m for label in labels], np.nan)[0].float(),
            "intrinsic_stable": _pad(
                [label.intrinsic_stable for label in labels], False
            )[0].bool(),
            "scene_executable": scene_executable.to(torch.int8),
            "anchor_visible_distance_m": _pad(
                [label.anchor_visible_distance_m for label in labels], np.nan
            )[0].float(),
            "valid_mask": global_valid.bool(),
            "label_set_complete": torch.tensor(
                [label.label_set_complete for label in labels], dtype=torch.bool
            ),
        },
        # Retained only for precise error diagnostics; Trainer._move leaves
        # these dataclass objects on CPU.
        "samples": samples,
    }
    # The neutral PTv3 stream still consumes the fused world cloud above, while
    # every GraspNet forward receives an independently sampled real-camera view.
    result.update(
        _camera_view_payload(
            observations,
            view_index=graspnet_view_index,
            grid_size_m=grid_size_m,
            training=training,
            point_count=graspnet_point_count,
        )
    )
    if grid_coord is not None:
        result["grid_coord"] = grid_coord.to(torch.int32)
    return result

