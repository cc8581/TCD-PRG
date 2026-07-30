"""Unified in-memory sample structures.

Arrays are NumPy on the data side. ``collate.py`` converts them to tensors. No
model is allowed to depend on dataset-specific filenames or HDF5 paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraParameters:
    sensor_type: str
    width: int
    height: int
    eye_world: np.ndarray
    target_world: np.ndarray
    up_world: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    near_m: float
    far_m: float


@dataclass(slots=True)
class SceneObservation:
    scene_id: int
    state_id: int
    task_index: int
    xyz: np.ndarray
    rgb: np.ndarray
    instance_id: np.ndarray
    target_mask: np.ndarray
    target_object: int
    task_region_id: int
    object_uuid: tuple[str, ...]
    object_pose: np.ndarray
    object_category_id: np.ndarray
    object_present: np.ndarray
    object_active: np.ndarray
    camera_parameters: tuple[CameraParameters, ...]
    point_valid: np.ndarray | None = None
    source_view: np.ndarray | None = None
    region_target: np.ndarray | None = None
    region_valid: np.ndarray | None = None
    task_region_visibility: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n = self.xyz.shape[0]
        if self.xyz.shape != (n, 3) or self.rgb.shape != (n, 3):
            raise ValueError("xyz/rgb must be [N,3]")
        for name, value in (("instance_id", self.instance_id), ("target_mask", self.target_mask)):
            if value.shape != (n,):
                raise ValueError(f"{name} must be [N]")
        if self.object_pose.ndim != 2 or self.object_pose.shape[1] != 7:
            raise ValueError("object_pose must be [O,7] xyzw")
        o = self.object_pose.shape[0]
        if (
            len(self.object_uuid) != o
            or self.object_present.shape != (o,)
            or self.object_active.shape != (o,)
            or self.object_category_id.shape != (o,)
        ):
            raise ValueError("object arrays disagree")
        if any(c.sensor_type.lower() == "oracle" for c in self.camera_parameters):
            raise ValueError("Oracle camera leakage in formal SceneObservation")
        if self.source_view is not None and np.any(self.source_view >= len(self.camera_parameters)):
            raise ValueError("source_view references a disallowed camera")
        if self.region_target is not None and self.region_target.shape != (n,):
            raise ValueError("region_target must be [N]")
        if self.region_valid is not None and self.region_valid.shape != (n,):
            raise ValueError("region_valid must be [N]")
        if (self.region_target is None) != (self.region_valid is None):
            raise ValueError("region_target and region_valid must be supplied together")


@dataclass(slots=True)
class StateLabels:
    relation_graph: np.ndarray
    task_block_graph: np.ndarray | None
    blockers: np.ndarray
    task_pressed: bool
    task_region_pressed: bool
    verified_positive_grasp_count: int
    required_grasp_count: int
    direct_goal_valid: bool
    terminal_goal_valid: bool
    potential_components: np.ndarray
    object_visible_pixels: np.ndarray
    sequence_depth: int
    target_visible_ratio: float
    relation_names: tuple[str, ...]
    direct_blocker_mask: np.ndarray
    indirect_blocker_mask: np.ndarray
    actionable_blocker_mask: np.ndarray
    prerequisite_object_order: np.ndarray
    sequence_topology_valid: bool

    @property
    def graspable(self) -> bool:
        return self.verified_positive_grasp_count >= self.required_grasp_count


@dataclass(slots=True)
class ActionCandidateGroup:
    candidate_action_ids: np.ndarray
    action_type: np.ndarray
    acted_object: np.ndarray
    valid_mask: np.ndarray
    evaluation_status: np.ndarray
    outcome_code: np.ndarray
    from_state: np.ndarray
    to_state: np.ndarray
    after_state_valid: np.ndarray
    after_pose_valid: np.ndarray
    potential_after_valid: np.ndarray
    acted_object_motion_valid: np.ndarray
    target_motion_valid: np.ndarray
    potential_delta: np.ndarray
    success_mask: np.ndarray
    action_parameters: dict[str, np.ndarray]

    @property
    def action_improves_state(self) -> np.ndarray:
        """Local action-effect label; never use this for policy cloning."""

        return self.success_mask

    def validate(self) -> None:
        n = len(self.candidate_action_ids)
        for name in (
            "action_type",
            "acted_object",
            "valid_mask",
            "evaluation_status",
            "outcome_code",
            "from_state",
            "to_state",
            "after_state_valid",
            "after_pose_valid",
            "potential_after_valid",
            "acted_object_motion_valid",
            "target_motion_valid",
            "success_mask",
        ):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must be [{n}]")
        if self.potential_delta.shape[0] != n:
            raise ValueError("potential_delta first dimension mismatch")
        unknown = self.evaluation_status < 0
        if np.any(self.success_mask[unknown]):
            raise ValueError("UNKNOWN_UNTESTED cannot be a successful supervised candidate")


@dataclass(slots=True)
class SequenceLabels:
    state_ids: np.ndarray
    transition_ids: np.ndarray
    policy_action_ids: np.ndarray
    terminal_action_ids: np.ndarray
    final_grasp_source_indices: np.ndarray
    sequence_topology_valid: bool
    task_index: int


@dataclass(slots=True)
class UnifiedSample:
    observation: SceneObservation
    state_labels: StateLabels
    candidates: ActionCandidateGroup
    sequences: tuple[SequenceLabels, ...]
