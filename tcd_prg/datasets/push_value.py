"""Offline state metadata and binary Stage-C PUSH-improvement labels."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from tcd_prg.constants import ActionType, OutcomeCode
from tcd_prg.push_improvement import (
    PUSH_IMPROVEMENT_COMPONENT_EPS,
    PUSH_IMPROVEMENT_COMPONENT_NAMES,
    PUSH_IMPROVEMENT_DEFINITION,
    improvement_event,
)


PUSH_VALUE_SCHEMA_VERSION = 1  # State-sidecar schema retained for compatibility.
PUSH_IMPROVEMENT_SCHEMA_VERSION = 8


@dataclass(frozen=True, slots=True)
class StateValues:
    """Frozen Stage-B teacher values aligned with one scene's published states."""

    graspability: np.ndarray
    directly_graspable: np.ndarray
    valid: np.ndarray
    stage_b_checkpoint_sha256: str
    render_protocol_sha256: str
    task_grasp_probability_threshold: float = 0.5

    def validate(self, state_count: int) -> "StateValues":
        for name in ("graspability", "directly_graspable", "valid"):
            if getattr(self, name).shape != (state_count,):
                raise ValueError(f"state value {name} must be [{state_count}]")
        if np.any(self.valid & ~np.isfinite(self.graspability)):
            raise ValueError("valid state graspability must be finite")
        if np.any(self.valid & ((self.graspability < 0) | (self.graspability > 1))):
            raise ValueError("valid state graspability must lie in [0,1]")
        if not self.stage_b_checkpoint_sha256 or not self.render_protocol_sha256:
            raise ValueError("state values require Stage-B and render provenance hashes")
        if not 0.0 <= self.task_grasp_probability_threshold <= 1.0:
            raise ValueError("Stage-B decision threshold must lie in [0,1]")
        return self


def load_state_values(path: str | Path, state_count: int) -> StateValues:
    with h5py.File(path, "r", swmr=True) as handle:
        if int(handle.attrs.get("schema_version", -1)) != PUSH_VALUE_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported state-value schema: {path}")
        result = StateValues(
            graspability=handle["graspability"][:].astype(np.float32),
            directly_graspable=handle["directly_graspable"][:].astype(bool),
            valid=handle["valid"][:].astype(bool),
            stage_b_checkpoint_sha256=str(handle.attrs.get("stage_b_checkpoint_sha256", "")),
            render_protocol_sha256=str(handle.attrs.get("render_protocol_sha256", "")),
            task_grasp_probability_threshold=float(
                handle.attrs.get("task_grasp_probability_threshold", float("nan"))
            ),
        )
    return result.validate(state_count)


def _atomic_h5(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as handle:
            writer(handle)
            handle.flush()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_state_values(path: str | Path, values: StateValues) -> None:
    values.validate(len(values.graspability))

    def writer(handle: h5py.File) -> None:
        handle.attrs["schema_version"] = PUSH_VALUE_SCHEMA_VERSION
        handle.attrs["stage_b_checkpoint_sha256"] = values.stage_b_checkpoint_sha256
        handle.attrs["render_protocol_sha256"] = values.render_protocol_sha256
        handle.attrs["task_grasp_probability_threshold"] = (
            values.task_grasp_probability_threshold
        )
        handle.attrs["graspability_definition"] = "max_valid_stage_b_probability"
        handle.create_dataset("graspability", data=values.graspability, compression="gzip")
        handle.create_dataset("directly_graspable", data=values.directly_graspable, compression="gzip")
        handle.create_dataset("valid", data=values.valid, compression="gzip")

    _atomic_h5(Path(path), writer)


def _ragged_state_values(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    return values[int(offsets[index]) : int(offsets[index + 1])]


def _structural_state_progress_keys(
    scene: h5py.Group,
    raw_relation_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build an auditable lexicographic task-progress key for every state.

    Higher is better.  Components are *not* summed or weighted:

    0. terminal/direct task goal reached;
    1. negative dependency-blocker count (direct + propagated prerequisites);
    2. negative direct task-blocker count;
    3. negative task-region-pressed flag;
    4. negative target-pressed flag;
    5. target visible ratio;
    6. clipped verified-grasp progress (tie-break only).

    The dependency closure is identical to ``TaskOrientedClutterAdapter``:
    direct task blockers are expanded upward through support/contact relations.
    This means removing a top prerequisite receives positive progress even when
    the target still has zero immediately executable grasps.
    """
    required_relations = {"support", "contact", "block_path"}
    missing = required_relations - set(raw_relation_names)
    if missing:
        raise ValueError(f"raw relation vocabulary is missing {sorted(missing)}")
    raw_index = {name: raw_relation_names.index(name) for name in raw_relation_names}

    states = scene["states"]
    state_task = states["task_index"][:].astype(np.int64)
    state_count = len(state_task)
    target_by_task = scene["catalog/task_object_index"][:].astype(np.int64)
    relation = states["relation_graph"][:].astype(np.float32)
    object_pose = states["object_pose"][:].astype(np.float32)
    if relation.shape[0] != state_count or object_pose.shape[0] != state_count:
        raise ValueError("state relation/object-pose arrays do not align")

    occlusion_values = states["occlusion_blockers"][:].astype(np.int64)
    occlusion_offsets = states["occlusion_blocker_offsets"][:].astype(np.int64)
    task_occ_values = states["task_occlusion_blockers"][:].astype(np.int64)
    task_occ_offsets = states["task_occlusion_blocker_offsets"][:].astype(np.int64)

    direct_goal = states["direct_goal_valid"][:].astype(bool)
    terminal_goal = states["terminal_goal_valid"][:].astype(bool)
    task_pressed = states["task_pressed"][:].astype(bool)
    region_pressed = states["task_region_pressed"][:].astype(bool)
    visibility = states["target_visible_ratio"][:].astype(np.float32)
    verified = states["verified_positive_grasp_count"][:].astype(np.float32)
    required = states["required_grasp_count"][:].astype(np.float32)

    keys = np.full((state_count, 7), np.nan, np.float32)
    valid = (
        np.isfinite(visibility)
        & np.isfinite(verified)
        & np.isfinite(required)
        & (verified >= 0)
        & (required > 0)
    )
    support_channel = raw_index["support"]
    contact_channel = raw_index["contact"]
    block_path_channel = raw_index["block_path"]

    for state_id in range(state_count):
        task_index = int(state_task[state_id])
        if task_index < 0 or task_index >= len(target_by_task):
            valid[state_id] = False
            continue
        target_object = int(target_by_task[task_index])
        raw_relation = relation[state_id]
        object_count = raw_relation.shape[0]
        if not 0 <= target_object < object_count:
            valid[state_id] = False
            continue

        occlusion = _ragged_state_values(
            occlusion_values, occlusion_offsets, state_id
        ).astype(np.int64, copy=False)
        task_occlusion = _ragged_state_values(
            task_occ_values, task_occ_offsets, state_id
        ).astype(np.int64, copy=False)
        approach = np.flatnonzero(
            raw_relation[:, target_object, block_path_channel] > 0.5
        ).astype(np.int64)
        direct = np.zeros(object_count, dtype=bool)
        direct[task_occlusion[(task_occlusion >= 0) & (task_occlusion < object_count)]] = True
        direct[occlusion[(occlusion >= 0) & (occlusion < object_count)]] = True
        direct[approach] = True

        dependency = direct.copy()
        frontier = list(np.flatnonzero(direct))
        while frontier:
            dependent = int(frontier.pop())
            above = (
                raw_relation[dependent, :, support_channel] > 0.5
            ) | (
                (raw_relation[dependent, :, contact_channel] > 0.5)
                & (
                    object_pose[state_id, :, 2]
                    > object_pose[state_id, dependent, 2] + 0.005
                )
            )
            for prerequisite in np.flatnonzero(above & ~dependency):
                dependency[int(prerequisite)] = True
                frontier.append(int(prerequisite))

        grasp_progress = float(np.clip(verified[state_id] / required[state_id], 0.0, 1.0))
        keys[state_id] = np.asarray(
            [
                float(direct_goal[state_id] or terminal_goal[state_id]),
                -float(dependency.sum()),
                -float(direct.sum()),
                -float(region_pressed[state_id]),
                -float(task_pressed[state_id]),
                float(np.clip(visibility[state_id], 0.0, 1.0)),
                grasp_progress,
            ],
            np.float32,
        )
    valid &= np.isfinite(keys).all(-1)
    return keys, valid


def build_push_improvement_sidecar(
    scene_label_path: str | Path,
    output_path: str | Path,
    *,
    raw_relation_names: tuple[str, ...],
) -> None:
    """Build binary single-step PUSH-improvement supervision.

    Only executed, linked PUSH transitions with a valid post-state and without
    an explicitly invalid/unsafe outcome supervise the evaluator.  Safety is
    *not* a target: such transitions are simply excluded because deployment
    safety is handled by deterministic certification.

    The model never receives the after-state.  The after-state exists only here,
    offline, to define the target order for the current ``(state, action)``.
    """
    with h5py.File(scene_label_path, "r", swmr=True) as handle:
        if len(handle.keys()) != 1:
            raise ValueError("scene label file must contain exactly one scene group")
        scene = handle[next(iter(handle.keys()))]
        states, actions = scene["states"], scene["actions"]
        state_count = len(states["task_index"])
        state_task = states["task_index"][:].astype(np.int64)
        state_key, state_key_valid = _structural_state_progress_keys(
            scene, tuple(raw_relation_names)
        )

        action_type = actions["action_type"][:].astype(np.int8)
        executed = actions["executed"][:].astype(bool)
        outcome = actions["outcome_code"][:].astype(np.int8)
        from_state = actions["from_state"][:].astype(np.int64)
        to_state = actions["to_state"][:].astype(np.int64)
        after_valid = actions["after_state_valid"][:].astype(bool)
        task = actions["task_index"][:].astype(np.int64)
        potential_improved = (
            actions["potential_improved"][:].astype(bool)
            if "potential_improved" in actions
            else np.zeros(len(action_type), dtype=bool)
        )

    clipped_from = from_state.clip(0, max(state_count - 1, 0))
    clipped_to = to_state.clip(0, max(state_count - 1, 0))
    linked = (
        executed
        & after_valid
        & (from_state >= 0)
        & (from_state < state_count)
        & (to_state >= 0)
        & (to_state < state_count)
        & (task == state_task[clipped_from])
        & (task == state_task[clipped_to])
    )
    # This is a data-validity filter, not a learned safety target.
    execution_valid = ~np.isin(
        outcome,
        np.asarray(
            [OutcomeCode.UNSTABLE, OutcomeCode.OUT_OF_WORKSPACE, OutcomeCode.OTHER_INVALID],
            np.int8,
        ),
    )
    improvement_valid = (
        linked
        & execution_valid
        & state_key_valid[clipped_from]
        & state_key_valid[clipped_to]
    )

    improvement_target = np.zeros(len(action_type), np.float32)
    component_delta = np.full((len(action_type), state_key.shape[1]), np.nan, np.float32)
    improvement_reason_mask = np.zeros(len(action_type), np.uint8)
    regression_mask = np.zeros(len(action_type), np.uint8)
    for action_index in np.flatnonzero(improvement_valid):
        source = int(from_state[action_index])
        target = int(to_state[action_index])
        improved, positive, regression, delta = improvement_event(
            state_key[source], state_key[target]
        )
        improvement_target[action_index] = float(improved)
        component_delta[action_index] = delta
        improvement_reason_mask[action_index] = sum(
            (1 << index) for index in np.flatnonzero(positive)
        )
        regression_mask[action_index] = sum(
            (1 << index) for index in np.flatnonzero(regression)
        )

    push = (action_type == int(ActionType.PUSH)) & executed
    action_ids = np.flatnonzero(push).astype(np.int64)

    def writer(output: h5py.File) -> None:
        output.attrs["schema_version"] = PUSH_IMPROVEMENT_SCHEMA_VERSION
        output.attrs["improvement_definition"] = PUSH_IMPROVEMENT_DEFINITION
        output.attrs["teacher"] = "none"
        output.attrs["learned_heads"] = "improvement_logit_only"
        output.attrs["component_definition"] = (
            "goal,-dependency_blockers,-direct_blockers,-task_region_pressed,"
            "-target_pressed,target_visibility,verified_grasp_progress"
        )
        output.attrs["component_eps"] = np.asarray(PUSH_IMPROVEMENT_COMPONENT_EPS, np.float64)
        output.attrs["visibility_eps"] = PUSH_IMPROVEMENT_COMPONENT_EPS[5]
        output.attrs["components"] = ",".join(PUSH_IMPROVEMENT_COMPONENT_NAMES)
        output.create_dataset("action_id", data=action_ids, compression="gzip")
        output.create_dataset("from_state", data=from_state[push], compression="gzip")
        output.create_dataset("to_state", data=to_state[push], compression="gzip")
        output.create_dataset("improvement_target", data=improvement_target[push], compression="gzip")
        output.create_dataset("improvement_valid", data=improvement_valid[push], compression="gzip")
        output.create_dataset("component_delta", data=component_delta[push], compression="gzip")
        output.create_dataset("improvement_reason_mask", data=improvement_reason_mask[push], compression="gzip")
        output.create_dataset("regression_mask", data=regression_mask[push], compression="gzip")
        output.create_dataset("outcome_code", data=outcome[push], compression="gzip")
        # Diagnostic only: never fed to the model/loss.
        output.create_dataset(
            "dataset_potential_improved", data=potential_improved[push], compression="gzip"
        )

    _atomic_h5(Path(output_path), writer)


class PushImprovementStore:
    """Read-only single-head Stage-C supervision lookup."""

    def __init__(self, root: str | Path, horizons: int | None = None) -> None:
        del horizons
        self.root = Path(root)

    def load_scene(self, scene_id: int) -> dict[str, np.ndarray]:
        path = self.root / f"scene_{int(scene_id):04d}.h5"
        with h5py.File(path, "r", swmr=True) as handle:
            if int(handle.attrs.get("schema_version", -1)) != PUSH_IMPROVEMENT_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported improvement schema: {path}; rebuild Stage-C binary sidecars"
                )
            if str(handle.attrs.get("improvement_definition", "")) != PUSH_IMPROVEMENT_DEFINITION:
                raise RuntimeError(f"Action-improvement definition mismatch: {path}")
            stored_eps = np.asarray(handle.attrs.get("component_eps", []), np.float64)
            expected_eps = np.asarray(PUSH_IMPROVEMENT_COMPONENT_EPS, np.float64)
            if stored_eps.shape != expected_eps.shape or not np.array_equal(stored_eps, expected_eps):
                raise RuntimeError(f"Action-improvement thresholds mismatch: {path}")
            if str(handle.attrs.get("teacher", "")) != "none":
                raise RuntimeError(f"Stage-C structural sidecar must not contain a teacher: {path}")
            required = {
                "action_id", "improvement_target", "improvement_valid",
                "component_delta", "improvement_reason_mask", "regression_mask",
            }
            missing = required - set(handle.keys())
            if missing:
                raise RuntimeError(f"Stage-C sidecar is missing {sorted(missing)}: {path}")
            return {name: handle[name][:] for name in handle.keys()}


class PushStateValueStore:
    """Read-only current-state Stage-B values used as causal Stage-C input."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def load_scene(self, scene_id: int) -> dict[str, np.ndarray]:
        path = self.root / f"scene_{int(scene_id):04d}.h5"
        with h5py.File(path, "r", swmr=True) as handle:
            if int(handle.attrs.get("schema_version", -1)) != PUSH_VALUE_SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported state-value schema: {path}")
            return {
                "graspability": handle["graspability"][:].astype(np.float32),
                "valid": handle["valid"][:].astype(bool),
            }
