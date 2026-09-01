"""Versioned offline state/action values for finite-horizon Stage-C training."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from tcd_prg.constants import ActionType, OutcomeCode


PUSH_VALUE_SCHEMA_VERSION = 1
PUSH_VALUE_HORIZONS = 5


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


def build_action_value_sidecar(
    scene_label_path: str | Path,
    state_value_path: str | Path,
    output_path: str | Path,
    *,
    gamma: float = 0.95,
    horizons: int = PUSH_VALUE_HORIZONS,
) -> None:
    """Propagate frozen state values over the published preparation-action graph.

    Both PUSH and PICK_REMOVE transitions contribute to continuation values,
    while only PUSH rows are emitted as Stage-C supervision. Unknown/unexecuted
    actions are never assigned a value. Unsafe outcomes receive zero continuation.
    """

    if not 0.0 < gamma <= 1.0 or horizons <= 0:
        raise ValueError("finite-horizon values require gamma in (0,1] and positive horizons")
    with h5py.File(scene_label_path, "r", swmr=True) as handle:
        if len(handle.keys()) != 1:
            raise ValueError("scene label file must contain exactly one scene group")
        scene = handle[next(iter(handle.keys()))]
        states, actions = scene["states"], scene["actions"]
        state_count = len(states["task_index"])
        teacher = load_state_values(state_value_path, state_count)
        action_type = actions["action_type"][:].astype(np.int8)
        executed = actions["executed"][:].astype(bool)
        outcome = actions["outcome_code"][:].astype(np.int8)
        from_state = actions["from_state"][:].astype(np.int64)
        to_state = actions["to_state"][:].astype(np.int64)
        after_valid = actions["after_state_valid"][:].astype(bool)
        task = actions["task_index"][:].astype(np.int64)
        state_task = states["task_index"][:].astype(np.int64)
        potential_delta = actions["potential_delta"][:].astype(np.float32)
        potential_valid = actions["potential_after_valid"][:].astype(bool)
        part_of_sequence = actions["part_of_success_sequence"][:].astype(bool)
        terminal = states["terminal_goal_valid"][:].astype(bool)
        if "direct_goal_valid" in states:
            terminal |= states["direct_goal_valid"][:].astype(bool)

    teacher.validate(state_count)
    base = np.where(terminal | teacher.directly_graspable, 1.0, teacher.graspability).astype(
        np.float32
    )
    base[~teacher.valid & ~terminal] = np.nan
    values = np.repeat(base[None], horizons + 1, axis=0)
    preparation = (action_type == int(ActionType.PUSH)) | (
        action_type == int(ActionType.PICK_REMOVE)
    )
    linked = (
        preparation
        & executed
        & after_valid
        & (from_state >= 0)
        & (from_state < state_count)
        & (to_state >= 0)
        & (to_state < state_count)
        & (task == state_task[from_state.clip(0, state_count - 1)])
        & (task == state_task[to_state.clip(0, state_count - 1)])
    )
    safe = executed & ~np.isin(
        outcome,
        np.asarray(
            [OutcomeCode.UNSTABLE, OutcomeCode.OUT_OF_WORKSPACE, OutcomeCode.OTHER_INVALID],
            np.int8,
        ),
    )
    action_q = np.full((len(action_type), horizons), np.nan, np.float32)
    for horizon in range(1, horizons + 1):
        valid_edge = linked & safe & np.isfinite(values[horizon - 1, to_state.clip(0, state_count - 1)])
        continuation = np.full(len(action_type), np.nan, np.float32)
        continuation[valid_edge] = gamma * values[
            horizon - 1, to_state[valid_edge]
        ]
        unsafe = preparation & executed & ~safe
        continuation[unsafe] = 0.0
        action_q[:, horizon - 1] = continuation
        next_value = base.copy()
        for action_index in np.flatnonzero(np.isfinite(continuation)):
            state_id = int(from_state[action_index])
            if state_id < 0 or state_id >= state_count:
                continue
            candidate = continuation[action_index]
            if not np.isfinite(next_value[state_id]) or candidate > next_value[state_id]:
                next_value[state_id] = candidate
        values[horizon] = next_value

    push = (action_type == int(ActionType.PUSH)) & executed
    action_ids = np.flatnonzero(push).astype(np.int64)

    def writer(output: h5py.File) -> None:
        output.attrs["schema_version"] = PUSH_VALUE_SCHEMA_VERSION
        output.attrs["horizons"] = int(horizons)
        output.attrs["gamma"] = float(gamma)
        output.attrs["stage_b_checkpoint_sha256"] = teacher.stage_b_checkpoint_sha256
        output.attrs["render_protocol_sha256"] = teacher.render_protocol_sha256
        output.attrs["task_grasp_probability_threshold"] = (
            teacher.task_grasp_probability_threshold
        )
        output.create_dataset("action_id", data=action_ids, compression="gzip")
        output.create_dataset("from_state", data=from_state[push], compression="gzip")
        output.create_dataset("to_state", data=to_state[push], compression="gzip")
        output.create_dataset("q_value", data=action_q[push], compression="gzip")
        output.create_dataset("q_valid", data=np.isfinite(action_q[push]), compression="gzip")
        output.create_dataset("safe", data=safe[push], compression="gzip")
        output.create_dataset("safety_valid", data=np.ones(len(action_ids), bool), compression="gzip")
        output.create_dataset("potential_delta", data=potential_delta[push], compression="gzip")
        output.create_dataset("potential_valid", data=potential_valid[push], compression="gzip")
        output.create_dataset(
            "part_of_success_sequence", data=part_of_sequence[push], compression="gzip"
        )

    _atomic_h5(Path(output_path), writer)


class PushActionValueStore:
    """Read-only per-scene action-value lookup used by Stage-C collators."""

    def __init__(self, root: str | Path, horizons: int = PUSH_VALUE_HORIZONS) -> None:
        self.root = Path(root)
        self.horizons = int(horizons)

    def load_scene(self, scene_id: int) -> dict[str, np.ndarray]:
        path = self.root / f"scene_{int(scene_id):04d}.h5"
        with h5py.File(path, "r", swmr=True) as handle:
            if int(handle.attrs.get("schema_version", -1)) != PUSH_VALUE_SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported action-value schema: {path}")
            if int(handle.attrs.get("horizons", -1)) != self.horizons:
                raise RuntimeError(f"Action-value horizon mismatch: {path}")
            return {name: handle[name][:] for name in handle.keys()}
