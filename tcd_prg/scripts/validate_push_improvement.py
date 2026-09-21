"""Validate binary PUSH-improvement sidecars before Stage-C training."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.constants import ActionType
from tcd_prg.datasets.push_value import PUSH_IMPROVEMENT_SCHEMA_VERSION, PushImprovementStore
from tcd_prg.runtime import create_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--improvement-root", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    adapter = create_adapter(load_config(args.config, args.overrides), allow_render=False)
    store = PushImprovementStore(args.improvement_root)
    expected = tuple(adapter.snapshot_scene_ids)
    expected_names = {f"scene_{scene_id:04d}.h5" for scene_id in expected}
    actual_names = {path.name for path in Path(args.improvement_root).glob("scene_*.h5")}
    if actual_names != expected_names:
        raise RuntimeError(
            "action sidecar coverage mismatch: "
            f"missing={sorted(expected_names-actual_names)[:10]}, "
            f"extra={sorted(actual_names-expected_names)[:10]}"
        )
    state_total = push_total = valid_total = improved = not_improved = 0
    unexecuted_total = unexecuted_positive = 0
    for scene_id in tqdm(expected, desc="Validate PUSH improvement", unit="scene"):
        with h5py.File(adapter._path_by_scene[int(scene_id)], "r", swmr=True) as handle:
            scene = handle[next(iter(handle.keys()))]
            state_total += len(scene["states/task_index"])
            action_type = scene["actions/action_type"][:].astype(np.int8)
            executed = scene["actions/executed"][:].astype(bool)
            sequence_actions = np.unique(np.concatenate((
                scene["sequences/policy_action_ids"][:].astype(np.int64),
                scene["sequences/terminal_action_ids"][:].astype(np.int64),
            )))
        payload = store.load_scene(scene_id)
        expected_ids = np.flatnonzero(action_type == int(ActionType.PUSH)).astype(np.int64)
        if not np.array_equal(payload["action_id"].astype(np.int64), expected_ids):
            raise RuntimeError(f"scene {scene_id}: PUSH action IDs are not source-aligned")
        target = payload["improvement_target"].astype(np.float32)
        valid = payload["improvement_valid"].astype(bool)
        stored_executed = payload["executed"].astype(bool)
        if target.shape != (len(expected_ids),) or valid.shape != target.shape:
            raise RuntimeError(f"scene {scene_id}: invalid improvement-target shape")
        if not valid.all() or np.any(~np.isin(target, (0.0, 1.0))):
            raise RuntimeError(f"scene {scene_id}: every PUSH target must be valid and binary")
        expected_target = np.isin(expected_ids, sequence_actions).astype(np.float32)
        if not np.array_equal(target, expected_target):
            raise RuntimeError(f"scene {scene_id}: labels do not match successful sequences")
        if not np.array_equal(stored_executed, executed[expected_ids]):
            raise RuntimeError(f"scene {scene_id}: executed diagnostics are not source-aligned")
        push_total += len(expected_ids)
        valid_total += int(valid.sum())
        improved += int((valid & (target > 0.5)).sum())
        not_improved += int((valid & (target <= 0.5)).sum())
        unexecuted_total += int((~stored_executed).sum())
        unexecuted_positive += int(((~stored_executed) & (target > 0.5)).sum())
    print({
        "action_schema_version": PUSH_IMPROVEMENT_SCHEMA_VERSION,
        "scenes": len(expected), "states": state_total, "push_actions": push_total,
        "valid": valid_total, "invalid": push_total-valid_total,
        "improved": improved, "not_improved": not_improved,
        "positive_fraction": improved/max(valid_total, 1),
        "unexecuted": unexecuted_total,
        "unexecuted_positive": unexecuted_positive,
        "definition": "successful_sequence_membership_binary_v1",
        "teacher": "none",
    }, flush=True)


if __name__ == "__main__":
    main()
