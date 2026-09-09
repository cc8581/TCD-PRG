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
from tcd_prg.push_improvement import PUSH_IMPROVEMENT_COMPONENT_NAMES
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
            f"action sidecar coverage mismatch: missing={sorted(expected_names-actual_names)[:10]}, "
            f"extra={sorted(actual_names-expected_names)[:10]}"
        )
    state_total = push_total = valid_total = improved = not_improved = 0
    reason_counts = np.zeros(len(PUSH_IMPROVEMENT_COMPONENT_NAMES), np.int64)
    regression_counts = np.zeros_like(reason_counts)
    for scene_id in tqdm(expected, desc="Validate PUSH improvement", unit="scene"):
        with h5py.File(adapter._path_by_scene[int(scene_id)], "r", swmr=True) as handle:
            scene = handle[next(iter(handle.keys()))]
            state_total += len(scene["states/task_index"])
            action_type = scene["actions/action_type"][:].astype(np.int8)
            executed = scene["actions/executed"][:].astype(bool)
        payload = store.load_scene(scene_id)
        expected_ids = np.flatnonzero((action_type == int(ActionType.PUSH)) & executed).astype(np.int64)
        if not np.array_equal(payload["action_id"].astype(np.int64), expected_ids):
            raise RuntimeError(f"scene {scene_id}: PUSH action IDs are not source-aligned")
        target = payload["improvement_target"].astype(np.float32)
        valid = payload["improvement_valid"].astype(bool)
        delta = payload["component_delta"].astype(np.float32)
        reasons = payload["improvement_reason_mask"].astype(np.uint8)
        regressions = payload["regression_mask"].astype(np.uint8)
        if target.shape != (len(expected_ids),) or valid.shape != target.shape:
            raise RuntimeError(f"scene {scene_id}: invalid improvement-target shape")
        if delta.shape != (len(expected_ids), len(PUSH_IMPROVEMENT_COMPONENT_NAMES)):
            raise RuntimeError(f"scene {scene_id}: invalid component-delta shape")
        if np.any(valid & ~np.isin(target, (0.0, 1.0))):
            raise RuntimeError(f"scene {scene_id}: valid targets must be binary")
        if np.any(~np.isfinite(delta[valid])) or np.any(np.isfinite(delta[~valid])):
            raise RuntimeError(f"scene {scene_id}: component-delta validity mismatch")
        strong_reason = reasons & 0b00011111 != 0
        structural_regression = regressions & 0b00011111 != 0
        if np.any((target > 0.5) & structural_regression & ~strong_reason):
            raise RuntimeError(f"scene {scene_id}: weak improvement overrode structural regression")
        push_total += len(expected_ids); valid_total += int(valid.sum())
        improved += int((valid & (target > 0.5)).sum())
        not_improved += int((valid & (target <= 0.5)).sum())
        for index in range(len(reason_counts)):
            reason_counts[index] += int((valid & (reasons & (1 << index) != 0)).sum())
            regression_counts[index] += int((valid & (regressions & (1 << index) != 0)).sum())
    print({
        "action_schema_version": PUSH_IMPROVEMENT_SCHEMA_VERSION,
        "scenes": len(expected), "states": state_total, "push_actions": push_total,
        "valid": valid_total, "invalid": push_total-valid_total,
        "improved": improved, "not_improved": not_improved,
        "positive_fraction": improved/max(valid_total, 1),
        "positive_event_counts": dict(zip(PUSH_IMPROVEMENT_COMPONENT_NAMES, reason_counts.tolist())),
        "regression_event_counts": dict(zip(PUSH_IMPROVEMENT_COMPONENT_NAMES, regression_counts.tolist())),
        "teacher": "none",
    }, flush=True)


if __name__ == "__main__":
    main()
