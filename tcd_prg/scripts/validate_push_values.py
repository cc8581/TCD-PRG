"""Validate GT action sidecars before formal Stage-C training."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.constants import ActionType
from tcd_prg.datasets.push_value import PUSH_ACTION_VALUE_SCHEMA_VERSION, PushActionValueStore
from tcd_prg.runtime import create_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--action-value-root", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    action_store = PushActionValueStore(args.action_value_root)
    expected = tuple(adapter.snapshot_scene_ids)
    expected_names = {f"scene_{scene_id:04d}.h5" for scene_id in expected}
    action_names = {path.name for path in Path(args.action_value_root).glob("scene_*.h5")}
    if action_names != expected_names:
        missing = sorted(expected_names - action_names)[:10]
        extra = sorted(action_names - expected_names)[:10]
        raise RuntimeError(f"action sidecar coverage mismatch: missing={missing}, extra={extra}")

    state_total = push_total = valid_total = 0
    improved = neutral = worsened = 0
    for scene_id in tqdm(expected, desc="Validate PUSH values", unit="scene"):
        label_path = adapter._path_by_scene[int(scene_id)]
        with h5py.File(label_path, "r", swmr=True) as handle:
            scene = handle[next(iter(handle.keys()))]
            state_count = len(scene["states/task_index"])
            action_type = scene["actions/action_type"][:].astype(np.int8)
            executed = scene["actions/executed"][:].astype(bool)
        state_total += state_count
        payload = action_store.load_scene(scene_id)
        expected_ids = np.flatnonzero(
            (action_type == int(ActionType.PUSH)) & executed
        ).astype(np.int64)
        if not np.array_equal(payload["action_id"].astype(np.int64), expected_ids):
            raise RuntimeError(f"scene {scene_id}: PUSH action IDs are not source-aligned")
        target = payload["value_target"].astype(np.float32)
        valid = payload["value_valid"].astype(bool)
        before = payload["before_rank_key"].astype(np.float32)
        after = payload["after_rank_key"].astype(np.float32)
        if target.shape != (len(expected_ids),) or valid.shape != target.shape:
            raise RuntimeError(f"scene {scene_id}: invalid value target shape")
        if before.shape != after.shape or before.shape != (len(expected_ids), 7):
            raise RuntimeError(f"scene {scene_id}: invalid rank-key shape")
        allowed = np.isin(target, np.asarray([-1.0, 0.0, 1.0], np.float32))
        if np.any(valid & (~np.isfinite(target) | ~allowed)):
            raise RuntimeError(f"scene {scene_id}: invalid core PUSH values")
        if np.any(~valid & np.isfinite(target)):
            raise RuntimeError(f"scene {scene_id}: invalid targets must remain NaN")
        if np.any(~np.isfinite(before[valid])) or np.any(~np.isfinite(after[valid])):
            raise RuntimeError(f"scene {scene_id}: valid rank keys must be finite")
        push_total += len(expected_ids)
        valid_total += int(valid.sum())
        improved += int((valid & (target > 0)).sum())
        neutral += int((valid & (target == 0)).sum())
        worsened += int((valid & (target < 0)).sum())

    print(
        {
            "action_schema_version": PUSH_ACTION_VALUE_SCHEMA_VERSION,
            "scenes": len(expected),
            "states": state_total,
            "push_actions": push_total,
            "valid_push_values": valid_total,
            "improved": improved,
            "neutral": neutral,
            "worsened": worsened,
            "teacher": "none",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
