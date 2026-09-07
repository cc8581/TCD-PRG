"""Validate GT action sidecars before formal Stage-C training."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.constants import ActionType
from tcd_prg.datasets.push_value import (
    PUSH_VALUE_HORIZONS,
    PUSH_ACTION_VALUE_SCHEMA_VERSION,
    PushActionValueStore,
)
from tcd_prg.runtime import create_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--action-value-root", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    action_store = PushActionValueStore(
        args.action_value_root, config.training.push_value_horizons
    )
    expected = tuple(adapter.snapshot_scene_ids)
    expected_names = {f"scene_{scene_id:04d}.h5" for scene_id in expected}
    action_names = {path.name for path in Path(args.action_value_root).glob("scene_*.h5")}
    if action_names != expected_names:
        missing = sorted(expected_names - action_names)[:10]
        extra = sorted(action_names - expected_names)[:10]
        raise RuntimeError(f"action sidecar coverage mismatch: missing={missing}, extra={extra}")

    state_total = push_total = valid_q_total = positive_q_total = 0
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
        q = payload["q_value"].astype(np.float32)
        q_valid = payload["q_valid"].astype(bool)
        expected_shape = (len(expected_ids), config.training.push_value_horizons)
        if q.shape != expected_shape or q_valid.shape != expected_shape:
            raise RuntimeError(f"scene {scene_id}: invalid Q shape")
        if np.any(q_valid & (~np.isfinite(q) | (q < 0) | (q > 1))):
            raise RuntimeError(f"scene {scene_id}: invalid finite-horizon Q values")
        if np.any(~q_valid & np.isfinite(q)):
            raise RuntimeError(f"scene {scene_id}: invalid Q entries must remain NaN")
        if np.any(np.diff(np.where(q_valid, q, 0.0), axis=1) < -1e-6):
            raise RuntimeError(f"scene {scene_id}: Q horizons must be monotonic")
        if not np.asarray(payload["safety_valid"], bool).all():
            raise RuntimeError(f"scene {scene_id}: missing PUSH safety labels")
        push_total += len(expected_ids)
        valid_q_total += int(q_valid.sum())
        positive_q_total += int(np.count_nonzero(q_valid & (q > 0)))

    print(
        {
            "action_schema_version": PUSH_ACTION_VALUE_SCHEMA_VERSION,
            "horizons": PUSH_VALUE_HORIZONS,
            "scenes": len(expected),
            "states": state_total,
            "push_actions": push_total,
            "valid_q_values": valid_q_total,
            "positive_q_values": positive_q_total,
            "teacher": "none",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
