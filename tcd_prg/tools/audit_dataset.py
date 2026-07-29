"""Bounded audit of published action HDF5 files and unified samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tcd_prg.config import load_config
from tcd_prg.constants import ActionType, CandidateStatus, PUSH_DISTANCE_M
from tcd_prg.runtime import create_adapter
from tcd_prg.observation import ObservationCacheMissError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--output", default="outputs/data_audit_100.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    units = list(islice(adapter.iter_action_groups(None), args.states))
    if len(units) < args.states:
        raise RuntimeError(f"Only {len(units)} completed groups are available")
    issues, types, strata, cache_misses = [], Counter(), Counter(), 0
    for unit in tqdm(units, desc="audit unified states"):
        try:
            labels = adapter.load_state_labels(unit[0], unit[1])
            group = adapter.load_action_group(unit[0], unit[3])
            adapter.load_sequences(unit[0], unit[2])
            group.validate()
            types.update(group.action_type.tolist())
            push = group.action_type == int(ActionType.PUSH)
            if np.any(np.abs(group.action_parameters["push_distance_m"][push] - PUSH_DISTANCE_M) > 1e-6):
                issues.append({"unit": unit, "issue": "push_distance"})
            unknown = group.evaluation_status == int(CandidateStatus.UNKNOWN_UNTESTED)
            if np.any(group.success_mask[unknown]):
                issues.append({"unit": unit, "issue": "unknown_marked_positive"})
            strata["graspable" if labels.graspable else "not_graspable"] += 1
            try:
                observation = adapter.load_observation(unit[0], unit[1], unit[2])
                if any(camera.sensor_type.lower() == "oracle"
                       for camera in observation.camera_parameters):
                    issues.append({"unit": unit, "issue": "oracle_leakage"})
                observation.validate()
            except ObservationCacheMissError:
                cache_misses += 1
        except Exception as error:
            issues.append({"unit": unit, "issue": type(error).__name__, "detail": str(error)})
    report = {
        "audited_groups": len(units),
        "published_scene_snapshot": list(adapter.snapshot_scene_ids),
        "action_type_counts": dict(types),
        "state_counts": dict(strata),
        "issue_count": len(issues),
        "observation_cache_miss_count": cache_misses,
        "issues": issues,
        "capabilities": adapter.capabilities.__dict__ if hasattr(adapter.capabilities, "__dict__") else {
            name: getattr(adapter.capabilities, name) for name in adapter.capabilities.__slots__
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console = {key: value for key, value in report.items()
               if key not in {"issues", "published_scene_snapshot"}}
    console["published_scene_count"] = len(adapter.snapshot_scene_ids)
    print(json.dumps(console, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
