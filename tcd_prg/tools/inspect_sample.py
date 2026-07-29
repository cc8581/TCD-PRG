"""Print the exact unified contract for one real state group."""

from __future__ import annotations

import argparse
import json

import numpy as np

from tcd_prg.config import load_config
from tcd_prg.runtime import create_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--scene-id", type=int)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=False)
    unit = next(
        unit for unit in adapter.iter_action_groups(None)
        if (args.scene_id is None or unit[0] == args.scene_id) and unit[3] == args.group_index
    )
    sample = adapter.load_sample(*unit)
    result = {
        "unit": unit,
        "observation": {
            "points": sample.observation.xyz.shape,
            "objects": len(sample.observation.object_uuid),
            "visible_instances": np.unique(sample.observation.instance_id).tolist(),
            "target_object": sample.observation.target_object,
            "task_region": sample.observation.task_region_id,
            "camera_types": [item.sensor_type for item in sample.observation.camera_parameters],
        },
        "state": {
            "graspable": sample.state_labels.graspable,
            "verified_positive_grasp_count": sample.state_labels.verified_positive_grasp_count,
            "required_grasp_count": sample.state_labels.required_grasp_count,
            "relations": sample.state_labels.relation_names,
            "topology_valid": sample.state_labels.sequence_topology_valid,
        },
        "candidates": {
            "count": len(sample.candidates.action_type),
            "type_counts": {
                str(kind): int(np.sum(sample.candidates.action_type == kind)) for kind in range(3)
            },
            "evaluated": int(np.sum(sample.candidates.evaluation_status >= 0)),
            "successful": int(np.sum(sample.candidates.success_mask)),
            "parameter_shapes": {
                key: list(value.shape) for key, value in sample.candidates.action_parameters.items()
            },
        },
        "sequence_count": len(sample.sequences),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
