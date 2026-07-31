"""Offline scene-level certification for task-free global grasp libraries.

This executable is intentionally separate from the GPU training loop. It uses
the same FR5/AG exact certifier as closed-loop execution and writes tri-state
cache rows aligned by ``(object_index, source_grasp_index)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcd_prg.config import load_config
from tcd_prg.constants import ActionType
from tcd_prg.runtime import create_action_certifier, create_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--scene-id", type=int)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-render", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute existing caches after certifier or geometry changes.",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    adapter = create_adapter(config, allow_render=args.allow_render)
    if not adapter.capabilities.has_global_grasps:
        raise RuntimeError("Generate generic_grasp_library_v1 before scene certification")
    certifier = create_action_certifier(config)
    units: dict[tuple[int, int], int] = {}
    for scene_id, state_id, task_index, _ in adapter.iter_action_groups():
        if args.scene_id is None or scene_id == args.scene_id:
            units.setdefault((scene_id, state_id), task_index)
    selected = sorted(units.items())
    if args.max_states is not None:
        selected = selected[: args.max_states]
    output_root = Path(config.dataset.root) / config.dataset.global_grasp_certification_subdir
    for (scene_id, state_id), task_index in selected:
        output = output_root / f"scene_{scene_id:04d}" / f"state_{state_id:04d}.npz"
        if output.is_file() and not args.overwrite:
            continue
        observation = adapter.load_observation(scene_id, state_id, task_index)
        labels = adapter.load_global_grasps(scene_id, state_id, observation, training=False)
        if labels is None:
            raise RuntimeError("Global labels disappeared after capability check")
        certifier.set_observation(observation)
        actions = [
            {
                "action_type": int(ActionType.PICK_REMOVE),
                "acted_object": int(labels.object_index[index]),
                "grasp_pose_world": labels.grasp_pose_world[index],
                "grasp_width_m": float(labels.width_m[index]),
            }
            for index in range(len(labels.object_index))
        ]
        results = []
        for start in range(0, len(actions), args.batch_size):
            results.extend(certifier.certify_many(actions[start : start + args.batch_size]))
        executable = np.asarray([int(ok) for ok, _ in results], np.int8)
        reasons = np.asarray([reason for _, reason in results], dtype="U64")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary, format=np.asarray("global_grasp_scene_certification_v1"),
            scene_id=np.int64(scene_id), state_id=np.int64(state_id),
            object_index=labels.object_index, source_grasp_index=labels.source_grasp_index,
            scene_executable=executable, reason=reasons,
            conversion_version=np.asarray(labels.conversion_version),
        )
        temporary.replace(output)


if __name__ == "__main__":
    main()
