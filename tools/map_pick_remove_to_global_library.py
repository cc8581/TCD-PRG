"""Associate legacy PICK_REMOVE grasps with the task-free global library.

No physics sequence is regenerated. Exact source-index matches are preferred;
SE(3)+width matching is used only when the original ACRONYM index is absent.
Unmatched actions are retained explicitly with their original world pose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tcd_prg.geometry.numpy_se3 import (
    compose_pose_with_transform, quaternion_xyzw_to_matrix_numpy,
)


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = quaternion_xyzw_to_matrix_numpy(first)
    b = quaternion_xyzw_to_matrix_numpy(second)
    symmetry = np.diag([-1.0, -1.0, 1.0])
    values = []
    for target in (b, b @ symmetry):
        cosine = np.clip((np.trace(a.T @ target) - 1) * 0.5, -1.0, 1.0)
        values.append(float(np.degrees(np.arccos(cosine))))
    return min(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-h5", type=Path, required=True)
    parser.add_argument("--scene-npz", type=Path, required=True)
    parser.add_argument("--global-library", type=Path, required=True)
    parser.add_argument("--step-labels", type=Path, required=True)
    parser.add_argument("--task-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translation-m", type=float, default=0.01)
    parser.add_argument("--rotation-deg", type=float, default=15.0)
    parser.add_argument("--width-m", type=float, default=0.005)
    args = parser.parse_args()
    with np.load(args.scene_npz, allow_pickle=False) as scene_npz:
        h5_names = tuple(str(x) for x in scene_npz["object_h5_name"])
        categories = tuple(str(x) for x in scene_npz["object_category_key"])
    with np.load(args.step_labels, allow_pickle=False) as steps:
        old_library_files = tuple(str(x) for x in steps["object_match_file"])
    with h5py.File(args.scene_h5, "r", swmr=True) as handle:
        scene_key = next(key for key in handle if key.startswith("scene_"))
        scene = handle[scene_key]
        actions = scene["actions"]
        action_type = actions["action_type"][:]
        action_ids = np.flatnonzero(action_type == 1)
        payload = actions["payload_index"][:][action_ids]
        object_index = actions["pick_remove/acted_object"][:][payload]
        old_source = actions["pick_remove/removal_grasp_source_index"][:][payload]
        old_pose = actions["pick_remove/removal_grasp_pose_world"][:][payload]
        from_state = actions["from_state"][:][action_ids]
        state_pose = scene["states/object_pose"][:]
    matched_row = np.full(len(action_ids), -1, np.int64)
    status = np.full(len(action_ids), "unmatched", dtype="U16")
    translation = np.full(len(action_ids), np.nan, np.float32)
    rotation = np.full(len(action_ids), np.nan, np.float32)
    width_error = np.full(len(action_ids), np.nan, np.float32)
    old_width = np.full(len(action_ids), np.nan, np.float32)
    library_file = np.empty(len(action_ids), dtype="U256")
    for index, (obj, source, pose, state) in enumerate(zip(object_index, old_source, old_pose, from_state, strict=True)):
        with np.load(args.task_library / old_library_files[int(obj)], allow_pickle=False) as legacy:
            legacy_rows = np.flatnonzero(legacy["source_grasp_index"] == source)
            if len(legacy_rows) == 1:
                old_width[index] = float(legacy["contact_span"][legacy_rows[0]])
        path = args.global_library / categories[int(obj)] / f"{Path(h5_names[int(obj)]).stem}.npz"
        library_file[index] = path.relative_to(args.global_library).as_posix()
        with np.load(path, allow_pickle=False) as library:
            rows = np.flatnonzero(library["source_grasp_index"] == source)
            if len(rows):
                candidates = rows
                status[index] = "source_index"
            else:
                candidates = np.arange(len(library["source_grasp_index"]))
                status[index] = "se3"
            best = None
            best_cost = float("inf")
            for row in candidates:
                world = compose_pose_with_transform(
                    state_pose[int(state), int(obj)], library["canonical_contact_pose_object"][row]
                )
                et = float(np.linalg.norm(world[:3] - pose[:3]))
                er = rotation_error_deg(world[3:], pose[3:])
                ew = abs(float(library["ag_width_m"][row]) - float(old_width[index]))
                cost = et / args.translation_m + er / args.rotation_deg + ew / args.width_m
                if cost < best_cost:
                    best_cost, best = cost, (int(row), et, er, ew)
            if (
                best is not None and best[1] <= args.translation_m
                and best[2] <= args.rotation_deg and best[3] <= args.width_m
            ):
                matched_row[index], translation[index], rotation[index], width_error[index] = best
            else:
                matched_row[index] = -1
                status[index] = "unmatched"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, format=np.asarray("pick_remove_global_grasp_association_v1"),
        action_id=action_ids.astype(np.int64), object_index=object_index.astype(np.int64),
        old_source_grasp_index=old_source.astype(np.int64), old_pose_world=old_pose.astype(np.float32),
        old_width_m=old_width,
        global_library_file=library_file, global_library_row=matched_row,
        match_status=status, translation_error_m=translation,
        rotation_error_deg=rotation, width_error_m=width_error,
    )


if __name__ == "__main__":
    main()
