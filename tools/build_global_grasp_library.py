"""Build task-unfiltered ACRONYM-to-AG global grasp libraries.

This tool reuses the dataset's contact-corridor geometry but intentionally
removes every functional-region purity filter. It writes both stable and
explicitly failed source grasps whenever contact geometry can be recovered.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np


FORMAT = "tcd_prg_global_grasp_library_v1"
CONVERSION_VERSION = "panda_to_ag_contact_midpoint_v1"


@dataclass(frozen=True)
class ContactGeometry:
    closing_center_z: float = 0.089
    closing_half_x: float = 0.041
    closing_half_y: float = 0.014
    closing_z_min: float = 0.065
    closing_z_max: float = 0.113
    query_radius: float = 0.065
    contact_band: float = 0.005
    minimum_corridor_points: int = 6
    minimum_points_per_side: int = 3


def _contact(points: np.ndarray, transform: np.ndarray, criteria: ContactGeometry):
    rotation, translation = transform[:3, :3], transform[:3, 3]
    center = translation + rotation @ np.asarray([0.0, 0.0, criteria.closing_center_z])
    distance_sq = np.sum((points - center) ** 2, axis=1)
    nearby = np.flatnonzero(distance_sq <= criteria.query_radius**2)
    if len(nearby) < criteria.minimum_corridor_points:
        return None
    local = (points[nearby] - translation) @ rotation
    inside = (
        (np.abs(local[:, 0]) <= criteria.closing_half_x)
        & (np.abs(local[:, 1]) <= criteria.closing_half_y)
        & (local[:, 2] >= criteria.closing_z_min)
        & (local[:, 2] <= criteria.closing_z_max)
    )
    nearby, local = nearby[inside], local[inside]
    if len(nearby) < criteria.minimum_corridor_points:
        return None
    negative = local[:, 0] <= float(local[:, 0].min()) + criteria.contact_band
    positive = local[:, 0] >= float(local[:, 0].max()) - criteria.contact_band
    if min(int(negative.sum()), int(positive.sum())) < criteria.minimum_points_per_side:
        return None
    contacts = np.stack((points[nearby[negative]].mean(0), points[nearby[positive]].mean(0)))
    return contacts, float(local[:, 0].max() - local[:, 0].min())


def _find_h5(acronym_root: Path, stem: str) -> Path:
    matches = list((acronym_root / "grasps").rglob(f"{stem}.h5"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one ACRONYM file for {stem}, found {len(matches)}")
    return matches[0]


def convert_one(task_adapter: Path, annotations: Path, acronym_root: Path, output: Path) -> dict:
    with np.load(task_adapter, allow_pickle=False) as metadata:
        category = str(metadata["category_key"])
        model_id = str(metadata["model_id"])
        scale = float(metadata["object_scale"])
        stem = task_adapter.stem
    annotation_path = annotations / category / f"{model_id}.npz"
    with np.load(annotation_path, allow_pickle=False) as annotation:
        points = np.asarray(annotation["point_xyz"], np.float64) * scale
    h5_path = _find_h5(acronym_root, stem)
    with h5py.File(h5_path, "r") as handle:
        transforms = np.asarray(handle["grasps/transforms"], np.float64)
        stable = np.asarray(handle["grasps/qualities/flex/object_in_gripper"], np.int8) == 1
        quality = stable.astype(np.float32)
        source_gripper = handle["gripper/type"][()].decode("utf-8")
    rows, contacts, spans = [], [], []
    for index, transform in enumerate(transforms):
        geometry = _contact(points, transform, ContactGeometry())
        if geometry is None:
            continue
        contact, span = geometry
        rows.append(index)
        contacts.append(contact)
        spans.append(span)
    rows_array = np.asarray(rows, np.int32)
    contact_array = np.asarray(contacts, np.float32).reshape(-1, 2, 3)
    span_array = np.asarray(spans, np.float32)
    canonical = transforms[rows_array].copy()
    canonical[:, :3, 3] = contact_array.mean(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format=np.asarray(FORMAT), conversion_version=np.asarray(CONVERSION_VERSION),
        model_id=np.asarray(model_id), category_key=np.asarray(category), object_scale=np.float32(scale),
        source_h5=np.asarray(h5_path.relative_to(acronym_root).as_posix()),
        source_gripper=np.asarray(source_gripper),
        source_grasp_index=rows_array,
        canonical_contact_pose_object=canonical.astype(np.float32),
        contact_points_object=contact_array,
        approach_direction_object=canonical[:, :3, 2].astype(np.float32),
        contact_span_m=span_array, ag_width_m=span_array,
        width_compatible=span_array <= np.float32(0.095 + 1e-6),
        stability_label=stable[rows_array], quality=quality[rows_array],
        contact_geometry_json=np.asarray(json.dumps(asdict(ContactGeometry()), sort_keys=True)),
    )
    return {"file": output.as_posix(), "candidate_count": len(rows), "stable_count": int(stable[rows_array].sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acronym-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--task-library", type=Path, required=True, help="Metadata index only; task grasps are not copied")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    records = []
    files = sorted(args.task_library.rglob("*.npz"))
    if args.max_files is not None:
        files = files[: args.max_files]
    for source in files:
        relative = source.relative_to(args.task_library)
        target = args.output / relative
        if target.exists() and not args.overwrite:
            continue
        record = convert_one(source, args.annotations, args.acronym_root, target)
        record["file"] = relative.as_posix()
        records.append(record)
    manifest = {
        "format": FORMAT, "conversion_version": CONVERSION_VERSION,
        "records": records, "source_file_count": len(files),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
