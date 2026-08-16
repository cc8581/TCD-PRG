"""Build 570 object-local positive/negative priors from original ACRONYM H5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from tcd_prg.constants import CandidateStatus
from tcd_prg.datasets.acronym_grasp_database import ACRONYM_DATABASE_FORMAT


CONTACT_CENTER_Z_M = 0.089
QUALITY_FIELDS = (
    "object_in_gripper",
    "object_motion_during_closing_angular",
    "object_motion_during_closing_linear",
    "object_motion_during_shaking_angular",
    "object_motion_during_shaking_linear",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.asset_manifest.read_text(encoding="utf-8"))["records"]
    output = args.output_root / "objects"
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, record in enumerate(records):
        with h5py.File(record["source_h5"], "r") as source:
            transform = np.asarray(source["grasps/transforms"], dtype=np.float32)
            quality = {
                key: np.asarray(source[f"grasps/qualities/flex/{key}"])
                for key in QUALITY_FIELDS
            }
        evaluated = np.asarray(quality["object_in_gripper"], dtype=np.int8)
        status = np.where(
            evaluated == 1, int(CandidateStatus.POSITIVE), int(CandidateStatus.NEGATIVE)
        ).astype(np.int8)
        rotation = transform[:, :3, :3]
        # ACRONYM stores the Panda reference frame.  TCD object-local identity
        # uses the center of the closing corridor, 89 mm along local approach.
        translation = transform[:, :3, 3] + rotation[:, :, 2] * CONTACT_CENTER_Z_M
        arrays = {
            "format": np.asarray(ACRONYM_DATABASE_FORMAT),
            "record_id": np.asarray(record["record_id"], np.int32),
            "category": np.asarray(record["category"]),
            "model_id": np.asarray(record["model_id"]),
            "object_scale": np.asarray(record["object_scale"], np.float32),
            "source_h5": np.asarray(record["source_h5"]),
            "source_index": np.arange(len(transform), dtype=np.int32),
            "translation_object": translation.astype(np.float32),
            "rotation_object": rotation.astype(np.float32),
            "status": status,
            **{f"quality_{key}": value for key, value in quality.items()},
        }
        destination = output / f"{record['record_id']:03d}.npz"
        temporary = destination.with_suffix(".work.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
        summary.append({
            "record_id": record["record_id"], "model_id": record["model_id"],
            "object_scale": float(record["object_scale"]),
            "positive": int(np.count_nonzero(status == 1)),
            "negative": int(np.count_nonzero(status == 0)), "path": str(destination),
        })
        if (index + 1) % 50 == 0:
            print(json.dumps({"progress": [index + 1, len(records)]}), flush=True)
    manifest = {
        "format": ACRONYM_DATABASE_FORMAT,
        "contact_center_z_m": CONTACT_CENTER_Z_M,
        "identity": "translation plus parallel-jaw-symmetric rotation; width excluded",
        "unmatched_semantics": "UNKNOWN",
        "records": summary,
        "summary": {"objects": len(summary), "grasps": sum(x["positive"] + x["negative"] for x in summary),
                    "positive": sum(x["positive"] for x in summary), "negative": sum(x["negative"] for x in summary)},
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
