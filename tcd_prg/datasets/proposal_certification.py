"""Versioned proposal-level grasp certification caches.

Object-local intrinsic evidence is reusable across scenes.  Scene execution
evidence is state-specific.  Every boolean target has an independent validity
mask so an untested proposal remains UNKNOWN rather than becoming a negative.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tcd_prg.constants import CandidateStatus


PROPOSAL_CERTIFICATION_FORMAT = "tcd_prg_proposal_certification_v1"

REQUIRED_FIELDS = (
    "proposal_pose_object", "proposal_pose_world", "graspnet_width_m",
    "graspnet_score", "intrinsic_status", "intrinsic_valid",
    "ag_width_target_m", "ag_width_valid", "contact_left_object",
    "contact_right_object", "contact_valid", "force_closure_score",
    "force_closure_valid", "collision_free", "collision_valid",
    "approach_feasible", "approach_valid", "reachable", "reachability_valid",
    "scene_executable", "scene_executable_valid", "task_status",
)


def object_local_key(
    model_id: str, object_scale: float, pose_object: np.ndarray, *,
    translation_quantization_m: float = 0.001,
    rotation_quantization: float = 1e-4,
) -> str:
    """Stable reuse key for nearby object-local proposal geometry."""

    pose = np.asarray(pose_object, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("pose_object must be one finite xyzw pose [7]")
    translation = np.rint(pose[:3] / translation_quantization_m).astype(np.int64)
    quaternion = pose[3:].copy()
    if quaternion[np.argmax(np.abs(quaternion))] < 0:
        quaternion *= -1
    rotation = np.rint(quaternion / rotation_quantization).astype(np.int64)
    payload = json.dumps(
        [str(model_id), round(float(object_scale), 10), translation.tolist(), rotation.tolist()],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _status_valid(status: np.ndarray) -> bool:
    return bool(np.isin(status, [int(x) for x in CandidateStatus]).all())


def validate_entry(entry: dict[str, np.ndarray]) -> int:
    missing = set(REQUIRED_FIELDS) - entry.keys()
    if missing:
        raise KeyError(f"Proposal certification entry is missing {sorted(missing)}")
    count = len(np.asarray(entry["intrinsic_status"]))
    for key in REQUIRED_FIELDS:
        if len(np.asarray(entry[key])) != count:
            raise ValueError(f"{key} does not have proposal count {count}")
    for key in ("intrinsic_status", "task_status"):
        if not _status_valid(np.asarray(entry[key])):
            raise ValueError(f"{key} contains an unsupported tri-state value")
    intrinsic_status = np.asarray(entry["intrinsic_status"], dtype=np.int8)
    intrinsic_valid = np.asarray(entry["intrinsic_valid"], dtype=bool)
    if np.any(~intrinsic_valid & (intrinsic_status != int(CandidateStatus.UNKNOWN_UNTESTED))):
        raise ValueError("Uncertified intrinsic proposals must remain UNKNOWN")
    executable = np.asarray(entry["scene_executable"], dtype=bool)
    executable_valid = np.asarray(entry["scene_executable_valid"], dtype=bool)
    components_valid = (
        np.asarray(entry["collision_valid"], bool)
        & np.asarray(entry["approach_valid"], bool)
        & np.asarray(entry["reachability_valid"], bool)
    )
    if np.any(executable_valid & ~components_valid):
        raise ValueError("scene_executable cannot be known before every scene component")
    expected = (
        np.asarray(entry["collision_free"], bool)
        & np.asarray(entry["approach_feasible"], bool)
        & np.asarray(entry["reachable"], bool)
    )
    if np.any(executable_valid & (executable != expected)):
        raise ValueError("scene_executable disagrees with certified components")
    return count


def save_entry(path: str | Path, entry: dict[str, Any], metadata: dict[str, Any]) -> Path:
    arrays = {key: np.asarray(value) for key, value in entry.items()}
    count = validate_entry(arrays)
    arrays["format"] = np.asarray(PROPOSAL_CERTIFICATION_FORMAT)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    arrays["proposal_count"] = np.asarray(count, dtype=np.int32)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".work.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    return output


def load_entry(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        if str(source["format"]) != PROPOSAL_CERTIFICATION_FORMAT:
            raise ValueError("Proposal certification cache format is incompatible")
        entry = {key: np.asarray(source[key]) for key in REQUIRED_FIELDS}
        metadata = json.loads(str(source["metadata_json"]))
    validate_entry(entry)
    return entry, metadata
