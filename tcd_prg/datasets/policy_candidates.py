"""Versioned generated-candidate caches and open-world policy matching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tcd_prg.config import ModelConfig, TCDPRGConfig
from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.geometry.se3 import parallel_jaw_rotation_distance, quaternion_xyzw_to_matrix
from tcd_prg.paths import project_path

POLICY_CANDIDATE_CACHE_FORMAT = "tcd_prg_generated_policy_candidates_v2_open_world"
POLICY_CANDIDATE_LABEL_VERSION = "conflict_unknown_effective_rows_v1"

RAW_FIELDS = (
    "type", "object", "contact_world", "direction_world", "pose_world",
    "destination_world", "width_m", "proposal_score", "point_index",
    "direction_bin", "direction_score", "evidence", "valid",
)
LABEL_FIELDS = (
    "label_status", "policy_success", "matched_teacher_index", "match_conflict",
)


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_sha256(path: str | Path) -> str:
    """Hash a file/tree by relative names and contents; missing is explicit."""

    resolved = project_path(path)
    if not resolved.exists():
        return f"missing:{resolved.as_posix()}"
    if resolved.is_file():
        return checkpoint_sha256(resolved)
    digest = hashlib.sha256()
    for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
        digest.update(child.relative_to(resolved).as_posix().encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def generator_signature(config: TCDPRGConfig) -> str:
    """Bind cached candidates to generator code, config, and consumed assets."""

    code_files = (
        "tcd_prg/planners/candidate_generator.py",
        "tcd_prg/planners/tcd_policy.py",
        "tcd_prg/models/tcd_prg.py",
        "tcd_prg/models/dependency_graph/hgt.py",
        "tcd_prg/datasets/policy_candidates.py",
        "tcd_prg/datasets/task_oriented_clutter.py",
        "tcd_prg/datasets/collate.py",
        "tcd_prg/rendering/pybullet_renderer.py",
    )
    fields: dict[str, Any] = {
        "model": asdict(config.model),
        "ablation": asdict(config.ablation),
        "graph": asdict(config.graph),
        "backbone": asdict(config.backbone),
        "observation": asdict(config.observation),
        "dataset": {
            key: value for key, value in asdict(config.dataset).items()
            if key not in {"root", "acronym_root", "functional_region_root"}
        },
        "push_distance_m": config.push_distance_m,
        "label_version": POLICY_CANDIDATE_LABEL_VERSION,
        "code": {path: _path_sha256(path) for path in code_files},
    }
    if config.ablation.use_gripper_scene_verifier:
        fields["gripper_geometry"] = _path_sha256(config.observation.gripper_cache_dir)
        fields["gripper_worker"] = _path_sha256(config.observation.gripper_worker_script)
        fields["robot_urdf"] = _path_sha256(config.dataset.fr5_ag_urdf)
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_cache_key(sample: Any) -> str:
    observation = sample.observation
    action_ids = np.asarray(sample.candidates.candidate_action_ids, dtype=np.int64)
    group = hashlib.sha1(action_ids.tobytes()).hexdigest()[:12]
    return (
        f"scene_{int(observation.scene_id):06d}_state_{int(observation.state_id):04d}_"
        f"task_{int(observation.task_index):04d}_{group}"
    )


def sample_source_signature(sample: Any) -> str:
    """Bind an entry to the exact observation and teacher action outcomes."""

    digest = hashlib.sha256()
    observation = sample.observation
    candidates = sample.candidates
    arrays = (
        observation.xyz, observation.rgb, observation.instance_id,
        observation.target_mask, observation.object_pose,
        observation.object_present, observation.object_active,
        candidates.candidate_action_ids, candidates.action_type,
        candidates.acted_object, candidates.evaluation_status,
        candidates.success_mask, candidates.potential_delta,
    )
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    for key, value in sorted(candidates.action_parameters.items()):
        array = np.ascontiguousarray(value)
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    for sequence in sample.sequences:
        for value in (
            sequence.state_ids, sequence.transition_ids,
            sequence.policy_action_ids, sequence.terminal_action_ids,
        ):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def cache_manifest(
    checkpoint: str | Path, config: TCDPRGConfig, *, certification_scope: str = "none"
) -> dict[str, Any]:
    if certification_scope != "none":
        raise ValueError(
            "Generated policy caches must not pre-filter candidates with robot approach "
            "certification; label generated actions from observed outcomes/rollouts instead"
        )
    return {
        "format": POLICY_CANDIDATE_CACHE_FORMAT,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "generator_signature": generator_signature(config),
        "certification_scope": certification_scope,
        "label_version": POLICY_CANDIDATE_LABEL_VERSION,
    }


def validate_cache_manifest(
    directory: str | Path, config: TCDPRGConfig, expected_checkpoint_sha256: str = ""
) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Generated policy cache manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != POLICY_CANDIDATE_CACHE_FORMAT:
        raise ValueError("Generated policy candidate cache format is incompatible")
    if manifest.get("certification_scope") != "none":
        raise ValueError("Generated policy cache has an unsupported certification scope")
    if manifest.get("label_version") != POLICY_CANDIDATE_LABEL_VERSION:
        raise ValueError("Generated policy candidate label semantics are incompatible")
    if manifest.get("generator_signature") != generator_signature(config):
        raise ValueError(
            "Generated policy candidate cache uses a different generator/matching config"
        )
    if (
        expected_checkpoint_sha256
        and manifest.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise ValueError("Generated policy candidate cache checkpoint SHA-256 mismatch")
    return manifest


def validate_generated_policy_coverage(
    manifest: dict[str, Any], split: str, *, minimum_positive: float,
    minimum_effective: float,
) -> dict[str, float]:
    """Fail fast before pure generated-policy training on a nearly empty subset."""

    statistics = manifest.get("splits", {}).get(split)
    if not statistics:
        raise ValueError(f"Generated policy cache has no coverage statistics for split={split}")
    groups = int(statistics.get("entries", 0))
    positive = int(statistics.get("groups_with_known_positive", 0))
    effective = int(statistics.get("effective_policy_rows", 0))
    positive_coverage = positive / max(1, groups)
    effective_coverage = effective / max(1, groups)
    if positive_coverage < minimum_positive:
        raise ValueError(
            f"Generated positive coverage {positive_coverage:.4f} is below required "
            f"{minimum_positive:.4f}"
        )
    if effective_coverage < minimum_effective:
        raise ValueError(
            f"Effective generated policy-row coverage {effective_coverage:.4f} is below "
            f"required {minimum_effective:.4f}"
        )
    return {
        "positive_coverage": positive_coverage,
        "effective_coverage": effective_coverage,
    }


def _teacher_pose(batch: dict[str, Any], row: int) -> Tensor:
    kind = batch["action_type"][row]
    parameters = batch["action_parameters"]
    return torch.where(
        (kind == int(ActionType.PICK_REMOVE)).unsqueeze(-1),
        parameters["removal_grasp_pose_world"][row],
        parameters["task_grasp_pose_world"][row],
    )


@torch.no_grad()
def match_generated_candidates(
    candidates: dict[str, Tensor], batch: dict[str, Any], row: int, config: ModelConfig
) -> dict[str, Tensor]:
    """Match generated actions to known outcomes without inventing negatives."""

    count = candidates["type"].shape[1]
    status = torch.full((count,), int(CandidateStatus.UNKNOWN_UNTESTED), dtype=torch.int8)
    successful = torch.zeros(count, dtype=torch.bool)
    matched = torch.full((count,), -1, dtype=torch.long)
    conflict = torch.zeros(count, dtype=torch.bool)
    teacher_type = batch["action_type"][row].cpu()
    teacher_object = batch["acted_object"][row].cpu()
    teacher_success = batch["policy_success_mask"][row].cpu()
    teacher_negative = (
        batch["evaluation_status"][row].cpu() == int(CandidateStatus.NEGATIVE)
    )
    teacher_valid = batch["candidate_mask"][row].cpu() & (
        teacher_success | teacher_negative
    )
    parameters = {key: value[row].cpu() for key, value in batch["action_parameters"].items()}
    teacher_pose = _teacher_pose(batch, row).cpu()
    generated = {
        key: value[0].detach().cpu()
        for key, value in candidates.items() if isinstance(value, Tensor)
    }

    for index in torch.nonzero(generated["valid"], as_tuple=False).flatten().tolist():
        kind = int(generated["type"][index])
        domain = teacher_valid & (teacher_type == kind) & (
            teacher_object == generated["object"][index]
        )
        teacher_indices = torch.nonzero(domain, as_tuple=False).flatten()
        if not len(teacher_indices):
            continue
        if kind == int(ActionType.PUSH):
            contact_distance = torch.linalg.vector_norm(
                parameters["push_contact_world"][teacher_indices]
                - generated["contact_world"][index], dim=-1
            )
            first = torch.nn.functional.normalize(
                parameters["push_direction_world"][teacher_indices, :2], dim=-1
            )
            second = torch.nn.functional.normalize(
                generated["direction_world"][index, :2], dim=-1
            )
            angle = torch.rad2deg(torch.acos((first * second).sum(-1).clamp(-1.0, 1.0)))
            compatible = (
                (contact_distance <= config.policy_push_match_contact_m)
                & (angle <= config.policy_push_match_direction_deg)
            )
            cost = (
                contact_distance / config.policy_push_match_contact_m
                + angle / config.policy_push_match_direction_deg
            )
        else:
            pose = teacher_pose[teacher_indices]
            finite = torch.isfinite(pose).all(-1) & torch.isfinite(
                parameters["grasp_width_m"][teacher_indices]
            )
            translation = torch.linalg.vector_norm(
                pose[:, :3] - generated["pose_world"][index, :3], dim=-1
            )
            rotation = torch.rad2deg(parallel_jaw_rotation_distance(
                quaternion_xyzw_to_matrix(torch.nan_to_num(pose[:, 3:], nan=0.0)),
                quaternion_xyzw_to_matrix(
                    generated["pose_world"][index, 3:].expand(len(pose), -1)
                ),
            ))
            width = (
                parameters["grasp_width_m"][teacher_indices]
                - generated["width_m"][index]
            ).abs()
            compatible = finite & (
                (translation <= config.policy_grasp_match_translation_m)
                & (rotation <= config.policy_grasp_match_rotation_deg)
                & (width <= config.policy_grasp_match_width_m)
            )
            cost = (
                translation / config.policy_grasp_match_translation_m
                + rotation / config.policy_grasp_match_rotation_deg
                + width / config.policy_grasp_match_width_m
            )
        matches = teacher_indices[compatible]
        if not len(matches):
            continue
        match_cost = cost[compatible]
        positive_mask = teacher_success[matches]
        negative_mask = teacher_negative[matches]
        if positive_mask.any() and negative_mask.any():
            # Geometrically indistinguishable teacher actions disagree about
            # outcome.  Without repeated-trial success probabilities this is
            # irreducible UNKNOWN, never a forced binary target.
            conflict[index] = True
            continue
        if positive_mask.any():
            positive = matches[positive_mask]
            chosen = positive[match_cost[positive_mask].argmin()]
            status[index] = int(CandidateStatus.POSITIVE)
            successful[index] = True
        else:
            chosen = matches[match_cost.argmin()]
            status[index] = int(CandidateStatus.NEGATIVE)
        matched[index] = chosen
    return {
        "label_status": status,
        "policy_success": successful,
        "matched_teacher_index": matched,
        "match_conflict": conflict,
    }


def save_candidate_entry(
    directory: str | Path, sample: Any, candidates: dict[str, Tensor], labels: dict[str, Tensor]
) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{sample_cache_key(sample)}.npz"
    arrays = {key: candidates[key][0].detach().cpu().numpy() for key in RAW_FIELDS}
    arrays.update({key: value.cpu().numpy() for key, value in labels.items()})
    arrays["source_signature"] = np.asarray(sample_source_signature(sample))
    np.savez_compressed(path, **arrays)
    return path


def load_candidate_batch(
    samples: list[Any], directory: str | Path, config: TCDPRGConfig,
    expected_checkpoint_sha256: str = "", *, validate_manifest: bool = True,
) -> dict[str, Tensor]:
    if validate_manifest:
        validate_cache_manifest(directory, config, expected_checkpoint_sha256)
    entries: list[dict[str, np.ndarray]] = []
    keys = (*RAW_FIELDS, *LABEL_FIELDS)
    for sample in samples:
        path = Path(directory) / f"{sample_cache_key(sample)}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Generated policy candidate cache entry is missing: {path}")
        with np.load(path, allow_pickle=False) as source:
            if str(source["source_signature"].item()) != sample_source_signature(sample):
                raise ValueError(
                    f"Generated policy candidate source data changed for {path.name}"
                )
            entries.append({key: source[key] for key in keys})
    maximum = max(len(entry["type"]) for entry in entries)
    result: dict[str, Tensor] = {}
    fill: dict[str, float | int | bool] = {
        "type": -1, "object": -1, "point_index": -1, "direction_bin": -1,
        "matched_teacher_index": -1,
        "label_status": int(CandidateStatus.UNKNOWN_UNTESTED),
        "match_conflict": False,
        "valid": False, "policy_success": False, "proposal_score": -1.0,
    }
    for key in keys:
        example = entries[0][key]
        value = np.full(
            (len(entries), maximum) + example.shape[1:],
            fill.get(key, np.nan), dtype=example.dtype,
        )
        for row, entry in enumerate(entries):
            value[row, :len(entry[key])] = entry[key]
        result[key] = torch.from_numpy(value)
    for key in ("type", "object", "point_index", "direction_bin", "matched_teacher_index"):
        result[key] = result[key].long()
    result["label_status"] = result["label_status"].to(torch.int8)
    result["valid"] = result["valid"].bool()
    result["policy_success"] = result["policy_success"].bool()
    result["match_conflict"] = result["match_conflict"].bool()
    return result
