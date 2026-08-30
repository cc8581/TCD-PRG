"""Strict deployment composition for independently trained A/B/C checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from tcd_prg.datasets.stageb_manifest import compatibility_provenance, stageb_compatibility

STAGE_PREFIXES = {
    "perception": ("encoder.", "region_head."),
    "grasp": ("task_grasp.",),
    "push": ("push.",),
}

PUSH_EVALUATOR_PROTOCOL_VERSION = 1


def resolve_staged_checkpoint_root(root: str | Path) -> dict[str, Path]:
    """Resolve the four stage-best checkpoints from one portable run directory."""

    root = Path(root).expanduser().resolve()
    manifest_path = root / "checkpoints.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("layout") != "tcd_prg_staged_run_v1":
            raise ValueError(f"Unsupported staged checkpoint manifest: {manifest_path}")
        configured = manifest.get("checkpoints", {})
    else:
        configured = {
            stage: f"{stage}/{stage}_best.pt"
            for stage in ("perception", "grasp", "push", "push_evaluator")
        }
    paths: dict[str, Path] = {}
    for stage in ("perception", "grasp", "push", "push_evaluator"):
        relative = configured.get(stage)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Checkpoint manifest is missing stage {stage!r}")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"Checkpoint path escapes the run directory: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {stage} checkpoint: {path}")
        paths[stage] = path
    return paths

PERCEPTION_RUNTIME_MODEL_FIELDS = (
    "instance_queries",
    "instance_decoder_layers",
    "instance_decoder_heads",
    "instance_objectness_threshold",
    "target_query_temperature",
    "target_prompt_radius_m",
    "target_prompt_sigma_m",
    "target_prompt_weight",
    "target_category_weight",
    "target_objectness_weight",
    "target_center_weight",
    "target_learned_weight",
    "target_reid_weight",
    "target_reid_center_weight",
    "target_reid_max_center_distance_m",
    "target_prompt_min_support",
    "target_prompt_min_margin",
)

PUSH_RUNTIME_MODEL_FIELDS = (
    "instance_queries",
    "num_categories",
    "num_task_regions",
    "num_direction_bins",
    "push_contact_match_max_distance_m",
    "push_direction_feature_dim",
    "push_direction_transformer_layers",
    "push_direction_transformer_heads",
    "push_direction_contact_topk",
    "push_object_topk",
    "push_candidates",
    "push_directions_per_contact",
    "max_push_candidates",
    "push_utility_threshold",
    "push_candidate_probability_threshold",
    "push_utility_temperature",
    "push_nms_contact_m",
    "push_nms_direction_deg",
)


def _load_stage(model: nn.Module, path: str | Path, stage: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 12 or payload.get("training_stage") != stage:
        raise RuntimeError(f"{path} is not a schema-12 {stage} checkpoint")
    source = payload.get("ema") or payload["model"]
    current = model.state_dict()
    prefixes = STAGE_PREFIXES[stage]
    required = {name for name in current if name.startswith(prefixes)}
    supplied = {name for name in source if name.startswith(prefixes)}
    if required != supplied:
        missing, extra = sorted(required - supplied), sorted(supplied - required)
        raise RuntimeError(
            f"{stage} checkpoint module mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    incompatible = [name for name in required if current[name].shape != source[name].shape]
    if incompatible:
        raise RuntimeError(f"{stage} checkpoint tensor shape mismatch: {incompatible[:5]}")
    model.load_state_dict({name: source[name] for name in required}, strict=False)
    return payload


def _require_fields(
    payload: dict, runtime_config, sections: dict[str, tuple[str, ...]], stage: str
) -> None:
    saved = payload.get("config", {})
    for section, fields in sections.items():
        current_section = getattr(runtime_config, section)
        saved_section = saved.get(section, {})
        for field in fields:
            if saved_section.get(field) != getattr(current_section, field):
                raise RuntimeError(f"{stage} runtime mismatch: {section}.{field}")


def validate_perception_stage_runtime(payload: dict, runtime_config) -> None:
    _require_fields(
        payload,
        runtime_config,
        {
            "model": PERCEPTION_RUNTIME_MODEL_FIELDS,
            "ablation": ("use_task_region_condition",),
            "backbone": ("backend", "grid_size_m", "patch_size", "attention_points"),
        },
        "perception",
    )


def validate_push_stage_runtime(payload: dict, runtime_config) -> None:
    """Reject shape-invariant decoder/proposal protocol drift."""
    _require_fields(
        payload,
        runtime_config,
        {"model": PUSH_RUNTIME_MODEL_FIELDS, "backbone": ("grid_size_m",)},
        "push",
    )
    saved_ablation = payload.get("config", {}).get("ablation", {})
    saved_use_potential = bool(saved_ablation.get("use_push_potential", False))
    runtime_use_potential = bool(runtime_config.ablation.use_push_potential)
    if saved_use_potential != runtime_use_potential:
        raise RuntimeError("push runtime mismatch: ablation.use_push_potential")
    if runtime_use_potential:
        saved_losses = payload.get("config", {}).get("losses", {})
        if float(saved_losses.get("push_potential", 0.0)) <= 0.0:
            raise RuntimeError(
                "push runtime requires a Stage-C checkpoint trained with push potential"
            )


def load_perception_stage(model: nn.Module, checkpoint: str | Path, runtime_config) -> dict:
    payload = _load_stage(model, checkpoint, "perception")
    validate_perception_stage_runtime(payload, runtime_config)
    return payload


def load_push_stage(model: nn.Module, checkpoint: str | Path, runtime_config) -> dict:
    payload = _load_stage(model, checkpoint, "push")
    validate_push_stage_runtime(payload, runtime_config)
    return payload


def load_staged_tcd_prg(
    model: nn.Module,
    stage_a_checkpoint: str | Path,
    stage_b_checkpoint: str | Path,
    stage_c_checkpoint: str | Path,
    runtime_config,
) -> float:
    stage_a = load_perception_stage(model, stage_a_checkpoint, runtime_config)
    stage_b = _load_stage(model, stage_b_checkpoint, "grasp")
    stage_c = load_push_stage(model, stage_c_checkpoint, runtime_config)
    _require_fields(
        stage_b,
        runtime_config,
        {"model": ("task_grasp_scene_points", "task_grasp_gripper_points")},
        "grasp",
    )
    del stage_a, stage_c
    saved_stageb = compatibility_provenance(stage_b.get("stageb_provenance", {}))
    runtime_stageb = stageb_compatibility(runtime_config)
    for key in ("scene_preprocess", "target_graspnet", "proposal_label_protocol"):
        if saved_stageb.get(key) != runtime_stageb.get(key):
            raise RuntimeError(f"grasp runtime provenance mismatch: {key}")
    if "task_grasp_probability_threshold" not in stage_b:
        raise RuntimeError("Stage-B checkpoint is missing its calibrated deployment threshold")
    return float(stage_b["task_grasp_probability_threshold"])


def push_checkpoint_fingerprint(checkpoint: str | Path) -> tuple[str, str]:
    """Fingerprint the exact Stage-C state used by deployment."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("ema") or payload.get("model", payload)
    source_kind = "ema" if payload.get("ema") else "model"
    tensors = {name: value for name, value in source.items() if name.startswith("push.")}
    if not tensors:
        raise RuntimeError(f"{checkpoint} does not contain Stage-C push.* tensors")
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest(), source_kind


def load_push_evaluator(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    proposal_checkpoint: str | Path | None = None,
) -> None:
    """Load an evaluator and optionally verify its exact Stage-C provenance."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("training_stage") != "push_evaluator":
        raise RuntimeError(f"{checkpoint} is not a push_evaluator checkpoint")
    version = int(payload.get("push_evaluator_protocol_version", -1))
    if version != PUSH_EVALUATOR_PROTOCOL_VERSION:
        raise RuntimeError(
            f"{checkpoint} uses PUSH evaluator protocol {version}; "
            f"runtime requires {PUSH_EVALUATOR_PROTOCOL_VERSION}. "
            "Retrain the evaluator under the current exact-action protocol."
        )
    if proposal_checkpoint is not None:
        expected = payload.get("proposal_state_fingerprint")
        expected_source = payload.get("proposal_state_source")
        if not expected:
            raise RuntimeError(
                "PUSH evaluator checkpoint lacks proposal provenance; retrain it "
                "against the Stage-C checkpoint used for deployment"
            )
        actual, actual_source = push_checkpoint_fingerprint(proposal_checkpoint)
        if actual != expected or (expected_source is not None and expected_source != actual_source):
            raise RuntimeError(
                "PUSH evaluator/Stage-C mismatch: evaluator and deployment use "
                "different proposal states"
            )
    model.push_evaluator.load_state_dict(payload["model"], strict=True)
    model.push_evaluator_ready = True
