"""Strict deployment composition for independently trained A/B/C checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from tcd_prg.datasets.stageb_manifest import compatibility_provenance, stageb_compatibility

STAGE_PREFIXES = {
    "perception": ("encoder.", "region_head."),
    "grasp": ("task_grasp.",),
    "push": ("push.",),
}


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


def load_staged_tcd_prg(
    model: nn.Module,
    stage_a_checkpoint: str | Path,
    stage_b_checkpoint: str | Path,
    stage_c_checkpoint: str | Path,
    runtime_config,
) -> float:
    stage_a = _load_stage(model, stage_a_checkpoint, "perception")
    stage_b = _load_stage(model, stage_b_checkpoint, "grasp")
    stage_c = _load_stage(model, stage_c_checkpoint, "push")
    perception_fields = (
        "instance_queries",
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
    _require_fields(
        stage_a,
        runtime_config,
        {
            "model": perception_fields,
            "ablation": ("use_task_region_condition",),
            "backbone": ("grid_size_m",),
        },
        "perception",
    )
    _require_fields(
        stage_c,
        runtime_config,
        {
            "model": (
                "instance_queries",
                "num_categories",
                "num_task_regions",
                "num_direction_bins",
                "push_direction_contact_topk",
                "push_object_topk",
                "push_utility_temperature",
            )
        },
        "push",
    )
    _require_fields(
        stage_b,
        runtime_config,
        {"model": ("task_grasp_scene_points", "task_grasp_gripper_points")},
        "grasp",
    )
    saved_ablation = stage_c.get("config", {}).get("ablation", {})
    saved_losses = stage_c.get("config", {}).get("losses", {})
    if runtime_config.ablation.use_push_potential and (
        not saved_ablation.get("use_push_potential", False)
        or float(saved_losses.get("push_potential", 0.0)) <= 0.0
    ):
        raise RuntimeError("push runtime requires a Stage-C checkpoint trained with push potential")
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
