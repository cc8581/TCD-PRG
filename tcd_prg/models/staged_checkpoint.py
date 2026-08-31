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
}

PUSH_EVALUATOR_PROTOCOL_VERSION = 2


def perception_geometry_fingerprint(payload: dict) -> str:
    source = payload.get("ema") or payload.get("model", {})
    prefix = "encoder.scene_backbone."
    tensors = {name[len(prefix):]: value for name, value in source.items() if name.startswith(prefix)}
    if not tensors:
        raise RuntimeError("Perception checkpoint contains no scene geometry encoder")
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def resolve_staged_checkpoint_root(root: str | Path) -> dict[str, Path]:
    """Resolve the three stage-best checkpoints from one portable run directory."""

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
            for stage in ("perception", "grasp", "push_evaluator")
        }
    paths: dict[str, Path] = {}
    for stage in ("perception", "grasp", "push_evaluator"):
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


def load_perception_stage(model: nn.Module, checkpoint: str | Path, runtime_config) -> dict:
    payload = _load_stage(model, checkpoint, "perception")
    validate_perception_stage_runtime(payload, runtime_config)
    return payload


def load_staged_tcd_prg(
    model: nn.Module,
    perception_checkpoint: str | Path,
    stage_b_checkpoint: str | Path,
    runtime_config,
) -> float:
    stage_a = load_perception_stage(model, perception_checkpoint, runtime_config)
    stage_b = _load_stage(model, stage_b_checkpoint, "grasp")
    _require_fields(
        stage_b,
        runtime_config,
        {"model": ("task_grasp_scene_points", "task_grasp_gripper_points")},
        "grasp",
    )
    model.perception_geometry_fingerprint = perception_geometry_fingerprint(stage_a)
    del stage_a
    saved_stageb = compatibility_provenance(stage_b.get("stageb_provenance", {}))
    runtime_stageb = stageb_compatibility(runtime_config)
    for key in ("scene_preprocess", "target_graspnet", "proposal_label_protocol"):
        if saved_stageb.get(key) != runtime_stageb.get(key):
            raise RuntimeError(f"grasp runtime provenance mismatch: {key}")
    if "task_grasp_probability_threshold" not in stage_b:
        raise RuntimeError("Stage-B checkpoint is missing its calibrated deployment threshold")
    return float(stage_b["task_grasp_probability_threshold"])


def load_push_evaluator(model, checkpoint):
    """Reject old proposal-dependent checkpoints and mismatched perception features."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("training_stage") != "push_evaluator":
        raise RuntimeError("Not a push_evaluator checkpoint")
    if payload.get("push_evaluator_protocol_version") != PUSH_EVALUATOR_PROTOCOL_VERSION:
        raise RuntimeError("Obsolete PUSH evaluator: retrain the independent action evaluator")
    expected = payload.get("perception_geometry_fingerprint")
    actual = getattr(model, "perception_geometry_fingerprint", None)
    if not expected or not actual or expected != actual:
        raise RuntimeError("PUSH evaluator/perception geometry mismatch")
    model.push_evaluator.load_state_dict(payload["model"], strict=True)
    model.push_evaluator_ready = True

def stage_training_state(model, state, stage):
    """Retain strict A/B state loading while discarding obsolete inactive PUSH tensors."""
    if stage not in {"perception", "grasp"}:
        return state
    current = model.state_dict()
    prefixes = ("push.", "push_evaluator.")
    old = {k: v for k, v in state.items() if k.startswith(prefixes)}
    new = {k: v for k, v in current.items() if k.startswith(prefixes)}
    if old.keys() == new.keys() and all(old[k].shape == new[k].shape for k in old):
        return state
    return {**{k: v for k, v in state.items() if not k.startswith(prefixes)}, **new}


def stage_training_optimizer_state(model, optimizer, payload):
    """Preserve A/B optimizer moments; replace only obsolete unused trailing evaluator slots."""
    import copy
    saved = payload["optimizer"]
    if payload.get("training_stage") not in {"perception", "grasp"}:
        return saved
    legacy = payload["model"]
    if not any(k.startswith("push.") for k in legacy):
        return saved
    old_count = sum(k.startswith("push_evaluator.") for k in legacy)
    new_count = sum(k.startswith("push_evaluator.") and p.requires_grad for k, p in model.named_parameters())
    current = optimizer.state_dict()
    if [len(g["params"]) for g in saved["param_groups"]] == [len(g["params"]) for g in current["param_groups"]]:
        return saved
    result = copy.deepcopy(saved)
    if len(result["param_groups"]) != len(current["param_groups"]):
        raise RuntimeError("A/B optimizer group mismatch")
    group, destination = result["param_groups"][-1], current["param_groups"][-1]
    if not old_count or len(group["params"]) - old_count + new_count != len(destination["params"]):
        raise RuntimeError("A/B optimizer mismatch beyond inactive PUSH evaluator")
    if any(p in result["state"] for p in group["params"][-old_count:]):
        raise RuntimeError("Old A/B checkpoint unexpectedly optimized its PUSH evaluator")
    if any(len(a["params"]) != len(b["params"]) for a, b in zip(result["param_groups"][:-1], current["param_groups"][:-1])):
        raise RuntimeError("A/B active optimizer groups changed")
    next_id = max(p for g in result["param_groups"] for p in g["params"]) + 1
    group["params"] = group["params"][:-old_count] + list(range(next_id, next_id + new_count))
    return result
