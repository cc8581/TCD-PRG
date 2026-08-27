"""Strict deployment composition for independently trained A/B/C checkpoints."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

STAGE_PREFIXES = {"perception": ("encoder.", "region_head."), "grasp": ("task_grasp.",), "push": ("push.",)}

def _load_stage(model: nn.Module, path: str | Path, stage: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 12 or payload.get("training_stage") != stage:
        raise RuntimeError(f"{path} is not a schema-12 {stage} checkpoint")
    source = payload.get("ema") or payload["model"]; current = model.state_dict(); prefixes = STAGE_PREFIXES[stage]
    required = {name for name in current if name.startswith(prefixes)}
    supplied = {name for name in source if name.startswith(prefixes)}
    if required != supplied:
        missing, extra = sorted(required-supplied), sorted(supplied-required)
        raise RuntimeError(f"{stage} checkpoint module mismatch: missing={missing[:5]} extra={extra[:5]}")
    incompatible = [name for name in required if current[name].shape != source[name].shape]
    if incompatible:
        raise RuntimeError(f"{stage} checkpoint tensor shape mismatch: {incompatible[:5]}")
    model.load_state_dict({name: source[name] for name in required}, strict=False)
    return payload

def load_staged_tcd_prg(model: nn.Module, stage_a_checkpoint: str | Path,
                        stage_b_checkpoint: str | Path, stage_c_checkpoint: str | Path) -> float:
    _load_stage(model, stage_a_checkpoint, "perception")
    stage_b = _load_stage(model, stage_b_checkpoint, "grasp")
    _load_stage(model, stage_c_checkpoint, "push")
    if "task_grasp_probability_threshold" not in stage_b:
        raise RuntimeError("Stage-B checkpoint is missing its calibrated deployment threshold")
    return float(stage_b["task_grasp_probability_threshold"])
