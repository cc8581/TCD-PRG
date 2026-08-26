"""Versioned provenance contract for immutable Stage-B binary data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from tcd_prg.config import TCDPRGConfig

SCHEMA_VERSION = "tcd_prg_stageb_binary_v2"
GEOMETRY_PROTOCOL_VERSION = "ag16095_geometric_close_v2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config: TCDPRGConfig) -> str:
    relevant = {
        "scene_points": config.dataset.scene_points,
        "backbone": asdict(config.backbone),
        "graspnet": asdict(config.graspnet),
        "model": {
            "feature_dim": config.model.feature_dim,
            "task_dim": config.model.task_dim,
            "target_prompt_jitter_std_m": config.model.target_prompt_jitter_std_m,
        },
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_provenance(
    config: TCDPRGConfig, stage_a_checkpoint: str | Path | None = None
) -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    result = {
        "graspnet_checkpoint_sha256": sha256_file(config.graspnet.checkpoint),
        "geometry_protocol_version": GEOMETRY_PROTOCOL_VERSION,
        "gripper_geometry_sha256": sha256_file(config.model.stageb_label_gripper_geometry),
        "config_fingerprint": config_fingerprint(config),
        "git_commit": commit,
    }
    if stage_a_checkpoint is not None:
        result["stage_a_checkpoint_sha256"] = sha256_file(stage_a_checkpoint)
    return result
