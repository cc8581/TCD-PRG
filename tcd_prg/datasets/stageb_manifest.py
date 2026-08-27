"""Versioned provenance contract for immutable Stage-B binary data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from tcd_prg.config import TCDPRGConfig

SCHEMA_VERSION = "tcd_prg_stageb_binary_v4"
GEOMETRY_PROTOCOL_VERSION = "ag16095_geometric_close_v2"
CAMERA_TRANSFER_PROTOCOL_VERSION = "fused_nearest_camera2_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config: TCDPRGConfig) -> str:
    relevant = {
        "dataset": asdict(config.dataset),
        "observation_renderer_version": config.observation.renderer_version,
        "graspnet": asdict(config.graspnet),
        "model": {
            "stageb_label_gripper_geometry": config.model.stageb_label_gripper_geometry,
        },
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_provenance(config: TCDPRGConfig) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "compatibility": {
            "graspnet_checkpoint_sha256": sha256_file(config.graspnet.checkpoint),
            "camera_transfer_protocol_version": CAMERA_TRANSFER_PROTOCOL_VERSION,
            "geometry_protocol_version": GEOMETRY_PROTOCOL_VERSION,
            "gripper_geometry_sha256": sha256_file(
                config.model.stageb_label_gripper_geometry
            ),
            "config_fingerprint": config_fingerprint(config),
        },
        "audit": {"producer_git_commits": [commit]},
    }


def compatibility_provenance(provenance: Mapping[str, Any]) -> dict[str, str]:
    value = provenance.get("compatibility")
    if not isinstance(value, Mapping):
        raise ValueError("Stage-B provenance is missing compatibility fields")
    return {str(key): str(item) for key, item in value.items()}
