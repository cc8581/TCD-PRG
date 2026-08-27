"""Versioned provenance contract for immutable Stage-B binary data."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

from tcd_prg.config import TCDPRGConfig

SCHEMA_VERSION = "tcd_prg_stageb_binary_v5"
GEOMETRY_PROTOCOL_VERSION = "ag16095_geometric_close_v2"
CAMERA_TRANSFER_PROTOCOL_VERSION = "fused_nearest_camera2_v1"
PROPOSAL_SAMPLING_VERSION = "all_valid_target_proposals_v1"
DATASET_PROTOCOL_VERSION = "task_oriented_clutter_stageb_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stageb_compatibility(config: TCDPRGConfig) -> dict[str, Any]:
    """Exact semantic inputs to Stage-B record construction; never local paths."""
    graspnet = config.graspnet
    return {
        "dataset_protocol_version": DATASET_PROTOCOL_VERSION,
        "scene_preprocess": {
            "scene_points": config.dataset.scene_points,
            "grid_size_m": config.backbone.grid_size_m,
            "camera_profile": config.observation.camera_profile,
            "renderer_version": config.observation.renderer_version,
            "render_width": config.observation.render_width,
            "render_height": config.observation.render_height,
        },
        "target_graspnet": {
            "checkpoint_sha256": sha256_file(graspnet.checkpoint),
            "target_input_points": graspnet.target_input_points,
            "target_proposals": graspnet.target_proposals,
            "target_selection_mode": graspnet.target_selection_mode,
            "diversity_quality_fraction": graspnet.diversity_quality_fraction,
            "diversity_translation_m": graspnet.diversity_translation_m,
            "diversity_rotation_deg": graspnet.diversity_rotation_deg,
            "diversity_pool_factor": graspnet.diversity_pool_factor,
            "camera_view_index": graspnet.camera_view_index,
            "target_crop_probability": graspnet.target_crop_probability,
            "target_min_crop_points": graspnet.target_min_crop_points,
            "camera_transfer_max_distance_m": graspnet.camera_transfer_max_distance_m,
            "num_view": graspnet.num_view,
            "num_angle": graspnet.num_angle,
            "num_depth": graspnet.num_depth,
            "cylinder_radius": graspnet.cylinder_radius,
            "hmin": graspnet.hmin,
            "hmax_list": list(graspnet.hmax_list),
        },
        "proposal_label_protocol": {
            "proposal_sampling_version": PROPOSAL_SAMPLING_VERSION,
            "camera_transfer_protocol_version": CAMERA_TRANSFER_PROTOCOL_VERSION,
            "geometry_protocol_version": GEOMETRY_PROTOCOL_VERSION,
            "gripper_geometry_sha256": sha256_file(
                config.model.stageb_label_gripper_geometry
            ),
        },
    }


def build_provenance(config: TCDPRGConfig) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "compatibility": {
            **stageb_compatibility(config),
        },
        "audit": {"producer_git_commits": [commit]},
    }


def compatibility_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    value = provenance.get("compatibility")
    if not isinstance(value, Mapping):
        raise ValueError("Stage-B provenance is missing compatibility fields")
    return dict(value)
