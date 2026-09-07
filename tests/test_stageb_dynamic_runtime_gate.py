from types import SimpleNamespace

import pytest

from tcd_prg.config import TCDPRGConfig
from tcd_prg.models import staged_checkpoint


def _payload(config):
    sections = {}
    fields = {
        "model": ("task_grasp_scene_points", "task_grasp_gripper_points"),
        "dataset": ("scene_points",),
        "backbone": ("grid_size_m",),
        "observation": ("camera_profile", "renderer_version", "render_width", "render_height"),
        "graspnet": (
            "scene_input_points", "target_input_points", "target_proposals",
            "target_selection_mode", "diversity_quality_fraction", "diversity_translation_m",
            "diversity_rotation_deg", "diversity_pool_factor", "camera_view_index",
            "target_crop_probability", "target_min_crop_points",
            "camera_transfer_max_distance_m", "num_view", "num_angle", "num_depth",
            "cylinder_radius", "hmin", "hmax_list",
        ),
    }
    for section, names in fields.items():
        source = getattr(config, section)
        sections[section] = {name: getattr(source, name) for name in names}
    return {
        "config": sections,
        "stageb_provenance": {"compatibility": {
            "dataset_protocol_version": "task_oriented_clutter_acronym_dynamic_v1"
        }},
        "task_grasp_probability_threshold": .5,
    }


def test_dynamic_stageb_runtime_uses_saved_inference_protocol(monkeypatch):
    config = TCDPRGConfig()
    config.backbone.backend = "legacy"
    payload = _payload(config)
    monkeypatch.setattr(staged_checkpoint, "load_perception_stage", lambda *args: {})
    monkeypatch.setattr(staged_checkpoint, "_load_stage", lambda *args: payload)
    model = SimpleNamespace()
    assert staged_checkpoint.load_staged_tcd_prg(model, "a.pt", "b.pt", config) == .5
    config.observation.renderer_version = "tcd_prg_pybullet_v3_sensor_only_instance_query"
    assert staged_checkpoint.load_staged_tcd_prg(model, "a.pt", "b.pt", config) == .5
    payload["config"]["observation"]["render_width"] += 1
    with pytest.raises(RuntimeError, match="observation.render_width"):
        staged_checkpoint.load_staged_tcd_prg(model, "a.pt", "b.pt", config)
