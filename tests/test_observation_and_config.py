from __future__ import annotations

from pathlib import Path

import numpy as np

from tcd_prg.config import load_config
from tcd_prg.observation.base import ObservationRequest
from tcd_prg.observation.cached import request_hash


def _request(**changes) -> ObservationRequest:
    values = dict(
        scene_id=1, state_id=2,
        object_pose=np.asarray([[0, 0, 0, 0, 0, 0, 1]], np.float32),
        object_active=np.asarray([True]), object_present=np.asarray([True]),
        object_asset_ids=("asset",), object_model_ids=("model",),
        object_scales=np.asarray([1.0], np.float32), render_seed=7,
        camera_profile="three_pro_s", point_count=2048, renderer_version="v1",
    )
    values.update(changes)
    return ObservationRequest(**values)


def test_observation_cache_hash_binds_geometry_camera_seed_and_sampling() -> None:
    base = request_hash(_request())
    variants = (
        _request(object_model_ids=("different",)),
        _request(object_scales=np.asarray([1.1], np.float32)),
        _request(render_seed=8), _request(camera_profile="other"),
        _request(point_count=1024), _request(renderer_version="v2"),
    )
    assert len({base, *(request_hash(item) for item in variants)}) == len(variants) + 1


def test_every_shipped_yaml_passes_strict_schema() -> None:
    paths = list(Path("configs").rglob("*.yaml"))
    assert paths
    for path in paths:
        load_config(path)
