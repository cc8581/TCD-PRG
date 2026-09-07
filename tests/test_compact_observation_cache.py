from __future__ import annotations

import numpy as np

from tcd_prg.observation.base import ObservationProvider, ObservationRequest, PointObservation
from tcd_prg.observation.cached import CachedObservationProvider, request_hash


class _Fallback(ObservationProvider):
    def __init__(self, observation: PointObservation) -> None:
        self.observation = observation

    def get(self, request: ObservationRequest) -> PointObservation:
        return self.observation


def _request() -> ObservationRequest:
    return ObservationRequest(
        scene_id=1,
        state_id=2,
        object_pose=np.asarray([[0, 0, 0, 0, 0, 0, 1]], np.float32),
        object_active=np.asarray([True]),
        object_present=np.asarray([True]),
        object_asset_ids=("asset",),
        object_model_ids=("model",),
        object_scales=np.asarray([1.0], np.float32),
        render_seed=3,
        camera_profile="camera",
        point_count=2,
        renderer_version="v3",
    )


def test_cache_compacts_exact_sensor_rgb_and_instance_ids_losslessly(tmp_path) -> None:
    observation = PointObservation(
        xyz=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float32),
        rgb=np.asarray([[1, 2, 3], [250, 251, 252]], np.float32) / 255.0,
        instance_id=np.asarray([-1, 17], np.int64),
        source_view=np.asarray([0, 2], np.int16),
    )
    request = _request()
    writable = CachedObservationProvider(
        tmp_path,
        fallback=_Fallback(observation),
        max_bytes=1 << 20,
        min_free_bytes=0,
    )
    returned = writable.get(request)
    np.testing.assert_array_equal(returned.rgb, observation.rgb)
    path = tmp_path / request_hash(request)[:2] / f"{request_hash(request)}.npz"
    with np.load(path, allow_pickle=False) as stored:
        assert stored["rgb"].dtype == np.uint8
        assert stored["instance_id"].dtype == np.int16
        assert stored["xyz"].dtype == np.float32

    loaded = CachedObservationProvider(tmp_path, fallback=None).get(request)
    np.testing.assert_array_equal(loaded.xyz, observation.xyz)
    np.testing.assert_array_equal(loaded.rgb, observation.rgb)
    np.testing.assert_array_equal(loaded.instance_id, observation.instance_id)
    np.testing.assert_array_equal(loaded.source_view, observation.source_view)
