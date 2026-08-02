from __future__ import annotations

from types import SimpleNamespace

import pytest

from tcd_prg.scripts.train import preflight_observation_cache


class _Adapter:
    def __init__(self, available):
        self.available = set(available)

    def observation_available(self, scene_id, state_id, task_index):
        return (scene_id, state_id, task_index) in self.available


def _dataset(*keys):
    return SimpleNamespace(units=[
        SimpleNamespace(scene_id=scene, state_id=state, task_index=task)
        for scene, state, task in keys
    ])


def test_cache_preflight_deduplicates_state_tasks() -> None:
    key = (1, 2, 3)
    checked = preflight_observation_cache(
        _Adapter({key}),
        ("train", _dataset(key, key)),
    )
    assert checked == 1


def test_cache_preflight_fails_before_training_on_first_missing_request() -> None:
    with pytest.raises(RuntimeError, match=r"split=train scene=1 state=2 task=3"):
        preflight_observation_cache(
            _Adapter(set()),
            ("train", _dataset((1, 2, 3))),
        )
