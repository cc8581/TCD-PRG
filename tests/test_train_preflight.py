from __future__ import annotations

import pytest

from tcd_prg.observation.cached import CachedObservationProvider
from tcd_prg.scripts.train import validate_read_through_observation_cache


class _Adapter:
    def __init__(self, provider) -> None:
        self.observation_provider = provider


def test_formal_training_accepts_bounded_read_through_cache(tmp_path) -> None:
    provider = CachedObservationProvider(
        tmp_path, fallback=object(), max_bytes=1 << 30, min_free_bytes=1,
    )
    status = validate_read_through_observation_cache(_Adapter(provider))
    assert status["mode"] == "read-through-lru"
    assert status["missing"] == "render-on-demand"
    assert status["directory"] == str(tmp_path.resolve())


def test_formal_training_rejects_cache_without_renderer(tmp_path) -> None:
    provider = CachedObservationProvider(tmp_path, fallback=None)
    with pytest.raises(RuntimeError, match="on-miss observation renderer"):
        validate_read_through_observation_cache(_Adapter(provider))


def test_formal_training_enforces_free_space_reserve(tmp_path) -> None:
    provider = CachedObservationProvider(
        tmp_path, fallback=object(), min_free_bytes=1 << 80,
    )
    with pytest.raises(OSError, match="below the configured reserve"):
        validate_read_through_observation_cache(_Adapter(provider))
