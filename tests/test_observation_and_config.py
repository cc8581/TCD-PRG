from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

from tcd_prg.config import ModelConfig, TCDPRGConfig, TrainingConfig, load_config
from tcd_prg.observation.base import ObservationRequest, PointObservation
from tcd_prg.observation.cached import CachedObservationProvider, request_hash
from tcd_prg.observation.saved import _resize_view_nearest, deterministic_stratified_sample
from tcd_prg.paths import PROJECT_ROOT


def _request(**changes) -> ObservationRequest:
    values = dict(
        scene_id=1,
        state_id=2,
        object_pose=np.asarray([[0, 0, 0, 0, 0, 0, 1]], np.float32),
        object_active=np.asarray([True]),
        object_present=np.asarray([True]),
        object_asset_ids=("asset",),
        object_model_ids=("model",),
        object_scales=np.asarray([1.0], np.float32),
        render_seed=7,
        camera_profile="three_pro_s",
        point_count=2048,
        renderer_version="v1",
    )
    values.update(changes)
    return ObservationRequest(**values)


def test_observation_cache_hash_binds_geometry_camera_seed_and_sampling() -> None:
    base = request_hash(_request())
    variants = (
        _request(object_model_ids=("different",)),
        _request(object_scales=np.asarray([1.1], np.float32)),
        _request(render_seed=8),
        _request(camera_profile="other"),
        _request(point_count=1024),
        _request(renderer_version="v2"),
    )
    assert len({base, *(request_hash(item) for item in variants)}) == len(variants) + 1


def test_cache_availability_is_read_only(tmp_path) -> None:
    missing_cache = tmp_path / "does-not-exist"
    provider = CachedObservationProvider(missing_cache, fallback=None)
    request = _request()
    # Strict cache-only setup must not even create the configured directory.
    assert not provider.is_available(request)
    assert not missing_cache.exists()


def test_read_only_cache_cannot_evict_or_clear(tmp_path) -> None:
    provider = CachedObservationProvider(tmp_path / "legacy", fallback=None)
    with pytest.raises(RuntimeError, match="Read-only"):
        provider.evict()
    with pytest.raises(RuntimeError, match="Read-only"):
        provider.clear_completed()


def test_cache_eviction_skips_entries_locked_by_another_worker(tmp_path, monkeypatch) -> None:
    provider = CachedObservationProvider(
        tmp_path,
        fallback=object(),
        max_bytes=0,
        min_free_bytes=0,
    )
    locked = tmp_path / "00" / "locked.npz"
    removable = tmp_path / "01" / "removable.npz"
    for path in (locked, removable):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cache-entry")
    os.utime(locked, (1, 1))
    os.utime(removable, (2, 2))

    original_unlink = Path.unlink

    def unlink_unless_locked(path: Path, *args, **kwargs) -> None:
        if path == locked:
            raise PermissionError("entry is open in another DataLoader worker")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_unless_locked)

    provider.evict()

    assert locked.exists()
    assert not removable.exists()


def test_cache_eviction_keeps_most_recent_entries_within_capacity(tmp_path) -> None:
    provider = CachedObservationProvider(
        tmp_path,
        fallback=object(),
        max_bytes=2 * len(b"cache-entry"),
        min_free_bytes=0,
    )
    oldest = tmp_path / "00" / "oldest.npz"
    middle = tmp_path / "01" / "middle.npz"
    newest = tmp_path / "02" / "newest.npz"
    for timestamp, path in enumerate((oldest, middle, newest), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cache-entry")
        os.utime(path, (timestamp, timestamp))

    provider.evict()

    assert not oldest.exists()
    assert middle.exists()
    assert newest.exists()


def test_cache_step_cleanup_removes_completed_entries_only(tmp_path, monkeypatch) -> None:
    provider = CachedObservationProvider(tmp_path, fallback=object())
    removable = tmp_path / "00" / "complete.npz"
    locked = tmp_path / "01" / "locked.npz"
    temporary = tmp_path / "02" / "active.123.tmp.npz"
    for path in (removable, locked, temporary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cache-entry")

    original_unlink = Path.unlink

    def unlink_unless_locked(path: Path, *args, **kwargs) -> None:
        if path == locked:
            raise PermissionError("entry is open in another DataLoader worker")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_unless_locked)
    result = provider.clear_completed()

    assert result == {
        "removed_files": 1,
        "removed_bytes": len(b"cache-entry"),
        "locked_files": 1,
    }
    assert not removable.exists()
    assert locked.exists()
    assert temporary.exists()


def test_zero_scene_point_limit_preserves_variable_length_observation() -> None:
    observation = PointObservation(
        xyz=np.arange(21, dtype=np.float32).reshape(7, 3),
        rgb=np.zeros((7, 3), dtype=np.float32),
        instance_id=np.arange(7, dtype=np.int64) % 2,
        source_view=np.zeros(7, dtype=np.int16),
    )
    sampled = deterministic_stratified_sample(observation, 0, seed=19)
    assert sampled is observation
    assert sampled.xyz.shape == (7, 3)


def test_formal_config_uses_bounded_variable_length_scenes() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    assert config.dataset.scene_points == 16_384


def test_formal_config_uses_strict_offline_cache_and_scene_splits() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    assert config.cache.max_gb == 10.0
    assert config.cache.eviction == "lru"
    assert config.observation.allow_render_on_miss is False
    assert config.training.scene_start == 0
    assert config.training.scene_count == 2500
    assert tuple(config.training.split_ratios) == (9.0, 1.0)
    assert config.training.max_validation_groups is None
    assert config.training.validation_scene_count == 20
    assert config.training.validation_scene_seed == 2026
    assert config.training.validation_num_workers == 0
    assert config.logging.validation_log_interval == 20


def test_validation_worker_count_must_be_non_negative() -> None:
    config = TCDPRGConfig()
    config.training.validation_num_workers = -1
    with pytest.raises(ValueError, match="validation_num_workers"):
        config.validate()


def test_legacy_renderer_protocol_forbids_render_fallback() -> None:
    config = TCDPRGConfig()
    config.observation.renderer_version = "tcd_prg_pybullet_v2_variable_grid"
    config.observation.allow_render_on_miss = True
    with pytest.raises(ValueError, match="read-only"):
        config.validate()


def test_legacy_cache_utility_contains_no_generation_path() -> None:
    source = (PROJECT_ROOT / "scripts" / "precompute_observation_cache.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "generate_missing",
        "write_request_npz",
        "PersistentRendererClient",
        "match-legacy-counts-from",
    )
    assert all(name not in source for name in forbidden)


def test_saved_state_zero_is_resized_to_formal_render_resolution() -> None:
    depth = np.arange(48, dtype=np.float32).reshape(6, 8)
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    instance = np.zeros((6, 8), dtype=np.int16)
    resized = _resize_view_nearest(depth, rgb, instance, width=4, height=3)
    assert resized[0].shape == (3, 4)
    assert resized[1].shape == (3, 4, 3)
    assert resized[2].shape == (3, 4)


def test_every_shipped_yaml_passes_strict_schema() -> None:
    paths = [
        path for path in Path("configs").rglob("*.yaml") if not path.name.startswith("local_paths")
    ]
    assert paths
    for path in paths:
        load_config(path)


def test_config_paths_are_project_relative_not_cwd_relative(tmp_path, monkeypatch) -> None:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    assert Path(config.output_dir).parent == PROJECT_ROOT / "outputs"
    assert Path(config.observation.worker_script) == (
        PROJECT_ROOT / "scripts" / "render_observation_worker_py38.py"
    )


def test_source_and_config_contain_no_absolute_drive_paths() -> None:
    pattern = re.compile(r"(?:^|[\"'=:\s])([A-Za-z]:[\\/])")
    roots = ("tcd_prg", "configs", "scripts", "tools")
    suffixes = {".py", ".yaml", ".yml", ".ps1", ".sh"}
    offenders = []
    for root in roots:
        for path in (PROJECT_ROOT / root).rglob("*"):
            if path.name.startswith("local_paths"):
                continue
            if path.suffix.lower() in suffixes:
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if pattern.search(line):
                        offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}")
    assert offenders == []


def test_task_grasp_candidate_capacity_covers_maximum_required_count() -> None:
    config = TCDPRGConfig(
        model=ModelConfig(
            task_grasp_candidates=19,
            default_required_grasp_count=19,
            max_required_grasp_count=20,
        )
    )
    with pytest.raises(ValueError, match="task_grasp_candidates"):
        config.validate()


def test_train_only_bootstrap_allows_zero_validation_interval() -> None:
    TCDPRGConfig(training=TrainingConfig(validation_interval=0)).validate()
    with pytest.raises(ValueError, match="validation_interval"):
        TCDPRGConfig(training=TrainingConfig(validation_interval=-1)).validate()


def test_scene_split_ratios_require_train_and_enabled_validation() -> None:
    TCDPRGConfig(training=TrainingConfig(split_ratios=(9.0, 1.0))).validate()
    with pytest.raises(ValueError, match="train/val"):
        TCDPRGConfig(training=TrainingConfig(split_ratios=(1.0,))).validate()
    with pytest.raises(ValueError, match="allocate scenes to val"):
        TCDPRGConfig(training=TrainingConfig(split_ratios=(1.0, 0.0))).validate()
