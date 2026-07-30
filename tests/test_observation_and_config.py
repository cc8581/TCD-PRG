from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from tcd_prg.config import load_config
from tcd_prg.paths import PROJECT_ROOT
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


def test_config_paths_are_project_relative_not_cwd_relative(tmp_path, monkeypatch) -> None:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    monkeypatch.setenv("TCD_DATASET_ROOT", str(tmp_path))
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
            if path.suffix.lower() in suffixes:
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if pattern.search(line):
                        offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}")
    assert offenders == []
