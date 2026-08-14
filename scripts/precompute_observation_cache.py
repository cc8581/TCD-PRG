#!/usr/bin/env python3
"""Read-only validator for the legacy TCD-PRG observation cache.

This historical filename is retained for operator compatibility.  The tool
cannot render, repair, migrate, evict, or write observation-cache entries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
# This machine's validator Python links NumPy and Torch against distinct Intel
# OpenMP copies. Scope the compatibility switch to this read-only CLI process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from tcd_prg.config import LEGACY_READ_ONLY_RENDERER, load_config  # noqa: E402
from tcd_prg.observation.cached import (  # noqa: E402
    CachedObservationProvider,
    ObservationCacheMissError,
)
from tcd_prg.observation.base import ObservationProvider  # noqa: E402
from tcd_prg.runtime import create_adapter  # noqa: E402


class _RequestCaptured(Exception):
    def __init__(self, request):
        super().__init__()
        self.request = request


class _CaptureProvider(ObservationProvider):
    """Stop adapter loading immediately after its exact request is constructed."""

    def get(self, request):
        raise _RequestCaptured(request)


def _local_paths(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Local paths file must contain a mapping: {path}")
    return {str(key): str(value) for key, value in values.items()}


def _quoted_override(name: str, value: object) -> str:
    return f"{name}={json.dumps(value, ensure_ascii=False)}"


def _load(args: argparse.Namespace):
    local = _local_paths(args.paths_config)
    overrides = [
        _quoted_override("observation.allow_render_on_miss", False),
    ]
    values = {
        "dataset.root": args.dataset_root or local.get("dataset_root"),
        "dataset.functional_region_root": (
            args.functional_region_root or local.get("functional_region_root")
        ),
        "cache.directory": args.cache_dir or local.get("observation_cache_dir"),
    }
    for name, value in values.items():
        if value:
            overrides.append(_quoted_override(name, value))
    if args.point_count is not None:
        overrides.append(f"dataset.scene_points={int(args.point_count)}")
    overrides.extend(args.override)
    config = load_config(args.config, overrides)
    if config.observation.renderer_version != LEGACY_READ_ONLY_RENDERER:
        raise ValueError(
            "This validator only accepts the legacy read-only renderer protocol "
            f"{LEGACY_READ_ONLY_RENDERER!r}; got {config.observation.renderer_version!r}"
        )
    return config


def _scene_ids(adapter, args: argparse.Namespace) -> list[int]:
    available = set(adapter.snapshot_scene_ids)
    if args.scene_ids_file:
        selected = sorted(
            {
                int(line.split(",")[0].split()[0])
                for raw in args.scene_ids_file.read_text(encoding="utf-8-sig").splitlines()
                if (line := raw.strip()) and not line.startswith("#")
            }
        )
    else:
        stop = None if args.scene_count is None else args.scene_start + args.scene_count
        selected = [
            scene
            for scene in sorted(available)
            if scene >= args.scene_start and (stop is None or scene < stop)
        ]
    missing = sorted(set(selected) - available)
    if missing:
        raise FileNotFoundError(f"Selected unpublished scenes: {missing[:20]}")
    if not selected:
        raise ValueError("Scene selection is empty")
    return selected


def _requests(adapter, scene_ids: list[int], split: str):
    rows = adapter._load_action_group_index()  # repository's formal training index
    keep = np.isin(rows[:, 1], np.asarray(scene_ids, dtype=np.int64))
    if split != "all":
        keep &= np.isin(rows[:, 1], np.asarray(adapter.scene_splits[split], dtype=np.int64))
    selected = rows[keep]
    if not len(selected):
        raise RuntimeError("No action-state groups match the requested scenes and split")
    # Cache requests are keyed by scene, state, and task; repeated action groups
    # reuse the same observation and must not inflate validation work.
    triples = np.unique(selected[:, [1, 4, 3]], axis=0)
    return [tuple(map(int, row)) for row in triples]


def validate(args: argparse.Namespace) -> int:
    config = _load(args)
    cache_dir = Path(config.cache.directory).resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Observation cache directory does not exist: {cache_dir}")

    # Explicitly inject a provider with no fallback. Its constructor, get path,
    # eviction, and cleanup operations are all non-mutating in this mode.
    provider = CachedObservationProvider(cache_dir, fallback=None)
    adapter = create_adapter(config, allow_render=False)
    adapter.observation_provider = _CaptureProvider()
    scene_ids = _scene_ids(adapter, args)
    requests = _requests(adapter, scene_ids, args.split)

    valid = 0
    missing: list[dict[str, int | str]] = []
    invalid: list[dict[str, int | str]] = []
    point_counts: list[int] = []
    for index, (scene_id, state_id, task_index) in enumerate(requests, start=1):
        try:
            try:
                adapter.load_observation(scene_id, state_id, task_index)
            except _RequestCaptured as captured:
                points = provider.get(captured.request)
            else:
                raise RuntimeError("Adapter failed to expose its observation request")
            xyz = np.asarray(points.xyz)
            rgb = np.asarray(points.rgb)
            instance = np.asarray(points.instance_id)
            source = np.asarray(points.source_view)
            count = len(xyz)
            object_count = len(captured.request.object_pose)
            if (
                xyz.dtype != np.float32
                or rgb.dtype != np.float32
                or xyz.shape != (count, 3)
                or rgb.shape != (count, 3)
                or instance.shape != (count,)
                or source.shape != (count,)
                or count == 0
                or not np.isfinite(xyz).all()
                or not np.isfinite(rgb).all()
                or float(rgb.min()) < -1e-6
                or float(rgb.max()) > 1.0 + 1e-6
                or not np.issubdtype(instance.dtype, np.integer)
                or int(instance.min()) < -1
                or int(instance.max()) >= object_count
                or not np.issubdtype(source.dtype, np.integer)
                or int(source.min()) < 0
                or int(source.max()) > 2
            ):
                raise ValueError("cache arrays violate the XYZ/RGB/instance/source contract")
            point_counts.append(count)
            valid += 1
        except ObservationCacheMissError as error:
            missing.append(
                {"scene_id": scene_id, "state_id": state_id, "task_index": task_index,
                 "error": str(error)}
            )
        except (OSError, ValueError, KeyError, EOFError) as error:
            invalid.append(
                {"scene_id": scene_id, "state_id": state_id, "task_index": task_index,
                 "error": f"{type(error).__name__}: {error}"}
            )
        if index % args.progress_interval == 0 or index == len(requests):
            print(
                f"[validate] {index:,}/{len(requests):,} valid={valid:,} "
                f"missing={len(missing):,} invalid={len(invalid):,}",
                flush=True,
            )

    summary = {
        "mode": "legacy-cache-read-only-validation",
        "cache_dir": str(cache_dir),
        "renderer_version": config.observation.renderer_version,
        "scene_count": len(scene_ids),
        "request_count": len(requests),
        "valid": valid,
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "point_count": {
            "minimum": min(point_counts) if point_counts else None,
            "median": float(np.median(point_counts)) if point_counts else None,
            "maximum": max(point_counts) if point_counts else None,
        },
        "missing_preview": missing[:20],
        "invalid_preview": invalid[:20],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if not missing and not invalid else 3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate legacy TCD-PRG observation cache without modifying it."
    )
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "config.yaml")
    parser.add_argument(
        "--paths-config", type=Path, default=PROJECT / "configs" / "local_paths.yaml"
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--functional-region-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--point-count", type=int)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-count", type=int)
    parser.add_argument("--scene-ids-file", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--override", action="append", default=[])
    # Accepted only so old validation shortcuts remain harmless and compatible.
    parser.add_argument("--verify-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.scene_start < 0:
        parser.error("--scene-start must be non-negative")
    if args.scene_count is not None and args.scene_count <= 0:
        parser.error("--scene-count must be positive")
    if args.point_count is not None and args.point_count < 0:
        parser.error("--point-count must be non-negative")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return validate(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
