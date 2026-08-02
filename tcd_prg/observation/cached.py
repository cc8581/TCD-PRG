"""Content-addressed observation cache with atomic writes and LRU eviction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .base import ObservationProvider, ObservationRequest, PointObservation


def request_hash(request: ObservationRequest) -> str:
    payload = {
        "scene_id": request.scene_id,
        "state_id": request.state_id,
        "object_pose": np.asarray(request.object_pose, dtype=np.float32).round(7).tolist(),
        "object_active": np.asarray(request.object_active, dtype=bool).tolist(),
        "object_present": np.asarray(request.object_present, dtype=bool).tolist(),
        "object_asset_ids": request.object_asset_ids,
        "object_model_ids": request.object_model_ids,
        "object_scales": np.asarray(request.object_scales, dtype=np.float32).round(7).tolist(),
        "render_seed": request.render_seed,
        "camera_profile": request.camera_profile,
        "point_count": request.point_count,
        "renderer_version": request.renderer_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ObservationCacheMissError(FileNotFoundError):
    """Raised when synchronous rendering is disabled and a state is not cached."""


class CachedObservationProvider(ObservationProvider):
    def __init__(self, cache_dir: str | Path, fallback: ObservationProvider | None = None,
                 max_bytes: int = 15 << 30, min_free_bytes: int = 20 << 30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fallback = fallback
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.npz"

    def is_available(self, request: ObservationRequest) -> bool:
        return self._path(request_hash(request)).is_file() or self.fallback is not None

    def get(self, request: ObservationRequest) -> PointObservation:
        key = request_hash(request)
        path = self._path(key)
        if path.exists():
            os.utime(path, None)
            with np.load(path, allow_pickle=False) as data:
                return PointObservation(data["xyz"], data["rgb"], data["instance_id"], data["source_view"])
        if self.fallback is None:
            raise ObservationCacheMissError(
                f"Observation {key} is not cached; run tcd-prg-prefetch before training"
            )
        observation = self.fallback.get(request)
        estimated_bytes = sum(
            int(value.nbytes)
            for value in (
                observation.xyz, observation.rgb,
                observation.instance_id, observation.source_view,
            )
        )
        self.evict(reserve_bytes=estimated_bytes)
        if shutil.disk_usage(self.cache_dir).free < self.min_free_bytes + estimated_bytes:
            raise OSError(
                "Observation cache write refused: configured free-space reserve "
                f"({self.min_free_bytes} bytes) cannot be maintained"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temp,
            xyz=observation.xyz,
            rgb=observation.rgb,
            instance_id=observation.instance_id,
            source_view=observation.source_view,
        )
        os.replace(temp, path)
        self.evict()
        return observation

    def evict(self, reserve_bytes: int = 0) -> None:
        files = list(self.cache_dir.glob("*/*.npz"))
        total = sum(p.stat().st_size for p in files)
        free = shutil.disk_usage(self.cache_dir).free
        if total <= self.max_bytes and free >= self.min_free_bytes + reserve_bytes:
            return
        for path in sorted(files, key=lambda p: p.stat().st_atime):
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
            free = shutil.disk_usage(self.cache_dir).free
            if total <= self.max_bytes and free >= self.min_free_bytes + reserve_bytes:
                break
