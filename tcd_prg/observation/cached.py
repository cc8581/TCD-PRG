"""Content-addressed observation cache with atomic writes and LRU eviction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

import numpy as np

from .base import ObservationProvider, ObservationRequest, PointObservation


def request_hash(request: ObservationRequest) -> str:
    # key 覆盖状态、相机和渲染版本，防止参数变化后误复用旧点云。
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
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


class ObservationCacheMissError(FileNotFoundError):
    """Raised when synchronous rendering is disabled and a state is not cached."""


class CachedObservationProvider(ObservationProvider):
    def __init__(self, cache_dir: str | Path, fallback: ObservationProvider | None = None,
                 max_bytes: int = 15 << 30, min_free_bytes: int = 20 << 30,
                 eviction_enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.fallback = fallback
        self.read_only = fallback is None
        if not self.read_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.eviction_enabled = eviction_enabled
        self._size_lock = threading.Lock()
        self._eviction_lock = threading.Lock()
        # Stat the cache once at startup.  Afterwards writes update this value
        # in O(1), avoiding periodic full scans of hundreds of thousands of
        # files while the cache is comfortably below its limit.
        self._cache_bytes = 0 if self.read_only else self._measure_cache_bytes()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_size_lock", None)
        state.pop("_eviction_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._size_lock = threading.Lock()
        self._eviction_lock = threading.Lock()

    def _measure_cache_bytes(self) -> int:
        total = 0
        for path in self.cache_dir.glob("*/*.npz"):
            if path.name.endswith(".tmp.npz"):
                continue
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.npz"

    def is_available(self, request: ObservationRequest) -> bool:
        """Return whether this request is already cached without rendering."""

        return self._path(request_hash(request)).is_file()

    def get(self, request: ObservationRequest) -> PointObservation:
        key = request_hash(request)
        path = self._path(key)
        if path.exists():
            try:
                # Strict cache-only training must not mutate legacy entries,
                # including their access/modified timestamps.
                if not self.read_only:
                    os.utime(path, None)
                with np.load(path, allow_pickle=False) as data:
                    rgb = data["rgb"]
                    if rgb.dtype == np.uint8:
                        rgb = rgb.astype(np.float32) / 255.0
                    return PointObservation(
                        data["xyz"], rgb, data["instance_id"], data["source_view"]
                    )
            except FileNotFoundError:
                # Another DataLoader worker may evict this entry between the
                # existence check and open; treat that race as a normal miss.
                pass
        # Evaluation can deliberately omit fallback and remain strictly cache-only.
        if self.fallback is None:
            raise ObservationCacheMissError(
                f"Observation {key} is not cached and no renderer fallback is configured"
            )
        observation = self.fallback.get(request)
        estimated_bytes = sum(
            int(value.nbytes)
            for value in (
                observation.xyz, observation.rgb,
                observation.instance_id, observation.source_view,
            )
        )
        free = shutil.disk_usage(self.cache_dir).free
        if free < self.min_free_bytes + estimated_bytes:
            if self.eviction_enabled:
                self.evict(reserve_bytes=estimated_bytes)
                free = shutil.disk_usage(self.cache_dir).free
        if free < self.min_free_bytes + estimated_bytes:
            raise OSError(
                "Observation cache write refused without deleting existing entries: "
                "configured free-space reserve "
                f"({self.min_free_bytes} bytes) cannot be maintained"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        # 先写进程唯一的临时文件再原子替换，读取方永远只看到完整条目。
        temp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        rgb = np.asarray(observation.rgb)
        quantized_rgb = np.rint(rgb * 255.0)
        if (
            np.issubdtype(rgb.dtype, np.floating)
            and np.isfinite(rgb).all()
            and np.all((rgb >= 0.0) & (rgb <= 1.0))
            and np.allclose(rgb * 255.0, quantized_rgb, atol=1e-5, rtol=0.0)
        ):
            rgb = quantized_rgb.astype(np.uint8)
        instance_id = np.asarray(observation.instance_id)
        if (
            np.issubdtype(instance_id.dtype, np.integer)
            and instance_id.size
            and instance_id.min() >= np.iinfo(np.int16).min
            and instance_id.max() <= np.iinfo(np.int16).max
        ):
            instance_id = instance_id.astype(np.int16)
        np.savez_compressed(
            temp,
            xyz=observation.xyz,
            rgb=rgb,
            instance_id=instance_id,
            source_view=observation.source_view,
        )
        os.replace(temp, path)
        written_bytes = path.stat().st_size
        with self._size_lock:
            self._cache_bytes += written_bytes
            over_capacity = self._cache_bytes > self.max_bytes
        if over_capacity and self.eviction_enabled:
            self.evict()
        return observation

    def evict(self, reserve_bytes: int = 0) -> None:
        if self.read_only:
            raise RuntimeError("Read-only observation cache cannot evict entries")
        if not self.eviction_enabled:
            raise RuntimeError("Observation cache eviction is disabled")
        with self._eviction_lock, self._size_lock:
            entries: list[tuple[Path, int, float]] = []
            for path in self.cache_dir.glob("*/*.npz"):
                if path.name.endswith(".tmp.npz"):
                    continue
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                entries.append((path, stat.st_size, stat.st_atime))
            total = sum(size for _, size, _ in entries)
            for path, size, _ in sorted(entries, key=lambda item: item[2]):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except PermissionError:
                    # Windows does not allow unlinking an npz while another
                    # DataLoader worker has it open.  A locked cache entry is not
                    # corrupt and will be eligible again during the next eviction
                    # pass, so continue with the remaining LRU candidates.
                    continue
                total -= size
                free = shutil.disk_usage(self.cache_dir).free
                if total <= self.max_bytes and free >= self.min_free_bytes + reserve_bytes:
                    break
            self._cache_bytes = total

    def clear_completed(self) -> dict[str, int]:
        """Remove completed observations while tolerating active worker readers."""

        if self.read_only:
            raise RuntimeError("Read-only observation cache cannot delete entries")

        removed_files = 0
        removed_bytes = 0
        locked_files = 0
        for path in self.cache_dir.glob("*/*.npz"):
            if path.name.endswith(".tmp.npz"):
                continue
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                continue
            except PermissionError:
                locked_files += 1
                continue
            removed_files += 1
            removed_bytes += size
        with self._size_lock:
            self._cache_bytes = max(0, self._cache_bytes - removed_bytes)
        return {
            "removed_files": removed_files,
            "removed_bytes": removed_bytes,
            "locked_files": locked_files,
        }
