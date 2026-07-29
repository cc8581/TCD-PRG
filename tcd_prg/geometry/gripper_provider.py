"""Content-addressed exact AG-160-95 geometry provider."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .gripper import AG16095Calibration, GripperGeometry


class ExactAG16095GeometryProvider:
    """Sample exact URDF collision meshes at a requested total opening."""

    VERSION = "ag16095_collision_surface_v2"

    def __init__(self, python_executable: str | Path, worker_script: str | Path,
                 urdf: str | Path, cache_dir: str | Path, point_count: int = 512,
                 seed: int = 2026, calibration: AG16095Calibration | None = None,
                 allow_generate: bool = True) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = Path(worker_script)
        self.urdf = Path(urdf)
        self.cache_dir = Path(cache_dir)
        self.point_count, self.seed = int(point_count), int(seed)
        self.calibration = calibration or AG16095Calibration()
        self.allow_generate = bool(allow_generate)
        for path in (self.python_executable, self.worker_script, self.urdf):
            if not path.is_file():
                raise FileNotFoundError(path)
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._urdf_digest = hashlib.sha256(self.urdf.read_bytes()).hexdigest()

    def _key(self, width_m: float) -> str:
        payload = {"version": self.VERSION, "urdf_sha256": self._urdf_digest,
                   "width_m": round(float(width_m), 7), "point_count": self.point_count,
                   "seed": self.seed}
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()

    def get(self, width_m: float) -> GripperGeometry:
        width = float(self.calibration.clamp_width(width_m))
        return self.get_many(np.asarray([width], dtype=np.float32))[0]

    def _load(self, width: float) -> GripperGeometry | None:
        key = self._key(width)
        path = self.cache_dir / key[:2] / f"{key}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            geometry = GripperGeometry(data["points_tcp"].astype(np.float32),
                                       float(data["width_m"]), self.urdf)
        geometry.validate(self.calibration)
        return geometry

    def get_many(self, widths_m: np.ndarray) -> tuple[GripperGeometry, ...]:
        widths = np.asarray(self.calibration.clamp_width(widths_m), dtype=np.float64).reshape(-1)
        unique = np.unique(np.round(widths, 7))
        geometries = {float(width): self._load(float(width)) for width in unique}
        missing = np.asarray([width for width, value in geometries.items() if value is None])
        if len(missing):
            if not self.allow_generate:
                raise FileNotFoundError(
                    f"{len(missing)} AG geometries are absent; run tcd-prg-prefetch before training"
                )
            with tempfile.TemporaryDirectory(dir=self.cache_dir) as temporary_dir:
                temporary = Path(temporary_dir)
                request, output = temporary / "widths.npz", temporary / "geometry.npz"
                np.savez_compressed(request, widths_m=missing)
                command = [str(self.python_executable), str(self.worker_script), "--urdf",
                           str(self.urdf), "--widths-npz", str(request), "--point-count",
                           str(self.point_count), "--seed", str(self.seed), "--output", str(output)]
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
                if completed.returncode != 0:
                    raise RuntimeError("Exact AG geometry worker failed\n"
                                       f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
                if not output.is_file():
                    raise RuntimeError("AG geometry worker returned no output")
                with np.load(output, allow_pickle=False) as data:
                    generated_points = data["points_tcp"].astype(np.float32)
                    generated_widths = data["widths_m"].astype(np.float32)
                for width, points in zip(generated_widths, generated_points, strict=True):
                    width = float(round(float(width), 7))
                    key = self._key(width)
                    path = self.cache_dir / key[:2] / f"{key}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pending = path.with_name(f"{path.stem}.tmp.npz")
                    np.savez_compressed(pending, points_tcp=points,
                                        width_m=np.asarray(width, np.float32),
                                        source_urdf=np.asarray(str(self.urdf.resolve())))
                    pending.replace(path)
                    geometries[width] = self._load(width)
        return tuple(geometries[float(round(float(width), 7))] for width in widths)  # type: ignore[misc]

    def prewarm(self, widths_m: np.ndarray) -> tuple[Path, ...]:
        widths = np.unique(np.round(np.asarray(widths_m, dtype=np.float64), 7))
        widths = widths[np.isfinite(widths)]
        self.get_many(widths)
        paths = []
        for width in widths:
            clamped = float(self.calibration.clamp_width(width))
            key = self._key(clamped)
            paths.append(self.cache_dir / key[:2] / f"{key}.npz")
        return tuple(paths)
