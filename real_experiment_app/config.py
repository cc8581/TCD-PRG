from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import tempfile
import yaml
import numpy as np

from .transforms import xyz_rpy_to_matrix


@dataclass(slots=True)
class AppConfig:
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(path, raw)

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.path.parent / path).resolve()

    def save(self) -> None:
        """Atomically persist operator settings using UTF-8 YAML."""
        payload = yaml.safe_dump(self.raw, allow_unicode=True, sort_keys=False)
        handle, temporary = tempfile.mkstemp(
            prefix=f".{self.path.stem}-", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def record_table_plane(
        self, normal: list[float] | np.ndarray, point_m: list[float] | np.ndarray
    ) -> None:
        """Record a calibrated tabletop plane in robot-base coordinates."""
        normal_value = np.asarray(normal, dtype=np.float64)
        point_value = np.asarray(point_m, dtype=np.float64)
        if normal_value.shape != (3,) or point_value.shape != (3,):
            raise ValueError("Table normal and point must each contain three values")
        if not np.isfinite(normal_value).all() or not np.isfinite(point_value).all():
            raise ValueError("Table calibration values must be finite")
        length = float(np.linalg.norm(normal_value))
        if length < 1e-8:
            raise ValueError("Table normal must be non-zero")
        normal_value /= length
        if normal_value[2] <= 0:
            raise ValueError("Table normal must point upward in robot-base coordinates")
        fusion = self.raw.setdefault("fusion", {})
        fusion["table_plane_base"] = {
            "normal": normal_value.tolist(),
            "offset_m": -float(normal_value @ point_value),
        }
        self.save()

    @property
    def tcp_transform(self) -> np.ndarray:
        tcp = self.raw["robot"]["model_tcp_to_robot_tcp"]
        if "matrix" in tcp:
            value = np.asarray(tcp["matrix"], np.float64)
            if value.shape != (4, 4): raise ValueError("TCP matrix must be 4x4")
            return value
        return xyz_rpy_to_matrix(tcp.get("xyz_mm_rpy_deg", [0,0,0,0,0,0]), 0.001)
