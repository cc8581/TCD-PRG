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

    @property
    def tcp_transform(self) -> np.ndarray:
        tcp = self.raw["robot"]["model_tcp_to_robot_tcp"]
        if "matrix" in tcp:
            value = np.asarray(tcp["matrix"], np.float64)
            if value.shape != (4, 4): raise ValueError("TCP matrix must be 4x4")
            return value
        return xyz_rpy_to_matrix(tcp.get("xyz_mm_rpy_deg", [0,0,0,0,0,0]), 0.001)
