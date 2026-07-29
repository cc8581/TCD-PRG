"""AG-160-95 kinematic opening calibration and gripper point-cloud contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class AG16095Calibration:
    min_width_m: float = 0.0
    max_width_m: float = 0.095
    urdf_open_inner_gap_m: float = 0.0892024577
    open_pad_center_distance_m: float = 0.1092001200
    closed_pad_center_distance_m: float = 0.0157364507
    open_joint_rad: float = 0.0
    closed_joint_rad: float = 0.93
    tcp_from_mount_m: tuple[float, float, float] = (0.0, 0.0, 0.19)

    def clamp_width(self, width_m: np.ndarray | float) -> np.ndarray:
        return np.clip(np.asarray(width_m, dtype=np.float32), self.min_width_m, self.max_width_m)

    def width_to_joint(self, width_m: np.ndarray | float) -> np.ndarray:
        """Map AG total commanded opening to the master linkage joint."""

        width = self.clamp_width(width_m)
        alpha = (self.max_width_m - width) / (self.max_width_m - self.min_width_m)
        return self.open_joint_rad + alpha * (self.closed_joint_rad - self.open_joint_rad)


@dataclass(slots=True)
class GripperGeometry:
    """Sampled gripper surface in the canonical TCP frame."""

    points_tcp: np.ndarray
    width_m: float
    source_urdf: Path

    def validate(self, calibration: AG16095Calibration) -> None:
        if self.points_tcp.ndim != 2 or self.points_tcp.shape[1] != 3:
            raise ValueError("points_tcp must be [G,3]")
        if not calibration.min_width_m <= self.width_m <= calibration.max_width_m:
            raise ValueError(f"width {self.width_m} outside AG limits")
