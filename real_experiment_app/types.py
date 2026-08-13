from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class RGBDFrame:
    camera_id: str
    color_rgb: np.ndarray
    depth_mm: np.ndarray
    intrinsics: dict[str, float]
    camera_to_base: np.ndarray


@dataclass(slots=True)
class SegmentationResult:
    instance_image: np.ndarray
    category_by_instance: dict[int, int]


@dataclass(slots=True)
class FusedScene:
    xyz_m: np.ndarray
    rgb: np.ndarray
    instance_id: np.ndarray
    source_view: np.ndarray
    category_by_instance: dict[int, int]

    @property
    def instance_ids(self) -> list[int]:
        return sorted(int(x) for x in np.unique(self.instance_id) if int(x) >= 0)


@dataclass(slots=True)
class Prediction:
    action: dict[str, Any]
    inference_seconds: float

