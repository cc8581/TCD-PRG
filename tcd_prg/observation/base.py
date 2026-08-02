"""Observation reconstruction protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    scene_id: int
    state_id: int
    object_pose: np.ndarray
    object_active: np.ndarray
    object_present: np.ndarray
    object_asset_ids: tuple[str, ...]
    object_model_ids: tuple[str, ...]
    object_scales: np.ndarray
    render_seed: int
    camera_profile: str
    point_count: int
    renderer_version: str


@dataclass(slots=True)
class PointObservation:
    xyz: np.ndarray
    rgb: np.ndarray
    instance_id: np.ndarray
    source_view: np.ndarray


class ObservationProvider(ABC):
    def is_available(self, request: ObservationRequest) -> bool:
        """Return whether ``get`` can serve the request without new rendering."""

        return True

    @abstractmethod
    def get(self, request: ObservationRequest) -> PointObservation:
        pass
