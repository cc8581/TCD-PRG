"""Fair-comparison policy interface used by all learned and external methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from tcd_prg.datasets.types import SceneObservation


@dataclass(frozen=True, slots=True)
class GlobalGraspPrediction:
    """Common task-free grasp output shared by TCD-PRG and baselines."""

    object_index: int
    contact_point_world: np.ndarray
    grasp_pose_world: np.ndarray
    width_m: float
    raw_score: float
    scene_score: float
    intrinsic_score: float | None
    certified: bool
    source: str

    @property
    def score(self) -> float:
        """Compatibility alias for execution-oriented scene ranking."""

        return self.scene_score


class ManipulationPolicy(ABC):
    @abstractmethod
    def encode_observation(self, observation: SceneObservation) -> Any: ...

    @abstractmethod
    def generate_candidates(self, encoded: Any) -> Any: ...

    @abstractmethod
    def select_action(self, candidates: Any) -> Any: ...

    @abstractmethod
    def predict_grasps(self, encoded: Any) -> Any: ...

    def predict_task_grasps(self, encoded: Any) -> Any:
        """Task-conditioned grasp API; legacy policies delegate here."""

        return self.predict_grasps(encoded)

    def predict_global_grasps(self, encoded: Any) -> list[GlobalGraspPrediction]:
        """Task-free grasp API. Baselines must opt in explicitly."""

        raise NotImplementedError(f"{type(self).__name__} has no task-free global grasp output")

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update_after_action(self, action: Any, observation: SceneObservation) -> None: ...
