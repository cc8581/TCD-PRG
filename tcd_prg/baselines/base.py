"""Fair-comparison policy interface used by all learned and external methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tcd_prg.datasets.types import SceneObservation


class ManipulationPolicy(ABC):
    @abstractmethod
    def encode_observation(self, observation: SceneObservation) -> Any: ...

    @abstractmethod
    def generate_candidates(self, encoded: Any) -> Any: ...

    @abstractmethod
    def select_action(self, candidates: Any) -> Any: ...

    @abstractmethod
    def predict_grasps(self, encoded: Any) -> Any: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update_after_action(self, action: Any, observation: SceneObservation) -> None: ...

