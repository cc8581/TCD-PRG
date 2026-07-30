"""Abstract adapter contract for every supported dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .capabilities import DatasetCapabilities
from .types import (
    ActionCandidateGroup, GlobalGraspLabels, SceneObservation, SequenceLabels,
    StateLabels, UnifiedSample,
)


class DatasetAdapter(ABC):
    capabilities: DatasetCapabilities

    @abstractmethod
    def iter_action_groups(self, split: str | None = None) -> Iterable[tuple[int, int, int, int]]:
        """Yield ``(scene_id, state_id, task_index, group_index)`` units."""

    @abstractmethod
    def load_observation(self, scene_id: int, state_id: int, task_index: int) -> SceneObservation:
        pass

    @abstractmethod
    def load_state_labels(self, scene_id: int, state_id: int) -> StateLabels:
        pass

    @abstractmethod
    def load_action_group(self, scene_id: int, group_index: int) -> ActionCandidateGroup:
        pass

    @abstractmethod
    def load_sequences(self, scene_id: int, task_index: int | None = None) -> tuple[SequenceLabels, ...]:
        pass

    def load_global_grasps(
        self, scene_id: int, state_id: int, observation: SceneObservation
    ) -> GlobalGraspLabels | None:
        """Optional task-free grasp supervision supplied by capable datasets."""

        return None

    def load_sample(self, scene_id: int, state_id: int, task_index: int, group_index: int) -> UnifiedSample:
        observation = self.load_observation(scene_id, state_id, task_index)
        sample = UnifiedSample(
            observation=observation,
            state_labels=self.load_state_labels(scene_id, state_id),
            candidates=self.load_action_group(scene_id, group_index),
            sequences=self.load_sequences(scene_id, task_index),
            global_grasps=self.load_global_grasps(scene_id, state_id, observation),
        )
        sample.observation.validate()
        sample.candidates.validate()
        if sample.global_grasps is not None:
            sample.global_grasps.validate()
        return sample
