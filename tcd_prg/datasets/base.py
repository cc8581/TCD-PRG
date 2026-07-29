"""Abstract adapter contract for every supported dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from .capabilities import DatasetCapabilities
from .types import ActionCandidateGroup, SceneObservation, SequenceLabels, StateLabels, UnifiedSample


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

    def load_sample(self, scene_id: int, state_id: int, task_index: int, group_index: int) -> UnifiedSample:
        sample = UnifiedSample(
            observation=self.load_observation(scene_id, state_id, task_index),
            state_labels=self.load_state_labels(scene_id, state_id),
            candidates=self.load_action_group(scene_id, group_index),
            sequences=self.load_sequences(scene_id, task_index),
        )
        sample.observation.validate()
        sample.candidates.validate()
        return sample

