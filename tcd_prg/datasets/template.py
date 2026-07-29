"""Documented starting point for adapting another manipulation dataset."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable

from .base import DatasetAdapter
from .capabilities import DatasetCapabilities
from .types import ActionCandidateGroup, SceneObservation, SequenceLabels, StateLabels


class DatasetAdapterTemplate(DatasetAdapter):
    """Strict adapter skeleton with the invariants required by model code.

    Implementations must map native labels into the unified dataclasses.  A
    missing capability must be declared false; it must never be replaced by a
    zero tensor pretending to be ground truth.
    """

    capabilities = DatasetCapabilities(
        has_instance_masks=False,
        has_task_regions=False,
        has_task_grasps=False,
        has_push_actions=False,
        has_pick_remove_actions=False,
        has_sequences=False,
        has_relation_graph=False,
        has_exact_ik=False,
        has_intermediate_observations=False,
        supports_closed_loop=False,
    )

    @abstractmethod
    def iter_action_groups(self, split: str | None = None) -> Iterable[tuple[int, int, int, int]]:
        """Yield local composite keys, never globally ordered numeric features."""

    @abstractmethod
    def load_observation(self, scene_id: int, state_id: int,
                         task_index: int) -> SceneObservation:
        """Return world-frame metres, xyzw quaternions, and non-Oracle cameras."""

    @abstractmethod
    def load_state_labels(self, scene_id: int, state_id: int) -> StateLabels:
        """Return only labels genuinely present in the native dataset."""

    @abstractmethod
    def load_action_group(self, scene_id: int, group_index: int) -> ActionCandidateGroup:
        """Preserve POSITIVE/NEGATIVE/UNKNOWN_UNTESTED candidate status."""

    @abstractmethod
    def load_sequences(self, scene_id: int,
                       task_index: int | None = None) -> tuple[SequenceLabels, ...]:
        """Return an empty tuple when has_sequences is false."""
