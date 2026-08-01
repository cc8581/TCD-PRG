"""Dataset-independent contracts and concrete adapters."""

from typing import Any

from .base import DatasetAdapter
from .capabilities import DatasetCapabilities
from .gapg_observation import GAPGObservationAdapter
from .policy_candidates import (
    POLICY_CANDIDATE_CACHE_FORMAT,
    load_candidate_batch,
    match_generated_candidates,
)
from .task_oriented_clutter import TaskOrientedClutterAdapter
from .template import DatasetAdapterTemplate
from .types import (
    ActionCandidateGroup,
    GlobalGraspLabels,
    SceneObservation,
    SequenceLabels,
    StateLabels,
)

__all__ = [
    "ActionCandidateGroup",
    "ActionStateGroupDataset",
    "collate_unified",
    "DatasetAdapter",
    "DatasetCapabilities",
    "DistributedEvaluationSampler",
    "DistributedWeightedStateSampler",
    "DatasetAdapterTemplate",
    "GAPGObservationAdapter",
    "GlobalGraspLabels",
    "POLICY_CANDIDATE_CACHE_FORMAT",
    "SceneObservation",
    "SequenceLabels",
    "StateLabels",
    "StateGroupUnit",
    "TaskOrientedClutterAdapter",
    "load_candidate_batch",
    "match_generated_candidates",
    "split_units_by_scene",
]


def __getattr__(name: str) -> Any:
    if name == "collate_unified":
        from .collate import collate_unified

        return collate_unified
    if name in {
        "ActionStateGroupDataset",
        "DistributedEvaluationSampler",
        "DistributedWeightedStateSampler",
        "StateGroupUnit",
        "split_units_by_scene",
    }:
        from .torch_dataset import (
            ActionStateGroupDataset,
            DistributedEvaluationSampler,
            DistributedWeightedStateSampler,
            StateGroupUnit,
            split_units_by_scene,
        )

        return {
            "ActionStateGroupDataset": ActionStateGroupDataset,
            "DistributedEvaluationSampler": DistributedEvaluationSampler,
            "DistributedWeightedStateSampler": DistributedWeightedStateSampler,
            "StateGroupUnit": StateGroupUnit,
            "split_units_by_scene": split_units_by_scene,
        }[name]
    raise AttributeError(name)
