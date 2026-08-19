"""Dataset-independent contracts and concrete adapters."""

from typing import Any

from .base import DatasetAdapter
from .acronym_grasp_database import load_object_grasps, match_object_grasp_priors
from .capabilities import DatasetCapabilities
from .gapg_observation import GAPGObservationAdapter
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
    "DistributedTaskStateBatchSampler",
    "DistributedWeightedStateSampler",
    "GlobalStateDataset",
    "DatasetAdapterTemplate",
    "GAPGObservationAdapter",
    "GlobalGraspLabels",
    "SceneObservation",
    "SequenceLabels",
    "StateLabels",
    "StateGroupUnit",
    "TaskOrientedClutterAdapter",
    "load_object_grasps",
    "match_object_grasp_priors",
]


def __getattr__(name: str) -> Any:
    if name == "collate_unified":
        from .collate import collate_unified

        return collate_unified
    if name in {
        "ActionStateGroupDataset",
        "DistributedEvaluationSampler",
        "DistributedTaskStateBatchSampler",
        "DistributedWeightedStateSampler",
        "GlobalStateDataset",
        "StateGroupUnit",
    }:
        from .torch_dataset import (
            ActionStateGroupDataset,
            DistributedEvaluationSampler,
            DistributedTaskStateBatchSampler,
            DistributedWeightedStateSampler,
            GlobalStateDataset,
            StateGroupUnit,
        )

        return {
            "ActionStateGroupDataset": ActionStateGroupDataset,
            "DistributedEvaluationSampler": DistributedEvaluationSampler,
            "DistributedTaskStateBatchSampler": DistributedTaskStateBatchSampler,
            "DistributedWeightedStateSampler": DistributedWeightedStateSampler,
            "GlobalStateDataset": GlobalStateDataset,
            "StateGroupUnit": StateGroupUnit,
        }[name]
    raise AttributeError(name)
