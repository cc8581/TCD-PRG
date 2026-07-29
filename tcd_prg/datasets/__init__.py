"""Dataset-independent contracts and concrete adapters."""

from typing import Any

from .base import DatasetAdapter
from .capabilities import DatasetCapabilities
from .task_oriented_clutter import TaskOrientedClutterAdapter
from .template import DatasetAdapterTemplate
from .gapg_observation import GAPGObservationAdapter
from .types import ActionCandidateGroup, SceneObservation, SequenceLabels, StateLabels

__all__ = [
    "ActionCandidateGroup",
    "ActionStateGroupDataset",
    "collate_unified",
    "DatasetAdapter",
    "DatasetCapabilities",
    "DatasetAdapterTemplate",
    "GAPGObservationAdapter",
    "SceneObservation",
    "SequenceLabels",
    "StateLabels",
    "StateGroupUnit",
    "TaskOrientedClutterAdapter",
    "split_units_by_scene",
]


def __getattr__(name: str) -> Any:
    if name == "collate_unified":
        from .collate import collate_unified

        return collate_unified
    if name in {"ActionStateGroupDataset", "StateGroupUnit", "split_units_by_scene"}:
        from .torch_dataset import ActionStateGroupDataset, StateGroupUnit, split_units_by_scene

        return {
            "ActionStateGroupDataset": ActionStateGroupDataset,
            "StateGroupUnit": StateGroupUnit,
            "split_units_by_scene": split_units_by_scene,
        }[name]
    raise AttributeError(name)
