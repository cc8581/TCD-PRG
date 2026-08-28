"""Exact scene-state/action samples for PUSH effectiveness training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from tcd_prg.constants import PUSH_DISTANCE_M, ActionType, CandidateStatus

from .types import SceneObservation


@dataclass(slots=True)
class PushEvaluatorSample:
    scene_id: int
    state_id: int
    observation: SceneObservation
    acted_object: int
    contact_world: np.ndarray
    direction_world: np.ndarray
    push_distance: float
    task_category_id: int
    task_region_id: int
    target_object: int
    evaluation_status: int
    action_improves_state: bool


class PushEffectivenessDataset(Dataset[PushEvaluatorSample]):
    """Flatten known evaluated PUSH actions from an immutable state dataset."""

    def __init__(self, state_dataset: Dataset) -> None:
        self.state_dataset = state_dataset
        self.index: list[tuple[int, int]] = []
        positives = 0
        for state_index in range(len(state_dataset)):
            sample = state_dataset[state_index]
            group = sample.candidates
            valid = (
                group.valid_mask
                & (group.action_type == int(ActionType.PUSH))
                & (group.evaluation_status != int(CandidateStatus.UNKNOWN_UNTESTED))
            )
            for action_index in np.flatnonzero(valid).tolist():
                self.index.append((state_index, action_index))
                positives += int(group.action_improves_state[action_index])
        self.positive_count = positives
        self.negative_count = len(self.index) - positives

    @property
    def pos_weight(self) -> float:
        if self.positive_count == 0:
            raise ValueError("Cannot train PUSH evaluator without positive actions")
        return self.negative_count / self.positive_count

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> PushEvaluatorSample:
        state_index, action_index = self.index[index]
        sample = self.state_dataset[state_index]
        observation = sample.observation
        group = sample.candidates
        parameters = group.action_parameters
        return PushEvaluatorSample(
            scene_id=int(observation.scene_id),
            state_id=int(observation.state_id),
            observation=observation,
            acted_object=int(group.acted_object[action_index]),
            contact_world=np.asarray(parameters["push_contact_world"][action_index], np.float32),
            direction_world=np.asarray(
                parameters["push_direction_world"][action_index], np.float32
            ),
            push_distance=float(PUSH_DISTANCE_M),
            task_category_id=int(observation.object_category_id[observation.target_object]),
            task_region_id=int(observation.task_region_id),
            target_object=int(observation.target_object),
            evaluation_status=int(group.evaluation_status[action_index]),
            action_improves_state=bool(group.action_improves_state[action_index]),
        )
