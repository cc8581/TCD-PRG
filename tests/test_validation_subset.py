from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.datasets.torch_dataset import ActionStateGroupDataset
from tcd_prg.datasets.types import UnifiedSample
from tcd_prg.scripts.train import load_or_create_validation_scene_subset

STRATA = (
    "direct_grasp",
    "pick_remove",
    "push",
    "push_failure",
    "unresolved_or_unknown",
)


class _Adapter:
    def __init__(self, reverse=False):
        groups = []
        self.strata = {}
        index = 0
        for stratum_index, stratum in enumerate(STRATA):
            for local in range(6):
                unit = (local % 3, 100 * stratum_index + local, 0, index)
                groups.append(unit)
                self.strata[unit] = stratum
                index += 1
        self.groups = list(reversed(groups)) if reverse else groups

    def iter_action_groups(self, split=None):
        del split
        yield from self.groups

    def action_group_strata(self, units):
        return {unit: self.strata[unit] for unit in units}

    def load_sample(self, *args, **kwargs):
        raise AssertionError("subset construction must not load observations")


def test_validation_subset_is_scene_diverse_persisted_and_reused(tmp_path):
    manifest = tmp_path / "validation_subset.json"
    quota = {name: 2 for name in STRATA}
    first = ActionStateGroupDataset(
        _Adapter(False),
        split="val",
        max_groups=10,
        stratified_max_groups=True,
        stratum_quota=quota,
        subset_manifest_path=manifest,
        subset_seed=2026,
    )
    assert len(first) == 10
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["groups"]) == 10
    for stratum in STRATA:
        scenes = {unit.scene_id for unit in first.units if unit.stratum == stratum}
        assert len(scenes) == 2

    # Iterator order changes, but resume must reuse the manifest exactly.
    second = ActionStateGroupDataset(
        _Adapter(True),
        split="val",
        max_groups=10,
        stratified_max_groups=True,
        stratum_quota=quota,
        subset_manifest_path=manifest,
        subset_seed=2026,
    )
    assert [(u.scene_id, u.state_id, u.task_index, u.group_index) for u in first.units] == [
        (u.scene_id, u.state_id, u.task_index, u.group_index) for u in second.units
    ]


def test_validation_scene_subset_is_random_fixed_and_complete(tmp_path):
    manifest = tmp_path / "validation_scene_subset.json"
    selected = load_or_create_validation_scene_subset(
        tuple(range(30)), 20, 2026, manifest
    )
    assert len(selected) == 20
    assert selected != tuple(range(20))

    dataset = ActionStateGroupDataset(_Adapter(False), split="val", scene_ids=frozenset({0, 2}))
    assert {unit.scene_id for unit in dataset.units} == {0, 2}
    assert len(dataset) == 20

    # Input ordering does not change the saved or deterministic subset.
    reused = load_or_create_validation_scene_subset(
        tuple(reversed(range(30))), 20, 2026, manifest
    )
    assert reused == selected


def test_validation_scene_subset_rejects_manifest_drift(tmp_path):
    manifest = tmp_path / "validation_scene_subset.json"
    load_or_create_validation_scene_subset(tuple(range(30)), 20, 2026, manifest)
    with pytest.raises(ValueError, match="seed"):
        load_or_create_validation_scene_subset(tuple(range(30)), 20, 7, manifest)


def test_push_object_state_aggregation_uses_groups_outside_subset():
    def group(acted_object, status, success):
        return SimpleNamespace(
            valid_mask=np.asarray([True]),
            action_type=np.asarray([int(ActionType.PUSH)]),
            acted_object=np.asarray([acted_object]),
            evaluation_status=np.asarray([int(status)]),
            success_mask=np.asarray([success]),
        )

    class Adapter:
        groups = ((4, 9, 2, 10), (4, 9, 2, 11))
        candidates = {
            10: group(1, CandidateStatus.NEGATIVE, False),
            11: group(2, CandidateStatus.POSITIVE, True),
        }

        def iter_action_groups(self, split=None):
            del split
            yield from self.groups

        @staticmethod
        def action_group_strata(units):
            return {}

        def load_action_group(self, scene_id, group_index):
            assert scene_id == 4
            return self.candidates[group_index]

        def load_sample(self, scene_id, state_id, task_index, group_index, **kwargs):
            del kwargs
            return UnifiedSample(
                observation=None,  # type: ignore[arg-type]
                state_labels=None,  # type: ignore[arg-type]
                candidates=self.load_action_group(scene_id, group_index),
                sequences=(),
            )

    dataset = ActionStateGroupDataset(
        Adapter(), max_groups=1, global_grasp_mode="never"  # type: ignore[arg-type]
    )
    state = dataset[0].push_object_state
    assert state is not None
    assert state.evaluated_object.tolist() == [1, 2]
    assert state.positive_object.tolist() == [2]
