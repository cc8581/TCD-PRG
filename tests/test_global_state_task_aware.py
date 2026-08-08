from __future__ import annotations

from types import SimpleNamespace

from tcd_prg.datasets.torch_dataset import GlobalStateDataset, StateGroupUnit


class Adapter:
    def __init__(self, supervised):
        self.supervised = frozenset(supervised)

    def global_grasp_supervised_task_states(self, split, min_width, max_width):
        assert split == "train"
        assert min_width == 0.01
        assert max_width == 0.08
        return self.supervised


def dataset(units, supervised):
    return SimpleNamespace(
        adapter=Adapter(supervised),
        units=tuple(units),
        split="train",
    )


def test_uses_certified_task_not_first_arbitrary_task():
    units = (
        StateGroupUnit(1, 7, 0, 10, "push"),
        StateGroupUnit(1, 7, 3, 11, "pick_remove"),
    )
    source = dataset(units, {(1, 7, 3)})
    global_ds = GlobalStateDataset(source, 0.01, 0.08)
    assert len(global_ds.units) == 1
    assert global_ds.units[0].task_index == 3
    assert global_ds.units[0].group_index == 11


def test_multiple_certified_tasks_still_yield_one_scene_state():
    units = (
        StateGroupUnit(1, 7, 2, 12, "pick_remove"),
        StateGroupUnit(1, 7, 1, 11, "pick_remove"),
        StateGroupUnit(1, 8, 4, 20, "pick_remove"),
    )
    source = dataset(units, {(1, 7, 1), (1, 7, 2), (1, 8, 4)})
    global_ds = GlobalStateDataset(source, 0.01, 0.08)
    assert [(u.scene_id, u.state_id) for u in global_ds.units] == [(1, 7), (1, 8)]
    assert global_ds.units[0].task_index == 1


def test_uncertified_task_variant_cannot_create_extra_weight():
    units = (
        StateGroupUnit(2, 5, 0, 1, "push"),
        StateGroupUnit(2, 5, 1, 2, "push"),
        StateGroupUnit(2, 5, 2, 3, "pick_remove"),
    )
    source = dataset(units, {(2, 5, 2)})
    global_ds = GlobalStateDataset(source, 0.01, 0.08)
    assert len(global_ds.units) == 1
    assert (global_ds.units[0].state_id, global_ds.units[0].task_index) == (5, 2)
