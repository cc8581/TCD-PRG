from __future__ import annotations

from tcd_prg.datasets.torch_dataset import ActionStateGroupDataset, GlobalStateDataset


class _Adapter:
    def __init__(self):
        self.groups = [
            (0, 0, 0, 0),
            (0, 0, 1, 1),  # same scene-state: must not duplicate Global supervision
            (0, 1, 0, 2),
            (1, 0, 0, 0),
        ]
        self.loaded = []

    def iter_action_groups(self, split=None):
        del split
        yield from self.groups

    def action_group_strata(self, units):
        return {unit: "direct_grasp" for unit in units}

    def global_grasp_supervised_states(self, split=None):
        del split
        return frozenset({(0, 0), (1, 0)})

    def load_sample(self, scene_id, state_id, task_index, group_index, *, include_global_grasps):
        self.loaded.append((scene_id, state_id, task_index, group_index, include_global_grasps))
        return self.loaded[-1]


def test_global_state_dataset_is_unique_and_supervised_only():
    adapter = _Adapter()
    action = ActionStateGroupDataset(adapter, split="train", global_grasp_mode="never")
    global_states = GlobalStateDataset(action)
    assert [(unit.scene_id, unit.state_id) for unit in global_states.units] == [(0, 0), (1, 0)]
    assert len(global_states) == 2
    _ = global_states[0]
    assert adapter.loaded[-1][-1] is True
