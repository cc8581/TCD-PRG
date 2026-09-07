import h5py
import numpy as np

from tcd_prg.constants import ActionType, OutcomeCode
from tcd_prg.datasets.push_value import (
    StateValues,
    build_action_value_sidecar,
    load_state_values,
    write_state_values,
)


def test_mixed_preparation_actions_propagate_to_push_value(tmp_path):
    scene_path = tmp_path / "scene_0000.h5"
    with h5py.File(scene_path, "w") as handle:
        scene = handle.create_group("scene_0000")
        states = scene.create_group("states")
        states.create_dataset("task_index", data=np.zeros(4, np.int32))
        states.create_dataset("terminal_goal_valid", data=[False, False, False, True])
        states.create_dataset("direct_goal_valid", data=np.zeros(4, bool))
        actions = scene.create_group("actions")
        # PUSH s0->s1, PICK_REMOVE s1->s2, PUSH s2->terminal s3, unsafe PUSH s0.
        actions.create_dataset("action_type", data=[0, 1, 0, 0])
        actions.create_dataset("executed", data=np.ones(4, bool))
        actions.create_dataset("outcome_code", data=[1, 1, 0, 4])
        actions.create_dataset("from_state", data=[0, 1, 2, 0])
        actions.create_dataset("to_state", data=[1, 2, 3, -1])
        actions.create_dataset("after_state_valid", data=[True, True, True, False])
        actions.create_dataset("task_index", data=np.zeros(4, np.int32))
        actions.create_dataset("potential_delta", data=np.zeros((4, 5), np.float32))
        actions.create_dataset("potential_after_valid", data=np.ones(4, bool))
        actions.create_dataset("part_of_success_sequence", data=[True, True, True, False])
    action_path = tmp_path / "action.h5"
    build_action_value_sidecar(scene_path, action_path, gamma=.9)
    with h5py.File(action_path, "r") as result:
        assert result.attrs["schema_version"] == 3
        assert result.attrs["teacher"] == "none"
        assert result["action_id"][:].tolist() == [0, 2, 3]
        q = result["q_value"][:]
        # Mixed PUSH -> PICK_REMOVE -> PUSH path credits the first PUSH at h=3.
        assert np.isclose(q[0, 2], .9 ** 3)
        assert np.isclose(q[1, 0], .9)
        assert np.all(q[1, 1:] >= q[1, :-1])
        assert np.all(q[2] == 0)
        assert not bool(result["safe"][2])


def test_unverified_improved_alternative_remains_unknown(tmp_path):
    scene_path = tmp_path / "scene_0000.h5"
    with h5py.File(scene_path, "w") as handle:
        scene = handle.create_group("scene_0000")
        states = scene.create_group("states")
        states.create_dataset("task_index", data=np.zeros(2, np.int32))
        states.create_dataset("terminal_goal_valid", data=[False, True])
        states.create_dataset("direct_goal_valid", data=[False, True])
        actions = scene.create_group("actions")
        actions.create_dataset("action_type", data=[0])
        actions.create_dataset("executed", data=[True])
        actions.create_dataset("outcome_code", data=[int(OutcomeCode.IMPROVED)])
        actions.create_dataset("from_state", data=[0])
        actions.create_dataset("to_state", data=[1])
        actions.create_dataset("after_state_valid", data=[True])
        actions.create_dataset("task_index", data=[0])
        actions.create_dataset("potential_delta", data=np.zeros((1, 5), np.float32))
        actions.create_dataset("potential_after_valid", data=[True])
        actions.create_dataset("part_of_success_sequence", data=[False])
    action_path = tmp_path / "action.h5"
    build_action_value_sidecar(scene_path, action_path, gamma=.95)
    with h5py.File(action_path, "r") as result:
        assert result.attrs["schema_version"] == 3
        assert not result["q_valid"][0].any()
        assert np.isnan(result["q_value"][0]).all()
