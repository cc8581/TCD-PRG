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
    state_path = tmp_path / "state.h5"
    write_state_values(
        state_path,
        StateValues(
            np.asarray([.1, .2, .3, 1.], np.float32),
            np.asarray([False, False, False, True]),
            np.ones(4, bool),
            "stageb-hash",
            "render-hash",
        ),
    )
    assert load_state_values(state_path, 4).stage_b_checkpoint_sha256 == "stageb-hash"
    action_path = tmp_path / "action.h5"
    build_action_value_sidecar(scene_path, state_path, action_path, gamma=.9)
    with h5py.File(action_path, "r") as result:
        assert result["action_id"][:].tolist() == [0, 2, 3]
        q = result["q_value"][:]
        # Mixed PUSH -> PICK_REMOVE -> PUSH path credits the first PUSH at h=3.
        assert np.isclose(q[0, 2], .9 ** 3)
        assert np.isclose(q[1, 0], .9)
        assert np.all(q[1, 1:] >= q[1, :-1])
        assert np.all(q[2] == 0)
        assert not bool(result["safe"][2])
