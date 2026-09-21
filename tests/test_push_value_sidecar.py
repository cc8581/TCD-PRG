import h5py
import numpy as np

from tcd_prg.constants import ActionType
from tcd_prg.datasets.push_value import (
    PUSH_IMPROVEMENT_SCHEMA_VERSION,
    PushImprovementStore,
    build_push_improvement_sidecar,
)
from tcd_prg.push_improvement import PUSH_IMPROVEMENT_DEFINITION


def _scene(path):
    with h5py.File(path, "w") as handle:
        scene = handle.create_group("scene_0000")
        actions = scene.create_group("actions")
        actions.create_dataset("action_type", data=np.asarray([
            ActionType.PUSH, ActionType.PUSH, ActionType.PUSH, ActionType.TASK_GRASP
        ], np.int8))
        actions.create_dataset("executed", data=np.asarray([True, True, False, True]))
        actions.create_dataset("from_state", data=np.asarray([0, 0, 0, 1], np.int64))
        actions.create_dataset("to_state", data=np.asarray([1, 1, -1, 2], np.int64))
        sequences = scene.create_group("sequences")
        sequences.create_dataset("policy_action_ids", data=np.asarray([0, 0], np.int64))
        sequences.create_dataset("policy_action_offsets", data=np.asarray([0, 1, 2], np.int64))
        sequences.create_dataset("terminal_action_ids", data=np.asarray([3, 3], np.int64))
        sequences.create_dataset("terminal_action_offsets", data=np.asarray([0, 1, 2], np.int64))


def test_sequence_union_labels_every_push_candidate(tmp_path):
    source = tmp_path / "scene_0000.h5"
    output = tmp_path / "value.h5"
    _scene(source)
    build_push_improvement_sidecar(source, output, raw_relation_names=())
    with h5py.File(output, "r") as result:
        assert int(result.attrs["schema_version"]) == PUSH_IMPROVEMENT_SCHEMA_VERSION
        assert result.attrs["improvement_definition"] == PUSH_IMPROVEMENT_DEFINITION
        np.testing.assert_array_equal(result["action_id"][:], [0, 1, 2])
        np.testing.assert_array_equal(result["improvement_target"][:], [1, 0, 0])
        np.testing.assert_array_equal(result["improvement_valid"][:], [True, True, True])
        np.testing.assert_array_equal(result["executed"][:], [True, True, False])
        assert "component_delta" not in result


def test_old_seven_metric_schema_is_rejected(tmp_path):
    root = tmp_path / "labels"
    root.mkdir()
    with h5py.File(root / "scene_0000.h5", "w") as handle:
        handle.attrs["schema_version"] = 8
    try:
        PushImprovementStore(root).load_scene(0)
    except RuntimeError as error:
        assert "rebuild" in str(error)
    else:
        raise AssertionError("schema 8 must not be accepted")
