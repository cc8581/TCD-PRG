import h5py
import numpy as np

from tcd_prg.constants import ActionType, OutcomeCode
from tcd_prg.datasets.push_value import build_push_improvement_sidecar


RELATIONS = ("near", "contact", "support", "occlude", "block_path")


def _scene(path, *, unsafe=False):
    with h5py.File(path, "w") as handle:
        scene = handle.create_group("scene_0000")
        catalog = scene.create_group("catalog")
        catalog.create_dataset("task_object_index", data=np.asarray([0], np.int64))
        states = scene.create_group("states")
        states.create_dataset("task_index", data=np.zeros(2, np.int32))
        pose = np.zeros((2, 3, 7), np.float32)
        pose[..., 6] = 1.0
        pose[:, 1, 2] = 0.05
        pose[:, 2, 2] = 0.10
        states.create_dataset("object_pose", data=pose)
        relation = np.zeros((2, 3, 3, len(RELATIONS)), np.float32)
        # state 0: object 1 directly blocks target; object 2 is an upper prerequisite.
        relation[0, 1, 2, RELATIONS.index("support")] = 1.0
        states.create_dataset("relation_graph", data=relation)
        states.create_dataset("occlusion_blockers", data=np.empty(0, np.int64))
        states.create_dataset("occlusion_blocker_offsets", data=np.asarray([0, 0, 0], np.int64))
        states.create_dataset("task_occlusion_blockers", data=np.asarray([1, 1], np.int64))
        states.create_dataset("task_occlusion_blocker_offsets", data=np.asarray([0, 1, 2], np.int64))
        for name in ("direct_goal_valid", "terminal_goal_valid", "task_pressed", "task_region_pressed"):
            states.create_dataset(name, data=np.zeros(2, bool))
        states.create_dataset("target_visible_ratio", data=np.asarray([0.2, 0.2], np.float32))
        states.create_dataset("verified_positive_grasp_count", data=np.zeros(2, np.int32))
        states.create_dataset("required_grasp_count", data=np.full(2, 5, np.int32))

        actions = scene.create_group("actions")
        actions.create_dataset("action_type", data=np.asarray([int(ActionType.PUSH)], np.int8))
        actions.create_dataset("executed", data=np.asarray([True]))
        actions.create_dataset(
            "outcome_code",
            data=np.asarray([
                int(OutcomeCode.UNSTABLE if unsafe else OutcomeCode.IMPROVED)
            ], np.int8),
        )
        actions.create_dataset("from_state", data=np.asarray([0], np.int64))
        actions.create_dataset("to_state", data=np.asarray([1], np.int64))
        actions.create_dataset("after_state_valid", data=np.asarray([True]))
        actions.create_dataset("task_index", data=np.asarray([0], np.int64))
        actions.create_dataset("potential_improved", data=np.asarray([True]))


def test_removing_indirect_top_blocker_is_positive_even_when_grasp_count_stays_zero(tmp_path):
    source = tmp_path / "scene_0000.h5"
    output = tmp_path / "value.h5"
    _scene(source)
    build_push_improvement_sidecar(source, output, raw_relation_names=RELATIONS)
    with h5py.File(output, "r") as result:
        assert int(result.attrs["schema_version"]) == 8
        np.testing.assert_array_equal(
            result.attrs["component_eps"],
            np.asarray([0, 0, 0, 0, 0, 0.01, 1e-6]),
        )
        assert result.attrs["teacher"] == "none"
        assert bool(result["improvement_valid"][0])
        assert float(result["improvement_target"][0]) == 1.0
        delta = result["component_delta"][0]
        assert delta[-1] == 0.0
        assert delta[1] == 1.0


def test_unsafe_transition_is_excluded_not_learned_as_negative(tmp_path):
    source = tmp_path / "scene_0000.h5"
    output = tmp_path / "value.h5"
    _scene(source, unsafe=True)
    build_push_improvement_sidecar(source, output, raw_relation_names=RELATIONS)
    with h5py.File(output, "r") as result:
        assert not bool(result["improvement_valid"][0])
        assert float(result["improvement_target"][0]) == 0.0


def test_subpercent_visibility_change_is_neutral(tmp_path):
    source = tmp_path / "scene_0000.h5"
    output = tmp_path / "value.h5"
    _scene(source)
    with h5py.File(source, "r+") as handle:
        handle["scene_0000/states/target_visible_ratio"][:] = [0.2, 0.205]
        # Keep blocker structure identical so visibility is the first possible
        # differing component.
        handle["scene_0000/states/relation_graph"][0] = (
            handle["scene_0000/states/relation_graph"][1]
        )
    build_push_improvement_sidecar(source, output, raw_relation_names=RELATIONS)
    with h5py.File(output, "r") as result:
        assert float(result["improvement_target"][0]) == 0.0
