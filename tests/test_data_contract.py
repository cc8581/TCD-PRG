from __future__ import annotations

import h5py
import numpy as np

from tcd_prg.constants import ActionType, CandidateStatus, PUSH_DISTANCE_M
from tcd_prg.datasets import TaskOrientedClutterAdapter


def test_real_hdf5_primary_keys_and_shapes(dataset_root) -> None:
    path = dataset_root / "task_positive_multistep_sequences" / "scene_labels" / "scene_0000.h5"
    with h5py.File(path, "r", swmr=True) as handle:
        scene = handle["scene_0000"]
        assert scene["states/object_pose"].shape[-1] == 7
        assert scene["states/relation_graph"].shape[-1] == 5
        assert len(scene["action_state_groups/action_offsets"]) == len(scene["action_state_groups/from_state"]) + 1
        assert "object_present" not in scene["states"]


def test_adapter_shapes_oracle_exclusion_and_uuid(dataset_root) -> None:
    adapter = TaskOrientedClutterAdapter(dataset_root, point_count=256)
    unit = next(iter(adapter.iter_action_groups()))
    sample = adapter.load_sample(*unit)
    assert sample.observation.xyz.shape == (256, 3)
    assert sample.observation.source_view.max() <= 2
    assert sample.observation.object_uuid[0] == "scene_0000/object_00"
    assert sample.observation.object_present.all()


def test_model_id_grasp_library_and_function_region_link(dataset_root) -> None:
    raw = dataset_root / "task_clutter_scenes_20_categories" / "scene_0000" / "scene.npz"
    label = dataset_root / "task_training_labels_steps1_6_v1" / "scene_labels" / "scene_0000_labels.npz"
    with np.load(raw, allow_pickle=False) as scene, np.load(label, allow_pickle=False) as steps:
        count = int(steps["object_count"])
        assert np.array_equal(scene["object_model_id"][:count], steps["object_model_id"])
        assert all(steps["object_match_file"] != "")


def test_action_types_push_distance_and_unknown_semantics(dataset_root) -> None:
    adapter = TaskOrientedClutterAdapter(dataset_root, point_count=64)
    kinds = set()
    for unit in adapter.iter_action_groups():
        if unit[0] != 0:
            break
        group = adapter.load_action_group(unit[0], unit[3])
        kinds.update(group.action_type.tolist())
        push = group.action_type == ActionType.PUSH
        if push.any():
            assert np.allclose(group.action_parameters["push_distance_m"][push], PUSH_DISTANCE_M)
        unknown = group.evaluation_status == CandidateStatus.UNKNOWN_UNTESTED
        assert not group.success_mask[unknown].any()
    assert kinds == {0, 1, 2}


def test_required_grasp_count_is_state_adaptive(dataset_root) -> None:
    adapter = TaskOrientedClutterAdapter(dataset_root, point_count=64)
    labels = adapter.load_state_labels(0, 0)
    assert labels.graspable == (
        labels.verified_positive_grasp_count >= labels.required_grasp_count
    )


def test_pick_remove_marks_object_inactive_but_present(dataset_root) -> None:
    adapter = TaskOrientedClutterAdapter(dataset_root, point_count=64)
    target_state = None
    with h5py.File(adapter._h5_path(0), "r", swmr=True) as handle:
        transitions = handle["scene_0000/transitions"]
        rows = np.flatnonzero(transitions["action_type"][:] == ActionType.PICK_REMOVE)
        if len(rows):
            target_state = int(transitions["to_state"][rows[0]])
            acted = int(transitions["acted_object"][rows[0]])
    if target_state is None:
        return
    with h5py.File(adapter._h5_path(0), "r", swmr=True) as handle:
        active = adapter._object_active(handle["scene_0000"], target_state)
    assert not active[acted]


def test_relation_names_are_canonical_and_press_is_support_transpose(dataset_root) -> None:
    adapter = TaskOrientedClutterAdapter(dataset_root, point_count=64)
    labels = adapter.load_state_labels(0, 0)
    assert labels.relation_names == ("near", "contact", "support", "press", "occlude")
    support = labels.relation_graph[..., 2]
    press = labels.relation_graph[..., 3]
    assert np.array_equal(press, support.T)


def test_initial_verifier_augmentation_uses_explicit_head_masks(dataset_root) -> None:
    adapter = TaskOrientedClutterAdapter(dataset_root, point_count=64)
    unit = next(item for item in adapter.iter_action_groups() if item[1] == 0)
    group = adapter.load_sample(*unit).candidates
    parameters = group.action_parameters
    for head in ("stability", "task_compatibility", "collision", "clearance", "approach", "overall"):
        target = parameters[f"verifier_{head}_target"]
        valid = parameters[f"verifier_{head}_valid"]
        assert np.isfinite(target[valid]).all()
