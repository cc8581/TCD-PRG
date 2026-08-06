from __future__ import annotations

import h5py
import numpy as np

from tcd_prg.constants import PUSH_DISTANCE_M, ActionType, CandidateStatus
from tcd_prg.datasets import TaskOrientedClutterAdapter
from tcd_prg.datasets.collate import collate_unified, grid_sample_indices
from tcd_prg.datasets.task_oriented_clutter import (
    ACTION_GROUP_INDEX_CACHE_VERSION,
    split_scene_ids,
)


def test_grid_sample_keeps_one_representative_per_voxel() -> None:
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    deterministic = grid_sample_indices(xyz, 0.01, training=False)
    assert deterministic.tolist() == [0, 2]
    state = np.random.get_state()
    np.random.seed(7)
    random_representative = grid_sample_indices(xyz, 0.01, training=True)
    np.random.set_state(state)
    assert len(random_representative) == 2
    assert random_representative[0] in (0, 1)
    assert random_representative[1] == 2


def test_authoritative_training_index_avoids_scene_hdf5_scan(tmp_path) -> None:
    index = tmp_path / "training_index.h5"
    rows = np.asarray(
        [
            [10, 10, 0, 3, 7, 0, 2],
            [10, 10, 1, 4, 9, 1, 5],
            [11, 11, 0, 2, 1, 2, 3],
        ],
        dtype=np.int32,
    )
    with h5py.File(index, "w") as handle:
        handle.attrs["format"] = "task_conditioned_action_training_index_v2"
        handle.attrs["generation_signature"] = "test-signature"
        handle.create_dataset("action_state_group", data=rows)
    adapter = object.__new__(TaskOrientedClutterAdapter)
    adapter.training_index_path = index
    adapter._action_group_index = None
    adapter._scene_ids = ()
    adapter.scene_splits = {"train": (11,), "val": (10,), "test": ()}

    assert list(adapter.iter_action_groups("val")) == [
        (10, 7, 3, 0),
        (10, 9, 4, 1),
    ]


def test_scene_split_is_seeded_scene_level_and_ignores_original_split_codes() -> None:
    first = split_scene_ids(range(10_000), 0.5, (8.0, 1.0, 1.0), seed=2026)
    repeated = split_scene_ids(range(10_000), 0.5, (8.0, 1.0, 1.0), seed=2026)

    assert first == repeated
    assert {name: len(values) for name, values in first.items()} == {
        "train": 4000,
        "val": 500,
        "test": 500,
    }
    split_sets = [set(first[name]) for name in ("train", "val", "test")]
    assert len(set.union(*split_sets)) == 5000
    assert all(
        split_sets[left].isdisjoint(split_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )


def test_two_way_scene_split_leaves_test_empty() -> None:
    splits = split_scene_ids(range(10), 1.0, (9.0, 1.0), seed=7)
    assert len(splits["train"]) == 9
    assert len(splits["val"]) == 1
    assert splits["test"] == ()


def test_action_strata_cache_is_reused_without_hdf5_scan(tmp_path) -> None:
    index = tmp_path / "training_index.h5"
    rows = np.asarray(
        [
            [10, 10, 0, 3, 7, 0, 2],
            [10, 10, 1, 4, 9, 1, 5],
        ],
        dtype=np.int32,
    )
    with h5py.File(index, "w") as handle:
        handle.attrs["format"] = "task_conditioned_action_training_index_v2"
        handle.attrs["generation_signature"] = "test-signature"
        handle.create_dataset("action_state_group", data=rows)
    adapter = object.__new__(TaskOrientedClutterAdapter)
    adapter.training_index_path = index
    adapter.action_root = tmp_path
    adapter.index_cache_dir = tmp_path / "cache"
    adapter.index_cache_dir.mkdir()
    adapter._action_group_index = None
    cache = adapter._strata_cache_path()
    np.savez_compressed(
        cache,
        version=np.asarray(ACTION_GROUP_INDEX_CACHE_VERSION),
        strata=np.asarray([0, 2], dtype=np.int8),
    )

    assert adapter.action_group_strata([(10, 9, 4, 1)]) == {(10, 9, 4, 1): "push"}


def test_first_strata_scan_builds_reusable_cache(tmp_path) -> None:
    rows = np.asarray(
        [
            [10, 10, 0, 3, 7, 0, 1],
            [10, 10, 1, 4, 9, 0, 1],
        ],
        dtype=np.int32,
    )
    scene_file = tmp_path / "scene_0010.h5"
    with h5py.File(scene_file, "w") as handle:
        scene = handle.create_group("scene_0010")
        groups = scene.create_group("action_state_groups")
        groups.create_dataset("action_offsets", data=np.asarray([0, 1, 2]))
        groups.create_dataset("action_ids", data=np.asarray([0, 1]))
        actions = scene.create_group("actions")
        actions.create_dataset(
            "action_type", data=np.asarray([ActionType.TASK_GRASP, ActionType.PUSH])
        )
        actions.create_dataset("executed", data=np.asarray([True, True]))
        actions.create_dataset("success", data=np.asarray([True, False]))
        actions.create_dataset("potential_improved", data=np.asarray([False, False]))
    adapter = object.__new__(TaskOrientedClutterAdapter)
    adapter._path_by_scene = {10: scene_file}
    cache = tmp_path / "strata.npz"

    codes = adapter._build_strata_cache(cache, rows)

    assert codes.tolist() == [0, 3]
    with np.load(cache, allow_pickle=False) as saved:
        assert saved["strata"].tolist() == [0, 3]


def test_real_hdf5_primary_keys_and_shapes(dataset_root) -> None:
    path = dataset_root / "task_positive_multistep_sequences" / "scene_labels" / "scene_0000.h5"
    with h5py.File(path, "r", swmr=True) as handle:
        scene = handle["scene_0000"]
        assert scene["states/object_pose"].shape[-1] == 7
        assert scene["states/relation_graph"].shape[-1] == 5
        assert (
            len(scene["action_state_groups/action_offsets"])
            == len(scene["action_state_groups/from_state"]) + 1
        )
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
    label = dataset_root / "task_training_labels" / "scene_labels" / "scene_0000_labels.npz"
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
    assert labels.graspable == (labels.verified_positive_grasp_count >= labels.required_grasp_count)


def test_policy_positive_mask_comes_only_from_successful_sequences(dataset_root) -> None:
    adapter = TaskOrientedClutterAdapter(dataset_root, point_count=64)
    sample = adapter.load_sample(*next(iter(adapter.iter_action_groups())))
    batch = collate_unified([sample])
    successful_ids = (
        np.unique(
            np.concatenate(
                [
                    np.concatenate((sequence.policy_action_ids, sequence.terminal_action_ids))
                    for sequence in sample.sequences
                ]
            )
        )
        if sample.sequences
        else np.empty(0, dtype=np.int64)
    )
    expected = np.isin(sample.candidates.candidate_action_ids, successful_ids)
    actual = batch["policy_success_mask"][0, : len(expected)].numpy()
    assert np.array_equal(actual, expected)


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
    for head in (
        "stability",
        "task_compatibility",
        "collision",
        "clearance",
        "approach",
        "overall",
    ):
        target = parameters[f"verifier_{head}_target"]
        valid = parameters[f"verifier_{head}_valid"]
        assert np.isfinite(target[valid]).all()
