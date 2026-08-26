"""Strata 分类与标签构建口径对齐、以及认证感知代表选择的回归测试。"""

from __future__ import annotations

import h5py
import numpy as np

from tcd_prg.constants import ActionType, OutcomeCode
from tcd_prg.datasets.task_oriented_clutter import TaskOrientedClutterAdapter
from tcd_prg.datasets.torch_dataset import ActionStateGroupDataset


class _Library:
    def __init__(self, width: float) -> None:
        self.contact_span_m = np.asarray([width], np.float32)

    @staticmethod
    def rows_for_source(sources) -> np.ndarray:
        return np.zeros(len(sources), np.int64)


class _Registry:
    def __init__(self, width: float) -> None:
        self._library = _Library(width)

    def load(self, match_file):
        del match_file
        return self._library


def _bare_adapter(width_bounds=None, library_width: float = 0.05):
    adapter = object.__new__(TaskOrientedClutterAdapter)
    adapter.grasp_width_bounds = width_bounds
    adapter.grasp_registry = _Registry(library_width)
    return adapter


def _width_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    return (
        np.array([0, 1, 2], np.int64),  # payload_index per action row
        np.array([0, 0, 0], np.int64),  # task_grasp target_object per payload
        np.array([5, 6, 7], np.int64),  # task_grasp grasp_source_index per payload
        ("object_0.npz",),
    )


def test_positive_mask_counts_terminal_positive() -> None:
    positive = TaskOrientedClutterAdapter._action_positive_mask(
        np.array([False, True, False]),
        np.array([False, False, True]),
        np.array(
            [
                int(OutcomeCode.TERMINAL_POSITIVE),
                int(OutcomeCode.SUCCESS),
                int(OutcomeCode.IMPROVED),
            ],
            np.int8,
        ),
    )
    assert positive.tolist() == [True, True, True]


def test_group_stratum_prefers_direct_grasp_and_falls_through() -> None:
    adapter = _bare_adapter()
    action_type = np.array(
        [int(ActionType.TASK_GRASP), int(ActionType.PICK_REMOVE), int(ActionType.PUSH)]
    )
    executed = np.array([True, True, True])
    positive = np.array([True, True, False])
    ids = np.array([0, 1, 2])

    assert adapter._group_stratum(ids, action_type, executed, positive, None) == "direct_grasp"
    assert adapter._group_stratum(ids[1:], action_type, executed, positive, None) == "pick_remove"
    assert adapter._group_stratum(ids[2:], action_type, executed, positive, None) == "push_failure"

    push_positive = np.array([False, False, True])
    assert adapter._group_stratum(ids[2:], action_type, executed, push_positive, None) == "push"

    not_executed = np.zeros(3, bool)
    assert (
        adapter._group_stratum(ids[2:], action_type, not_executed, positive, None)
        == "unresolved_or_unknown"
    )


def test_direct_grasp_stratum_respects_width_window() -> None:
    action_type = np.array([int(ActionType.TASK_GRASP), int(ActionType.PICK_REMOVE)])
    executed = np.array([True, True])
    positive = np.array([True, True])
    ids = np.array([0, 1])

    in_bounds = _bare_adapter(width_bounds=(0.01, 0.08), library_width=0.05)
    assert (
        in_bounds._group_stratum(ids, action_type, executed, positive, _width_payload())
        == "direct_grasp"
    )

    # 唯一正 TASK_GRASP 超出模型宽度窗口，动作分层应相应降级。
    out_of_bounds = _bare_adapter(width_bounds=(0.01, 0.08), library_width=0.20)
    assert (
        out_of_bounds._group_stratum(ids, action_type, executed, positive, _width_payload())
        == "pick_remove"
    )


def test_strata_build_skips_scenes_outside_configured_snapshot(tmp_path) -> None:
    """training_index 覆盖完整生成结果时，构建只扫描配置快照内的场景。"""

    rows = np.asarray(
        [
            [10, 10, 0, 3, 7, 0, 1],
            [10, 10, 1, 4, 9, 0, 1],
            # 快照外场景：没有对应 HDF5，也绝不能被扫描（否则构建半途崩溃）。
            [9999, 9999, 0, 0, 1, 0, 1],
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
        actions.create_dataset(
            "outcome_code",
            data=np.asarray([OutcomeCode.SUCCESS, OutcomeCode.OTHER_INVALID], np.int8),
        )
    adapter = _bare_adapter()
    adapter._path_by_scene = {10: scene_file}

    codes = adapter._build_strata_cache(tmp_path / "strata.npz", rows)

    # direct_grasp=0、push_failure=3；快照外场景保持 unresolved_or_unknown=4。
    assert codes.tolist() == [0, 3, 4]


class _CertifiedAdapter:
    def __init__(self, supervised) -> None:
        self.supervised = frozenset(supervised)

    @staticmethod
    def iter_action_groups(split=None):
        del split
        return iter(((1, 7, 0, 10), (1, 7, 3, 11), (1, 8, 0, 12)))

    @staticmethod
    def action_group_strata(units):
        return {}

    def global_grasp_supervised_task_states(self, split, min_width, max_width):
        assert split == "val"
        assert (min_width, max_width) == (0.01, 0.08)
        return self.supervised


def test_representative_prefers_certified_task_when_bounds_provided() -> None:
    adapter = _CertifiedAdapter({(1, 7, 3)})
    dataset = ActionStateGroupDataset(  # type: ignore[arg-type]
        adapter, split="val", global_grasp_width_bounds=(0.01, 0.08)
    )
    # (1,7) 选择认证 task=3 的 unit（index 1）而非首个 unit（index 0）；
    # (1,8) 无认证三元组，退回首个 unit（index 2）。
    assert dataset._global_grasp_representatives == frozenset({1, 2})


def test_representative_legacy_first_unit_without_bounds() -> None:
    adapter = _CertifiedAdapter({(1, 7, 3)})
    dataset = ActionStateGroupDataset(adapter, split="val")  # type: ignore[arg-type]
    assert dataset._global_grasp_representatives == frozenset({0, 2})
