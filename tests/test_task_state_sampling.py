from __future__ import annotations

from collections import Counter

from tcd_prg.datasets.torch_dataset import (
    DistributedTaskStateBatchSampler,
    StateGroupUnit,
)


def _add_state(
    units: list[StateGroupUnit],
    *,
    scene: int,
    task: int,
    state: int,
    strata: tuple[str, ...],
) -> None:
    for stratum in strata:
        index = len(units)
        units.append(StateGroupUnit(scene, state, task, index, stratum))


def _multi_path_units() -> tuple[StateGroupUnit, ...]:
    units: list[StateGroupUnit] = []
    # Task 0 deliberately owns far more groups/states (many alternative paths).
    for state in range(8):
        _add_state(
            units,
            scene=0,
            task=0,
            state=state,
            strata=("direct_grasp", "push", "pick_remove"),
        )
    # Other tasks have fewer decision states and fewer action groups.
    for task in (1, 2, 3):
        for state in range(2):
            _add_state(
                units,
                scene=0,
                task=task,
                state=100 * task + state,
                strata=("direct_grasp",),
            )
    return tuple(units)


def _task_state(units: tuple[StateGroupUnit, ...], index: int) -> tuple[int, int, int]:
    unit = units[index]
    return unit.scene_id, unit.task_index, unit.state_id


def test_many_path_groups_do_not_weight_task_sampling_probability():
    units = _multi_path_units()
    sampler = DistributedTaskStateBatchSampler(
        units,
        batch_size=2,
        coverage_strata=(),
        rank=0,
        world_size=1,
        seed=17,
    )
    task_counts = Counter()
    for batch in sampler:
        for index in batch:
            task_counts[units[index].task_index] += 1
    # The raw action-group counts are highly imbalanced, but task draws are
    # round-robin balanced (difference at most one over one sampler epoch).
    assert max(task_counts.values()) - min(task_counts.values()) <= 1
    assert task_counts[0] < 8 * 3


def test_coverage_preference_cannot_change_selected_task_state_schedule():
    units: list[StateGroupUnit] = []
    for task in range(6):
        for state in range(3):
            _add_state(
                units,
                scene=0,
                task=task,
                state=100 * task + state,
                strata=("direct_grasp", "push"),
            )
    packed = tuple(units)
    plain = DistributedTaskStateBatchSampler(
        packed, 3, (), rank=0, world_size=1, seed=23
    )
    covered = DistributedTaskStateBatchSampler(
        packed,
        3,
        ("direct_grasp", "push"),
        rank=0,
        world_size=1,
        seed=23,
    )
    plain_states = [
        sorted(_task_state(packed, index) for index in batch)
        for batch in plain
    ]
    covered_states = [
        sorted(_task_state(packed, index) for index in batch)
        for batch in covered
    ]
    assert plain_states == covered_states


def test_best_effort_coverage_uses_alternative_groups_of_selected_states():
    units: list[StateGroupUnit] = []
    for task in range(4):
        _add_state(
            units,
            scene=0,
            task=task,
            state=task,
            strata=("direct_grasp", "push"),
        )
    packed = tuple(units)
    sampler = DistributedTaskStateBatchSampler(
        packed,
        batch_size=2,
        coverage_strata=("direct_grasp", "push"),
        rank=0,
        world_size=1,
        seed=3,
    )
    first = next(iter(sampler))
    assert {packed[index].stratum for index in first} == {"direct_grasp", "push"}
    assert len({_task_state(packed, index) for index in first}) == 2


def test_ddp_ranks_slice_one_duplicate_free_global_state_batch():
    units: list[StateGroupUnit] = []
    for task in range(12):
        for state in range(2):
            _add_state(
                units,
                scene=0,
                task=task,
                state=100 * task + state,
                strata=("direct_grasp", "push"),
            )
    packed = tuple(units)
    rank0 = DistributedTaskStateBatchSampler(
        packed, 2, ("push",), rank=0, world_size=2, seed=41
    )
    rank1 = DistributedTaskStateBatchSampler(
        packed, 2, ("push",), rank=1, world_size=2, seed=41
    )
    batch0 = next(iter(rank0))
    batch1 = next(iter(rank1))
    states0 = {_task_state(packed, index) for index in batch0}
    states1 = {_task_state(packed, index) for index in batch1}
    assert states0.isdisjoint(states1)
    assert len(states0 | states1) == 4


def test_sampler_epoch_is_deterministic_and_changes_with_epoch():
    units = _multi_path_units()
    first = DistributedTaskStateBatchSampler(
        units, 2, ("push",), rank=0, world_size=1, seed=7
    )
    second = DistributedTaskStateBatchSampler(
        units, 2, ("push",), rank=0, world_size=1, seed=7
    )
    first.set_epoch(3)
    second.set_epoch(3)
    epoch3 = list(first)
    assert epoch3 == list(second)
    second.set_epoch(4)
    assert epoch3 != list(second)
