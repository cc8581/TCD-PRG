"""PyTorch datasets and deterministic state-group sampling.

The primary action-stream sampling unit is always
``(scene_id, state_id, task_index, action_state_group)``.  Global Grasp direct
supervision is exposed through a separate unique ``(scene_id, state_id)``
stream so task multiplicity cannot change its statistical weight.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from tcd_prg.constants import ActionType, CandidateStatus

from .base import DatasetAdapter
from .types import GlobalGraspSample, PushObjectStateGroup, UnifiedSample


@dataclass(frozen=True, slots=True)
class StateGroupUnit:
    scene_id: int
    state_id: int
    task_index: int
    group_index: int
    stratum: str = "unclassified"


def _deterministic_fraction_indices(size: int, fraction: float, seed: int) -> np.ndarray:
    """Select an exact deterministic subset without replacement."""

    if size < 0:
        raise ValueError("size must be non-negative")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    if size == 0 or fraction == 1.0:
        return np.arange(size, dtype=np.int64)
    count = max(1, min(size, int(round(size * fraction))))
    selected = np.random.default_rng(seed).choice(size, size=count, replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def _scene_diverse_stratified_units(
    raw_units: list[tuple[int, int, int, int]],
    strata: Mapping[tuple[int, int, int, int], str],
    quota: Mapping[str, int],
    seed: int,
) -> list[tuple[int, int, int, int]]:
    """Select a deterministic no-replacement subset, round-robin over scenes."""

    rng = np.random.default_rng(seed)
    selected: list[tuple[int, int, int, int]] = []
    for stratum, requested in quota.items():
        requested = int(requested)
        if requested <= 0:
            continue
        by_scene: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for unit in raw_units:
            if strata.get(unit, "unclassified") == stratum:
                by_scene[int(unit[0])].append(unit)
        available = sum(len(values) for values in by_scene.values())
        if available < requested:
            raise ValueError(
                f"Validation stratum {stratum!r} requests {requested} groups but only "
                f"{available} are available"
            )
        scenes = list(by_scene)
        rng.shuffle(scenes)
        for scene_id in scenes:
            order = rng.permutation(len(by_scene[scene_id]))
            by_scene[scene_id] = [by_scene[scene_id][int(i)] for i in order]
        offsets = {scene_id: 0 for scene_id in scenes}
        taken = 0
        while taken < requested:
            progressed = False
            for scene_id in scenes:
                offset = offsets[scene_id]
                values = by_scene[scene_id]
                if offset >= len(values):
                    continue
                selected.append(values[offset])
                offsets[scene_id] = offset + 1
                taken += 1
                progressed = True
                if taken >= requested:
                    break
            if not progressed:
                raise RuntimeError(f"Unable to fill validation quota for {stratum}")
    return selected


def _load_or_create_validation_subset(
    raw_units: list[tuple[int, int, int, int]],
    strata: Mapping[tuple[int, int, int, int], str],
    quota: Mapping[str, int],
    seed: int,
    manifest_path: str | Path | None,
) -> list[tuple[int, int, int, int]]:
    """Use an existing manifest on resume, otherwise create one deterministically."""

    path = Path(manifest_path) if manifest_path is not None else None
    expected_count = sum(int(value) for value in quota.values())
    raw_map = {unit: strata.get(unit, "unclassified") for unit in raw_units}
    if path is not None and path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("seed", -1)) != int(seed):
            raise ValueError("validation_subset.json seed does not match current config")
        if payload.get("quota") != {key: int(value) for key, value in quota.items()}:
            raise ValueError("validation_subset.json quota does not match current config")
        if int(payload.get("source_group_count", -1)) != len(raw_units):
            raise ValueError(
                "validation_subset.json source group count does not match current split"
            )
        entries = payload.get("groups", [])
        if len(entries) != expected_count:
            raise ValueError("validation_subset.json group count does not match configured quota")
        selected: list[tuple[int, int, int, int]] = []
        for entry in entries:
            unit = (
                int(entry["scene_id"]),
                int(entry["state_id"]),
                int(entry["task_index"]),
                int(entry["group_index"]),
            )
            if unit not in raw_map:
                raise ValueError(f"validation_subset.json references unavailable group {unit}")
            if str(entry["stratum"]) != raw_map[unit]:
                raise ValueError(f"validation_subset.json stratum drift for group {unit}")
            selected.append(unit)
        return selected

    selected = _scene_diverse_stratified_units(raw_units, strata, quota, seed)
    if path is not None:
        is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if is_primary:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "seed": int(seed),
                "quota": {key: int(value) for key, value in quota.items()},
                "source_group_count": len(raw_units),
                "groups": [
                    {
                        "scene_id": unit[0],
                        "state_id": unit[1],
                        "task_index": unit[2],
                        "group_index": unit[3],
                        "stratum": strata.get(unit, "unclassified"),
                    }
                    for unit in selected
                ],
            }
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, path)
    return selected


class ActionStateGroupDataset(Dataset[UnifiedSample]):
    """Immutable snapshot of completed action-state groups."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        split: str | None = None,
        max_groups: int | None = None,
        include_strata: bool = True,
        allowed_strata: tuple[str, ...] = (),
        deduplicate_state_task: bool = False,
        fraction: float = 1.0,
        subset_seed: int = 2026,
        scene_ids: set[int] | frozenset[int] | None = None,
        global_grasp_mode: str = "representative",
        stratified_max_groups: bool = False,
        stratum_quota: Mapping[str, int] | None = None,
        subset_manifest_path: str | Path | None = None,
        global_grasp_width_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.adapter = adapter
        self.split = split
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0,1]")
        if max_groups is not None and max_groups <= 0:
            raise ValueError("max_groups must be positive")
        if global_grasp_mode not in {"representative", "always", "never"}:
            raise ValueError("global_grasp_mode must be representative, always or never")
        if stratified_max_groups and max_groups is None:
            raise ValueError("stratified_max_groups requires max_groups")
        if stratified_max_groups and not stratum_quota:
            raise ValueError("stratified_max_groups requires stratum_quota")

        iterator = adapter.iter_action_groups(split)
        if scene_ids is not None:
            allowed_scenes = frozenset(int(scene_id) for scene_id in scene_ids)
            if not allowed_scenes:
                raise ValueError("scene_ids must not be empty")
            iterator = (unit for unit in iterator if int(unit[0]) in allowed_scenes)
        stage_filtering = bool(allowed_strata) or deduplicate_state_task
        strata: dict[tuple[int, int, int, int], str] = {}
        if (
            fraction == 1.0
            and max_groups is not None
            and not stratified_max_groups
            and not stage_filtering
        ):
            raw_units = list(islice(iterator, max_groups))
            self.source_group_count: int | None = None
        else:
            raw_units = list(iterator)
            self.source_group_count = len(raw_units)
            if fraction < 1.0:
                selected = _deterministic_fraction_indices(len(raw_units), fraction, subset_seed)
                raw_units = [raw_units[int(index)] for index in selected]
            if stage_filtering:
                strata_method = getattr(adapter, "action_group_strata", None)
                if not include_strata or strata_method is None:
                    raise ValueError("stage filtering requires action-group strata")
                strata = strata_method(raw_units)
                if allowed_strata:
                    allowed = frozenset(str(value) for value in allowed_strata)
                    raw_units = [
                        unit for unit in raw_units
                        if strata.get(unit, "unclassified") in allowed
                    ]
                    if not raw_units:
                        raise RuntimeError(
                            f"Stage filter {sorted(allowed)} removed every action-state group"
                        )
                if deduplicate_state_task:
                    unique: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
                    for unit in raw_units:
                        unique.setdefault((unit[0], unit[1], unit[2]), unit)
                    raw_units = list(unique.values())
            if max_groups is not None and not stratified_max_groups:
                raw_units = raw_units[:max_groups]

        self.requested_fraction = float(fraction)
        strata_method = getattr(adapter, "action_group_strata", None)
        if not strata:
            strata = strata_method(raw_units) if include_strata and strata_method else {}
        else:
            strata = {unit: strata.get(unit, "unclassified") for unit in raw_units}
        if stratified_max_groups:
            assert stratum_quota is not None
            if sum(int(value) for value in stratum_quota.values()) != int(max_groups):
                raise ValueError("validation stratum quota must sum to max_groups")
            raw_units = _load_or_create_validation_subset(
                raw_units, strata, stratum_quota, subset_seed, subset_manifest_path
            )
            strata = {unit: strata.get(unit, "unclassified") for unit in raw_units}
        self.selected_group_count = len(raw_units)
        self.units = tuple(
            StateGroupUnit(*unit, stratum=strata.get(unit, "unclassified")) for unit in raw_units
        )
        # Push-object labels describe the whole (scene,state,task), not whichever
        # action group happened to survive a train/validation subset. Re-read the
        # immutable published index so UNKNOWN/excluded groups cannot silently turn
        # known objects into unsupervised objects.
        push_iterator = adapter.iter_action_groups(split)
        if scene_ids is not None:
            allowed_push_scenes = frozenset(int(scene_id) for scene_id in scene_ids)
            push_iterator = (
                item for item in push_iterator if int(item[0]) in allowed_push_scenes
            )
        push_state_units: dict[tuple[int, int, int], list[StateGroupUnit]] = defaultdict(list)
        for raw_unit in push_iterator:
            unit = StateGroupUnit(*raw_unit)
            push_state_units[(unit.scene_id, unit.state_id, unit.task_index)].append(unit)
        self._push_state_units = {
            key: tuple(values) for key, values in push_state_units.items()
        }
        self._push_state_cache: dict[tuple[int, int, int], PushObjectStateGroup] = {}
        self.global_grasp_mode = global_grasp_mode
        representatives: dict[tuple[int, int], int] = {}
        certified: frozenset[tuple[int, int, int]] | None = None
        if global_grasp_mode == "representative" and global_grasp_width_bounds is not None:
            supervised_method = getattr(adapter, "global_grasp_supervised_task_states", None)
            if supervised_method is not None:
                certified = supervised_method(
                    split,
                    float(global_grasp_width_bounds[0]),
                    float(global_grasp_width_bounds[1]),
                )
        if certified:
            # physical_active 由 task 相关的转移图重建，任意 task 的代表可能把
            # 认证监督行全部过滤掉。优先选择认证三元组对应的 unit 作为代表；
            # 没有认证 unit 的 state 退回首个 unit（与旧行为一致，标签为空时
            # 下游 sample_valid 会安全跳过）。
            preferred: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
            for index, unit in enumerate(self.units):
                triple = (unit.scene_id, unit.state_id, unit.task_index)
                if triple not in certified:
                    continue
                key = (unit.scene_id, unit.state_id)
                rank = (unit.task_index, unit.group_index)
                if key not in preferred or rank < preferred[key][0]:
                    preferred[key] = (rank, index)
            for key, (_, index) in preferred.items():
                representatives[key] = index
        for index, unit in enumerate(self.units):
            representatives.setdefault((unit.scene_id, unit.state_id), index)
        self._global_grasp_representatives = frozenset(representatives.values())

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> UnifiedSample:
        unit = self.units[index]
        include_global = self.global_grasp_mode == "always" or (
            self.global_grasp_mode == "representative"
            and index in self._global_grasp_representatives
        )
        sample = self.adapter.load_sample(
            unit.scene_id,
            unit.state_id,
            unit.task_index,
            unit.group_index,
            include_global_grasps=include_global,
        )
        key = (unit.scene_id, unit.state_id, unit.task_index)
        push_state = self._push_state_cache.get(key)
        if push_state is None:
            evaluated_objects: set[int] = set()
            positive_objects: set[int] = set()
            for state_unit in self._push_state_units[key]:
                if state_unit.group_index == unit.group_index:
                    group = sample.candidates
                else:
                    group = self.adapter.load_action_group(
                        state_unit.scene_id, state_unit.group_index
                    )
                push = group.valid_mask & (
                    group.action_type == int(ActionType.PUSH)
                )
                evaluated = push & (
                    group.evaluation_status != int(CandidateStatus.UNKNOWN_UNTESTED)
                )
                evaluated_objects.update(
                    int(value) for value in group.acted_object[evaluated] if int(value) >= 0
                )
                positive_objects.update(
                    int(value)
                    for value in group.acted_object[evaluated & group.success_mask]
                    if int(value) >= 0
                )
            push_state = PushObjectStateGroup(
                scene_id=unit.scene_id,
                state_id=unit.state_id,
                task_index=unit.task_index,
                evaluated_object=np.asarray(sorted(evaluated_objects), np.int64),
                positive_object=np.asarray(sorted(positive_objects), np.int64),
            )
            self._push_state_cache[key] = push_state
        return replace(sample, push_object_state=push_state)

    def balanced_sampler(
        self, seed: int = 2026, samples: int | None = None
    ) -> DistributedWeightedStateSampler:
        """Legacy inverse-frequency weighted ordering kept for compatibility.

        Formal training no longer uses this method; use
        :class:`DistributedTaskStateBatchSampler` for formal task/state-first sampling.
        """

        counts = Counter(unit.stratum for unit in self.units)
        if not counts:
            raise ValueError("Cannot sample an empty dataset")
        weights = torch.tensor(
            [1.0 / counts[unit.stratum] for unit in self.units], dtype=torch.double
        )
        return DistributedWeightedStateSampler(
            weights,
            rank=0,
            world_size=1,
            total_samples=samples or len(self.units),
            seed=seed,
        )

    def distributed_balanced_sampler(
        self, rank: int, world_size: int, seed: int = 2026, samples: int | None = None
    ) -> DistributedWeightedStateSampler:
        counts = Counter(unit.stratum for unit in self.units)
        weights = torch.tensor(
            [1.0 / counts[unit.stratum] for unit in self.units], dtype=torch.double
        )
        return DistributedWeightedStateSampler(
            weights, rank, world_size, samples or len(self.units), seed
        )

    @property
    def stratum_counts(self) -> dict[str, int]:
        return dict(Counter(unit.stratum for unit in self.units))


class GlobalStateDataset(Dataset[UnifiedSample]):
    """One task-valid representative per unique Global ``(scene,state)``.

    Global Grasp supervision is task-free, but the current observation contract
    reconstructs ``physical_active`` through a task-specific transition graph.
    Therefore an arbitrary task representative is unsafe.  The adapter first
    identifies task/state pairs that can actually produce certified labels; this
    dataset then deduplicates those candidates back to one physical scene-state.
    """

    def __init__(
        self,
        action_dataset: ActionStateGroupDataset,
        min_grasp_width_m: float,
        max_grasp_width_m: float,
    ) -> None:
        self.adapter = action_dataset.adapter
        representative: dict[tuple[int, int, int], StateGroupUnit] = {}
        for unit in action_dataset.units:
            representative.setdefault(
                (unit.scene_id, unit.state_id, unit.task_index), unit
            )

        supervised_method = getattr(
            self.adapter, "global_grasp_supervised_task_states", None
        )
        if supervised_method is not None:
            supervised = supervised_method(
                action_dataset.split,
                float(min_grasp_width_m),
                float(max_grasp_width_m),
            )
            candidates: dict[tuple[int, int], list[StateGroupUnit]] = defaultdict(list)
            for key, unit in representative.items():
                if key in supervised:
                    candidates[(unit.scene_id, unit.state_id)].append(unit)
            units = [
                min(
                    candidates[key],
                    key=lambda unit: (unit.task_index, unit.group_index),
                )
                for key in sorted(candidates)
            ]
        else:
            # Compatibility path for external adapters that have not adopted the
            # task-aware index.  TCD's formal adapter always takes the branch above.
            old_method = getattr(self.adapter, "global_grasp_supervised_states", None)
            if old_method is None:
                raise AttributeError(
                    "GlobalStateDataset requires a Global Grasp supervision index method"
                )
            supervised = old_method(action_dataset.split)
            scene_state_rep: dict[tuple[int, int], StateGroupUnit] = {}
            for unit in action_dataset.units:
                scene_state_rep.setdefault((unit.scene_id, unit.state_id), unit)
            units = [
                unit for key, unit in scene_state_rep.items() if key in supervised
            ]

        self.units = tuple(units)
        if not self.units:
            raise ValueError("No scene-state has certified Global Grasp supervision")
        keys = [(unit.scene_id, unit.state_id) for unit in self.units]
        if len(keys) != len(set(keys)):
            raise RuntimeError("GlobalStateDataset must contain unique scene-state keys")

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> UnifiedSample | GlobalGraspSample:
        unit = self.units[index]
        lightweight = getattr(self.adapter, "load_global_sample", None)
        if lightweight is not None:
            return lightweight(unit.scene_id, unit.state_id, unit.task_index)
        # Compatibility path for external test/adapters that do not inherit the
        # DatasetAdapter lightweight Global-stream method.
        return self.adapter.load_sample(
            unit.scene_id,
            unit.state_id,
            unit.task_index,
            unit.group_index,
            include_global_grasps=True,
        )


class DistributedTaskStateBatchSampler(Sampler[list[int]]):
    """Task-first, unique-state sampling with best-effort supervision coverage.

    The sampling hierarchy is:

    1. choose ``(scene_id, task_index)`` approximately uniformly;
    2. cycle through unique ``state_id`` values inside that task;
    3. choose exactly one complete ``action_state_group`` for the selected state.

    A task with many successful paths/action groups therefore does not receive a
    larger sampling probability merely because it generated more groups.  If a
    selected state owns several groups, ``coverage_strata`` may prefer a group
    that fills a currently missing local-batch supervision type.  Crucially, that
    preference is *not allowed* to replace the already selected task/state.

    The yielded integer is still an ``ActionStateGroupDataset`` index, so no
    action row or trajectory fragment is ever exposed as an independent sample.
    Policy supervision continues to see the complete candidate set stored in the
    chosen action-state group.
    """

    def __init__(
        self,
        units: tuple[StateGroupUnit, ...] | list[StateGroupUnit],
        batch_size: int,
        coverage_strata: tuple[str, ...] | list[str],
        rank: int,
        world_size: int,
        seed: int = 2026,
    ) -> None:
        if not units:
            raise ValueError("Cannot sample an empty dataset")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if len(set(coverage_strata)) != len(tuple(coverage_strata)):
            raise ValueError("coverage_strata must not contain duplicates")

        self.units = tuple(units)
        self.local_batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.global_batch_size = self.local_batch_size * self.world_size
        self.coverage_strata = tuple(str(value) for value in coverage_strata)
        self.seed = int(seed)
        self.epoch = 0

        groups_by_state: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, unit in enumerate(self.units):
            key = (int(unit.scene_id), int(unit.task_index), int(unit.state_id))
            groups_by_state[key].append(index)
        self.groups_by_state = {key: tuple(indices) for key, indices in groups_by_state.items()}

        states_by_task: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for state_key in self.groups_by_state:
            task_key = (state_key[0], state_key[1])
            states_by_task[task_key].append(state_key)
        self.states_by_task = {key: tuple(sorted(values)) for key, values in states_by_task.items()}
        self.task_keys = tuple(sorted(self.states_by_task))
        self.task_count = len(self.task_keys)
        self.unique_state_count = len(self.groups_by_state)
        if self.unique_state_count < self.global_batch_size:
            raise ValueError(
                "Task/state sampler needs at least one unique decision state per "
                f"global mini-batch: states={self.unique_state_count}, "
                f"global_batch={self.global_batch_size}"
            )

        # Keep one sampler epoch close to one pass over unique decision states,
        # not one pass over action groups. Extra groups from alternative paths do
        # not make an epoch longer or increase that task's statistical weight.
        self.batches_per_epoch = max(1, math.ceil(self.unique_state_count / self.global_batch_size))
        self.global_samples_per_epoch = self.batches_per_epoch * self.global_batch_size

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _next_from_cycle(
        source: tuple[object, ...],
        generator: torch.Generator,
        state: dict[str, object],
    ) -> object:
        order = state.get("order")
        position = int(state.get("position", 0))
        if order is None or position >= len(order):
            permutation = torch.randperm(len(source), generator=generator).tolist()
            order = [source[int(index)] for index in permutation]
            position = 0
            state["order"] = order
        assert isinstance(order, list)
        value = order[position]
        state["position"] = position + 1
        return value

    def _draw_state(
        self,
        task_key: tuple[int, int],
        generator: torch.Generator,
        state_cycles: dict[tuple[int, int], dict[str, object]],
        forbidden: set[tuple[int, int, int]],
    ) -> tuple[int, int, int] | None:
        states = self.states_by_task[task_key]
        cycle = state_cycles[task_key]
        # A task can be revisited inside one global batch only when there are
        # fewer tasks than slots. Never duplicate a decision state when another
        # state of this task is available.
        for _ in range(max(1, len(states) * 2)):
            candidate = self._next_from_cycle(states, generator, cycle)
            assert isinstance(candidate, tuple)
            if candidate not in forbidden:
                return candidate
        return None

    def _select_group_for_state(
        self,
        state_key: tuple[int, int, int],
        missing_coverage: set[str],
        generator: torch.Generator,
        usage: dict[int, int],
    ) -> int:
        candidates = self.groups_by_state[state_key]
        preferred = tuple(
            index for index in candidates if self.units[index].stratum in missing_coverage
        )
        pool = preferred or candidates
        minimum_usage = min(usage[index] for index in pool)
        least_used = [index for index in pool if usage[index] == minimum_usage]
        chosen = least_used[int(torch.randint(len(least_used), (1,), generator=generator).item())]
        usage[chosen] += 1
        missing_coverage.discard(self.units[chosen].stratum)
        return chosen

    def __iter__(self):
        # Schedule RNG must stay identical across ranks so every rank sees the
        # same global task/state schedule before deterministic rank slicing.
        schedule_generator = torch.Generator().manual_seed(self.seed + self.epoch)
        group_generator = torch.Generator().manual_seed(
            self.seed + self.epoch + 1_000_003 * (self.rank + 1)
        )
        task_cycle: dict[str, object] = {}
        state_cycles = {task: {} for task in self.task_keys}
        group_usage: dict[int, int] = defaultdict(int)

        for _ in range(self.batches_per_epoch):
            selected_states: list[tuple[int, int, int]] = []
            selected_set: set[tuple[int, int, int]] = set()
            attempts = 0
            maximum_attempts = max(
                self.global_batch_size * max(4, self.task_count * 2),
                self.unique_state_count * 2,
            )
            while len(selected_states) < self.global_batch_size:
                attempts += 1
                if attempts > maximum_attempts:
                    raise RuntimeError(
                        "Unable to construct a duplicate-free global task/state batch"
                    )
                task = self._next_from_cycle(self.task_keys, schedule_generator, task_cycle)
                assert isinstance(task, tuple)
                state_key = self._draw_state(task, schedule_generator, state_cycles, selected_set)
                if state_key is None:
                    continue
                selected_states.append(state_key)
                selected_set.add(state_key)

            # Randomize rank assignment without altering the selected state set.
            permutation = torch.randperm(
                len(selected_states), generator=schedule_generator
            ).tolist()
            selected_states = [selected_states[int(index)] for index in permutation]
            start = self.rank * self.local_batch_size
            local_states = selected_states[start : start + self.local_batch_size]

            # Coverage is local and best-effort. It may only choose among groups
            # belonging to an already selected state; it cannot swap tasks/states.
            missing = set(self.coverage_strata)
            local_groups = [
                self._select_group_for_state(state_key, missing, group_generator, group_usage)
                for state_key in local_states
            ]
            order = torch.randperm(len(local_groups), generator=group_generator).tolist()
            yield [local_groups[int(index)] for index in order]


class DistributedWeightedStateSampler(Sampler[int]):
    """Legacy deterministic inverse-frequency weighted ordering.

    Kept to avoid breaking external scripts. Formal training uses
    ``DistributedTaskStateBatchSampler`` so action-group multiplicity from
    alternative paths cannot silently weight a task more heavily.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        rank: int,
        world_size: int,
        total_samples: int,
        seed: int = 2026,
    ) -> None:
        if weights.numel() == 0:
            raise ValueError("weights must not be empty")
        if total_samples <= 0:
            raise ValueError("total_samples must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if total_samples > weights.numel():
            raise ValueError(
                "total_samples cannot exceed the dataset size for sampling without replacement"
            )
        if total_samples < world_size:
            raise ValueError("total_samples must be at least world_size")
        if total_samples % world_size != 0:
            raise ValueError(
                "total_samples must be divisible by world_size for multi-GPU training: "
                f"total_samples={total_samples}, world_size={world_size}"
            )
        self.weights, self.rank, self.world_size = weights, rank, world_size
        self.samples_per_rank = total_samples // world_size
        self.total_size = self.samples_per_rank * world_size
        self.seed, self.epoch = seed, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights, self.total_size, replacement=False, generator=generator
        )
        return iter(global_indices[self.rank : self.total_size : self.world_size].tolist())

    def __len__(self) -> int:
        return self.samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class DistributedEvaluationSampler(Sampler[int]):
    """Shard validation exactly, without replication or padding."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.size = int(size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return (self.size - 1 - self.rank) // self.world_size + 1
