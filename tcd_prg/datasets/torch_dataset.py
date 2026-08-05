"""PyTorch datasets and state-group balanced sampling.

The sampling unit is always ``(scene_id, state_id, task_index,
action_state_group)``.  No action row is exposed as an independent sample.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import islice

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .base import DatasetAdapter
from .types import UnifiedSample


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


class ActionStateGroupDataset(Dataset[UnifiedSample]):
    """Immutable snapshot of completed action-state groups."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        split: str | None = None,
        max_groups: int | None = None,
        include_strata: bool = True,
        fraction: float = 1.0,
        subset_seed: int = 2026,
    ) -> None:
        self.adapter = adapter
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0,1]")
        if max_groups is not None and max_groups <= 0:
            raise ValueError("max_groups must be positive")
        iterator = adapter.iter_action_groups(split)
        if fraction == 1.0:
            if max_groups is not None:
                raw_units = list(islice(iterator, max_groups))
                self.source_group_count: int | None = None
            else:
                raw_units = list(iterator)
                self.source_group_count = len(raw_units)
        else:
            raw_units = list(iterator)
            self.source_group_count = len(raw_units)
            selected = _deterministic_fraction_indices(
                len(raw_units), fraction, subset_seed
            )
            raw_units = [raw_units[int(index)] for index in selected]
            if max_groups is not None:
                raw_units = raw_units[:max_groups]
        self.requested_fraction = float(fraction)
        self.selected_group_count = len(raw_units)
        strata_method = getattr(adapter, "action_group_strata", None)
        strata = strata_method(raw_units) if include_strata and strata_method else {}
        self.units = tuple(
            StateGroupUnit(*unit, stratum=strata.get(unit, "unclassified")) for unit in raw_units
        )
        representatives: dict[tuple[int, int], int] = {}
        for index, unit in enumerate(self.units):
            representatives.setdefault((unit.scene_id, unit.state_id), index)
        self._global_grasp_representatives = frozenset(representatives.values())

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> UnifiedSample:
        unit = self.units[index]
        return self.adapter.load_sample(
            unit.scene_id,
            unit.state_id,
            unit.task_index,
            unit.group_index,
            include_global_grasps=index in self._global_grasp_representatives,
        )

    def balanced_sampler(
        self, seed: int = 2026, samples: int | None = None
    ) -> "DistributedWeightedStateSampler":
        """Inverse-frequency sampler without duplicate groups within an epoch."""

        counts = Counter(unit.stratum for unit in self.units)
        if not counts:
            raise ValueError("Cannot sample an empty dataset")
        weights = torch.tensor([1.0 / counts[unit.stratum] for unit in self.units], dtype=torch.double)
        return DistributedWeightedStateSampler(
            weights, rank=0, world_size=1,
            total_samples=samples or len(self.units), seed=seed,
        )

    def distributed_balanced_sampler(
        self, rank: int, world_size: int, seed: int = 2026, samples: int | None = None
    ) -> "DistributedWeightedStateSampler":
        counts = Counter(unit.stratum for unit in self.units)
        weights = torch.tensor([1.0 / counts[unit.stratum] for unit in self.units], dtype=torch.double)
        return DistributedWeightedStateSampler(
            weights, rank, world_size, samples or len(self.units), seed
        )

    @property
    def stratum_counts(self) -> dict[str, int]:
        return dict(Counter(unit.stratum for unit in self.units))


class DistributedWeightedStateSampler(Sampler[int]):
    """Deterministic inverse-frequency sampler sharded without rank overlap."""

    def __init__(self, weights: torch.Tensor, rank: int, world_size: int,
                 total_samples: int, seed: int = 2026) -> None:
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
