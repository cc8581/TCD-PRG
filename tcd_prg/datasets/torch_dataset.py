"""PyTorch datasets and state-group balanced sampling.

The sampling unit is always ``(scene_id, state_id, task_index,
action_state_group)``.  No action row is exposed as an independent sample.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice

import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from .base import DatasetAdapter
from .types import UnifiedSample


@dataclass(frozen=True, slots=True)
class StateGroupUnit:
    scene_id: int
    state_id: int
    task_index: int
    group_index: int
    stratum: str = "unclassified"


class ActionStateGroupDataset(Dataset[UnifiedSample]):
    """Immutable snapshot of completed action-state groups."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        split: str | None = None,
        max_groups: int | None = None,
        include_strata: bool = True,
    ) -> None:
        self.adapter = adapter
        iterator = adapter.iter_action_groups(split)
        if max_groups is not None:
            if max_groups <= 0:
                raise ValueError("max_groups must be positive")
            raw_units = list(islice(iterator, max_groups))
        else:
            raw_units = list(iterator)
        strata_method = getattr(adapter, "action_group_strata", None)
        strata = strata_method(raw_units) if include_strata and strata_method else {}
        self.units = tuple(
            StateGroupUnit(*unit, stratum=strata.get(unit, "unclassified")) for unit in raw_units
        )

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int) -> UnifiedSample:
        unit = self.units[index]
        return self.adapter.load_sample(
            unit.scene_id, unit.state_id, unit.task_index, unit.group_index
        )

    def balanced_sampler(self, seed: int = 2026, samples: int | None = None) -> WeightedRandomSampler:
        """Inverse-frequency sampler over state semantics, not individual actions."""

        counts = Counter(unit.stratum for unit in self.units)
        if not counts:
            raise ValueError("Cannot sample an empty dataset")
        weights = torch.tensor([1.0 / counts[unit.stratum] for unit in self.units], dtype=torch.double)
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(
            weights,
            num_samples=samples or len(self.units),
            replacement=True,
            generator=generator,
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


def split_units_by_scene(
    units: Sequence[StateGroupUnit],
    validation_fraction: float = 0.1,
    seed: int = 2026,
) -> tuple[list[int], list[int]]:
    """Return train/validation indices with scene-disjoint deterministic splitting."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    scene_ids = sorted({unit.scene_id for unit in units})
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(scene_ids), generator=generator).tolist()
    validation_count = max(1, round(len(scene_ids) * validation_fraction))
    validation_scenes = {scene_ids[index] for index in order[:validation_count]}
    train_indices = [index for index, unit in enumerate(units) if unit.scene_id not in validation_scenes]
    validation_indices = [index for index, unit in enumerate(units) if unit.scene_id in validation_scenes]
    return train_indices, validation_indices


class DistributedWeightedStateSampler(Sampler[int]):
    """Deterministic inverse-frequency sampler sharded without rank overlap."""

    def __init__(self, weights: torch.Tensor, rank: int, world_size: int,
                 total_samples: int, seed: int = 2026) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.weights, self.rank, self.world_size = weights, rank, world_size
        self.samples_per_rank = (total_samples + world_size - 1) // world_size
        self.total_size = self.samples_per_rank * world_size
        self.seed, self.epoch = seed, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        )
        return iter(global_indices[self.rank : self.total_size : self.world_size].tolist())

    def __len__(self) -> int:
        return self.samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
