"""Capability-aware aggregation for the minimal TCD-PRG objective set."""
from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from tcd_prg.config import AblationConfig
from tcd_prg.datasets.capabilities import DatasetCapabilities


class MultiTaskLoss(nn.Module):
    DEFAULT_WEIGHTS = {
        "instance": 1.0,
        "region": 1.0,
        "task_grasp": 1.0,
        "push_object": 1.0,
        "push_contact": 1.0,
        "push_direction": 1.0,
        "push_potential": 1.0,
    }

    def __init__(
        self,
        capabilities: DatasetCapabilities,
        ablation: AblationConfig,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.capabilities = capabilities
        self.ablation = ablation
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.weights.update(weights or {})

    def enabled(self, family: str) -> bool:
        if family not in self.weights:
            raise KeyError(family)
        if float(self.weights[family]) == 0.0:
            return False
        requirement = {
            "instance": "instance",
            "region": "region",
            "task_grasp": "proposal",
            "push_object": "push",
            "push_contact": "push",
            "push_direction": "push",
            "push_potential": "push",
        }[family]
        if family == "push_potential" and not self.ablation.use_push_potential:
            return False
        return self.capabilities.loss_available(requirement)

    def forward(
        self, families: Mapping[str, Mapping[str, Tensor]]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        selected: dict[str, Tensor] = {}
        total: Tensor | None = None
        for family, values in families.items():
            if not self.enabled(family):
                continue
            family_total = values["loss"]
            selected[f"loss_{family}"] = family_total.detach()
            for name, value in values.items():
                if name != "loss":
                    selected[name] = value.detach()
            weighted = self.weights[family] * family_total
            selected[f"weighted_loss_{family}"] = weighted.detach()
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No applicable loss objective for this dataset/configuration")
        selected["loss_total"] = total
        return total, selected
