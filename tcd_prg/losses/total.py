"""Capability-aware aggregation of the eleven paper-level objectives."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from tcd_prg.config import AblationConfig
from tcd_prg.datasets.capabilities import DatasetCapabilities


class MultiTaskLoss(nn.Module):
    DEFAULT_WEIGHTS = {
        "region": 1.0,
        "task_grasp": 1.0,
        "global_grasp": 1.0,
        "physical_edge": 1.0,
        "task_edge": 1.0,
        "verify_overall": 1.0,
        "push_object": 1.0,
        "push_contact": 1.0,
        "push_direction": 1.0,
        "push_potential": 1.0,
        "policy_candidate": 1.0,
    }

    def __init__(
        self, capabilities: DatasetCapabilities, ablation: AblationConfig,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.capabilities = capabilities
        self.ablation = ablation
        self.weights = dict(self.DEFAULT_WEIGHTS)
        self.weights.update(weights or {})

    def enabled(self, family: str) -> bool:
        requirement = {
            "region": "region",
            "task_grasp": "proposal",
            "global_grasp": "global_grasp",
            "physical_edge": "graph",
            "task_edge": "graph",
            "verify_overall": "verify",
            "push_object": "push",
            "push_contact": "push",
            "push_direction": "push",
            "push_potential": "push",
            "policy_candidate": "policy",
        }[family]
        if requirement == "graph" and not self.ablation.use_dependency_graph:
            return False
        if family == "push_potential" and not self.ablation.use_push_potential:
            return False
        return self.capabilities.loss_available(requirement)

    def forward(self, families: Mapping[str, Mapping[str, Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
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
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No applicable loss objective for this dataset/configuration")
        selected["loss_total"] = total
        return total, selected
