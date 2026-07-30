"""Capability- and ablation-aware multi-task loss aggregation."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from tcd_prg.config import AblationConfig
from tcd_prg.datasets.capabilities import DatasetCapabilities


class MultiTaskLoss(nn.Module):
    DEFAULT_WEIGHTS = {
        "region": 1.0,
        "proposal": 1.0,
        "global_grasp": 1.0,
        "verify": 1.0,
        "graph": 1.0,
        "remove": 1.0,
        "push": 1.0,
        "policy": 1.0,
        "potential": 1.0,
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
        if family == "graph" and not self.ablation.use_dependency_graph:
            return False
        if family == "potential" and not self.ablation.use_push_potential:
            return False
        return self.capabilities.loss_available("push" if family == "potential" else family)

    def forward(self, families: Mapping[str, Mapping[str, Tensor]]) -> tuple[Tensor, dict[str, Tensor]]:
        selected: dict[str, Tensor] = {}
        total: Tensor | None = None
        for family, values in families.items():
            if not self.enabled(family):
                continue
            if "loss" not in values:
                raise KeyError(f"Loss family {family!r} did not provide an explicit subtotal")
            family_total = values["loss"]
            selected[f"loss_{family}"] = family_total.detach()
            for name, value in values.items():
                if name != "loss":
                    selected[name] = value.detach()
            weighted = self.weights.get(family, 1.0) * family_total
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No applicable loss family for this dataset/configuration")
        selected["loss_total"] = total
        return total, selected
