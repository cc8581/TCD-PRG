"""End-to-end supervised objective for one action-state group batch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from tcd_prg.config import AblationConfig, ModelConfig, RegionHeadConfig
from tcd_prg.constants import CandidateStatus
from tcd_prg.datasets.capabilities import DatasetCapabilities

from .actions import PickRemoveLoss, PushLoss
from .graph import DependencyGraphLoss
from .labels import (
    build_graph_labels,
    build_grasp_proposal_labels,
    build_push_supervision,
    build_region_labels,
    build_remove_labels,
    build_verifier_labels,
)
from .policy import HierarchicalSetPolicyLoss
from .proposal import GraspProposalLoss
from .region import TaskRegionLoss
from .verifier import GraspVerifierLoss
from .total import MultiTaskLoss


class TCDPRGObjective(nn.Module):
    """Construct labels lazily and aggregate only capability-valid losses."""

    def __init__(
        self,
        capabilities: DatasetCapabilities,
        model_config: ModelConfig,
        ablation: AblationConfig,
        weights: Mapping[str, float] | None = None,
        region_config: RegionHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.ablation = ablation
        region_config = region_config or RegionHeadConfig()
        self.region = TaskRegionLoss(
            region_config.focal_alpha, region_config.focal_gamma, region_config.dice_weight
        )
        self.proposal = GraspProposalLoss()
        self.graph = DependencyGraphLoss()
        self.push = PushLoss()
        self.remove = PickRemoveLoss()
        self.verify = GraspVerifierLoss()
        self.policy = HierarchicalSetPolicyLoss()
        self.total = MultiTaskLoss(capabilities, ablation, weights)

    def forward(self, model: nn.Module, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        output = model(batch)
        families: dict[str, dict[str, Tensor]] = {}
        region_labels = build_region_labels(batch)
        if region_labels is not None and self.total.enabled("region"):
            families["region"] = self.region(output["region"], region_labels)
        if self.total.enabled("proposal"):
            task_losses = self.proposal(
                output["task_grasp"], build_grasp_proposal_labels(batch, self.model_config)
            )
            generic_losses = self.proposal(
                output["generic_grasp"],
                build_grasp_proposal_labels(batch, self.model_config, generic_remove=True),
            )
            families["proposal"] = {
                **{f"task_{key}": value for key, value in task_losses.items()},
                **{f"generic_{key}": value for key, value in generic_losses.items()},
            }
        if output["verifier"] is not None and self.total.enabled("verify"):
            families["verify"] = self.verify(
                output["verifier"], build_verifier_labels(batch)
            )
        if output["graph"] is not None and self.total.enabled("graph"):
            families["graph"] = self.graph(output["graph"], build_graph_labels(batch))
        if self.total.enabled("push"):
            push_output, push_labels = build_push_supervision(
                output["push"],
                batch,
                self.ablation.use_push_potential,
                self.ablation.use_push_risk,
            )
            push_losses = self.push(push_output, push_labels)  # type: ignore[arg-type]
            potential = {key: value for key, value in push_losses.items() if key == "push_potential"}
            families["push"] = {
                key: value for key, value in push_losses.items() if key != "push_potential"
            }
            if potential:
                families["potential"] = potential
        if self.total.enabled("remove"):
            families["remove"] = self.remove(output["pick_remove"], build_remove_labels(batch))
        if (
            "router" in output
            and self.total.enabled("policy")
            and self.ablation.router_type != "fixed_priority"
        ):
            evaluated = batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED)
            families["policy"] = self.policy(
                output["router"],
                batch["action_type"],
                batch["acted_object"],
                batch["success_mask"],
                evaluated,
                batch["remaining_steps_target"],
                batch["remaining_steps_valid"],
            )
        return self.total(families)
