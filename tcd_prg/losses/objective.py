"""End-to-end supervised objective for one action-state group batch."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.config import AblationConfig, LossConfig, ModelConfig, RegionHeadConfig
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
from .proposal import GraspProposalLoss, StateGraspabilityLoss
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
        loss_config: LossConfig | None = None,
        region_config: RegionHeadConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.ablation = ablation
        loss_config = loss_config or LossConfig()
        self.internal_weights = dict(loss_config.internal)
        region_config = region_config or RegionHeadConfig()
        self.region = TaskRegionLoss(
            region_config.focal_alpha, region_config.focal_gamma, region_config.dice_weight
        )
        self.proposal = GraspProposalLoss()
        self.state_graspability = StateGraspabilityLoss()
        self.graph = DependencyGraphLoss()
        self.push = PushLoss()
        self.remove = PickRemoveLoss()
        self.verify = GraspVerifierLoss()
        self.policy = HierarchicalSetPolicyLoss()
        self.total = MultiTaskLoss(capabilities, ablation, loss_config.family_weights())

    @staticmethod
    def _listwise_active(positive: Tensor, valid: Tensor) -> Tensor:
        valid = valid.bool()
        positive = positive.bool() & valid
        return (valid.any(-1) & positive.any(-1)).any()

    def _subtotal(
        self, values: dict[str, Tensor], active: dict[str, Tensor | bool]
    ) -> dict[str, Tensor]:
        """Return a weighted mean over child losses with valid supervision.

        A zero-valued but valid loss remains active. An unavailable child loss
        contributes neither its value nor its configured weight to the family
        denominator, preventing large families from dominating merely because
        they expose more heads.
        """

        if values.keys() != active.keys():
            missing = values.keys() - active.keys()
            extra = active.keys() - values.keys()
            raise KeyError(f"Loss activity mismatch: missing={missing}, extra={extra}")
        reference = next(iter(values.values()), None)
        if reference is None:
            raise ValueError("Cannot construct an empty loss family")
        numerator = reference.new_zeros(())
        active_weight = reference.new_zeros(())
        for name, value in values.items():
            flag = torch.as_tensor(active[name], dtype=torch.bool, device=value.device).any()
            flag_float = flag.to(value.dtype)
            weight = float(self.internal_weights.get(name, 1.0))
            numerator = numerator + weight * value * flag_float
            active_weight = active_weight + weight * flag_float
        total = numerator / active_weight.clamp_min(torch.finfo(reference.dtype).eps)
        return {"loss": total, **values}

    @staticmethod
    def _proposal_activity(labels: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
        proposal = labels["proposal_valid"].bool()
        mode = labels["mode_valid"].bool()
        compatibility = labels.get("compatibility_valid", proposal).bool()
        return {
            f"{prefix}_proposal_contact": proposal.any(),
            f"{prefix}_proposal_approach": (
                mode & torch.isfinite(labels["approach_target"]).all(-1)
            ).any(),
            f"{prefix}_proposal_rotation": (mode & (labels["rotation_bin"] >= 0)).any(),
            f"{prefix}_proposal_width": (
                mode & labels["width_valid"].bool() & torch.isfinite(labels["width_target_m"])
            ).any(),
            f"{prefix}_proposal_confidence": proposal.any(),
            f"{prefix}_proposal_task_compatibility": (
                compatibility & torch.isfinite(labels["compatibility_target"])
            ).any(),
        }

    def forward(self, model: nn.Module, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        exceeds_declared_maximum = (
            batch["required_grasp_count"] > self.model_config.max_required_grasp_count
        ).any()
        if bool(exceeds_declared_maximum):
            maximum = int(batch["required_grasp_count"].max())
            raise ValueError(
                f"Batch required_grasp_count={maximum} exceeds declared "
                f"max_required_grasp_count={self.model_config.max_required_grasp_count}"
            )
        if bool((batch["required_grasp_count"] > self.model_config.task_grasp_candidates).any()):
            maximum = int(batch["required_grasp_count"].max())
            raise ValueError(
                f"Batch requires {maximum} unique grasps but task_grasp_candidates="
                f"{self.model_config.task_grasp_candidates}"
            )
        batch["contact_heatmap_sigma_m"] = self.model_config.contact_heatmap_sigma_m
        output = model(batch)
        families: dict[str, dict[str, Tensor]] = {}
        region_labels = build_region_labels(batch)
        if region_labels is not None and self.total.enabled("region"):
            region_losses = self.region(output["region"], region_labels)
            families["region"] = self._subtotal(region_losses, {
                "region_focal": region_labels["region_valid"].any(),
                "region_dice": region_labels["region_valid"].any(),
                "region_visibility": region_labels["visibility_valid"].any(),
            })
        if self.total.enabled("proposal"):
            task_labels = build_grasp_proposal_labels(batch, self.model_config)
            generic_labels = build_grasp_proposal_labels(
                batch, self.model_config, generic_remove=True
            )
            task_losses = self.proposal(
                output["task_grasp"], task_labels
            )
            generic_losses = self.proposal(
                output["generic_grasp"], generic_labels,
            )
            state_losses = self.state_graspability(
                output["state_graspability"],
                {
                    "graspable_target": (
                        batch["verified_positive_grasp_count"] >= batch["required_grasp_count"]
                    ),
                    "verified_count_target": batch["verified_positive_grasp_count"],
                },
            )
            proposal_losses = {
                **{f"task_{key}": value for key, value in task_losses.items()},
                **{f"generic_{key}": value for key, value in generic_losses.items()},
                **state_losses,
            }
            proposal_active = {
                **self._proposal_activity(task_labels, "task"),
                **self._proposal_activity(generic_labels, "generic"),
                "proposal_state_graspable": torch.ones(
                    (), dtype=torch.bool, device=batch["xyz"].device
                ),
                "proposal_verified_count": torch.ones(
                    (), dtype=torch.bool, device=batch["xyz"].device
                ),
            }
            families["proposal"] = self._subtotal(proposal_losses, proposal_active)
        if output["verifier"] is not None and self.total.enabled("verify"):
            verifier_labels = build_verifier_labels(batch)
            verifier_losses = self.verify(output["verifier"], verifier_labels)
            families["verify"] = self._subtotal(verifier_losses, {
                f"verify_{head}": verifier_labels[f"{head}_valid"].any()
                for head in self.verify.HEADS
            })
        if output["graph"] is not None and self.total.enabled("graph"):
            graph_labels = build_graph_labels(batch)
            graph_losses = self.graph(output["graph"], graph_labels)
            families["graph"] = self._subtotal(graph_losses, {
                "graph_physical_edge": graph_labels["physical_edge_valid"].any(),
                "graph_task_edge": graph_labels["task_edge_valid"].any(),
                "graph_direct_blocker": graph_labels["blocker_valid"].any(),
                "graph_indirect_blocker": graph_labels["blocker_valid"].any(),
                "graph_actionable": graph_labels["blocker_valid"].any(),
                "graph_topology_order": (
                    graph_labels["topology_edge_valid"]
                    & graph_labels["sequence_topology_valid"][:, None, None]
                ).any(),
            })
        if self.total.enabled("push"):
            push_output, push_labels = build_push_supervision(
                output["push"],
                batch,
                self.ablation.use_push_potential,
                self.ablation.use_push_risk,
            )
            push_losses = self.push(push_output, push_labels)  # type: ignore[arg-type]
            push_active = {
                "push_object": self._listwise_active(
                    push_labels["object_positive"], push_labels["object_valid_mask"]
                ),
                "push_contact": push_labels["contact_valid"].any(),
                "push_direction_bin": push_labels["direction_valid"].any(),
                "push_direction_residual": push_labels["direction_valid"].any(),
            }
            if "push_risk" in push_losses:
                push_active["push_risk"] = push_labels["risk_valid"].any()
            if "push_potential" in push_losses:
                push_active["push_potential"] = push_labels["potential_after_valid"].any()
            potential = {
                key: value for key, value in push_losses.items() if key == "push_potential"
            }
            main_push = {
                key: value for key, value in push_losses.items() if key != "push_potential"
            }
            families["push"] = self._subtotal(
                main_push, {key: push_active[key] for key in main_push}
            )
            if potential:
                families["potential"] = self._subtotal(
                    potential, {"push_potential": push_active["push_potential"]}
                )
        if self.total.enabled("remove"):
            remove_labels = build_remove_labels(batch)
            remove_losses = self.remove(output["pick_remove"], remove_labels)
            remove_active = {
                "remove_object": self._listwise_active(
                    remove_labels["object_positive"], remove_labels["object_valid_mask"]
                )
            }
            if "remove_candidate" in remove_losses:
                remove_active["remove_candidate"] = self._listwise_active(
                    remove_labels["candidate_positive"], remove_labels["candidate_valid"]
                )
            families["remove"] = self._subtotal(remove_losses, remove_active)
        if (
            "router" in output
            and self.total.enabled("policy")
            and self.ablation.router_type != "fixed_priority"
        ):
            evaluated = batch["evaluation_status"] != int(CandidateStatus.UNKNOWN_UNTESTED)
            policy_losses = self.policy(
                output["router"],
                batch["action_type"],
                batch["acted_object"],
                batch["policy_success_mask"],
                evaluated,
                batch["remaining_steps_target"],
                batch["remaining_steps_valid"],
            )
            policy_candidate_valid = output["router"].candidate_valid_mask & evaluated
            type_positive = torch.stack([
                (
                    batch["policy_success_mask"]
                    & (batch["action_type"] == kind)
                ).any(-1)
                for kind in range(3)
            ], -1)
            policy_active = {
                "policy_candidate": self._listwise_active(
                    batch["policy_success_mask"], policy_candidate_valid
                ),
                "policy_type": self._listwise_active(
                    type_positive, output["router"].type_valid_mask
                ),
                "policy_object": batch["policy_success_mask"].any(),
            }
            if "policy_remaining_steps" in policy_losses:
                policy_active["policy_remaining_steps"] = (
                    batch["remaining_steps_valid"]
                    & torch.isfinite(batch["remaining_steps_target"])
                ).any()
            families["policy"] = self._subtotal(policy_losses, policy_active)
        return self.total(families)
