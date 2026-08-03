"""End-to-end objective exposing exactly eleven paper-level loss modules."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.config import AblationConfig, LossConfig, ModelConfig, RegionHeadConfig
from tcd_prg.constants import CandidateStatus
from tcd_prg.datasets.capabilities import DatasetCapabilities

from .actions import PushLoss
from .global_grasp import GlobalGraspLoss
from .graph import DependencyGraphLoss
from .labels import (
    build_global_grasp_labels,
    build_graph_labels,
    build_grasp_proposal_labels,
    build_push_supervision,
    build_region_labels,
    build_verifier_labels,
)
from .policy import HierarchicalSetPolicyLoss
from .proposal import GraspProposalLoss
from .region import TaskRegionLoss
from .total import MultiTaskLoss
from .verifier import GraspVerifierLoss


class TCDPRGObjective(nn.Module):
    """Construct labels lazily and combine the eleven module objectives."""

    MODULE_OBJECTIVES = (
        "region", "task_grasp", "global_grasp", "physical_edge", "task_edge",
        "verify_overall", "push_object", "push_contact", "push_direction",
        "push_potential", "policy_candidate",
    )

    def __init__(
        self, capabilities: DatasetCapabilities, model_config: ModelConfig,
        ablation: AblationConfig, loss_config: LossConfig | None = None,
        region_config: RegionHeadConfig | None = None,
        generated_policy_candidate_ratio: float = 0.0,
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
        self.task_grasp = GraspProposalLoss(
            negative_translation_m=model_config.grasp_nms_translation_m,
            negative_rotation_deg=model_config.grasp_nms_rotation_deg,
            negative_width_m=model_config.grasp_nms_width_m,
        )
        self.global_grasp = GlobalGraspLoss(
            negative_translation_m=model_config.global_grasp_nms_translation_m,
            negative_rotation_deg=model_config.global_grasp_nms_rotation_deg,
            negative_width_m=model_config.global_grasp_nms_width_m,
        )
        self.graph = DependencyGraphLoss()
        self.push = PushLoss()
        self.verify = GraspVerifierLoss()
        self.policy = HierarchicalSetPolicyLoss()
        self.generated_policy_candidate_ratio = float(generated_policy_candidate_ratio)
        self.total = MultiTaskLoss(capabilities, ablation, loss_config.family_weights())

    @staticmethod
    def _listwise_active(positive: Tensor, valid: Tensor) -> Tensor:
        positive = positive.bool() & valid.bool()
        return (positive.any(-1) & (valid & ~positive).any(-1)).any()

    def _subtotal(
        self, values: dict[str, Tensor], active: dict[str, Tensor | bool]
    ) -> dict[str, Tensor]:
        if values.keys() != active.keys():
            raise KeyError("Loss activity keys do not match loss values")
        reference = next(iter(values.values()))
        numerator = reference.new_zeros(())
        denominator = reference.new_zeros(())
        for name, value in values.items():
            flag = torch.as_tensor(active[name], dtype=torch.bool, device=value.device).any()
            weight = float(self.internal_weights.get(name, 1.0))
            numerator = numerator + weight * value * flag.to(value.dtype)
            denominator = denominator + weight * flag.to(value.dtype)
        return {"loss": numerator / denominator.clamp_min(torch.finfo(reference.dtype).eps), **values}

    def forward(self, model: nn.Module, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        if bool((batch["required_grasp_count"] > self.model_config.task_grasp_candidates).any()):
            maximum = int(batch["required_grasp_count"].max())
            raise ValueError(
                f"Batch requires {maximum} unique grasps but task_grasp_candidates="
                f"{self.model_config.task_grasp_candidates}"
            )
        # 纯 generated-policy 阶段跳过抓取、图、Verifier 和 Push head 的前向计算。
        generated_only = (
            self.generated_policy_candidate_ratio == 1.0
            and self.total.enabled("policy_candidate")
            and not any(
                self.total.enabled(name)
                for name in self.MODULE_OBJECTIVES if name != "policy_candidate"
            )
        )
        output = model(batch, forward_mode="generated_policy" if generated_only else "full")
        families: dict[str, dict[str, Tensor]] = {}

        region_labels = build_region_labels(batch)
        if region_labels is not None and self.total.enabled("region"):
            region_losses = self.region(output["region"], region_labels)
            families["region"] = self._subtotal(region_losses, {
                "region_focal": region_labels["region_valid"].any(),
                "region_dice": region_labels["region_valid"].any(),
                "region_visibility": region_labels["visibility_valid"].any(),
            })

        if self.total.enabled("task_grasp"):
            task_labels = build_grasp_proposal_labels(batch, self.model_config)
            if task_labels["sample_valid"].any():
                families["task_grasp"] = self.task_grasp(output["task_grasp"], task_labels)

        if self.total.enabled("global_grasp"):
            global_labels = build_global_grasp_labels(batch, self.model_config)
            if global_labels is not None and global_labels["sample_valid"].any():
                families["global_grasp"] = self.global_grasp(output["global_grasp"], global_labels)

        if output["graph"] is not None and self.total.enabled("physical_edge"):
            graph_labels = build_graph_labels(batch)
            graph_losses = self.graph(output["graph"], graph_labels)
            families["physical_edge"] = {"loss": graph_losses["physical_edge"]}
            families["task_edge"] = {"loss": graph_losses["task_edge"]}

        if output["verifier"] is not None and self.total.enabled("verify_overall"):
            verifier_labels = build_verifier_labels(batch)
            families["verify_overall"] = {
                "loss": self.verify(output["verifier"], verifier_labels)
            }

        if self.total.enabled("push_object"):
            push_output, push_labels = build_push_supervision(
                output["push"], batch, self.model_config
            )
            push_losses = self.push(push_output, push_labels)
            families["push_object"] = {"loss": push_losses["push_object"]}
            families["push_contact"] = {"loss": push_losses["push_contact"]}
            families["push_direction"] = {
                "loss": push_losses["push_direction"],
                "push_direction_bin_diagnostic": push_losses["push_direction_bin_diagnostic"],
                "push_direction_residual_diagnostic": push_losses["push_direction_residual_diagnostic"],
            }
            if self.total.enabled("push_potential"):
                families["push_potential"] = {"loss": push_losses["push_potential"]}

        if self.total.enabled("policy_candidate") and self.ablation.router_type != "fixed_priority":
            teacher_loss = None
            if "router" in output:
                evaluated = batch["policy_success_mask"] | (
                    batch["evaluation_status"] == int(CandidateStatus.NEGATIVE)
                )
                teacher_loss = self.policy(
                    output["router"], batch["policy_success_mask"], evaluated
                )
            generated_loss = None
            generated = batch.get("generated_policy_candidates")
            if "generated_router" in output and generated is not None:
                # 三态标签：UNKNOWN 不进入分母；只有认证后的正/负候选参与排序损失。
                generated_evaluated = (
                    generated["label_status"] != int(CandidateStatus.UNKNOWN_UNTESTED)
                )
                generated_loss = self.policy(
                    output["generated_router"], generated["policy_success"],
                    generated_evaluated,
                )
            # 课程学习可在干净 teacher candidates 与真实生成误差的 candidates 间插值。
            ratio = self.generated_policy_candidate_ratio
            if generated_loss is not None and teacher_loss is not None:
                policy_loss = (1.0 - ratio) * teacher_loss + ratio * generated_loss
            elif generated_loss is not None:
                policy_loss = generated_loss
            elif teacher_loss is not None and ratio == 0.0:
                policy_loss = teacher_loss
            else:
                raise RuntimeError(
                    "Generated policy training is enabled but the batch has no generated candidates"
                )
            families["policy_candidate"] = {"loss": policy_loss}
        total, terms = self.total(families)
        generated = batch.get("generated_policy_candidates")
        if generated is not None:
            valid = generated["valid"]
            positive = valid & generated["policy_success"]
            negative = valid & (
                generated["label_status"] == int(CandidateStatus.NEGATIVE)
            )
            positive_rows = positive.any(-1)
            negative_rows = negative.any(-1)
            terms.update({
                "generated_states": valid.new_tensor(valid.shape[0], dtype=torch.float32),
                "generated_states_with_positive": positive_rows.sum().float(),
                "generated_effective_policy_rows": (
                    positive_rows & negative_rows
                ).sum().float(),
                "generated_known_candidates": (positive | negative).sum().float(),
                "generated_unknown_candidates": (
                    valid & ~positive & ~negative
                ).sum().float(),
                "generated_conflict_candidates": generated.get(
                    "match_conflict", torch.zeros_like(valid)
                ).sum().float(),
            })
        return total, terms
