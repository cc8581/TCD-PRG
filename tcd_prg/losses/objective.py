"""End-to-end objective with strict sensor/task/GT separation."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.models.stageb_condition import stageb_condition_from_gt
from tcd_prg.models.push_condition import push_condition_from_gt

from tcd_prg.config import AblationConfig, LossConfig, ModelConfig, RegionHeadConfig
from tcd_prg.datasets.capabilities import DatasetCapabilities

from .instance import (
    InstanceSetLoss,
    build_instance_targets,
)
from .labels import (
    build_region_labels,
)
from .task_grasp_binary import TaskGraspBinaryLoss
from .region import TaskRegionLoss
from .total import MultiTaskLoss


class TCDPRGObjective(nn.Module):
    """GT is consumed here; perception forward consumes sensor/task only."""

    MODULE_OBJECTIVES = (
        "instance",
        "region",
        "task_grasp",
        "push_object",
        "push_contact",
        "push_direction",
        "push_potential",
    )

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

        self.instance = InstanceSetLoss(
            matching_points=model_config.instance_matching_points,
            objectness_weight=float(self.internal_weights.get("instance_objectness", 1.0)),
            mask_weight=float(self.internal_weights.get("instance_mask", 2.0)),
            dice_weight=float(self.internal_weights.get("instance_dice", 2.0)),
            category_weight=float(self.internal_weights.get("instance_category", 1.0)),
            target_weight=float(self.internal_weights.get("target_query", 1.0)),
            same_category_target_weight=model_config.target_same_category_loss_boost,
            auxiliary_weight=float(self.internal_weights.get("instance_auxiliary", 0.5)),
        )
        self.region = TaskRegionLoss(
            region_config.focal_alpha,
            region_config.focal_gamma,
            region_config.dice_weight,
        )
        self.task_grasp = TaskGraspBinaryLoss()
        self.total = MultiTaskLoss(
            capabilities,
            ablation,
            loss_config.family_weights(),
        )

    @staticmethod
    def _row_active(valid: Tensor) -> Tensor:
        valid = valid.bool()
        if valid.ndim == 0:
            return valid.reshape(1)
        return valid.reshape(valid.shape[0], -1).any(-1)

    @staticmethod
    def _listwise_active_rows(positive: Tensor, valid: Tensor) -> Tensor:
        """Rows with both a known positive and a known negative competitor."""
        valid = valid.bool()
        positive = positive.bool() & valid
        negative = valid & ~positive
        return positive.any(-1) & negative.any(-1)

    def _subtotal(
        self,
        values: dict[str, Tensor],
        active: dict[str, Tensor | bool],
    ) -> dict[str, Tensor]:
        if values.keys() != active.keys():
            raise KeyError("Loss activity keys do not match loss values")
        reference = next(iter(values.values()))
        numerator = reference.new_zeros(())
        denominator = reference.new_zeros(())
        for name, value in values.items():
            flag = torch.as_tensor(
                active[name],
                dtype=torch.bool,
                device=value.device,
            ).any()
            weight = float(self.internal_weights.get(name, 1.0))
            numerator = numerator + weight * value * flag.to(value.dtype)
            denominator = denominator + weight * flag.to(value.dtype)
        return {
            "loss": numerator / denominator.clamp_min(torch.finfo(reference.dtype).eps),
            **values,
        }

    def _synthesize_target_prompt(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        """Create an observable positive 3D task prompt from GT during training.

        The GT mask is used only to choose one coordinate, analogous to generating
        a click annotation online. The mask itself never enters model_inputs or
        task_inputs. Existing HDF5 action/physics labels therefore stay unchanged.
        """
        xyz = batch["xyz"]
        valid_target = batch["point_mask"].bool() & batch["target_mask"].bool()
        b = xyz.shape[0]
        prompt_xyz = xyz.new_zeros((b, 1, 3))
        prompt_label = torch.ones((b, 1), dtype=torch.long, device=xyz.device)
        prompt_valid = torch.zeros((b, 1), dtype=torch.bool, device=xyz.device)
        for row in range(b):
            candidates = torch.nonzero(valid_target[row], as_tuple=False).flatten()
            if not len(candidates):
                continue
            if self.training:
                choice = candidates[torch.randint(len(candidates), (1,), device=xyz.device)[0]]
            else:
                # Deterministic validation prompt: the observed target point closest
                # to its visible centroid, not an off-cloud synthetic coordinate.
                points = xyz[row, candidates]
                centroid = points.mean(0)
                choice = candidates[torch.linalg.vector_norm(points - centroid, dim=-1).argmin()]
            prompt_xyz[row, 0] = xyz[row, choice]
            prompt_valid[row, 0] = True
        jitter = float(self.model_config.target_prompt_jitter_std_m)
        if self.training and jitter > 0:
            noise = torch.randn_like(prompt_xyz) * jitter
            prompt_xyz = torch.where(prompt_valid[..., None], prompt_xyz + noise, prompt_xyz)
        return prompt_xyz, prompt_label, prompt_valid

    def _model_view(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Strict view: fused sensor data + task semantics + observable prompt."""
        model_inputs = {
            "xyz": batch["xyz"],
            "rgb": batch["rgb"],
            "point_mask": batch["point_mask"],
        }
        for key in (
            "source_view",
            "graspnet_xyz_world",
            "graspnet_point_mask",
            "camera2_eye_world",
            "camera2_target_world",
            "camera2_up_world",
            "camera2_valid",
        ):
            if key in batch:
                model_inputs[key] = batch[key]
        if "grid_coord" in batch:
            model_inputs["grid_coord"] = batch["grid_coord"]
        view: dict[str, Any] = {"model_inputs": model_inputs}
        task_keys = ("task_category_id", "task_region_id")
        if all(key in batch for key in task_keys):
            if all(
                key in batch
                for key in ("target_prompt_xyz", "target_prompt_label", "target_prompt_valid")
            ):
                prompt_xyz = batch["target_prompt_xyz"]
                prompt_label = batch["target_prompt_label"]
                prompt_valid = batch["target_prompt_valid"]
            else:
                prompt_xyz, prompt_label, prompt_valid = self._synthesize_target_prompt(batch)
            view["task_inputs"] = {key: batch[key] for key in task_keys}
            view["task_inputs"].update(
                {
                    "target_prompt_xyz": prompt_xyz,
                    "target_prompt_label": prompt_label,
                    "target_prompt_valid": prompt_valid,
                }
            )
        if "stageb_candidates" in batch:
            view["grasp_candidates"] = {
                key: value for key, value in batch["stageb_candidates"].items()
            }
        return view

    def _match_instances(
        self,
        output_instance: Any,
        batch: dict[str, Any],
        target_query_logits: Tensor | None,
    ):
        targets = build_instance_targets(batch, output_instance.mask_logits.shape[1])
        values, match = self.instance(
            output_instance,
            targets,
            target_query_logits=target_query_logits,
        )
        return values, match, targets

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        *,
        return_output: bool = False,
        return_family_losses: bool = False,
    ) -> Any:
        enabled = {name for name in self.MODULE_OBJECTIVES if self.total.enabled(name)}
        push_families = {"push_object", "push_contact", "push_direction", "push_potential"}
        has_push = bool(enabled & push_families)
        has_grasp = "task_grasp" in enabled
        if has_push:
            raise RuntimeError("PUSH generation training was removed; use train_push_evaluator.py")
        forward_mode = "grasp" if has_grasp else "perception"
        model_view = self._model_view(batch)
        if forward_mode == "grasp":
            model_view["stageb_condition"] = stageb_condition_from_gt(batch)
        output = model(model_view, forward_mode=forward_mode)

        families: dict[str, dict[str, Tensor]] = {}
        reference = batch["xyz"]
        activity: dict[str, Tensor] = {
            f"active_loss_{name}": reference.new_zeros(()) for name in enabled
        }

        match = None
        instance_targets = None
        if self.total.enabled("instance"):
            instance_values, match, instance_targets = self._match_instances(
                output["instance"],
                batch,
                output["encoded"].target_query_logits,
            )
            output["instance_match"] = match
            families["instance"] = instance_values
            activity["active_loss_instance"] = instance_targets["visible"].any(-1).float().mean()

        region_labels = build_region_labels(batch)
        if region_labels is not None and self.total.enabled("region"):
            region_losses = self.region(output["region"], region_labels)
            families["region"] = self._subtotal(
                region_losses,
                {
                    "region_focal": region_labels["region_valid"].any(),
                    "region_dice": region_labels["region_valid"].any(),
                    "region_visibility": region_labels["visibility_valid"].any(),
                },
            )
            activity["active_loss_region"] = (
                (
                    self._row_active(region_labels["region_valid"])
                    | self._row_active(region_labels["visibility_valid"])
                )
                .float()
                .mean()
            )

        if self.total.enabled("task_grasp"):
            if output.get("task_grasp") is None:
                raise RuntimeError("task_grasp loss enabled but Grasp forward was skipped")
            if "stageb_label" not in batch or "stageb_candidate_valid" not in batch:
                raise RuntimeError("Stage-B training requires the binary candidate dataset")
            score_losses = self.task_grasp(
                output["task_grasp"], batch["stageb_label"], batch["stageb_candidate_valid"]
            )
            families["task_grasp"] = score_losses
            activity["active_loss_task_grasp"] = (
                batch["stageb_candidate_valid"].any(-1).float().mean()
            )

        total, terms = self.total(families)
        terms.update(activity)
        terms["forward_mode_code"] = reference.new_tensor(
            {"perception": 0.0, "grasp": 1.0, "push": 2.0, "full": 3.0}[forward_mode]
        )
        weighted_family_losses = {
            name: float(self.total.weights[name]) * values["loss"]
            for name, values in families.items()
            if self.total.enabled(name)
        }

        if return_output and return_family_losses:
            return total, terms, output, weighted_family_losses
        if return_output:
            return total, terms, output
        if return_family_losses:
            return total, terms, weighted_family_losses
        return total, terms
