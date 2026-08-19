"""End-to-end objective with strict sensor/task/GT separation."""
from __future__ import annotations

from typing import Any
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from tcd_prg.config import (
    AblationConfig, LossConfig, ModelConfig, RegionHeadConfig
)
from tcd_prg.constants import CandidateStatus
from tcd_prg.datasets.capabilities import DatasetCapabilities
from tcd_prg.datasets.acronym_grasp_database import (
    load_object_grasps, match_object_grasp_priors,
)
from tcd_prg.geometry.se3 import quaternion_xyzw_to_matrix

from .actions import PushLoss
from .ag_width import AGWidthLoss
from .instance import (
    InstanceSetLoss,
    build_instance_targets,
    build_object_query_push_supervision,
)
from .labels import (
    build_grasp_proposal_labels,
    build_push_training_hints,
    build_region_labels,
)
from .task_grasp_score import TaskGraspScoringLoss
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
        acronym_object_grasp_database: str = "",
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.ablation = ablation
        loss_config = loss_config or LossConfig()
        self.internal_weights = dict(loss_config.internal)
        region_config = region_config or RegionHeadConfig()

        self.instance = InstanceSetLoss(
            matching_points=model_config.instance_matching_points,
            objectness_weight=float(
                self.internal_weights.get(
                    "instance_objectness", 1.0
                )
            ),
            mask_weight=float(
                self.internal_weights.get(
                    "instance_mask", 2.0
                )
            ),
            dice_weight=float(
                self.internal_weights.get(
                    "instance_dice", 2.0
                )
            ),
            category_weight=float(
                self.internal_weights.get(
                    "instance_category", 1.0
                )
            ),
            target_weight=float(
                self.internal_weights.get(
                    "target_query", 1.0
                )
            ),
            same_category_target_weight=model_config.target_same_category_loss_boost,
            auxiliary_weight=float(
                self.internal_weights.get("instance_auxiliary", 0.5)
            ),
        )
        self.region = TaskRegionLoss(
            region_config.focal_alpha,
            region_config.focal_gamma,
            region_config.dice_weight,
        )
        self.task_grasp = TaskGraspScoringLoss(
            translation_m=model_config.task_grasp_match_translation_m,
            rotation_deg=model_config.task_grasp_match_rotation_deg,
            width_m=model_config.task_grasp_match_width_m,
        )
        self.ag_width = AGWidthLoss(
            translation_m=model_config.task_grasp_match_translation_m,
            rotation_deg=model_config.task_grasp_match_rotation_deg,
        )
        self.push = PushLoss()
        self.acronym_database_root = (
            Path(acronym_object_grasp_database)
            if acronym_object_grasp_database else None
        )
        self._acronym_paths: dict[str, list[tuple[float, Path]]] | None = None
        self._acronym_cache: dict[
            tuple[str, str], tuple[Tensor, Tensor, Tensor]
        ] = {}
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

    def _acronym_path_map(self) -> dict[str, list[tuple[float, Path]]]:
        if self.acronym_database_root is None:
            return {}
        if self._acronym_paths is None:
            manifest = json.loads(
                (self.acronym_database_root / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            paths: dict[str, list[tuple[float, Path]]] = {}
            for record in manifest["records"]:
                model_id = str(record["model_id"])
                record_path = Path(record["path"])
                if not record_path.is_absolute():
                    record_path = self.acronym_database_root / record_path
                if not record_path.is_file():
                    raise FileNotFoundError(
                        f"ACRONYM object-grasp record does not exist: {record_path}"
                    )
                scale = record.get("object_scale")
                if scale is None:
                    scale = float(load_object_grasps(record_path)["object_scale"])
                entries = paths.setdefault(model_id, [])
                if any(abs(existing_scale - float(scale)) <= 1e-9 for existing_scale, _ in entries):
                    raise ValueError(
                        "Duplicate ACRONYM object-grasp model/scale in manifest: "
                        f"{model_id} scale={scale}"
                    )
                entries.append((float(scale), record_path))
            self._acronym_paths = paths
        return self._acronym_paths

    def _acronym_tensors(
        self, model_id: str, object_scale: float, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        entries = self._acronym_path_map()[model_id]
        scale, path = min(entries, key=lambda item: abs(item[0] - object_scale))
        tolerance = max(1e-7, abs(object_scale) * 1e-4)
        if abs(scale - object_scale) > tolerance:
            raise KeyError(
                f"No ACRONYM grasp record for {model_id} scale={object_scale}; "
                f"nearest={scale}"
            )
        key = (f"{model_id}:{scale:.12g}", str(device))
        if key not in self._acronym_cache:
            database = load_object_grasps(path)
            self._acronym_cache[key] = (
                torch.from_numpy(database["translation_object"]).to(device),
                torch.from_numpy(database["rotation_object"]).to(device),
                torch.from_numpy(database["status"]).to(device),
            )
        return self._acronym_cache[key]

    @torch.no_grad()
    def _acronym_proposal_status(
        self, prediction: dict[str, Tensor], batch: dict[str, Any]
    ) -> Tensor | None:
        if self.acronym_database_root is None or "target_model_id" not in batch:
            return None
        proposal_status = torch.full_like(
            prediction["valid"], int(CandidateStatus.UNKNOWN_UNTESTED),
            dtype=torch.int8,
        )
        for row, model_id in enumerate(batch["target_model_id"]):
            object_scale = float(batch["target_object_scale"][row])
            pose = batch["object_pose"][row, batch["target_object"][row]]
            rotation_world_object = quaternion_xyzw_to_matrix(pose[3:])
            translation_object = torch.einsum(
                "ij,kj->ki", rotation_world_object.transpose(0, 1),
                prediction["translation_world"][row] - pose[:3],
            )
            rotation_object = torch.einsum(
                "ij,kjl->kil", rotation_world_object.transpose(0, 1),
                prediction["rotation_matrix"][row],
            )
            intrinsic = match_object_grasp_priors(
                translation_object, rotation_object, prediction["valid"][row],
                *self._acronym_tensors(
                    str(model_id), object_scale, translation_object.device
                ),
                translation_m=self.task_grasp.translation_m,
                rotation_deg=self.task_grasp.rotation_deg,
            )["status"]
            status = intrinsic.clone()
            intrinsic_positive = intrinsic == int(CandidateStatus.POSITIVE)
            if "region_valid" in batch and bool(intrinsic_positive.any()):
                domain = (
                    batch["point_mask"][row].bool()
                    & batch["target_mask"][row].bool()
                )
                indices = torch.nonzero(domain, as_tuple=False).flatten()
                if len(indices):
                    nearest = torch.cdist(
                        prediction["translation_world"][row].float(),
                        batch["xyz"][row, indices].float(),
                    ).argmin(-1)
                    point_index = indices[nearest]
                    region_known = batch["region_valid"][row, point_index].bool()
                    in_region = batch["region_target"][row, point_index].bool()
                    status[intrinsic_positive & ~region_known] = int(
                        CandidateStatus.UNKNOWN_UNTESTED
                    )
                    status[intrinsic_positive & region_known & ~in_region] = int(
                        CandidateStatus.NEGATIVE
                    )
            proposal_status[row] = status
        return proposal_status

    @classmethod
    def _listwise_active_rows(
        cls, positive: Tensor, valid: Tensor
    ) -> Tensor:
        positive = positive.bool() & valid.bool()
        effective = (
            positive.any(-1)
            & (valid & ~positive).any(-1)
        )
        return cls._row_active(effective)

    def _subtotal(
        self,
        values: dict[str, Tensor],
        active: dict[str, Tensor | bool],
    ) -> dict[str, Tensor]:
        if values.keys() != active.keys():
            raise KeyError(
                "Loss activity keys do not match loss values"
            )
        reference = next(iter(values.values()))
        numerator = reference.new_zeros(())
        denominator = reference.new_zeros(())
        for name, value in values.items():
            flag = torch.as_tensor(
                active[name],
                dtype=torch.bool,
                device=value.device,
            ).any()
            weight = float(
                self.internal_weights.get(name, 1.0)
            )
            numerator = (
                numerator
                + weight * value * flag.to(value.dtype)
            )
            denominator = (
                denominator
                + weight * flag.to(value.dtype)
            )
        return {
            "loss": numerator
            / denominator.clamp_min(
                torch.finfo(reference.dtype).eps
            ),
            **values,
        }

    def _synthesize_target_prompt(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor]:
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
            prompt_xyz = torch.where(
                prompt_valid[..., None], prompt_xyz + noise, prompt_xyz
            )
        return prompt_xyz, prompt_label, prompt_valid

    def _model_view(
        self, batch: dict[str, Any], *, include_grasp_teacher: bool = False
    ) -> dict[str, Any]:
        """Strict view: fused sensor data + task semantics + observable prompt."""
        model_inputs = {
            "xyz": batch["xyz"],
            "rgb": batch["rgb"],
            "point_mask": batch["point_mask"],
        }
        for key in (
            "source_view", "graspnet_xyz_world", "graspnet_point_mask",
            "camera2_eye_world", "camera2_target_world", "camera2_up_world",
            "camera2_valid",
        ):
            if key in batch:
                model_inputs[key] = batch[key]
        if (
            include_grasp_teacher
            and self.training
            and "graspnet_instance_id" in batch
        ):
            teacher_crop = (
                batch["graspnet_point_mask"].bool()
                & (
                    batch["graspnet_instance_id"].long()
                    == batch["target_object"][:, None].long()
                )
            )
            model_inputs["teacher_target_crop_mask"] = teacher_crop
            model_inputs["teacher_target_identity_valid"] = (
                teacher_crop.sum(-1) >= 32
            )
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
            view["task_inputs"] = {
                key: batch[key] for key in task_keys
            }
            view["task_inputs"].update({
                "target_prompt_xyz": prompt_xyz,
                "target_prompt_label": prompt_label,
                "target_prompt_valid": prompt_valid,
            })
        return view

    def _match_instances(
        self,
        output_instance: Any,
        batch: dict[str, Any],
        target_query_logits: Tensor | None,
    ):
        targets = build_instance_targets(
            batch, output_instance.mask_logits.shape[1]
        )
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
        enabled = {
            name for name in self.MODULE_OBJECTIVES if self.total.enabled(name)
        }
        push_families = {
            "push_object", "push_contact", "push_direction", "push_potential"
        }
        has_push = bool(enabled & push_families)
        has_grasp = "task_grasp" in enabled
        if has_push and has_grasp:
            forward_mode = "full"
        elif has_push:
            forward_mode = "push"
        elif has_grasp:
            forward_mode = "grasp"
        else:
            forward_mode = "perception"

        # GT contact forcing is a training-only sparse-compute aid. Validation
        # and inference must depend exclusively on predicted contact top-k.
        training_hints = (
            build_push_training_hints(batch)
            if model.training and has_push
            else None
        )
        model_view = self._model_view(
            batch, include_grasp_teacher=has_grasp
        )
        if training_hints is not None:
            model_view["training_hints"] = {
                "push_direction_point_mask": training_hints[
                    "push_direction_point_mask"
                ]
            }
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
            activity["active_loss_instance"] = (
                instance_targets["visible"].any(-1).float().mean()
            )
        elif has_push:
            # Stage C needs only the frozen GT-object -> predicted-query assignment
            # for loss-side Push object supervision. Avoid computing the full
            # instance BCE/Dice/category objective when its family is disabled.
            instance_targets = build_instance_targets(
                batch, output["instance"].mask_logits.shape[1]
            )
            match = self.instance.match(
                output["instance"],
                instance_targets,
                output["encoded"].target_query_logits,
            )
            output["instance_match"] = match

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
                self._row_active(region_labels["region_valid"])
                | self._row_active(region_labels["visibility_valid"])
            ).float().mean()

        if self.total.enabled("task_grasp"):
            if output.get("task_grasp") is None:
                raise RuntimeError("task_grasp loss enabled but Grasp forward was skipped")
            task_labels = build_grasp_proposal_labels(batch, self.model_config)
            acronym_status = self._acronym_proposal_status(output["task_grasp"], batch)
            if acronym_status is not None:
                task_labels["proposal_status"] = acronym_status
            score_losses = self.task_grasp(output["task_grasp"], task_labels)
            width_losses = self.ag_width(output["task_grasp"], task_labels)
            score_losses["task_grasp_score_loss"] = score_losses["loss"].detach()
            score_losses["ag_width_loss"] = width_losses["loss"].detach()
            score_losses["loss"] = score_losses["loss"] + width_losses["loss"]
            score_losses.update(
                {key: value for key, value in width_losses.items() if key != "loss"}
            )
            task_output = output["task_grasp"]
            score_losses.update({
                "graspnet_valid_proposals_per_row": (
                    task_output["valid"].float().sum(-1).mean().detach()
                ),
                "target_crop_points": task_output[
                    "target_crop_points"
                ].float().mean().detach(),
                "camera2_transfer_coverage": task_output[
                    "camera_transfer_coverage"
                ].float().mean().detach(),
                "target_identity_valid_rate": task_output[
                    "target_identity_valid"
                ].float().mean().detach(),
            })
            families["task_grasp"] = score_losses
            activity["active_loss_task_grasp"] = (
                score_losses["task_grasp_supervised_rows"]
                / max(1, task_output["quality_logit"].shape[0])
            )

        if has_push:
            if output.get("push") is None or match is None:
                raise RuntimeError("Push loss enabled but Push/instance match is unavailable")
            push_output, push_labels = build_object_query_push_supervision(
                output["push"],
                batch,
                self.model_config,
                match,
                training_hints=training_hints,
            )
            push_losses = self.push(push_output, push_labels)

            if self.total.enabled("push_object"):
                families["push_object"] = {
                    "loss": push_losses["push_object"],
                    "push_object_effective_rows": push_losses[
                        "push_object_effective_rows"
                    ],
                    "push_known_objects_per_state": push_losses[
                        "push_known_objects_per_state"
                    ],
                    "push_multiobject_states": push_losses["push_multiobject_states"],
                    "push_positive_objects": push_losses["push_positive_objects"],
                    "push_negative_objects": push_losses["push_negative_objects"],
                }
                activity["active_loss_push_object"] = self._listwise_active_rows(
                    push_labels["object_positive"],
                    push_labels["object_valid_mask"],
                ).float().mean()

            if self.total.enabled("push_contact"):
                families["push_contact"] = {
                    "loss": push_losses["push_contact"],
                    "push_contact_positive_points": push_losses[
                        "push_contact_positive_points"
                    ],
                    "push_contact_negative_points": push_losses[
                        "push_contact_negative_points"
                    ],
                    "push_contact_valid_points": push_losses[
                        "push_contact_valid_points"
                    ],
                }
                activity["active_loss_push_contact"] = self._row_active(
                    push_labels["contact_valid"]
                ).float().mean()

            if self.total.enabled("push_direction"):
                families["push_direction"] = {
                    "loss": push_losses["push_direction"],
                    "push_direction_bin_diagnostic": push_losses[
                        "push_direction_bin_diagnostic"
                    ],
                    "push_direction_residual_diagnostic": push_losses[
                        "push_direction_residual_diagnostic"
                    ],
                    "push_direction_effective_rows": push_losses[
                        "push_direction_effective_rows"
                    ],
                    "push_direction_residual_targets": push_losses[
                        "push_direction_residual_targets"
                    ],
                }
                direction_rank_rows = self._listwise_active_rows(
                    push_labels["direction_positive"],
                    push_labels["direction_evaluated"],
                )
                activity["active_loss_push_direction"] = (
                    direction_rank_rows
                    | self._row_active(push_labels["direction_residual_valid"])
                ).float().mean()

            if self.total.enabled("push_potential"):
                families["push_potential"] = {
                    "loss": push_losses["push_potential"],
                    "push_potential_valid_candidates": push_losses[
                        "push_potential_valid_candidates"
                    ],
                }
                activity["active_loss_push_potential"] = self._row_active(
                    push_labels["utility_valid"]
                ).float().mean()

        total, terms = self.total(families)
        terms.update(activity)
        terms["forward_mode_code"] = reference.new_tensor(
            {"perception": 0.0, "grasp": 1.0, "push": 2.0, "full": 3.0}[
                forward_mode
            ]
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
