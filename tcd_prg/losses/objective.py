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
from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.datasets.capabilities import DatasetCapabilities
from tcd_prg.datasets.acronym_grasp_database import (
    load_object_grasps, match_object_grasp_priors,
)
from tcd_prg.geometry.se3 import quaternion_xyzw_to_matrix

from .actions import PushLoss
from .ag_width import AGWidthLoss
from .graph import DependencyGraphLoss
from .instance import (
    InstanceSetLoss,
    build_instance_targets,
    build_object_query_push_supervision,
    remap_graph_targets,
)
from .labels import (
    build_grasp_proposal_labels,
    build_region_labels,
    build_verifier_labels,
)
from .policy import HierarchicalSetPolicyLoss
from .task_grasp_score import TaskGraspScoringLoss
from .region import TaskRegionLoss
from .total import MultiTaskLoss
from .verifier import GraspVerifierLoss


class TCDPRGObjective(nn.Module):
    """GT is consumed here; perception forward consumes sensor/task only."""

    MODULE_OBJECTIVES = (
        "instance",
        "region",
        "task_grasp",
        "physical_edge",
        "task_edge",
        "verify_overall",
        "push_object",
        "push_contact",
        "push_direction",
        "push_potential",
        "policy_candidate",
    )

    def __init__(
        self,
        capabilities: DatasetCapabilities,
        model_config: ModelConfig,
        ablation: AblationConfig,
        loss_config: LossConfig | None = None,
        region_config: RegionHeadConfig | None = None,
        generated_policy_candidate_ratio: float = 0.0,
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
            translation_m=model_config.policy_grasp_match_translation_m,
            rotation_deg=model_config.policy_grasp_match_rotation_deg,
            width_m=model_config.policy_grasp_match_width_m,
        )
        self.ag_width = AGWidthLoss(
            translation_m=model_config.policy_grasp_match_translation_m,
            rotation_deg=model_config.policy_grasp_match_rotation_deg,
        )
        self.graph = DependencyGraphLoss()
        self.push = PushLoss()
        self.verify = GraspVerifierLoss()
        self.policy = HierarchicalSetPolicyLoss()
        self.generated_policy_candidate_ratio = float(
            generated_policy_candidate_ratio
        )
        self.acronym_database_root = (
            Path(acronym_object_grasp_database)
            if acronym_object_grasp_database else None
        )
        self._acronym_paths: dict[str, Path] | None = None
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

    def _acronym_path_map(self) -> dict[str, Path]:
        if self.acronym_database_root is None:
            return {}
        if self._acronym_paths is None:
            manifest = json.loads(
                (self.acronym_database_root / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self._acronym_paths = {
                str(record["model_id"]): Path(record["path"])
                for record in manifest["records"]
            }
        return self._acronym_paths

    def _acronym_tensors(
        self, model_id: str, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        key = (model_id, str(device))
        if key not in self._acronym_cache:
            database = load_object_grasps(self._acronym_path_map()[model_id])
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
                *self._acronym_tensors(str(model_id), translation_object.device),
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

    def _model_view(self, batch: dict[str, Any]) -> dict[str, Any]:
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
        if self.training and "graspnet_instance_id" in batch:
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
        task_keys = (
            "task_category_id", "task_region_id",
            "remaining_steps", "required_grasp_count",
        )
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
        generated = batch.get("generated_policy_candidates")
        if generated is not None:
            allowed = {
                "type", "object", "contact_world", "direction_world",
                "pose_world", "destination_world", "width_m", "valid",
                "previous_action", "evidence",
            }
            view["generated_policy_candidates"] = {
                key: value
                for key, value in generated.items()
                if key in allowed
            }
        return view

    @staticmethod
    def _teacher_candidates(
        batch: dict[str, Any]
    ) -> dict[str, Tensor]:
        """Geometry-only teacher candidate set for router/verifier training.

        These tensors are supervision-side candidate geometry, not perception
        observations.  `acted_object` and all candidate outcome labels are
        intentionally excluded; model assigns each candidate to a predicted
        object query from geometry.
        """
        parameters = batch["action_parameters"]
        kind = batch["action_type"]
        remove = kind == int(ActionType.PICK_REMOVE)
        pose = torch.where(
            remove.unsqueeze(-1),
            parameters["removal_grasp_pose_world"],
            parameters["task_grasp_pose_world"],
        )
        return {
            "type": kind,
            "valid": batch["candidate_mask"].bool(),
            "contact_world": parameters[
                "push_contact_world"
            ],
            "direction_world": parameters[
                "push_direction_world"
            ],
            "pose_world": pose,
            "destination_world": parameters[
                "removal_destination_world"
            ],
            "width_m": parameters["grasp_width_m"],
        }

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
        required = batch.get(
            "required_grasp_count",
            batch.get("task_inputs", {}).get(
                "required_grasp_count"
            ),
        )
        if required is not None and bool(
            (
                required
                > self.model_config.task_grasp_candidates
            ).any()
        ):
            maximum = int(required.max())
            raise ValueError(
                f"Batch requires {maximum} unique grasps but "
                f"task_grasp_candidates="
                f"{self.model_config.task_grasp_candidates}"
            )

        perception_only = (
            self.total.enabled("instance")
            and not any(
                self.total.enabled(name)
                for name in self.MODULE_OBJECTIVES
                if name != "instance"
            )
        )

        generated_only = (
            self.generated_policy_candidate_ratio == 1.0
            and self.total.enabled("policy_candidate")
            and not any(
                self.total.enabled(name)
                for name in self.MODULE_OBJECTIVES
                if name != "policy_candidate"
            )
        )

        teacher_needed = (
            not generated_only
            and self.total.enabled("policy_candidate")
            and self.generated_policy_candidate_ratio < 1.0
            and self.ablation.router_type
            != "fixed_priority"
        )
        teacher_candidates = (
            self._teacher_candidates(batch)
            if teacher_needed
            else None
        )

        model_view = self._model_view(batch)
        output = model(
            model_view,
            candidate_inputs=teacher_candidates,
            forward_mode=(
                "generated_policy"
                if generated_only
                else ("perception" if perception_only else "full")
            ),
        )
        families: dict[str, dict[str, Tensor]] = {}
        reference = batch["xyz"]
        activity: dict[str, Tensor] = {
            f"active_loss_{name}": reference.new_zeros(())
            for name in self.MODULE_OBJECTIVES
            if self.total.enabled(name)
        }

        if generated_only:
            # No upstream heads exist in this stage. Instance matching is not
            # needed because the generated cache already uses predicted query ids.
            match = None
        else:
            instance_values, match, instance_targets = (
                self._match_instances(
                    output["instance"],
                    batch,
                    output["encoded"].target_query_logits,
                )
            )
            output["instance_match"] = match
            if self.total.enabled("instance"):
                families["instance"] = instance_values
                activity["active_loss_instance"] = (
                    instance_targets["visible"]
                    .any(-1)
                    .float()
                    .mean()
                )

            region_labels = build_region_labels(batch)
            if (
                region_labels is not None
                and self.total.enabled("region")
            ):
                region_losses = self.region(
                    output["region"], region_labels
                )
                families["region"] = self._subtotal(
                    region_losses,
                    {
                        "region_focal": (
                            region_labels[
                                "region_valid"
                            ].any()
                        ),
                        "region_dice": (
                            region_labels[
                                "region_valid"
                            ].any()
                        ),
                        "region_visibility": (
                            region_labels[
                                "visibility_valid"
                            ].any()
                        ),
                    },
                )
                activity["active_loss_region"] = (
                    (
                        self._row_active(
                            region_labels["region_valid"]
                        )
                        | self._row_active(
                            region_labels[
                                "visibility_valid"
                            ]
                        )
                    )
                    .float()
                    .mean()
                )

            if self.total.enabled("task_grasp"):
                task_labels = build_grasp_proposal_labels(
                    batch, self.model_config
                )
                acronym_status = self._acronym_proposal_status(
                    output["task_grasp"], batch
                )
                if acronym_status is not None:
                    task_labels["proposal_status"] = acronym_status
                # Run even when the sparse label set has no match. This keeps
                # crop/proposal health metrics present on every task batch while
                # the loss remains a connected zero for UNKNOWN-only rows.
                score_losses = self.task_grasp(
                    output["task_grasp"], task_labels
                )
                width_losses = self.ag_width(output["task_grasp"], task_labels)
                score_losses["task_grasp_score_loss"] = score_losses[
                    "loss"
                ].detach()
                score_losses["ag_width_loss"] = width_losses["loss"].detach()
                score_losses["loss"] = score_losses["loss"] + width_losses["loss"]
                score_losses.update({
                    key: value
                    for key, value in width_losses.items()
                    if key != "loss"
                })
                task_output = output["task_grasp"]
                score_losses["graspnet_valid_proposals_per_row"] = (
                    task_output["valid"].float().sum(-1).mean().detach()
                )
                score_losses["target_crop_points"] = task_output[
                    "target_crop_points"
                ].float().mean().detach()
                score_losses["camera2_transfer_coverage"] = task_output[
                    "camera_transfer_coverage"
                ].float().mean().detach()
                score_losses["target_identity_valid_rate"] = task_output[
                    "target_identity_valid"
                ].float().mean().detach()
                score_losses["target_identity_teacher_forced_rate"] = task_output.get(
                    "target_identity_teacher_forced",
                    torch.zeros_like(task_output["target_identity_valid"]),
                ).float().mean().detach()
                if "target_mask" in batch:
                    hard_target = task_output["target_hard_mask_fused"].bool()
                    target_gt = batch["target_mask"].bool() & batch[
                        "point_mask"
                    ].bool()
                    intersection = (hard_target & target_gt).float().sum()
                    score_losses["hard_target_mask_purity_fused"] = (
                        intersection / hard_target.float().sum().clamp_min(1.0)
                    ).detach()
                    score_losses["hard_target_mask_recall_fused"] = (
                        intersection / target_gt.float().sum().clamp_min(1.0)
                    ).detach()
                if "graspnet_instance_id" in batch:
                    crop = task_output["target_crop_mask"].bool()
                    instance = batch["graspnet_instance_id"]
                    target_object = batch["target_object"][:, None]
                    score_losses["target_crop_purity"] = (
                        ((instance == target_object) & crop).float().sum()
                        / crop.float().sum().clamp_min(1.0)
                    ).detach()
                    attention = task_output["graspnet_attention_point_index"]
                    rows = torch.arange(
                        attention.shape[0], device=attention.device
                    )[:, None]
                    proposal_target = (
                        instance[rows, attention] == target_object
                    ) & task_output["valid"].bool()
                    score_losses["target_proposal_ratio"] = (
                        proposal_target.float().sum()
                        / task_output["valid"].float().sum().clamp_min(1.0)
                    ).detach()
                    transfer_valid = task_output[
                        "camera_transfer_valid"
                    ].bool()
                    reference = task_output[
                        "camera_transfer_reference_index"
                    ]
                    transfer_rows = torch.arange(
                        reference.shape[0], device=reference.device
                    )[:, None]
                    same_instance = (
                        batch["instance_id"][transfer_rows, reference]
                        == instance
                    ) & transfer_valid
                    score_losses["camera2_mask_transfer_accuracy"] = (
                        same_instance.float().sum()
                        / transfer_valid.float().sum().clamp_min(1.0)
                    ).detach()
                families["task_grasp"] = score_losses
                supervised_rows = score_losses[
                    "task_grasp_supervised_rows"
                ]
                activity["active_loss_task_grasp"] = (
                    supervised_rows
                    / max(1, output["task_grasp"]["quality_logit"].shape[0])
                )

            if (
                output["graph"] is not None
                and self.total.enabled(
                    "physical_edge"
                )
            ):
                graph_labels = remap_graph_targets(
                    batch["relation_graph"],
                    batch["task_block_graph"],
                    match,
                    output["encoded"].object_tokens.shape[1],
                )
                output["graph_labels_aligned"] = graph_labels
                graph_losses = self.graph(
                    output["graph"], graph_labels
                )
                families["physical_edge"] = {
                    "loss": graph_losses["physical_edge"]
                }
                families["task_edge"] = {
                    "loss": graph_losses["task_edge"]
                }
                activity[
                    "active_loss_physical_edge"
                ] = (
                    graph_labels["physical_edge_valid"]
                    .reshape(
                        graph_labels[
                            "physical_edge_valid"
                        ].shape[0],
                        -1,
                    )
                    .any(-1)
                    .float()
                    .mean()
                )
                activity["active_loss_task_edge"] = (
                    self._row_active(
                        graph_labels["task_edge_valid"]
                    )
                    .float()
                    .mean()
                )

            # Verifier candidate geometry is created by the collator from labels,
            # but target/region evidence inside the model is prediction-only.
            if (
                self.total.enabled("verify_overall")
                and self.ablation.use_gripper_scene_verifier
                and "verifier_inputs" in batch
            ):
                output["verifier"] = model.verify_cached(
                    model_view, output, batch["verifier_inputs"]
                )
                if (
                    teacher_needed
                    and "candidate_inputs" in output
                ):
                    # Re-route so teacher policy sees learned verifier evidence.
                    output["router"] = model.route_cached(
                        model_view,
                        output,
                        output["candidate_inputs"],
                    )
                verifier_labels = build_verifier_labels(
                    batch
                )
                families["verify_overall"] = self.verify(
                    output["verifier"],
                    verifier_labels,
                )
                activity[
                    "active_loss_verify_overall"
                ] = (
                    verifier_labels["overall_valid"]
                    .reshape(
                        verifier_labels[
                            "overall_valid"
                        ].shape[0],
                        -1,
                    )
                    .any(-1)
                    .float()
                    .mean()
                )

            if self.total.enabled("push_object"):
                push_output, push_labels = (
                    build_object_query_push_supervision(
                        output["push"],
                        batch,
                        self.model_config,
                        match,
                    )
                )
                push_losses = self.push(
                    push_output, push_labels
                )
                families["push_object"] = {
                    "loss": push_losses["push_object"],
                    "push_object_effective_rows": (
                        push_losses[
                            "push_object_effective_rows"
                        ]
                    ),
                    "push_known_objects_per_state": push_losses[
                        "push_known_objects_per_state"
                    ],
                    "push_multiobject_states": push_losses[
                        "push_multiobject_states"
                    ],
                    "push_positive_objects": push_losses[
                        "push_positive_objects"
                    ],
                    "push_negative_objects": push_losses[
                        "push_negative_objects"
                    ],
                }
                families["push_contact"] = {
                    "loss": push_losses["push_contact"],
                    "push_contact_positive_points": (
                        push_losses[
                            "push_contact_positive_points"
                        ]
                    ),
                    "push_contact_negative_points": (
                        push_losses[
                            "push_contact_negative_points"
                        ]
                    ),
                    "push_contact_valid_points": (
                        push_losses[
                            "push_contact_valid_points"
                        ]
                    ),
                }
                families["push_direction"] = {
                    "loss": push_losses[
                        "push_direction"
                    ],
                    "push_direction_bin_diagnostic": (
                        push_losses[
                            "push_direction_bin_diagnostic"
                        ]
                    ),
                    "push_direction_residual_diagnostic": (
                        push_losses[
                            "push_direction_residual_diagnostic"
                        ]
                    ),
                    "push_direction_effective_rows": (
                        push_losses[
                            "push_direction_effective_rows"
                        ]
                    ),
                    "push_direction_residual_targets": (
                        push_losses[
                            "push_direction_residual_targets"
                        ]
                    ),
                }
                if self.total.enabled(
                    "push_potential"
                ):
                    families["push_potential"] = {
                        "loss": push_losses[
                            "push_potential"
                        ],
                        "push_potential_valid_candidates": (
                            push_losses[
                                "push_potential_valid_candidates"
                            ]
                        ),
                    }

                object_active_rows = (
                    self._listwise_active_rows(
                        push_labels[
                            "object_positive"
                        ],
                        push_labels[
                            "object_valid_mask"
                        ],
                    )
                )
                direction_rank_rows = (
                    self._listwise_active_rows(
                        push_labels[
                            "direction_positive"
                        ],
                        push_labels[
                            "direction_evaluated"
                        ],
                    )
                )
                activity.update({
                    "active_loss_push_object": (
                        object_active_rows
                        .float()
                        .mean()
                    ),
                    "active_loss_push_contact": (
                        self._row_active(
                            push_labels[
                                "contact_valid"
                            ]
                        )
                        .float()
                        .mean()
                    ),
                    "active_loss_push_direction": (
                        (
                            direction_rank_rows
                            | self._row_active(
                                push_labels[
                                    "direction_residual_valid"
                                ]
                            )
                        )
                        .float()
                        .mean()
                    ),
                })
                if self.total.enabled(
                    "push_potential"
                ):
                    activity[
                        "active_loss_push_potential"
                    ] = (
                        self._row_active(
                            push_labels[
                                "utility_valid"
                            ]
                        )
                        .float()
                        .mean()
                    )

        if (
            self.total.enabled("policy_candidate")
            and self.ablation.router_type
            != "fixed_priority"
        ):
            teacher_loss = None
            teacher_active = None
            if "router" in output:
                evaluated = (
                    batch["policy_success_mask"]
                    | (
                        batch["evaluation_status"]
                        == int(CandidateStatus.NEGATIVE)
                    )
                )
                teacher_loss = self.policy(
                    output["router"],
                    batch["policy_success_mask"],
                    evaluated,
                )
                teacher_active = (
                    self._listwise_active_rows(
                        batch["policy_success_mask"],
                        evaluated,
                    )
                )

            generated_loss = None
            generated_active = None
            generated = batch.get(
                "generated_policy_candidates"
            )
            if (
                "generated_router" in output
                and generated is not None
            ):
                generated_evaluated = (
                    generated["label_status"]
                    != int(
                        CandidateStatus.UNKNOWN_UNTESTED
                    )
                )
                generated_loss = self.policy(
                    output["generated_router"],
                    generated["policy_success"],
                    generated_evaluated,
                )
                generated_active = (
                    self._listwise_active_rows(
                        generated[
                            "policy_success"
                        ],
                        generated_evaluated,
                    )
                )

            ratio = (
                self.generated_policy_candidate_ratio
            )
            if (
                generated_loss is not None
                and teacher_loss is not None
            ):
                policy_loss = (
                    (1.0 - ratio) * teacher_loss
                    + ratio * generated_loss
                )
            elif generated_loss is not None:
                policy_loss = generated_loss
            elif (
                teacher_loss is not None
                and ratio == 0.0
            ):
                policy_loss = teacher_loss
            else:
                raise RuntimeError(
                    "Generated policy training is enabled "
                    "but the batch has no generated candidates"
                )

            families["policy_candidate"] = {
                "loss": policy_loss
            }
            policy_active = torch.zeros(
                reference.shape[0],
                dtype=torch.bool,
                device=policy_loss.device,
            )
            if (
                teacher_active is not None
                and ratio < 1.0
            ):
                policy_active |= teacher_active
            if (
                generated_active is not None
                and ratio > 0.0
            ):
                policy_active |= generated_active
            families["policy_candidate"][
                "policy_effective_rows"
            ] = policy_active.sum().float()
            activity[
                "active_loss_policy_candidate"
            ] = policy_active.float().mean()

        total, terms = self.total(families)
        terms.update(activity)

        weighted_family_losses = {
            name: float(self.total.weights[name])
            * values["loss"]
            for name, values in families.items()
            if self.total.enabled(name)
        }

        generated = batch.get(
            "generated_policy_candidates"
        )
        if generated is not None:
            valid = generated["valid"]
            positive = (
                valid & generated["policy_success"]
            )
            negative = (
                valid
                & (
                    generated["label_status"]
                    == int(CandidateStatus.NEGATIVE)
                )
            )
            positive_rows = positive.any(-1)
            negative_rows = negative.any(-1)
            terms.update({
                "generated_states": valid.new_tensor(
                    valid.shape[0],
                    dtype=torch.float32,
                ),
                "generated_states_with_positive": (
                    positive_rows.sum().float()
                ),
                "generated_effective_policy_rows": (
                    (positive_rows & negative_rows)
                    .sum()
                    .float()
                ),
                "generated_known_candidates": (
                    (positive | negative).sum().float()
                ),
                "generated_unknown_candidates": (
                    (
                        valid
                        & ~positive
                        & ~negative
                    )
                    .sum()
                    .float()
                ),
                "generated_conflict_candidates": (
                    generated.get(
                        "match_conflict",
                        torch.zeros_like(valid),
                    )
                    .sum()
                    .float()
                ),
            })

        if (
            return_output
            and return_family_losses
        ):
            return (
                total,
                terms,
                output,
                weighted_family_losses,
            )
        if return_output:
            return total, terms, output
        if return_family_losses:
            return (
                total,
                terms,
                weighted_family_losses,
            )
        return total, terms
