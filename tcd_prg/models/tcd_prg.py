"""Complete shared-feature TCD-PRG model assembly."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.config import AblationConfig, BackboneConfig, GraphConfig, ModelConfig, RouterConfig

from .backbones import TaskConditionedPointTransformer
from .common import ActionCandidateEncoder
from .dependency_graph import TaskConditionedDependencyGraph
from .grasp_proposal import GlobalGraspProposalHead, TaskGraspProposalHead
from .grasp_verifier import GripperSceneTaskVerifier
from .policy import FlatCandidateClassifier, MaskedHierarchicalCandidateRouter, fixed_priority_output
from .push import PushHead
from .region import TaskRegionHead


class TCDPRGModel(nn.Module):
    """All learned modules; the global point backbone is evaluated exactly once."""

    def __init__(self, config: ModelConfig | None = None,
                 ablation: AblationConfig | None = None,
                 graph_config: GraphConfig | None = None,
                 router_config: RouterConfig | None = None,
                 backbone_config: BackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.ablation = ablation or AblationConfig()
        graph_config = graph_config or GraphConfig()
        router_config = router_config or RouterConfig()
        backbone_config = backbone_config or BackboneConfig()
        c = self.config
        self.encoder = TaskConditionedPointTransformer(
            dim=c.feature_dim,
            task_dim=c.task_dim,
            num_categories=c.num_categories,
            num_regions=c.num_task_regions,
            attention_points=backbone_config.attention_points,
            activation_checkpointing=c.activation_checkpointing,
        )
        self.region_head = TaskRegionHead(c.feature_dim)
        self.task_grasp = TaskGraspProposalHead(c.feature_dim, c.task_grasp_candidates)
        self.global_grasp = GlobalGraspProposalHead(
            c.feature_dim, c.global_grasp_candidates, c.global_grasp_input_mode,
        )
        self.verifier = GripperSceneTaskVerifier(c.feature_dim, c.feature_dim)
        self.graph = TaskConditionedDependencyGraph(
            c.feature_dim, physical_relations=len(graph_config.physical_relations),
            task_relations=len(graph_config.task_relations), layers=graph_config.layers,
            heads=graph_config.heads, edge_threshold=c.graph_edge_threshold,
        )
        self.push = PushHead(c.feature_dim, c.num_direction_bins)
        self.router = MaskedHierarchicalCandidateRouter(
            c.feature_dim, layers=router_config.layers, heads=router_config.heads
        )
        self.flat_router = FlatCandidateClassifier(
            c.feature_dim, layers=router_config.layers, heads=router_config.heads
        )
        self.candidate_encoder = ActionCandidateEncoder(c.feature_dim)
        self.candidate_evidence = nn.Sequential(
            nn.Linear(7, c.feature_dim), nn.GELU(), nn.Linear(c.feature_dim, c.feature_dim)
        )

    def _candidate_evidence_from_batch(
        self, batch: dict[str, Tensor], result: dict[str, Any], kind: Tensor,
        acted_object: Tensor, pose: Tensor
    ) -> Tensor:
        """Gather proposal/PUSH evidence for labelled training candidates."""

        b, k = kind.shape
        evidence = torch.zeros((b, k, 7), device=kind.device)
        parameters = batch["action_parameters"]
        for row in range(b):
            for candidate in torch.nonzero(batch["candidate_mask"][row], as_tuple=False).flatten().tolist():
                is_push = int(kind[row, candidate]) == 0
                is_remove = int(kind[row, candidate]) == 1
                if is_remove and not bool(parameters.get(
                    "removal_global_match_valid", torch.zeros_like(kind, dtype=torch.bool)
                )[row, candidate]):
                    continue
                query = parameters["push_contact_world"][row, candidate] if is_push else pose[row, candidate, :3]
                if not torch.isfinite(query).all():
                    continue
                mask = batch["point_mask"][row] & (
                    batch["instance_id"][row] == acted_object[row, candidate]
                )
                points = torch.nonzero(mask, as_tuple=False).flatten()
                if not len(points):
                    continue
                if not is_push and "grasp_contact_points_world" in parameters:
                    contacts = parameters["grasp_contact_points_world"][row, candidate]
                    if torch.isfinite(contacts).all():
                        contact_distance = torch.cdist(contacts, batch["xyz"][row, points])
                        query = contacts[contact_distance.min(-1).values.argmin()]
                delta = batch["xyz"][row, points] - query
                point = points[(delta * delta).sum(-1).argmin()]
                if is_push:
                    direction = parameters["push_direction_world"][row, candidate]
                    angle = torch.atan2(direction[1], direction[0]).remainder(2 * torch.pi)
                    direction_bin = torch.floor(
                        angle * self.config.num_direction_bins / (2 * torch.pi)
                    ).long().remainder(self.config.num_direction_bins)
                    evidence[row, candidate, 0] = result["push"]["utility_delta"][
                        row, point, direction_bin
                    ]
                    evidence[row, candidate, 1] = torch.sigmoid(
                        result["push"]["contact_logits"][row, point]
                    )
                    evidence[row, candidate, 3] = torch.softmax(
                        result["push"]["direction_logits"][row, point], dim=-1
                    )[direction_bin]
                else:
                    head = result["global_grasp"] if is_remove else result["task_grasp"]
                    nearest = torch.linalg.vector_norm(
                        head["translation_world"][row] - pose[row, candidate, :3], dim=-1
                    ).argmin()
                    evidence[row, candidate, 1] = torch.sigmoid(head["quality_logit"][row, nearest])
        return evidence

    def verify_cached(
        self, batch: dict[str, Tensor], result: dict[str, Any], verifier_inputs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Verify candidates from cached shared features without a second backbone pass."""

        encoded, region = result["encoded"], result["region"]
        point_index = verifier_inputs["scene_point_index"]
        batch_rows = torch.arange(point_index.shape[0], device=point_index.device)[:, None, None]
        verifier_features = encoded.point_features[batch_rows, point_index]
        verifier_target = batch["target_mask"][batch_rows, point_index]
        verifier_region = region["region_probability"][batch_rows, point_index]
        outputs: dict[str, list[Tensor]] = {}
        micro = self.config.verifier_candidate_micro_batch
        for start in range(0, point_index.shape[1], micro):
            stop = min(point_index.shape[1], start + micro)
            chunk = self.verifier(
                verifier_inputs["scene_xyz_grasp"][:, start:stop],
                verifier_inputs["gripper_xyz_grasp"][:, start:stop],
                verifier_features[:, start:stop],
                verifier_target[:, start:stop],
                verifier_region[:, start:stop],
                encoded.task_token,
                verifier_inputs["scene_valid"][:, start:stop],
                verifier_inputs["gripper_valid"][:, start:stop],
            )
            for key, value in chunk.items():
                outputs.setdefault(key, []).append(value)
        return {key: torch.cat(values, dim=1) for key, values in outputs.items()}

    def route_cached(
        self, batch: dict[str, Tensor], result: dict[str, Any], candidate_inputs: dict[str, Any]
    ) -> Any:
        """Route externally generated candidates using the already encoded scene."""

        encoded = result["encoded"]
        evidence = candidate_inputs.get("evidence")
        if evidence is None:
            evidence = torch.zeros(
                candidate_inputs["tokens"].shape[:2] + (7,),
                device=candidate_inputs["tokens"].device,
            )
        else:
            evidence = evidence.clone()
        if (
            result.get("verifier") is not None
            and not candidate_inputs.get("verifier_evidence_cached", False)
        ):
            evidence[..., 2] = torch.sigmoid(result["verifier"]["overall_logit"])
        routed_tokens = candidate_inputs["tokens"] + self.candidate_evidence(evidence)
        router_args = (
            encoded.task_token,
            encoded.global_scene_token,
            encoded.object_tokens,
            encoded.object_mask & batch["object_active"],
            routed_tokens,
            candidate_inputs["type"],
            candidate_inputs["object"],
            candidate_inputs["valid"],
            batch["remaining_steps"],
            candidate_inputs.get("previous_action"),
        )
        if self.ablation.router_type == "hierarchical":
            return self.router(*router_args)
        if self.ablation.router_type == "flat_candidate_classifier":
            return self.flat_router(*router_args)
        if self.ablation.router_type == "fixed_priority":
            return fixed_priority_output(
                candidate_inputs["type"], candidate_inputs["object"],
                candidate_inputs["valid"], encoded.object_mask & batch["object_active"],
            )
        raise ValueError(f"Unsupported router type {self.ablation.router_type}")

    def _external_candidate_inputs(
        self, encoded: Any, candidates: dict[str, Tensor]
    ) -> dict[str, Any]:
        flags = torch.stack((
            torch.isfinite(candidates["contact_world"]).all(-1),
            torch.isfinite(candidates["direction_world"]).all(-1),
            torch.isfinite(candidates["pose_world"]).all(-1),
            torch.isfinite(candidates["destination_world"]).all(-1),
            torch.isfinite(candidates["width_m"]),
        ), -1)
        return {
            **candidates,
            "verifier_evidence_cached": True,
            "tokens": self.candidate_encoder(
                encoded.object_tokens, candidates["type"], candidates["object"],
                candidates["contact_world"], candidates["direction_world"],
                candidates["pose_world"], candidates["destination_world"],
                flags, encoded.task_token,
            ),
        }

    def forward(self, batch: dict[str, Tensor], candidate_inputs: dict[str, Tensor] | None = None) -> dict[str, Any]:
        object_present = batch.get("object_present", batch["object_mask"])
        physical_active = object_present & batch["object_active"]
        encoded = self.encoder(
            batch["xyz"],
            batch["rgb"],
            batch["instance_id"],
            batch["point_mask"],
            batch["target_mask"],
            physical_active,
            batch["task_category_id"],
            batch["task_region_id"],
            use_task_region_condition=self.ablation.use_task_region_condition,
            target_object=batch["target_object"],
        )
        region = self.region_head(
            encoded.point_features, encoded.target_token, encoded.task_token, batch["target_mask"]
        )
        task_grasp = self.task_grasp(
            encoded.point_features,
            batch["xyz"],
            encoded.target_token,
            encoded.task_token,
            region["region_probability"],
            batch["target_mask"],
        )
        task_grasp["width_m"] = self.task_grasp.decode_width(
            task_grasp["width_raw"], self.config.min_grasp_width_m, self.config.max_grasp_width_m
        )
        if self.config.global_grasp_input_mode == "scene_only":
            global_mask = batch["point_mask"]
        else:
            valid_instance = (
                (batch["instance_id"] >= 0)
                & (batch["instance_id"] < physical_active.shape[1])
            )
            global_mask = (
                batch["point_mask"] & valid_instance
                & physical_active.gather(
                    1, batch["instance_id"].clamp(0, physical_active.shape[1] - 1)
                )
            )
        global_grasp = self.global_grasp(
            encoded.scene_point_features,
            batch["xyz"],
            encoded.scene_object_tokens,
            encoded.scene_global_token,
            batch["instance_id"],
            global_mask,
            physical_active,
        )
        global_grasp["width_m"] = self.global_grasp.decode_width(
            global_grasp["width_raw"], self.config.min_grasp_width_m, self.config.max_grasp_width_m
        )
        if self.ablation.use_dependency_graph:
            graph = self.graph(
                encoded.object_tokens,
                encoded.object_mask & batch["object_active"],
                encoded.task_token,
                batch["target_object"],
                batch.get("relation_graph"),
                self.ablation.use_indirect_dependency_reasoning,
            )
            graph_context = graph.node_features[:, :-1]
        else:
            graph = None
            graph_context = encoded.object_tokens
        push = self.push(
            encoded.point_features,
            batch["xyz"],
            batch["instance_id"],
            batch["point_mask"],
            encoded.object_tokens,
            encoded.object_mask & batch["object_active"],
            encoded.task_token,
            encoded.target_token,
            graph_context,
            batch["remaining_steps"],
        )
        result: dict[str, Any] = {
            "encoded": encoded,
            "region": region,
            "task_grasp": task_grasp,
            "global_grasp": global_grasp,
            "graph": graph,
            "push": push,
            "verifier": None,
        }
        verifier_inputs = batch.get("verifier_inputs")
        if verifier_inputs is not None and self.ablation.use_gripper_scene_verifier:
            result["verifier"] = self.verify_cached(batch, result, verifier_inputs)
        if candidate_inputs is None and "action_parameters" in batch:
            parameters = batch["action_parameters"]
            kind = batch["action_type"]
            push = kind == 0
            remove_mask = kind == 1
            pose = torch.where(
                remove_mask.unsqueeze(-1), parameters["removal_grasp_pose_world"],
                parameters["task_grasp_pose_world"]
            )
            flags = torch.stack(
                (
                    torch.isfinite(parameters["push_contact_world"]).all(-1),
                    torch.isfinite(parameters["push_direction_world"]).all(-1),
                    torch.isfinite(pose).all(-1),
                    torch.isfinite(parameters["removal_destination_world"]).all(-1),
                    torch.isfinite(parameters["grasp_width_m"]),
                ),
                -1,
            )
            candidate_inputs = {
                "tokens": self.candidate_encoder(
                    encoded.object_tokens, kind, batch["acted_object"],
                    parameters["push_contact_world"], parameters["push_direction_world"], pose,
                    parameters["removal_destination_world"], flags, encoded.task_token
                ),
                "type": kind,
                "object": batch["acted_object"],
                "valid": batch["candidate_mask"],
            }
            candidate_inputs["evidence"] = self._candidate_evidence_from_batch(
                batch, result, kind, batch["acted_object"], pose
            )
        if candidate_inputs is not None:
            result["router"] = self.route_cached(batch, result, candidate_inputs)
        generated = batch.get("generated_policy_candidates")
        if generated is not None:
            generated_inputs = self._external_candidate_inputs(encoded, generated)
            result["generated_router"] = self.route_cached(
                batch, result, generated_inputs
            )
        return result
