"""Complete shared-feature TCD-PRG model assembly."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.config import AblationConfig, BackboneConfig, GraphConfig, ModelConfig, RouterConfig

from .backbones import TaskConditionedPointTransformer
from .common import ActionCandidateEncoder
from .dependency_graph import TaskConditionedDependencyGraph
from .grasp_proposal import StateGraspabilityHead, TaskGraspProposalHead
from .grasp_verifier import GripperSceneTaskVerifier
from .pick_remove import PickRemoveHead
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
        self.task_grasp = TaskGraspProposalHead(c.feature_dim, c.num_grasp_rotation_bins)
        self.generic_grasp = TaskGraspProposalHead(c.feature_dim, c.num_grasp_rotation_bins)
        self.state_graspability = StateGraspabilityHead(c.feature_dim)
        self.verifier = GripperSceneTaskVerifier(c.feature_dim, c.feature_dim)
        self.graph = TaskConditionedDependencyGraph(
            c.feature_dim, physical_relations=len(graph_config.physical_relations),
            task_relations=len(graph_config.task_relations), layers=graph_config.layers,
            heads=graph_config.heads, edge_threshold=c.graph_edge_threshold,
        )
        self.pick_remove = PickRemoveHead(c.feature_dim)
        self.push = PushHead(c.feature_dim, c.num_direction_bins)
        self.router = MaskedHierarchicalCandidateRouter(
            c.feature_dim, layers=router_config.layers, heads=router_config.heads
        )
        self.flat_router = FlatCandidateClassifier(
            c.feature_dim, layers=router_config.layers, heads=router_config.heads
        )
        self.candidate_encoder = ActionCandidateEncoder(c.feature_dim)
        self.candidate_evidence = nn.Sequential(
            nn.Linear(22, c.feature_dim), nn.GELU(), nn.Linear(c.feature_dim, c.feature_dim)
        )

    def _candidate_evidence_from_batch(
        self, batch: dict[str, Tensor], result: dict[str, Any], kind: Tensor,
        acted_object: Tensor, pose: Tensor
    ) -> Tensor:
        """Gather proposal/PUSH evidence for labelled training candidates."""

        b, k = kind.shape
        evidence = torch.zeros((b, k, 22), device=kind.device)
        parameters = batch["action_parameters"]
        for row in range(b):
            for candidate in torch.nonzero(batch["candidate_mask"][row], as_tuple=False).flatten().tolist():
                is_push = int(kind[row, candidate]) == 0
                query = (
                    parameters["push_contact_world"][row, candidate]
                    if is_push else pose[row, candidate, :3]
                )
                if not torch.isfinite(query).all():
                    continue
                mask = batch["point_mask"][row] & (
                    batch["instance_id"][row] == acted_object[row, candidate]
                )
                points = torch.nonzero(mask, as_tuple=False).flatten()
                if not len(points):
                    continue
                delta = batch["xyz"][row, points] - query
                point = points[(delta * delta).sum(-1).argmin()]
                if is_push:
                    evidence[row, candidate, 7:12] = result["push"]["potential_delta"][row, point]
                    evidence[row, candidate, 12:15] = torch.sigmoid(
                        result["push"]["risk_logits"][row, point]
                    )
                    evidence[row, candidate, 15] = torch.sigmoid(
                        result["push"]["contact_logits"][row, point]
                    )
                else:
                    head = result["generic_grasp"] if int(kind[row, candidate]) == 1 else result["task_grasp"]
                    if "proposal_confidence_logit" in head:
                        evidence[row, candidate, 15] = torch.sigmoid(
                            head["proposal_confidence_logit"][row, point]
                        )
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
        self, batch: dict[str, Tensor], result: dict[str, Any], candidate_inputs: dict[str, Tensor]
    ) -> Any:
        """Route externally generated candidates using the already encoded scene."""

        encoded = result["encoded"]
        graph_context = (
            result["graph"].node_features[:, :-1]
            if result["graph"] is not None
            else encoded.object_tokens
        )
        remove_candidates = candidate_inputs["valid"] & (candidate_inputs["type"] == 1)
        preliminary_remove = self.pick_remove(
            encoded.object_tokens,
            encoded.object_mask & batch["object_active"],
            encoded.task_token,
            graph_context,
            candidate_inputs["tokens"],
            candidate_inputs["object"],
            remove_candidates,
        )
        evidence = candidate_inputs.get("evidence")
        if evidence is None:
            evidence = torch.zeros(
                candidate_inputs["tokens"].shape[:2] + (22,),
                device=candidate_inputs["tokens"].device,
            )
        else:
            evidence = evidence.clone()
        if "candidate_logits" in preliminary_remove:
            evidence[..., 15] = torch.where(
                remove_candidates,
                torch.sigmoid(preliminary_remove["candidate_logits"]),
                evidence[..., 15],
            )
        if result.get("verifier") is not None:
            verifier_evidence = torch.stack(
                [torch.sigmoid(result["verifier"][f"{name}_logit"])
                 for name in self.verifier.HEADS], -1
            )
            evidence[..., 16:22] = verifier_evidence
        routed_tokens = candidate_inputs["tokens"] + self.candidate_evidence(evidence)
        result["pick_remove"] = self.pick_remove(
            encoded.object_tokens,
            encoded.object_mask & batch["object_active"],
            encoded.task_token,
            graph_context,
            routed_tokens,
            candidate_inputs["object"],
            remove_candidates,
        )
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

    def forward(self, batch: dict[str, Tensor], candidate_inputs: dict[str, Tensor] | None = None) -> dict[str, Any]:
        encoded = self.encoder(
            batch["xyz"],
            batch["rgb"],
            batch["instance_id"],
            batch["point_mask"],
            batch["target_mask"],
            batch["object_mask"],
            batch["task_category_id"],
            batch["task_region_id"],
            use_task_region_condition=self.ablation.use_task_region_condition,
        )
        region = self.region_head(
            encoded.point_features, encoded.target_token, encoded.task_token, batch["target_mask"]
        )
        task_grasp = self.task_grasp(
            encoded.point_features,
            encoded.target_token,
            encoded.task_token,
            region["region_probability"],
            batch["target_mask"],
        )
        task_grasp["width_m"] = self.task_grasp.decode_width(
            task_grasp["width_raw"], self.config.min_grasp_width_m, self.config.max_grasp_width_m
        )
        generic_mask = batch["point_mask"] & batch["object_active"][
            torch.arange(batch["xyz"].shape[0], device=batch["xyz"].device)[:, None],
            batch["instance_id"].clamp(0, batch["object_active"].shape[1] - 1),
        ]
        generic_grasp = self.generic_grasp(
            encoded.point_features,
            encoded.global_scene_token,
            encoded.task_token,
            torch.ones_like(region["region_probability"]),
            generic_mask,
            generic_remove=True,
        )
        generic_grasp["width_m"] = self.generic_grasp.decode_width(
            generic_grasp["width_raw"], self.config.min_grasp_width_m, self.config.max_grasp_width_m
        )
        state_graspability = self.state_graspability(
            encoded.global_scene_token, encoded.target_token, encoded.task_token
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
        remove = self.pick_remove(
            encoded.object_tokens,
            encoded.object_mask & batch["object_active"],
            encoded.task_token,
            graph_context,
        )
        result: dict[str, Any] = {
            "encoded": encoded,
            "region": region,
            "task_grasp": task_grasp,
            "generic_grasp": generic_grasp,
            "state_graspability": state_graspability,
            "graph": graph,
            "push": push,
            "pick_remove": remove,
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
            task_grasp_mask = kind == 2
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
        return result
