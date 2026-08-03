"""Complete shared-feature TCD-PRG model assembly."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from tcd_prg.config import AblationConfig, BackboneConfig, GraphConfig, ModelConfig, RouterConfig
from tcd_prg.constants import ActionType

from .backbones import (
    PointTransformerV3SceneGeometryBackbone,
    TaskConditionedPointTransformer,
)
from .common import ActionCandidateEncoder
from .dependency_graph import TaskConditionedDependencyGraph
from .grasp_proposal import (
    GlobalGraspProposalHead,
    M2T2GraspDecoder,
    TaskGraspProposalHead,
)
from .grasp_verifier import GripperSceneTaskVerifier
from .policy import (
    FlatCandidateClassifier,
    MaskedHierarchicalCandidateRouter,
    fixed_priority_output,
)
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
        # 单元测试可使用轻量 legacy 骨干；正式训练入口始终显式传入 PTv3 配置。
        backbone_config = backbone_config or BackboneConfig(backend="legacy")
        c = self.config
        scene_backbone = None
        if backbone_config.backend == "point_transformer_v3":
            scene_backbone = PointTransformerV3SceneGeometryBackbone(
                dim=c.feature_dim,
                source_root=backbone_config.source_root,
                grid_size_m=backbone_config.grid_size_m,
                enable_flash_attention=backbone_config.enable_flash_attention,
                patch_size=backbone_config.patch_size,
                activation_checkpointing=c.activation_checkpointing,
            )
        self.encoder = TaskConditionedPointTransformer(
            dim=c.feature_dim,
            task_dim=c.task_dim,
            num_categories=c.num_categories,
            num_regions=c.num_task_regions,
            attention_points=backbone_config.attention_points,
            activation_checkpointing=c.activation_checkpointing,
            scene_backbone=scene_backbone,
        )
        self.region_head = TaskRegionHead(c.feature_dim)
        # Task/Global Grasp 共享同一个 M2T2 解码器参数，但各自保留独立 query 与输出头。
        # 这样只做一次场景编码，也不会把“任务抓取”和“移除抓取”的查询语义混在一起。
        self.grasp_decoder = M2T2GraspDecoder(
            c.feature_dim, c.grasp_decoder_layers, c.grasp_decoder_heads
        )
        self.task_grasp = TaskGraspProposalHead(c.feature_dim, c.task_grasp_candidates)
        self.global_grasp = GlobalGraspProposalHead(
            c.feature_dim, c.global_grasp_candidates, c.global_grasp_input_mode,
        )
        self.verifier = GripperSceneTaskVerifier(
            c.feature_dim,
            c.feature_dim,
            c.verifier_transformer_layers,
            c.verifier_transformer_heads,
        )
        self.graph = TaskConditionedDependencyGraph(
            c.feature_dim, physical_relations=len(graph_config.physical_relations),
            task_relations=len(graph_config.task_relations), layers=graph_config.layers,
            heads=graph_config.heads, edge_threshold=c.graph_edge_threshold,
        )
        self.push = PushHead(
            c.feature_dim,
            c.num_direction_bins,
            c.push_direction_feature_dim,
            c.push_direction_transformer_layers,
            c.push_direction_transformer_heads,
            c.push_direction_contact_topk,
        )
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
        candidate_valid = verifier_inputs["candidate_valid"].bool()
        coordinates = torch.nonzero(candidate_valid, as_tuple=False)
        batch_size, candidates = candidate_valid.shape
        outputs = {
            f"{head}_logit": encoded.point_features.new_full(
                (batch_size, candidates), -30.0
            )
            for head in self.verifier.HEADS
        }
        if not coordinates.numel():
            return outputs

        # 只打包真实抓取候选；PUSH 和 padding 不再进入局部 Transformer。
        micro = self.config.verifier_candidate_micro_batch
        for start in range(0, coordinates.shape[0], micro):
            selected = coordinates[start:start + micro]
            rows, candidate_indices = selected[:, 0], selected[:, 1]
            point_index = verifier_inputs["scene_point_index"][rows, candidate_indices]
            verifier_features = encoded.point_features[rows[:, None], point_index]
            verifier_target = batch["target_mask"][rows[:, None], point_index]
            verifier_region = region["region_probability"][rows[:, None], point_index]
            chunk = self.verifier(
                verifier_inputs["scene_xyz_grasp"][rows, candidate_indices][:, None],
                verifier_inputs["gripper_xyz_grasp"][rows, candidate_indices][:, None],
                verifier_features[:, None],
                verifier_target[:, None],
                verifier_region[:, None],
                encoded.task_token[rows],
                verifier_inputs["scene_valid"][rows, candidate_indices][:, None],
                verifier_inputs["gripper_valid"][rows, candidate_indices][:, None],
            )
            for key, value in chunk.items():
                outputs[key] = outputs[key].index_put(
                    (rows, candidate_indices), value.squeeze(1)
                )
        return outputs

    @torch.no_grad()
    def _push_supervision_point_indices(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, ...] | None:
        """Return nearest scene points for every labelled PUSH candidate."""

        required = {"action_parameters", "action_type", "candidate_mask", "acted_object"}
        if not required.issubset(batch):
            return None
        contacts = batch["action_parameters"]["push_contact_world"]
        forced: list[Tensor] = []
        for row in range(batch["xyz"].shape[0]):
            selected: list[Tensor] = []
            push_candidates = batch["candidate_mask"][row] & (
                batch["action_type"][row] == int(ActionType.PUSH)
            ) & torch.isfinite(contacts[row]).all(-1)
            for candidate in torch.nonzero(push_candidates, as_tuple=False).flatten():
                object_index = batch["acted_object"][row, candidate]
                domain = batch["point_mask"][row] & (
                    batch["instance_id"][row] == object_index
                )
                points = torch.nonzero(domain, as_tuple=False).flatten()
                if points.numel():
                    distance = batch["xyz"][row, points] - contacts[row, candidate]
                    selected.append(points[distance.square().sum(-1).argmin()])
            forced.append(
                torch.unique(torch.stack(selected))
                if selected else torch.empty(
                    0, dtype=torch.long, device=batch["xyz"].device
                )
            )
        return tuple(forced)

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

    def _encode_scene(self, batch: dict[str, Tensor]) -> tuple[Any, Tensor]:
        object_present = batch.get("object_present", batch["object_mask"])
        # 只有同时存在且尚未被移除的物体才参与点云几何、抓取分配和碰撞语义。
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
        return encoded, physical_active

    def forward_generated_policy(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        """Router-only stage: shared encoder plus cached generated candidates."""

        generated = batch.get("generated_policy_candidates")
        if generated is None:
            raise RuntimeError("Generated-policy forward requires cached generated candidates")
        # Policy-only 阶段若冻结场景编码器，则关闭其 autograd，避免保存无用激活。
        encoder_frozen = not any(parameter.requires_grad for parameter in self.encoder.parameters())
        if encoder_frozen:
            with torch.no_grad():
                encoded, _ = self._encode_scene(batch)
        else:
            encoded, _ = self._encode_scene(batch)
        result: dict[str, Any] = {"encoded": encoded, "verifier": None}
        generated_inputs = self._external_candidate_inputs(encoded, generated)
        result["generated_router"] = self.route_cached(batch, result, generated_inputs)
        return result

    def forward(
        self, batch: dict[str, Tensor], candidate_inputs: dict[str, Tensor] | None = None,
        *, forward_mode: str = "full",
    ) -> dict[str, Any]:
        if forward_mode == "generated_policy":
            return self.forward_generated_policy(batch)
        if forward_mode != "full":
            raise ValueError(f"Unsupported forward_mode={forward_mode}")
        # PTv3/场景编码仅执行一次，后续所有 head 复用同一组点、物体和任务特征。
        encoded, physical_active = self._encode_scene(batch)
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
            self.grasp_decoder,
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
            self.grasp_decoder,
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
            self._push_supervision_point_indices(batch),
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
