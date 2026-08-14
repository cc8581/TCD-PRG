"""Sensor-only scene backbone plus predicted object queries and task adapter.

Formal perception input is fused XYZRGB (+ point validity/grid coordinates).
GT instance ids, target masks and simulator object state are deliberately absent
from every forward signature in this file.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..common import MaskedAttentionPool, masked_softmax
from ..instance_segmentation import InstanceMaskDecoder, InstanceQueryOutput
from ..target_prompt import TargetPromptSelector


def _expand_bits_10(value: Tensor) -> Tensor:
    value = value.long() & 0x3FF
    value = (value | (value << 16)) & 0x030000FF
    value = (value | (value << 8)) & 0x0300F00F
    value = (value | (value << 4)) & 0x030C30C3
    return (value | (value << 2)) & 0x09249249


def _morton_codes(points: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    scale = (upper - lower).clamp_min(1e-6)
    quantized = (((points - lower) / scale).clamp(0.0, 1.0) * 1023.0).round().long()
    return (
        _expand_bits_10(quantized[..., 0])
        | (_expand_bits_10(quantized[..., 1]) << 1)
        | (_expand_bits_10(quantized[..., 2]) << 2)
    )


def _local_knn(
    query: Tensor, reference: Tensor, reference_mask: Tensor, k: int,
    candidate_window: int = 256,
) -> Tensor:
    neighbor_count = min(k, reference.shape[1])
    rows = []
    with torch.no_grad():
        for batch_row in range(query.shape[0]):
            valid_index = torch.nonzero(reference_mask[batch_row], as_tuple=False).squeeze(-1)
            if not len(valid_index):
                raise ValueError("A scene has no valid reference points")
            valid_reference = reference[batch_row, valid_index]
            lower = valid_reference.amin(0)
            upper = valid_reference.amax(0)
            reference_code = _morton_codes(valid_reference, lower, upper)
            order = reference_code.argsort(stable=True)
            sorted_code = reference_code[order]
            query_code = _morton_codes(query[batch_row], lower, upper)
            insertion = torch.searchsorted(sorted_code, query_code)
            window = min(len(valid_index), max(candidate_window, 4 * neighbor_count))
            offsets = torch.arange(window, device=query.device) - window // 2
            candidate_position = (insertion[:, None] + offsets).clamp(0, len(valid_index) - 1)
            candidate_index = valid_index[order[candidate_position]]
            candidate_xyz = reference[batch_row, candidate_index]
            squared_distance = ((query[batch_row, :, None] - candidate_xyz) ** 2).sum(-1)
            nearest = squared_distance.topk(min(neighbor_count, window), largest=False).indices
            selected = candidate_index.gather(1, nearest)
            if selected.shape[1] < neighbor_count:
                selected = torch.cat(
                    (selected, selected[:, :1].expand(-1, neighbor_count - selected.shape[1])), 1
                )
            rows.append(selected)
    return torch.stack(rows)


def _farthest_point_indices(xyz: Tensor, mask: Tensor, count: int) -> tuple[Tensor, Tensor]:
    index_rows, valid_rows = [], []
    with torch.no_grad():
        for batch_row in range(mask.shape[0]):
            valid_index = torch.nonzero(mask[batch_row], as_tuple=False).squeeze(-1)
            if not len(valid_index):
                raise ValueError("A scene has no valid points")
            points = xyz[batch_row, valid_index]
            selected_count = min(count, len(points))
            centroid = points.mean(0)
            farthest = ((points - centroid) ** 2).sum(-1).argmax()
            minimum_distance = torch.full(
                (len(points),), float("inf"), dtype=points.dtype, device=points.device
            )
            selected = []
            for _ in range(selected_count):
                selected.append(valid_index[farthest])
                distance = ((points - points[farthest]) ** 2).sum(-1)
                minimum_distance = torch.minimum(minimum_distance, distance)
                farthest = minimum_distance.argmax()
            row = torch.stack(selected)
            row_valid = torch.ones(selected_count, dtype=torch.bool, device=mask.device)
            if selected_count < count:
                row = torch.cat((row, row[:1].expand(count - selected_count)))
                row_valid = torch.cat(
                    (row_valid, torch.zeros(
                        count - selected_count, dtype=torch.bool, device=mask.device
                    ))
                )
            index_rows.append(row)
            valid_rows.append(row_valid)
    return torch.stack(index_rows), torch.stack(valid_rows)


def _gather_neighbors(features: Tensor, index: Tensor) -> Tensor:
    batch = torch.arange(features.shape[0], device=features.device)[:, None, None]
    return features[batch, index]


class PointTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, k: int = 16) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim, self.k = heads, dim // heads, k
        self.q = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.output = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, xyz: Tensor, features: Tensor, valid: Tensor) -> Tensor:
        index = _local_knn(xyz, xyz, valid, self.k)
        neighbor_xyz = _gather_neighbors(xyz, index)
        neighbor_features = _gather_neighbors(features, index)
        relative = neighbor_xyz - xyz[:, :, None]
        positional = self.position(relative)
        q = self.q(features).reshape(*features.shape[:2], self.heads, self.head_dim)
        k = self.k_proj(neighbor_features).reshape(
            *neighbor_features.shape[:3], self.heads, self.head_dim
        )
        v = (self.v(neighbor_features) + positional).reshape(
            *neighbor_features.shape[:3], self.heads, self.head_dim
        )
        logits = ((q[:, :, None] - k) * positional.reshape(
            *positional.shape[:3], self.heads, self.head_dim
        )).sum(-1)
        neighbor_valid = _gather_neighbors(
            valid.unsqueeze(-1).float(), index
        ).squeeze(-1).bool()
        attention = masked_softmax(
            logits.permute(0, 1, 3, 2), neighbor_valid[:, :, None], dim=-1
        )
        aggregated = torch.einsum("bnhk,bnkhd->bnhd", attention, v).flatten(-2)
        features = self.norm1(features + self.output(aggregated))
        return self.norm2(features + self.ffn(features)) * valid.unsqueeze(-1)


@dataclass(slots=True)
class SceneGeometryOutput:
    point_features: Tensor
    global_scene_token: Tensor


@dataclass(slots=True)
class EncoderOutput:
    point_features: Tensor
    object_tokens: Tensor
    object_mask: Tensor
    target_token: Tensor
    target_probability: Tensor
    target_query_logits: Tensor
    target_query_weights: Tensor
    target_query_index: Tensor
    target_selection_margin: Tensor
    target_prompt_support: Tensor
    target_prompt_used: Tensor
    global_scene_token: Tensor
    task_token: Tensor
    scene_point_features: Tensor
    scene_object_tokens: Tensor
    scene_global_token: Tensor
    instance: InstanceQueryOutput


class TaskFreeSceneGeometryBackbone(nn.Module):
    """Legacy fallback. Inputs are strictly XYZ/RGB/point validity."""

    def __init__(
        self, dim: int = 256, blocks: int = 3, heads: int = 4, neighbors: int = 16,
        attention_points: int = 1024, activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.attention_points = attention_points
        self.activation_checkpointing = activation_checkpointing
        self.input_projection = nn.Sequential(
            nn.Linear(7, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            PointTransformerBlock(dim, heads, neighbors) for _ in range(blocks)
        )
        self.context_projection = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.pool_query = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.global_pool = MaskedAttentionPool(dim, dim)

    def forward(
        self, xyz: Tensor, rgb: Tensor, point_mask: Tensor,
        grid_coord: Tensor | None = None,
    ) -> SceneGeometryOutput:
        del grid_coord
        b, n, _ = xyz.shape
        base = self.input_projection(
            torch.cat((xyz, rgb, point_mask.unsqueeze(-1).float()), -1)
        )
        base = base * point_mask.unsqueeze(-1)
        anchor_count = min(self.attention_points, n)
        anchor_index, anchor_valid = _farthest_point_indices(
            xyz, point_mask, anchor_count
        )
        row = torch.arange(b, device=xyz.device)[:, None]
        anchor_xyz = xyz[row, anchor_index]
        anchor_features = base[row, anchor_index]
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                anchor_features = checkpoint(
                    block, anchor_xyz, anchor_features, anchor_valid,
                    use_reentrant=False,
                )
            else:
                anchor_features = block(anchor_xyz, anchor_features, anchor_valid)
        nearest = _local_knn(xyz, anchor_xyz, anchor_valid, 1).squeeze(-1)
        context = anchor_features[row, nearest]
        point_features = self.context_projection(
            torch.cat((base, context), -1)
        ) * point_mask.unsqueeze(-1)
        query = self.pool_query.unsqueeze(0).expand(b, -1)
        global_token = self.global_pool(point_features, point_mask, query)
        return SceneGeometryOutput(point_features, global_token)


class TaskConditioningAdapter(nn.Module):
    """Condition predicted object queries on the closed-vocabulary task."""

    def __init__(
        self, dim: int, task_dim: int, num_categories: int, num_regions: int,
        target_temperature: float = 0.25,
        target_prompt_radius_m: float = 0.030,
        target_prompt_sigma_m: float = 0.012,
        target_prompt_weight: float = 4.0,
        target_category_weight: float = 1.0,
        target_objectness_weight: float = 0.5,
        target_center_weight: float = 0.5,
        target_learned_weight: float = 0.20,
        target_reid_weight: float = 0.75,
        target_reid_center_weight: float = 0.35,
        target_reid_max_center_distance_m: float = 0.15,
    ) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(num_categories, task_dim)
        self.region_embedding = nn.Embedding(num_regions, task_dim)
        self.task_projection = nn.Sequential(
            nn.Linear(2 * task_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        # Kept for V1 state-dict compatibility, but region semantics are no longer
        # allowed to identify an instance. It receives a category-only task token
        # and contributes only a small residual prior.
        self.target_selector = nn.Sequential(
            nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 1)
        )
        self.target_temperature = float(target_temperature)
        self.prompt_selector = TargetPromptSelector(
            radius_m=target_prompt_radius_m,
            sigma_m=target_prompt_sigma_m,
            prompt_weight=target_prompt_weight,
            category_weight=target_category_weight,
            objectness_weight=target_objectness_weight,
            center_weight=target_center_weight,
            learned_weight=target_learned_weight,
            reid_weight=target_reid_weight,
            reid_center_weight=target_reid_center_weight,
            reid_max_center_distance_m=target_reid_max_center_distance_m,
            temperature=target_temperature,
            hard_inference=True,
        )
        self.point_film = nn.Linear(2 * dim + 1, 2 * dim)
        self.object_film = nn.Linear(2 * dim + 1, 2 * dim)
        self.global_film = nn.Linear(2 * dim, 2 * dim)
        self.point_norm = nn.LayerNorm(dim)
        self.object_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)

    @staticmethod
    def _film(value: Tensor, parameters: Tensor, norm: nn.LayerNorm) -> Tensor:
        scale, shift = parameters.chunk(2, -1)
        return norm(value * (1.0 + 0.1 * torch.tanh(scale)) + shift)

    def forward(
        self,
        scene: SceneGeometryOutput,
        instance: InstanceQueryOutput,
        xyz: Tensor,
        point_mask: Tensor,
        task_category_id: Tensor,
        task_region_id: Tensor,
        use_task_region_condition: bool = True,
        target_prompt_xyz: Tensor | None = None,
        target_prompt_label: Tensor | None = None,
        target_prompt_valid: Tensor | None = None,
        target_reid_token: Tensor | None = None,
        target_reid_center: Tensor | None = None,
        target_reid_valid: Tensor | None = None,
    ) -> EncoderOutput:
        category = self.category_embedding(
            task_category_id.clamp(0, self.category_embedding.num_embeddings - 1)
        )
        region = self.region_embedding(
            task_region_id.clamp(0, self.region_embedding.num_embeddings - 1)
        )
        if not use_task_region_condition:
            region = torch.zeros_like(region)
        task = self.task_projection(torch.cat((category, region), -1))

        object_tokens = instance.object_tokens
        # Target identity is category + observable prompt, never the requested
        # functional region.  The old learned selector remains only as a weak
        # category-conditioned residual so V1 checkpoints load strictly.
        category_only_task = self.task_projection(
            torch.cat((category, torch.zeros_like(region)), -1)
        )
        category_object = category_only_task[:, None].expand_as(object_tokens)
        selector_features = torch.cat(
            (object_tokens, category_object, object_tokens - category_object,
             object_tokens * category_object), -1
        )
        learned = self.target_selector(selector_features).squeeze(-1)
        selection = self.prompt_selector(
            instance_probability=instance.mask_probability,
            objectness_logits=instance.objectness_logits,
            category_logits=instance.category_logits,
            centers_world=instance.centers_world,
            object_tokens=instance.object_tokens,
            xyz=xyz,
            point_mask=point_mask,
            task_category_id=task_category_id,
            target_prompt_xyz=target_prompt_xyz,
            target_prompt_label=target_prompt_label,
            target_prompt_valid=target_prompt_valid,
            target_reid_token=target_reid_token,
            target_reid_center=target_reid_center,
            target_reid_valid=target_reid_valid,
            learned_semantic_logit=learned,
        )
        target_logits = selection.logits
        target_weights = selection.weights
        target = torch.einsum("bq,bqd->bd", target_weights, object_tokens)
        target_probability = torch.einsum(
            "bq,bqn->bn", target_weights, instance.mask_probability
        ) * point_mask.to(scene.point_features.dtype)

        point_condition = torch.cat(
            (
                task[:, None].expand(-1, scene.point_features.shape[1], -1),
                target[:, None].expand(-1, scene.point_features.shape[1], -1),
                target_probability.unsqueeze(-1),
            ),
            -1,
        )
        point_features = self._film(
            scene.point_features, self.point_film(point_condition), self.point_norm
        ) * point_mask.unsqueeze(-1)

        object_condition = torch.cat(
            (
                task[:, None].expand_as(object_tokens),
                target[:, None].expand_as(object_tokens),
                target_weights.unsqueeze(-1),
            ),
            -1,
        )
        conditioned_objects = self._film(
            object_tokens, self.object_film(object_condition), self.object_norm
        ) * instance.object_mask.unsqueeze(-1)

        global_token = self._film(
            scene.global_scene_token,
            self.global_film(torch.cat((task, target), -1)),
            self.global_norm,
        )
        return EncoderOutput(
            point_features=point_features,
            object_tokens=conditioned_objects,
            object_mask=instance.object_mask,
            target_token=target,
            target_probability=target_probability,
            target_query_logits=target_logits,
            target_query_weights=target_weights,
            target_query_index=selection.query_index,
            target_selection_margin=selection.margin,
            target_prompt_support=selection.positive_prompt_support,
            target_prompt_used=selection.used_prompt,
            global_scene_token=global_token,
            task_token=task,
            scene_point_features=scene.point_features,
            scene_object_tokens=instance.object_tokens,
            scene_global_token=scene.global_scene_token,
            instance=instance,
        )


class TaskConditionedPointTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        task_dim: int = 128,
        num_categories: int = 64,
        num_regions: int = 64,
        blocks: int = 3,
        heads: int = 4,
        neighbors: int = 16,
        attention_points: int = 1024,
        activation_checkpointing: bool = True,
        scene_backbone: nn.Module | None = None,
        instance_queries: int = 32,
        instance_decoder_layers: int = 6,
        instance_decoder_heads: int = 8,
        instance_objectness_threshold: float = 0.5,
        target_temperature: float = 0.25,
        target_prompt_radius_m: float = 0.030,
        target_prompt_sigma_m: float = 0.012,
        target_prompt_weight: float = 4.0,
        target_category_weight: float = 1.0,
        target_objectness_weight: float = 0.5,
        target_center_weight: float = 0.5,
        target_learned_weight: float = 0.20,
        target_reid_weight: float = 0.75,
        target_reid_center_weight: float = 0.35,
        target_reid_max_center_distance_m: float = 0.15,
    ) -> None:
        super().__init__()
        self.scene_backbone = scene_backbone or TaskFreeSceneGeometryBackbone(
            dim, blocks, heads, neighbors, attention_points, activation_checkpointing
        )
        self.instance_head = InstanceMaskDecoder(
            dim, instance_queries, num_categories,
            layers=instance_decoder_layers,
            heads=instance_decoder_heads,
            objectness_threshold=instance_objectness_threshold,
            auxiliary_loss=True,
        )
        self.task_adapter = TaskConditioningAdapter(
            dim, task_dim, num_categories, num_regions,
            target_temperature=target_temperature,
            target_prompt_radius_m=target_prompt_radius_m,
            target_prompt_sigma_m=target_prompt_sigma_m,
            target_prompt_weight=target_prompt_weight,
            target_category_weight=target_category_weight,
            target_objectness_weight=target_objectness_weight,
            target_center_weight=target_center_weight,
            target_learned_weight=target_learned_weight,
            target_reid_weight=target_reid_weight,
            target_reid_center_weight=target_reid_center_weight,
            target_reid_max_center_distance_m=target_reid_max_center_distance_m,
        )

    def forward_scene_geometry(
        self,
        xyz: Tensor,
        rgb: Tensor,
        point_mask: Tensor,
        grid_coord: Tensor | None = None,
    ) -> SceneGeometryOutput:
        if grid_coord is None:
            return self.scene_backbone(xyz, rgb, point_mask)
        return self.scene_backbone(
            xyz, rgb, point_mask, grid_coord=grid_coord
        )

    def forward_scene_instances(
        self,
        xyz: Tensor,
        rgb: Tensor,
        point_mask: Tensor,
        grid_coord: Tensor | None = None,
    ) -> tuple[SceneGeometryOutput, InstanceQueryOutput]:
        scene = self.forward_scene_geometry(
            xyz, rgb, point_mask, grid_coord=grid_coord
        )
        instance = self.instance_head(scene.point_features, xyz, point_mask)
        return scene, instance

    def forward(
        self,
        xyz: Tensor,
        rgb: Tensor,
        point_mask: Tensor,
        task_category_id: Tensor,
        task_region_id: Tensor,
        use_task_region_condition: bool = True,
        grid_coord: Tensor | None = None,
        target_prompt_xyz: Tensor | None = None,
        target_prompt_label: Tensor | None = None,
        target_prompt_valid: Tensor | None = None,
        target_reid_token: Tensor | None = None,
        target_reid_center: Tensor | None = None,
        target_reid_valid: Tensor | None = None,
    ) -> EncoderOutput:
        scene, instance = self.forward_scene_instances(
            xyz, rgb, point_mask, grid_coord=grid_coord
        )
        return self.task_adapter(
            scene, instance, xyz, point_mask, task_category_id, task_region_id,
            use_task_region_condition,
            target_prompt_xyz=target_prompt_xyz,
            target_prompt_label=target_prompt_label,
            target_prompt_valid=target_prompt_valid,
            target_reid_token=target_reid_token,
            target_reid_center=target_reid_center,
            target_reid_valid=target_reid_valid,
        )
