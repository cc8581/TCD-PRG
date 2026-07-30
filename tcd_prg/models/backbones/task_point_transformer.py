"""Single-pass task-free scene backbone plus a lightweight task adapter.

The expensive point-neighbour computation is deliberately independent of the
target instance and task.  Instance membership is only used after the point
backbone for token pooling, which allows the same features to support both the
strict scene-only and instance-assisted global-grasp evaluation tracks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..common import MaskedAttentionPool, masked_softmax


def _chunked_knn(query: Tensor, reference: Tensor, reference_mask: Tensor, k: int, chunk: int = 512) -> Tensor:
    chunks = []
    for start in range(0, query.shape[1], chunk):
        distance = torch.cdist(query[:, start : start + chunk], reference)
        distance = distance.masked_fill(~reference_mask[:, None, :], float("inf"))
        chunks.append(distance.topk(min(k, reference.shape[1]), largest=False).indices)
    return torch.cat(chunks, dim=1)


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
        index = _chunked_knn(xyz, xyz, valid, self.k)
        neighbor_xyz = _gather_neighbors(xyz, index)
        neighbor_features = _gather_neighbors(features, index)
        relative = neighbor_xyz - xyz[:, :, None]
        positional = self.position(relative)
        q = self.q(features).reshape(*features.shape[:2], self.heads, self.head_dim)
        k = self.k_proj(neighbor_features).reshape(*neighbor_features.shape[:3], self.heads, self.head_dim)
        v = (self.v(neighbor_features) + positional).reshape(
            *neighbor_features.shape[:3], self.heads, self.head_dim
        )
        logits = ((q[:, :, None] - k) * positional.reshape(
            *positional.shape[:3], self.heads, self.head_dim
        )).sum(-1)
        neighbor_valid = _gather_neighbors(valid.unsqueeze(-1).float(), index).squeeze(-1).bool()
        attention = masked_softmax(logits.permute(0, 1, 3, 2), neighbor_valid[:, :, None], dim=-1)
        aggregated = torch.einsum("bnhk,bnkhd->bnhd", attention, v).flatten(-2)
        features = self.norm1(features + self.output(aggregated))
        return self.norm2(features + self.ffn(features)) * valid.unsqueeze(-1)


@dataclass(slots=True)
class SceneGeometryOutput:
    """Task-free representation produced by the expensive scene backbone."""

    point_features: Tensor
    object_tokens: Tensor
    object_mask: Tensor
    global_scene_token: Tensor


@dataclass(slots=True)
class EncoderOutput:
    """Task-conditioned view retaining explicit access to neutral features."""

    point_features: Tensor
    object_tokens: Tensor
    object_mask: Tensor
    target_token: Tensor
    global_scene_token: Tensor
    task_token: Tensor
    scene_point_features: Tensor
    scene_object_tokens: Tensor
    scene_global_token: Tensor


class TaskFreeSceneGeometryBackbone(nn.Module):
    """Encode XYZ/RGB/validity without target, task or object-pose leakage."""

    def __init__(
        self, dim: int = 256, blocks: int = 3, heads: int = 4, neighbors: int = 16,
        attention_points: int = 4096, activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.attention_points = attention_points
        self.activation_checkpointing = activation_checkpointing
        # XYZ + RGB + point-valid. Instance membership is intentionally absent.
        self.input_projection = nn.Sequential(nn.Linear(7, dim), nn.LayerNorm(dim), nn.GELU())
        self.blocks = nn.ModuleList(PointTransformerBlock(dim, heads, neighbors) for _ in range(blocks))
        self.context_projection = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pool_query = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.pool_query, std=0.02)
        self.object_pool = MaskedAttentionPool(dim, dim)
        self.global_pool = MaskedAttentionPool(dim, dim)

    @staticmethod
    def _anchor_indices(mask: Tensor, count: int) -> Tensor:
        rows = []
        for batch_row in range(mask.shape[0]):
            valid = torch.nonzero(mask[batch_row], as_tuple=False).squeeze(-1)
            if len(valid) == 0:
                raise ValueError("A scene has no valid points")
            if len(valid) >= count:
                position = torch.linspace(0, len(valid) - 1, count, device=mask.device).round().long()
                rows.append(valid[position])
            else:
                rows.append(valid[torch.arange(count, device=mask.device) % len(valid)])
        return torch.stack(rows)

    def forward(
        self, xyz: Tensor, rgb: Tensor, instance_id: Tensor, point_mask: Tensor, object_mask: Tensor
    ) -> SceneGeometryOutput:
        b, n, _ = xyz.shape
        base = self.input_projection(torch.cat((xyz, rgb, point_mask.unsqueeze(-1).float()), -1))
        base = base * point_mask.unsqueeze(-1)
        anchor_count = min(self.attention_points, n)
        anchor_index = self._anchor_indices(point_mask, anchor_count)
        row = torch.arange(b, device=xyz.device)[:, None]
        anchor_xyz = xyz[row, anchor_index]
        anchor_features = base[row, anchor_index]
        anchor_valid = point_mask[row, anchor_index]
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                anchor_features = checkpoint(block, anchor_xyz, anchor_features, anchor_valid, use_reentrant=False)
            else:
                anchor_features = block(anchor_xyz, anchor_features, anchor_valid)
        nearest = _chunked_knn(xyz, anchor_xyz, anchor_valid, 1).squeeze(-1)
        context = anchor_features[row, nearest]
        point_features = self.context_projection(torch.cat((base, context), -1)) * point_mask.unsqueeze(-1)
        query = self.pool_query.unsqueeze(0).expand(b, -1)
        objects = [
            self.object_pool(point_features, point_mask & (instance_id == index), query)
            for index in range(object_mask.shape[1])
        ]
        object_tokens = torch.stack(objects, 1) * object_mask.unsqueeze(-1)
        global_token = self.global_pool(point_features, point_mask, query)
        return SceneGeometryOutput(point_features, object_tokens, object_mask, global_token)


class TaskConditioningAdapter(nn.Module):
    """Cheap FiLM adapter that adds target/task context after neutral encoding."""

    def __init__(self, dim: int, task_dim: int, num_categories: int, num_regions: int) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(num_categories, task_dim)
        self.region_embedding = nn.Embedding(num_regions, task_dim)
        self.task_projection = nn.Sequential(nn.Linear(2 * task_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.target_pool = MaskedAttentionPool(dim, dim)
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
        self, scene: SceneGeometryOutput, instance_id: Tensor, point_mask: Tensor,
        target_mask: Tensor, target_object: Tensor, task_category_id: Tensor,
        task_region_id: Tensor, use_task_region_condition: bool = True,
    ) -> EncoderOutput:
        category = self.category_embedding(task_category_id.clamp(0, self.category_embedding.num_embeddings - 1))
        region = self.region_embedding(task_region_id.clamp(0, self.region_embedding.num_embeddings - 1))
        if not use_task_region_condition:
            region = torch.zeros_like(region)
        task = self.task_projection(torch.cat((category, region), -1))
        target = self.target_pool(scene.point_features, point_mask & target_mask, task)
        point_condition = torch.cat((
            task[:, None].expand(-1, scene.point_features.shape[1], -1),
            target[:, None].expand(-1, scene.point_features.shape[1], -1),
            target_mask.unsqueeze(-1).float(),
        ), -1)
        point_features = self._film(scene.point_features, self.point_film(point_condition), self.point_norm)
        point_features = point_features * point_mask.unsqueeze(-1)
        object_is_target = torch.arange(scene.object_tokens.shape[1], device=target_object.device)[None] == target_object[:, None]
        object_condition = torch.cat((
            task[:, None].expand_as(scene.object_tokens),
            target[:, None].expand_as(scene.object_tokens),
            object_is_target.unsqueeze(-1).float(),
        ), -1)
        object_tokens = self._film(scene.object_tokens, self.object_film(object_condition), self.object_norm)
        object_tokens = object_tokens * scene.object_mask.unsqueeze(-1)
        global_token = self._film(
            scene.global_scene_token,
            self.global_film(torch.cat((task, target), -1)),
            self.global_norm,
        )
        return EncoderOutput(
            point_features, object_tokens, scene.object_mask, target, global_token, task,
            scene.point_features, scene.object_tokens, scene.global_scene_token,
        )


class TaskConditionedPointTransformer(nn.Module):
    """Compatibility wrapper exposing one neutral backbone and one task adapter."""

    def __init__(
        self, dim: int = 256, task_dim: int = 128, num_categories: int = 64,
        num_regions: int = 64, blocks: int = 3, heads: int = 4, neighbors: int = 16,
        attention_points: int = 4096, activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.scene_backbone = TaskFreeSceneGeometryBackbone(
            dim, blocks, heads, neighbors, attention_points, activation_checkpointing
        )
        self.task_adapter = TaskConditioningAdapter(dim, task_dim, num_categories, num_regions)

    def forward(
        self, xyz: Tensor, rgb: Tensor, instance_id: Tensor, point_mask: Tensor,
        target_mask: Tensor, object_mask: Tensor, task_category_id: Tensor,
        task_region_id: Tensor, use_task_region_condition: bool = True,
        target_object: Tensor | None = None,
    ) -> EncoderOutput:
        scene = self.scene_backbone(xyz, rgb, instance_id, point_mask, object_mask)
        if target_object is None:
            counts = torch.stack([
                (target_mask & (instance_id == index)).sum(-1)
                for index in range(object_mask.shape[1])
            ], -1)
            target_object = counts.argmax(-1)
        return self.task_adapter(
            scene, instance_id, point_mask, target_mask, target_object,
            task_category_id, task_region_id, use_task_region_condition,
        )
