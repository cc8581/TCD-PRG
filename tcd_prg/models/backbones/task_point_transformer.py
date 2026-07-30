"""Windows-compatible task-conditioned point transformer.

The implementation uses an optional PyTorch3D KNN backend and a chunked Torch
fallback. Context attention operates on a configurable anchor set and is
interpolated to all input points, keeping 16k-point scenes practical on 24 GB.
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
        k = self.k_proj(neighbor_features).reshape(
            *neighbor_features.shape[:3], self.heads, self.head_dim
        )
        v = (self.v(neighbor_features) + positional).reshape(
            *neighbor_features.shape[:3], self.heads, self.head_dim
        )
        logits = ((q[:, :, None] - k) * positional.reshape(*positional.shape[:3], self.heads, self.head_dim)).sum(-1)
        neighbor_valid = _gather_neighbors(valid.unsqueeze(-1).float(), index).squeeze(-1).bool()
        attention = masked_softmax(logits.permute(0, 1, 3, 2), neighbor_valid[:, :, None], dim=-1)
        aggregated = torch.einsum("bnhk,bnkhd->bnhd", attention, v).flatten(-2)
        features = self.norm1(features + self.output(aggregated))
        return self.norm2(features + self.ffn(features)) * valid.unsqueeze(-1)


@dataclass(slots=True)
class EncoderOutput:
    point_features: Tensor
    object_tokens: Tensor
    object_mask: Tensor
    target_token: Tensor
    global_scene_token: Tensor
    task_token: Tensor


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
        attention_points: int = 4096,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.attention_points = attention_points
        self.activation_checkpointing = activation_checkpointing
        self.category_embedding = nn.Embedding(num_categories, task_dim)
        self.region_embedding = nn.Embedding(num_regions, task_dim)
        self.task_projection = nn.Sequential(nn.Linear(2 * task_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.input_projection = nn.Sequential(nn.Linear(11, dim), nn.LayerNorm(dim), nn.GELU())
        self.blocks = nn.ModuleList(PointTransformerBlock(dim, heads, neighbors) for _ in range(blocks))
        self.context_projection = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.object_pool = MaskedAttentionPool(dim, dim)
        self.target_pool = MaskedAttentionPool(dim, dim)
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
        self,
        xyz: Tensor,
        rgb: Tensor,
        instance_id: Tensor,
        point_mask: Tensor,
        target_mask: Tensor,
        object_mask: Tensor,
        task_category_id: Tensor,
        task_region_id: Tensor,
        use_task_region_condition: bool = True,
    ) -> EncoderOutput:
        b, n, _ = xyz.shape
        task_category_id = task_category_id.clamp(0, self.category_embedding.num_embeddings - 1)
        task_region_id = task_region_id.clamp(0, self.region_embedding.num_embeddings - 1)
        category = self.category_embedding(task_category_id)
        region = self.region_embedding(task_region_id)
        if not use_task_region_condition:
            region = torch.zeros_like(region)
        task = self.task_projection(torch.cat((category, region), dim=-1))
        object_count = object_mask.shape[1]
        safe_instance = instance_id.clamp(0, object_count - 1)
        row = torch.arange(b, device=xyz.device)[:, None]
        # Formal perception provides instance membership, not object 6D poses.
        # Build translation-invariant object-local coordinates exclusively from
        # the visible instance points so simulation pose truth cannot leak into
        # the learned policy.
        instance_valid = point_mask & (instance_id >= 0) & (instance_id < object_count)
        centers = xyz.new_zeros((b, object_count, 3))
        counts = xyz.new_zeros((b, object_count))
        centers.scatter_add_(
            1,
            safe_instance.unsqueeze(-1).expand(-1, -1, 3),
            xyz * instance_valid.unsqueeze(-1),
        )
        counts.scatter_add_(1, safe_instance, instance_valid.to(xyz.dtype))
        centers = centers / counts.clamp_min(1.0).unsqueeze(-1)
        local_xyz = (xyz - centers[row, safe_instance]) * instance_valid.unsqueeze(-1)
        input_features = torch.cat(
            (xyz, rgb, target_mask.unsqueeze(-1).float(), local_xyz, point_mask.unsqueeze(-1).float()), dim=-1
        )
        base = self.input_projection(input_features) * point_mask.unsqueeze(-1)
        anchor_count = min(self.attention_points, n)
        anchor_index = self._anchor_indices(point_mask, anchor_count)
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
        point_features = self.context_projection(torch.cat((base, context), dim=-1)) * point_mask.unsqueeze(-1)
        object_tokens = []
        for object_index in range(object_count):
            mask = point_mask & (instance_id == object_index)
            object_tokens.append(self.object_pool(point_features, mask, task))
        objects = torch.stack(object_tokens, dim=1) * object_mask.unsqueeze(-1)
        target = self.target_pool(point_features, point_mask & target_mask, task)
        global_token = self.global_pool(point_features, point_mask, task)
        return EncoderOutput(point_features, objects, object_mask, target, global_token, task)
