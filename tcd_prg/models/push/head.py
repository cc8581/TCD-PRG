"""Standalone task-conditioned PUSH predictor."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def push_contact_joint_score(contact_logits: Tensor, membership: Tensor) -> Tensor:
    """Shared sparse-contact ranking: log P(contact)+log P(object membership)."""
    return torch.nn.functional.logsigmoid(contact_logits) + membership.clamp_min(1e-6).log()


class PushHead(nn.Module):
    def __init__(
        self,
        dim=256,
        direction_bins=16,
        direction_dim=64,
        direction_layers=1,
        direction_heads=4,
        direction_contact_topk=32,
        direction_object_topk=4,
        num_categories=64,
        num_task_regions=64,
    ) -> None:
        super().__init__()
        self.direction_bins, self.direction_contact_topk, self.direction_object_topk = (
            int(direction_bins),
            int(direction_contact_topk),
            int(direction_object_topk),
        )
        self.point_encoder = nn.Sequential(
            nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, dim)
        )
        self.category_embedding, self.region_embedding = (
            nn.Embedding(num_categories, dim // 2),
            nn.Embedding(num_task_regions, dim // 2),
        )
        self.task_projection = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.relative_geometry = nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
        self.object_context = nn.Sequential(
            nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.object_pointer = nn.Linear(dim, 1)
        self.point = nn.Sequential(
            nn.Linear(5 * dim + 8, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.contact = nn.Linear(dim, 1)
        self.direction_context = nn.Linear(dim, direction_dim)
        self.direction_embedding = nn.Embedding(self.direction_bins, direction_dim)
        layer = nn.TransformerEncoderLayer(
            direction_dim,
            direction_heads,
            4 * direction_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.direction_transformer = nn.TransformerEncoder(
            layer, direction_layers, norm=nn.LayerNorm(direction_dim)
        )
        self.direction_score = nn.Linear(direction_dim, 1)
        self.direction_residual = nn.Linear(direction_dim, 2)
        self.utility = nn.Linear(direction_dim, 1)

    @staticmethod
    def _pool(features, weights):
        weights = weights.clamp_min(0)
        return torch.einsum("bn,bnd->bd", weights, features) / weights.sum(
            -1, keepdim=True
        ).clamp_min(1e-6)

    def forward(self, sensor, condition, forced_direction_point_mask=None):
        xyz, rgb, point_mask = sensor["xyz"], sensor["rgb"], sensor["point_mask"].bool()
        condition.validate(xyz.shape[1])
        valid = point_mask.to(xyz.dtype)
        target, region = condition.target_probability * valid, condition.region_probability * valid
        point_features = self.point_encoder(
            torch.cat((rgb, target[..., None], region[..., None]), -1)
        )
        raw = (
            condition.object_probability
            * condition.object_valid[:, :, None].to(xyz.dtype)
            * valid[:, None]
        )
        mass = raw.sum(-1, keepdim=True).clamp_min(1e-6)
        object_tokens = torch.einsum("bqn,bnd->bqd", raw, point_features) / mass
        object_center = torch.einsum("bqn,bnd->bqd", raw, xyz) / mass
        assignment = raw / raw.sum(1, keepdim=True).clamp_min(1e-6)
        target_token, region_token = (
            self._pool(point_features, target),
            self._pool(point_features, region),
        )
        target_center = self._pool(xyz, target)
        raw_region_center = self._pool(xyz, region)
        region_center = torch.where(
            region.sum(-1, keepdim=True) > 1e-6, raw_region_center, target_center
        )
        task_token = self.task_projection(
            torch.cat(
                (
                    self.category_embedding(condition.task_category_id),
                    self.region_embedding(condition.task_region_id),
                ),
                -1,
            )
        )
        offset = object_center - target_center[:, None]
        region_offset = (region_center - target_center)[:, None].expand_as(offset)
        rel = self.relative_geometry(
            torch.cat(
                (
                    offset,
                    torch.linalg.vector_norm(offset, dim=-1, keepdim=True),
                    region_offset,
                    torch.linalg.vector_norm(region_offset, dim=-1, keepdim=True),
                ),
                -1,
            )
        )
        q, n = object_tokens.shape[1], xyz.shape[1]
        object_context = self.object_context(
            torch.cat(
                (
                    object_tokens,
                    target_token[:, None].expand(-1, q, -1),
                    task_token[:, None].expand(-1, q, -1),
                    region_token[:, None].expand(-1, q, -1),
                    rel,
                ),
                -1,
            )
        )
        object_logits = (
            self.object_pointer(object_context)
            .squeeze(-1)
            .masked_fill(~condition.object_valid, -30.0)
        )
        point_object = torch.einsum("bqn,bqd->bnd", assignment, object_context)
        point_context = self.point(
            torch.cat(
                (
                    point_features,
                    point_object,
                    task_token[:, None].expand(-1, n, -1),
                    target_token[:, None].expand(-1, n, -1),
                    region_token[:, None].expand(-1, n, -1),
                    xyz - target_center[:, None],
                    xyz - region_center[:, None],
                    target[..., None],
                    region[..., None],
                ),
                -1,
            )
        )
        contact_logits = self.contact(point_context).squeeze(-1).masked_fill(~point_mask, -30.0)
        forced = (
            torch.zeros_like(point_mask)
            if forced_direction_point_mask is None
            else forced_direction_point_mask.bool() & point_mask
        )
        hard_owner = raw.argmax(1)
        masks = []
        logits_rows = []
        residual_rows = []
        utility_rows = []
        token_rows = []
        for b in range(xyz.shape[0]):
            objects = torch.nonzero(condition.object_valid[b], as_tuple=False).flatten()
            if len(objects) > self.direction_object_topk:
                objects = objects[
                    object_logits[b, objects].topk(self.direction_object_topk).indices
                ]
            picks = []
            for obj in objects:
                member = raw[b, obj]
                domain = point_mask[b] & (hard_owner[b] == obj) & (member >= 0.5)
                if not bool(domain.any()):
                    domain = point_mask[b] & (hard_owner[b] == obj)
                ids = torch.nonzero(domain, as_tuple=False).flatten()
                count = min(self.direction_contact_topk, len(ids))
                if count:
                    picks.append(
                        ids[
                            push_contact_joint_score(contact_logits[b, ids], member[ids])
                            .topk(count)
                            .indices
                        ]
                    )
            predicted = (
                torch.unique(torch.cat(picks), sorted=True)
                if picks
                else torch.empty(0, dtype=torch.long, device=xyz.device)
            )
            selected = torch.unique(
                torch.cat((predicted, torch.nonzero(forced[b], as_tuple=False).flatten())),
                sorted=True,
            )
            selected_mask = torch.zeros(n, dtype=torch.bool, device=xyz.device)
            selected_mask[selected] = True
            masks.append(selected_mask)
            logits = point_context.new_full((n, self.direction_bins), -30.0)
            residual = point_context.new_zeros((n, self.direction_bins, 2))
            utility = point_context.new_zeros((n, self.direction_bins))
            direction_tokens = point_context.new_zeros(
                (n, self.direction_bins, self.direction_context.out_features)
            )
            if len(selected):
                tokens = self.direction_transformer(
                    self.direction_context(point_context[b, selected])[:, None]
                    + self.direction_embedding.weight[None]
                )
                # Autocast may produce fp16/bf16 Transformer outputs while these
                # sparse destination tensors inherit the fp32 point-context dtype.
                # Indexed assignment does not promote automatically.
                logits[selected] = self.direction_score(tokens).squeeze(-1).to(logits.dtype)
                residual[selected] = torch.tanh(self.direction_residual(tokens)).to(
                    residual.dtype
                )
                utility[selected] = self.utility(tokens).squeeze(-1).to(utility.dtype)
                direction_tokens[selected] = tokens.to(direction_tokens.dtype)
            logits_rows.append(logits)
            residual_rows.append(residual)
            utility_rows.append(utility)
            token_rows.append(direction_tokens)
        return {
            "object_logits": object_logits,
            "contact_logits": contact_logits,
            "direction_logits": torch.stack(logits_rows),
            "direction_residual": torch.stack(residual_rows),
            "utility_delta": torch.stack(utility_rows),
            "direction_point_mask": torch.stack(masks),
            "point_object_probability": assignment,
            "target_center_world": target_center,
            "region_center_world": region_center,
            "proposal_direction_feature": torch.stack(token_rows),
            "proposal_object_feature": object_context,
            "proposal_point_feature": point_context,
            "proposal_task_feature": task_token,
        }
