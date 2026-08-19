"""Task-conditioned hierarchical PUSH prediction without Graph/Verifier/Router."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class PushHead(nn.Module):
    """Predict PUSH object -> contact -> direction -> utility.

    The branch consumes only prediction-side perception outputs. During training,
    ``forced_direction_point_mask`` may add evaluated GT contact locations to the
    sparse direction set; no GT coordinates or object ids are otherwise consumed
    by this module.
    """

    def __init__(
        self,
        dim: int = 256,
        direction_bins: int = 16,
        direction_dim: int = 64,
        direction_layers: int = 1,
        direction_heads: int = 4,
        direction_contact_topk: int = 32,
        direction_object_topk: int = 4,
    ) -> None:
        super().__init__()
        self.direction_bins = int(direction_bins)
        self.direction_contact_topk = int(direction_contact_topk)
        self.direction_object_topk = int(direction_object_topk)
        if self.direction_contact_topk <= 0 or self.direction_object_topk <= 0:
            raise ValueError("direction sparse budgets must be positive")

        # [object-target xyz + norm, region-target xyz + norm]
        self.relative_geometry = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.object_context = nn.Sequential(
            nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.object_pointer = nn.Linear(dim, 1)

        # point feature + object/task/target/region context + two relative xyz
        # vectors + target/region probabilities.
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
    def _weighted_pool(features: Tensor, weight: Tensor) -> Tensor:
        weight = weight.clamp_min(0)
        return torch.einsum("bn,bnd->bd", weight, features) / weight.sum(
            -1, keepdim=True
        ).clamp_min(1e-6)

    @staticmethod
    def _weighted_center(xyz: Tensor, weight: Tensor) -> Tensor:
        weight = weight.clamp_min(0)
        return torch.einsum("bn,bnd->bd", weight, xyz) / weight.sum(
            -1, keepdim=True
        ).clamp_min(1e-6)

    def forward(
        self,
        point_features: Tensor,
        xyz: Tensor,
        instance_probability: Tensor,
        point_mask: Tensor,
        object_tokens: Tensor,
        object_mask: Tensor,
        task_token: Tensor,
        target_token: Tensor,
        target_instance_probability: Tensor,
        region_probability: Tensor,
        forced_direction_point_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        valid_points = point_mask.to(xyz.dtype)
        raw_assignment = (
            instance_probability
            * object_mask[:, :, None].to(instance_probability.dtype)
            * valid_points[:, None]
        )
        # Object geometry must use each query's own predicted mask probability.
        # Cross-query normalization would give low-probability soft tails the same
        # total point mass as confident object pixels and bias object centers.
        object_mass = raw_assignment.sum(-1, keepdim=True).clamp_min(1e-6)
        object_center = torch.einsum(
            "bqn,bnd->bqd", raw_assignment, xyz
        ) / object_mass

        # For blending object context into each point, normalize across queries;
        # padded points remain zero.
        assignment = raw_assignment / raw_assignment.sum(
            1, keepdim=True
        ).clamp_min(1e-6)

        target_weight = target_instance_probability * valid_points
        target_center = self._weighted_center(xyz, target_weight)

        # RegionHead.region_probability already includes the predicted target
        # probability. Multiplying it by target probability again would square the
        # target confidence and suppress boundary points.
        region_weight = region_probability * valid_points
        region_token = self._weighted_pool(point_features, region_weight)
        raw_region_center = self._weighted_center(xyz, region_weight)
        region_visible = region_weight.sum(-1, keepdim=True) > 1e-6
        # If the requested functional region is currently invisible, an origin
        # fallback would create a fictitious large geometric offset. Treat the
        # unknown region center as the target center; the zero region token still
        # tells the network that no region evidence is visible.
        region_center = torch.where(region_visible, raw_region_center, target_center)

        object_to_target = object_center - target_center[:, None]
        region_to_target = region_center - target_center
        region_to_target_expanded = region_to_target[:, None].expand_as(object_to_target)
        rel_feature = self.relative_geometry(
            torch.cat(
                (
                    object_to_target,
                    torch.linalg.vector_norm(object_to_target, dim=-1, keepdim=True),
                    region_to_target_expanded,
                    torch.linalg.vector_norm(
                        region_to_target_expanded, dim=-1, keepdim=True
                    ),
                ),
                -1,
            )
        )

        q = object_tokens.shape[1]
        object_context = self.object_context(
            torch.cat(
                (
                    object_tokens,
                    target_token[:, None].expand(-1, q, -1),
                    task_token[:, None].expand(-1, q, -1),
                    region_token[:, None].expand(-1, q, -1),
                    rel_feature,
                ),
                -1,
            )
        )
        object_logits = self.object_pointer(object_context).squeeze(-1)
        object_logits = object_logits.masked_fill(~object_mask, -30.0)

        point_object = torch.einsum("bqn,bqd->bnd", assignment, object_context)
        point_to_target = xyz - target_center[:, None]
        point_to_region = xyz - region_center[:, None]
        n = xyz.shape[1]
        point_context = self.point(
            torch.cat(
                (
                    point_features,
                    point_object,
                    task_token[:, None].expand(-1, n, -1),
                    target_token[:, None].expand(-1, n, -1),
                    region_token[:, None].expand(-1, n, -1),
                    point_to_target,
                    point_to_region,
                    target_instance_probability[..., None],
                    region_probability[..., None],
                ),
                -1,
            )
        )
        contact_logits = self.contact(point_context).squeeze(-1)
        contact_logits = contact_logits.masked_fill(~point_mask, -30.0)

        if forced_direction_point_mask is None:
            forced_direction_point_mask = torch.zeros_like(point_mask)
        else:
            forced_direction_point_mask = forced_direction_point_mask.bool() & point_mask

        direction_logits_rows: list[Tensor] = []
        direction_residual_rows: list[Tensor] = []
        utility_rows: list[Tensor] = []
        direction_point_masks: list[Tensor] = []

        # Q is small. Sparse selection is intentionally per object so one large
        # exposed object cannot consume the full direction budget.  Selection is
        # restricted to the predicted object's hard point domain; otherwise tiny
        # sigmoid mask tails would make every object select unrelated scene points.
        hard_owner = raw_assignment.argmax(1)
        for batch_row in range(point_context.shape[0]):
            selected_per_object: list[Tensor] = []
            active_objects = torch.nonzero(
                object_mask[batch_row], as_tuple=False
            ).flatten()
            if active_objects.numel() > self.direction_object_topk:
                active_objects = active_objects[
                    object_logits[batch_row, active_objects].topk(
                        self.direction_object_topk
                    ).indices
                ]
            for object_index in active_objects:
                membership = raw_assignment[batch_row, object_index]
                domain = (
                    point_mask[batch_row]
                    & (hard_owner[batch_row] == object_index)
                    & (membership >= 0.5)
                )
                # Preserve a recovery path for under-confident instance masks while
                # still forbidding points owned by another predicted instance.
                if not bool(domain.any()):
                    domain = point_mask[batch_row] & (hard_owner[batch_row] == object_index)
                point_ids = torch.nonzero(domain, as_tuple=False).flatten()
                count = min(self.direction_contact_topk, int(point_ids.numel()))
                if count:
                    joint = (
                        contact_logits[batch_row, point_ids]
                        + membership[point_ids].clamp_min(1e-6).log()
                    )
                    selected_per_object.append(
                        point_ids[torch.topk(joint, k=count).indices]
                    )

            predicted = (
                torch.unique(torch.cat(selected_per_object), sorted=True)
                if selected_per_object
                else torch.empty(0, dtype=torch.long, device=point_context.device)
            )
            predicted = predicted[point_mask[batch_row, predicted]]
            forced = torch.nonzero(
                forced_direction_point_mask[batch_row], as_tuple=False
            ).flatten()
            selected = torch.unique(torch.cat((predicted, forced)), sorted=True)

            selected_mask = torch.zeros(n, dtype=torch.bool, device=point_context.device)
            selected_mask[selected] = True
            direction_point_masks.append(selected_mask)

            logits = point_context.new_full((n, self.direction_bins), -30.0)
            residual = point_context.new_zeros((n, self.direction_bins, 2))
            utility = point_context.new_zeros((n, self.direction_bins))
            if selected.numel():
                direction_tokens = (
                    self.direction_context(point_context[batch_row, selected])[:, None]
                    + self.direction_embedding.weight[None]
                )
                direction_tokens = self.direction_transformer(direction_tokens)
                logits[selected] = self.direction_score(direction_tokens).squeeze(-1)
                residual[selected] = torch.tanh(self.direction_residual(direction_tokens))
                utility[selected] = self.utility(direction_tokens).squeeze(-1)

            direction_logits_rows.append(logits)
            direction_residual_rows.append(residual)
            utility_rows.append(utility)

        return {
            "object_logits": object_logits,
            "contact_logits": contact_logits,
            "direction_logits": torch.stack(direction_logits_rows),
            "direction_residual": torch.stack(direction_residual_rows),
            "utility_delta": torch.stack(utility_rows),
            "direction_point_mask": torch.stack(direction_point_masks),
            "point_object_probability": assignment,
            "target_center_world": target_center,
            "region_center_world": region_center,
        }
