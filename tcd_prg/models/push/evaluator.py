"""Action-conditioned PUSH effectiveness evaluator."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def nearest_object_contact_point(
    xyz: Tensor,
    point_mask: Tensor,
    object_probability: Tensor,
    object_index: int,
    contact_world: Tensor,
) -> Tensor:
    """Use the same hard-owner/membership domain as the deployment decoder."""
    if object_index < 0 or object_index >= object_probability.shape[0]:
        raise IndexError(f"PUSH evaluator acted_object out of range: {object_index}")
    membership = object_probability[object_index]
    owner = object_probability.argmax(0)
    domain = point_mask.bool() & (owner == object_index) & (membership >= 0.5)
    if not bool(domain.any()):
        domain = point_mask.bool() & (owner == object_index)
    points = torch.nonzero(domain, as_tuple=False).flatten()
    if not len(points):
        raise ValueError(f"PUSH evaluator object {object_index} has no visible owned scene point")
    distance = torch.linalg.vector_norm(xyz[points] - contact_world, dim=-1)
    return points[distance.argmin()]


def canonical_push_direction(
    direction_world: Tensor,
    num_bins: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Canonicalize a physical final direction independently of proposal parameters."""
    if direction_world.ndim != 2 or direction_world.shape[-1] != 3:
        raise ValueError("PUSH direction_world must be [K,3]")
    if num_bins <= 0:
        raise ValueError("PUSH evaluator requires at least one direction bin")
    planar_raw = direction_world[:, :2]
    planar_norm = torch.linalg.vector_norm(planar_raw, dim=-1, keepdim=True)
    if not bool(torch.isfinite(planar_norm).all()) or bool((planar_norm <= 1e-8).any()):
        raise ValueError("PUSH evaluator requires finite non-zero planar directions")
    actual = planar_raw / planar_norm
    angle = torch.remainder(torch.atan2(actual[:, 1], actual[:, 0]), 2.0 * math.pi)
    direction_bin = (
        torch.floor(angle * num_bins / (2.0 * math.pi)).long().clamp_max(num_bins - 1)
    )
    center_angle = (direction_bin.to(angle.dtype) + 0.5) * 2.0 * math.pi / num_bins
    center = torch.stack((torch.cos(center_angle), torch.sin(center_angle)), dim=-1)
    residual = actual - center
    canonical_world = torch.cat(
        (actual, direction_world.new_zeros((len(actual), 1))), dim=-1
    )
    return canonical_world, direction_bin, residual


class PushEffectivenessEvaluator(nn.Module):
    """Score a complete physical PUSH independently of Stage-C parameterization."""

    def __init__(self, feature_dim: int = 256, direction_dim: int = 64) -> None:
        super().__init__()
        input_dim = direction_dim + 3 * feature_dim + 12
        self.network = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1),
        )

    def forward(
        self,
        push: dict[str, Tensor],
        candidates: dict[str, Tensor],
        *,
        batch_index: int,
    ) -> Tensor:
        point = candidates["point_index"].long()
        acted_object = candidates["object"].long()
        num_bins = push["proposal_direction_feature"].shape[2]
        direction_world, direction_bin, direction_residual = canonical_push_direction(
            candidates["direction_world"], num_bins
        )
        direction_feature = push["proposal_direction_feature"][
            batch_index, point, direction_bin
        ]
        object_feature = push["proposal_object_feature"][batch_index, acted_object]
        point_feature = push["proposal_point_feature"][batch_index, point]
        task_feature = push["proposal_task_feature"][batch_index].expand(len(point), -1)
        contact = candidates["contact_world"]
        target_rel = contact - push["target_center_world"][batch_index]
        region_rel = contact - push["region_center_world"][batch_index]
        action_geometry = torch.cat(
            (
                target_rel,
                region_rel,
                direction_world,
                direction_residual,
                candidates["push_distance"][:, None],
            ),
            dim=-1,
        )
        evaluator_input = torch.cat(
            (
                direction_feature.detach(),
                object_feature.detach(),
                point_feature.detach(),
                action_geometry,
                task_feature.detach(),
            ),
            dim=-1,
        )
        return self.network(evaluator_input).squeeze(-1)

    def score_exact_actions(
        self,
        sensor: dict[str, Tensor],
        push: dict[str, Tensor],
        *,
        batch_index: Tensor,
        acted_object: Tensor,
        contact_world: Tensor,
        direction_world: Tensor,
        push_distance: Tensor,
        point_index: Tensor | None = None,
    ) -> Tensor:
        """Condition on logged actions using the same final-action encoding as deployment."""
        rows: list[Tensor] = []
        for action in range(len(batch_index)):
            row = int(batch_index[action])
            if row < 0 or row >= sensor["xyz"].shape[0]:
                raise IndexError(f"PUSH evaluator batch_index out of range: {row}")
            object_index = int(acted_object[action])
            if object_index < 0 or object_index >= push["proposal_object_feature"].shape[1]:
                raise IndexError(f"PUSH evaluator acted_object out of range: {object_index}")
            if point_index is None:
                point = nearest_object_contact_point(
                    sensor["xyz"][row],
                    sensor["point_mask"][row],
                    push["point_object_probability"][row],
                    object_index,
                    contact_world[action],
                )
            else:
                point = point_index[action].long()
                if int(point) < 0 or int(point) >= sensor["xyz"].shape[1]:
                    raise IndexError(f"PUSH evaluator point_index out of range: {int(point)}")
                owner = push["point_object_probability"][row, :, point].argmax()
                if int(owner) != object_index:
                    raise ValueError(
                        "PUSH evaluator exact-action anchor does not belong to acted_object"
                    )
            candidate = {
                "point_index": point[None],
                "object": acted_object[action, None].long(),
                "contact_world": contact_world[action, None],
                "direction_world": direction_world[action, None],
                "push_distance": push_distance[action, None],
            }
            rows.append(self.forward(push, candidate, batch_index=row))
        return torch.cat(rows) if rows else contact_world.new_empty(0)
