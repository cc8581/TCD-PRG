"""Action-conditioned PUSH effectiveness evaluator."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PushEffectivenessEvaluator(nn.Module):
    """Score a decoded complete PUSH without updating proposal features."""

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
        direction_bin = candidates["direction_bin"].long()
        acted_object = candidates["object"].long()
        direction_feature = push["proposal_direction_feature"][batch_index, point, direction_bin]
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
                candidates["direction_world"],
                candidates["direction_residual"],
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
    ) -> Tensor:
        """Condition on logged actions, preserving their exact scene-state pairing."""
        rows: list[Tensor] = []
        num_bins = push["proposal_direction_feature"].shape[2]
        for action in range(len(batch_index)):
            row = int(batch_index[action])
            if row < 0 or row >= sensor["xyz"].shape[0]:
                raise IndexError(f"PUSH evaluator batch_index out of range: {row}")
            object_index = int(acted_object[action])
            if object_index < 0 or object_index >= push["proposal_object_feature"].shape[1]:
                raise IndexError(f"PUSH evaluator acted_object out of range: {object_index}")
            valid_points = torch.nonzero(sensor["point_mask"][row], as_tuple=False).flatten()
            if not len(valid_points):
                raise ValueError("PUSH evaluator received a scene row with no valid points")
            distance = torch.linalg.vector_norm(
                sensor["xyz"][row, valid_points] - contact_world[action], dim=-1
            )
            point = valid_points[distance.argmin()]
            planar_raw = direction_world[action, :2]
            planar_norm = torch.linalg.vector_norm(planar_raw)
            if not bool(torch.isfinite(planar_norm)) or float(planar_norm) <= 1e-8:
                raise ValueError("PUSH evaluator requires a finite non-zero planar direction")
            actual = planar_raw / planar_norm
            canonical_direction = torch.cat((actual, direction_world.new_zeros(1)), dim=0)
            angle = torch.atan2(actual[1], actual[0])
            angle = torch.remainder(angle, 2.0 * math.pi)
            direction_bin = (
                torch.floor(angle * num_bins / (2.0 * math.pi)).long().clamp_max(num_bins - 1)
            )
            center_angle = (direction_bin.to(angle.dtype) + 0.5) * 2.0 * math.pi / num_bins
            center = torch.stack((torch.cos(center_angle), torch.sin(center_angle)))
            candidate = {
                "point_index": point[None],
                "direction_bin": direction_bin[None],
                "object": acted_object[action, None].long(),
                "contact_world": contact_world[action, None],
                "direction_world": canonical_direction[None],
                "direction_residual": (actual - center)[None],
                "push_distance": push_distance[action, None],
            }
            rows.append(self.forward(push, candidate, batch_index=row))
        return torch.cat(rows) if rows else contact_world.new_empty(0)
