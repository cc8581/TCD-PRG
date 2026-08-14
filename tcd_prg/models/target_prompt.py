"""Observable 3D target-prompt selection for predicted instance queries.

A prompt identifies *which physical instance* is the task target.  The task
region remains a separate semantic condition.  The selector is deliberately
parameter-free so V1 checkpoints remain state-dict compatible.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class TargetSelectionOutput:
    logits: Tensor                  # [B,Q]
    weights: Tensor                 # [B,Q]
    query_index: Tensor             # [B]
    margin: Tensor                  # [B] top1-top2 logit margin
    prompt_support: Tensor          # [B,Q]
    positive_prompt_support: Tensor # [B,Q]
    negative_prompt_support: Tensor # [B,Q]
    used_prompt: Tensor             # [B]
    used_reid: Tensor               # [B]


class TargetPromptSelector(nn.Module):
    """Select a predicted object query from 3D positive/negative prompts.

    The dominant term is local instance-mask support around the prompt.  Closed
    vocabulary category/objectness are priors, not target identity.  Optional
    cross-frame re-identification is a continuity prior only.
    """

    def __init__(
        self,
        *,
        radius_m: float = 0.030,
        sigma_m: float = 0.012,
        prompt_weight: float = 4.0,
        negative_prompt_weight: float = 1.0,
        category_weight: float = 1.0,
        objectness_weight: float = 0.5,
        center_weight: float = 0.5,
        learned_weight: float = 0.20,
        reid_weight: float = 0.75,
        reid_center_weight: float = 0.35,
        reid_max_center_distance_m: float = 0.15,
        temperature: float = 0.25,
        hard_inference: bool = True,
    ) -> None:
        super().__init__()
        if radius_m <= 0 or sigma_m <= 0:
            raise ValueError("prompt radius/sigma must be positive")
        if temperature <= 0:
            raise ValueError("target selection temperature must be positive")
        self.radius_m = float(radius_m)
        self.sigma_m = float(sigma_m)
        self.prompt_weight = float(prompt_weight)
        self.negative_prompt_weight = float(negative_prompt_weight)
        self.category_weight = float(category_weight)
        self.objectness_weight = float(objectness_weight)
        self.center_weight = float(center_weight)
        self.learned_weight = float(learned_weight)
        self.reid_weight = float(reid_weight)
        self.reid_center_weight = float(reid_center_weight)
        self.reid_max_center_distance_m = float(reid_max_center_distance_m)
        self.temperature = float(temperature)
        self.hard_inference = bool(hard_inference)

    @staticmethod
    def _empty_prompt(
        xyz: Tensor,
        prompt_xyz: Tensor | None,
        prompt_label: Tensor | None,
        prompt_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b = xyz.shape[0]
        if prompt_xyz is None:
            prompt_xyz = xyz.new_zeros((b, 1, 3))
        if prompt_xyz.ndim == 2:
            prompt_xyz = prompt_xyz[:, None]
        if prompt_xyz.ndim != 3 or prompt_xyz.shape[0] != b or prompt_xyz.shape[-1] != 3:
            raise ValueError("target_prompt_xyz must have shape [B,P,3]")
        p = prompt_xyz.shape[1]
        if prompt_label is None:
            prompt_label = torch.ones((b, p), dtype=torch.long, device=xyz.device)
        if prompt_valid is None:
            prompt_valid = torch.zeros((b, p), dtype=torch.bool, device=xyz.device)
        if prompt_label.shape != (b, p) or prompt_valid.shape != (b, p):
            raise ValueError("target prompt label/valid must have shape [B,P]")
        return prompt_xyz, prompt_label.long(), prompt_valid.bool()

    def _prompt_support(
        self,
        instance_probability: Tensor,
        centers_world: Tensor,
        xyz: Tensor,
        point_mask: Tensor,
        prompt_xyz: Tensor,
        prompt_label: Tensor,
        prompt_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # distance: [B,P,N]
        distance = torch.cdist(prompt_xyz.float(), xyz.float()).to(xyz.dtype)
        neighborhood = (
            (distance <= self.radius_m)
            & point_mask[:, None]
            & prompt_valid[:, :, None]
            & torch.isfinite(prompt_xyz).all(-1)[:, :, None]
        )
        gaussian = torch.exp(
            -0.5 * distance.square() / max(self.sigma_m ** 2, 1e-12)
        ) * neighborhood.to(distance.dtype)
        denominator = gaussian.sum(-1).clamp_min(1e-6)  # [B,P]
        # [B,P,Q]
        per_prompt = torch.einsum(
            "bpn,bqn->bpq", gaussian, instance_probability
        ) / denominator[:, :, None]

        positive = prompt_valid & (prompt_label > 0)
        negative = prompt_valid & (prompt_label <= 0)
        positive_count = positive.sum(-1, keepdim=True).clamp_min(1)
        negative_count = negative.sum(-1, keepdim=True).clamp_min(1)
        positive_support = (
            per_prompt * positive[:, :, None].to(per_prompt.dtype)
        ).sum(1) / positive_count.to(per_prompt.dtype)
        negative_support = (
            per_prompt * negative[:, :, None].to(per_prompt.dtype)
        ).sum(1) / negative_count.to(per_prompt.dtype)
        signed_support = positive_support - self.negative_prompt_weight * negative_support

        # Distance to the closest positive prompt is a weak spatial prior.  It is
        # not allowed to dominate the instance-mask support.
        center_distance = torch.cdist(centers_world.float(), prompt_xyz.float()).to(xyz.dtype)
        center_distance = center_distance.masked_fill(~positive[:, None], float("inf"))
        nearest_center = center_distance.amin(-1)
        has_positive = positive.any(-1)
        # The clicked point may be on a handle/rim far from the object centroid;
        # use a broad scale rather than the small prompt-neighborhood radius.
        center_scale = max(4.0 * self.radius_m, 0.10)
        center_score = -nearest_center / center_scale
        center_score = torch.where(
            has_positive[:, None] & torch.isfinite(center_score),
            center_score.clamp(min=-4.0, max=0.0),
            torch.zeros_like(center_score),
        )
        # Prompt is only trusted if a valid neighborhood exists around at least
        # one positive point.
        positive_has_neighbors = (
            (neighborhood & positive[:, :, None]).any(-1).any(-1)
        )
        return (
            signed_support,
            positive_support,
            negative_support,
            center_score,
            positive_has_neighbors,
        )

    def forward(
        self,
        *,
        instance_probability: Tensor,
        objectness_logits: Tensor,
        category_logits: Tensor,
        centers_world: Tensor,
        object_tokens: Tensor,
        xyz: Tensor,
        point_mask: Tensor,
        task_category_id: Tensor,
        target_prompt_xyz: Tensor | None = None,
        target_prompt_label: Tensor | None = None,
        target_prompt_valid: Tensor | None = None,
        target_reid_token: Tensor | None = None,
        target_reid_center: Tensor | None = None,
        target_reid_valid: Tensor | None = None,
        learned_semantic_logit: Tensor | None = None,
    ) -> TargetSelectionOutput:
        prompt_xyz, prompt_label, prompt_valid = self._empty_prompt(
            xyz, target_prompt_xyz, target_prompt_label, target_prompt_valid
        )
        (
            prompt_support,
            positive_support,
            negative_support,
            center_score,
            used_prompt,
        ) = self._prompt_support(
            instance_probability,
            centers_world,
            xyz,
            point_mask,
            prompt_xyz,
            prompt_label,
            prompt_valid,
        )

        requested = task_category_id.clamp(0, category_logits.shape[-1] - 1)
        # Bounded priors keep an imperfect category/objectness prediction from
        # overpowering an explicit geometric prompt.
        category_prior = torch.softmax(category_logits, -1).gather(
            2,
            requested[:, None, None].expand(-1, category_logits.shape[1], 1),
        ).squeeze(-1)
        object_prior = torch.sigmoid(objectness_logits)

        logits = (
            self.prompt_weight * prompt_support
            + self.category_weight * category_prior
            + self.objectness_weight * object_prior
            + self.center_weight * center_score
        )
        if learned_semantic_logit is not None and self.learned_weight:
            logits = logits + self.learned_weight * torch.tanh(learned_semantic_logit)

        b, q, dim = object_tokens.shape
        if target_reid_valid is None:
            target_reid_valid = torch.zeros(b, dtype=torch.bool, device=xyz.device)
        target_reid_valid = target_reid_valid.bool()
        used_reid = target_reid_valid.clone()
        if target_reid_token is not None and target_reid_valid.any():
            reference = torch.nn.functional.normalize(target_reid_token, dim=-1)
            candidates = torch.nn.functional.normalize(object_tokens, dim=-1)
            similarity = torch.einsum("bd,bqd->bq", reference, candidates)
            similarity = torch.where(
                target_reid_valid[:, None], similarity, torch.zeros_like(similarity)
            )
            logits = logits + self.reid_weight * similarity
        if target_reid_center is not None and target_reid_valid.any():
            distance = torch.linalg.vector_norm(
                centers_world - target_reid_center[:, None], dim=-1
            )
            normalized = -distance / max(self.reid_max_center_distance_m, 1e-6)
            normalized = normalized.clamp(min=-3.0, max=0.0)
            normalized = torch.where(
                target_reid_valid[:, None], normalized, torch.zeros_like(normalized)
            )
            logits = logits + self.reid_center_weight * normalized

        soft_weights = torch.softmax(logits / self.temperature, -1)
        query_index = logits.argmax(-1)
        if not self.training and self.hard_inference:
            weights = torch.nn.functional.one_hot(query_index, q).to(soft_weights.dtype)
        else:
            weights = soft_weights
        if q > 1:
            top2 = logits.topk(2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
        else:
            margin = torch.full((b,), float("inf"), device=xyz.device, dtype=xyz.dtype)

        return TargetSelectionOutput(
            logits=logits,
            weights=weights,
            query_index=query_index,
            margin=margin,
            prompt_support=prompt_support,
            positive_prompt_support=positive_support,
            negative_prompt_support=negative_support,
            used_prompt=used_prompt,
            used_reid=used_reid,
        )
