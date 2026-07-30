"""Task-conditioned 6D grasp proposal head compatible with GraspNet concepts."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TaskGraspProposalHead(nn.Module):
    """Predict a canonical 6D contact frame and AG total opening.

    The dataset canonicalizes the translation at the two-contact midpoint.
    Its legacy ``grasp_depth_m`` is the offset to the source Panda frame, not
    an AG-160-95 TCP quantity, and is deliberately not predicted here.
    """

    def __init__(self, dim: int = 256, rotation_bins: int = 12) -> None:
        super().__init__()
        self.rotation_bins = rotation_bins
        self.shared = nn.Sequential(nn.Linear(3 * dim + 1, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim), nn.GELU())
        self.contact = nn.Linear(dim, 1)
        self.approach = nn.Linear(dim, 3)
        self.rotation = nn.Linear(dim, rotation_bins)
        self.width = nn.Linear(dim, 1)
        self.confidence = nn.Linear(dim, 1)
        self.task_compatibility = nn.Linear(dim, 1)

    def forward(
        self,
        point_features: Tensor,
        target_token: Tensor,
        task_token: Tensor,
        region_probability: Tensor,
        target_mask: Tensor,
        generic_remove: bool = False,
    ) -> dict[str, Tensor]:
        n = point_features.shape[1]
        task_context = torch.zeros_like(task_token) if generic_remove else task_token
        x = torch.cat(
            (
                point_features,
                target_token[:, None].expand(-1, n, -1),
                task_context[:, None].expand(-1, n, -1),
                region_probability.unsqueeze(-1),
            ),
            -1,
        )
        x = self.shared(x)
        contact = self.contact(x).squeeze(-1).masked_fill(~target_mask, -30.0)
        approach = torch.nn.functional.normalize(self.approach(x), dim=-1, eps=1e-6)
        return {
            "contact_logits": contact,
            "approach_direction": approach,
            "rotation_logits": self.rotation(x),
            "width_raw": self.width(x).squeeze(-1),
            "proposal_confidence_logit": self.confidence(x).squeeze(-1),
            "task_compatibility_logit": self.task_compatibility(x).squeeze(-1),
        }

    @staticmethod
    def decode_width(width_raw: Tensor, min_width_m: float, max_width_m: float) -> Tensor:
        return min_width_m + torch.sigmoid(width_raw) * (max_width_m - min_width_m)

    def topk(self, output: dict[str, Tensor], xyz: Tensor, k: int) -> dict[str, Tensor]:
        score = (
            torch.sigmoid(output["contact_logits"])
            * torch.sigmoid(output["proposal_confidence_logit"])
            * torch.sigmoid(output["task_compatibility_logit"])
        )
        k = min(k, score.shape[1])
        values, index = score.topk(k, dim=1)
        row = torch.arange(score.shape[0], device=score.device)[:, None]
        return {
            "point_index": index,
            "score": values,
            "contact_point": xyz[row, index],
            "approach_direction": output["approach_direction"][row, index],
            "rotation_logits": output["rotation_logits"][row, index],
            "width_raw": output["width_raw"][row, index],
        }


class GlobalGraspProposalHead(nn.Module):
    """Task-free dense global grasp head with multiple modes per contact point.

    ``input_mode='scene_only'`` consumes only neutral point/global features.
    ``input_mode='instance_assisted'`` additionally gathers the neutral token
    pooled from the externally supplied instance mask.  Neither mode accepts a
    target mask, task token, task-region feature, or graph feature.
    """

    VALID_INPUT_MODES = {"scene_only", "instance_assisted"}

    def __init__(
        self, dim: int = 256, rotation_bins: int = 12, modes_per_point: int = 4,
        input_mode: str = "scene_only",
    ) -> None:
        super().__init__()
        if modes_per_point < 2:
            raise ValueError("Global grasp prediction requires at least two modes per point")
        if input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(f"Unsupported global grasp input mode: {input_mode}")
        self.rotation_bins = rotation_bins
        self.modes_per_point = modes_per_point
        self.input_mode = input_mode
        self.generic_token = nn.Parameter(torch.zeros(dim))
        self.mode_embedding = nn.Parameter(torch.zeros(modes_per_point, dim))
        nn.init.normal_(self.generic_token, std=0.02)
        nn.init.normal_(self.mode_embedding, std=0.02)
        self.shared = nn.Sequential(
            nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim), nn.GELU()
        )
        self.contact = nn.Linear(dim, 1)
        self.mode_shared = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.approach = nn.Linear(dim, 3)
        self.rotation = nn.Linear(dim, rotation_bins)
        self.width = nn.Linear(dim, 1)
        self.scene_confidence = nn.Linear(dim, 1)
        self.intrinsic_confidence = nn.Linear(dim, 1)

    def forward(
        self, point_features: Tensor, object_tokens: Tensor, global_scene_token: Tensor,
        instance_id: Tensor, point_domain: Tensor,
    ) -> dict[str, Tensor]:
        b, n, dim = point_features.shape
        safe_instance = instance_id.clamp(0, object_tokens.shape[1] - 1)
        row = torch.arange(b, device=point_features.device)[:, None]
        per_point_object = object_tokens[row, safe_instance]
        if self.input_mode == "scene_only":
            per_point_object = torch.zeros_like(per_point_object)
        context = torch.cat((
            point_features,
            per_point_object,
            global_scene_token[:, None].expand(-1, n, -1),
            self.generic_token[None, None].expand(b, n, -1),
        ), -1)
        shared = self.shared(context)
        contact = self.contact(shared).squeeze(-1).masked_fill(~point_domain, -30.0)
        modes = self.mode_shared(shared[:, :, None] + self.mode_embedding[None, None])
        approach = torch.nn.functional.normalize(self.approach(modes), dim=-1, eps=1e-6)
        return {
            "contact_logits": contact,
            "approach_direction": approach,
            "rotation_logits": self.rotation(modes),
            "width_raw": self.width(modes).squeeze(-1),
            "scene_confidence_logit": self.scene_confidence(modes).squeeze(-1),
            "intrinsic_confidence_logit": self.intrinsic_confidence(modes).squeeze(-1),
            "point_domain": point_domain,
        }

    @staticmethod
    def decode_width(width_raw: Tensor, min_width_m: float, max_width_m: float) -> Tensor:
        return min_width_m + torch.sigmoid(width_raw) * (max_width_m - min_width_m)
