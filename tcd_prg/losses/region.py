"""Focal BCE, Dice and target-region visibility losses."""

import torch
from torch import Tensor, nn

from .masked import masked_mean, safe_bce_with_logits


class TaskRegionLoss(nn.Module):
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 dice_weight: float = 1.0) -> None:
        super().__init__()
        self.alpha, self.gamma = focal_alpha, focal_gamma
        self.dice_weight = dice_weight

    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        logits = output["region_logits"]
        target = labels["region_target"].float()
        valid = labels["region_valid"].bool()
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, safe_target, reduction="none")
        probability = torch.sigmoid(logits)
        pt = torch.where(safe_target > 0.5, probability, 1 - probability)
        alpha = torch.where(safe_target > 0.5, self.alpha, 1 - self.alpha)
        focal = masked_mean(alpha * (1 - pt).pow(self.gamma) * bce, valid)
        intersection = (probability * safe_target * valid).sum(-1)
        denominator = ((probability + safe_target) * valid).sum(-1)
        row_valid = valid.any(-1)
        dice = masked_mean(1 - (2 * intersection + 1) / (denominator + 1), row_valid)
        visibility = safe_bce_with_logits(
            output["visibility_logit"], labels["visibility_target"].float(), labels["visibility_valid"]
        )
        return {"region_focal": focal, "region_dice": self.dice_weight * dice,
                "region_visibility": visibility}
