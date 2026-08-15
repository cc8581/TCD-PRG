"""Supervision for the learned GraspNet-to-AG width adapter."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tcd_prg.geometry.se3 import parallel_jaw_rotation_distance


class AGWidthLoss(nn.Module):
    def __init__(self, translation_m: float = 0.02, rotation_deg: float = 20.0) -> None:
        super().__init__()
        self.translation_m = float(translation_m)
        self.rotation_deg = float(rotation_deg)

    def forward(
        self, prediction: dict[str, Tensor], labels: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        width = prediction["ag_width_m"]
        valid = prediction["valid"].bool()
        target = torch.zeros_like(width)
        matched = torch.zeros_like(valid)
        for row in range(width.shape[0]):
            candidates = torch.nonzero(valid[row], as_tuple=False).flatten()
            targets = torch.nonzero(
                labels["target_valid"][row] & labels["width_valid"][row],
                as_tuple=False,
            ).flatten()
            if not len(candidates) or not len(targets):
                continue
            translation = torch.cdist(
                prediction["translation_world"][row, candidates].float(),
                labels["translation_world"][row, targets].float(),
            )
            rotation = torch.rad2deg(
                parallel_jaw_rotation_distance(
                    prediction["rotation_matrix"][row, candidates, None].float(),
                    labels["rotation_matrix"][row, None, targets].float(),
                )
            )
            compatible = (
                (translation <= self.translation_m)
                & (rotation <= self.rotation_deg)
            )
            cost = translation / self.translation_m + rotation / self.rotation_deg
            cost = cost.masked_fill(~compatible, float("inf"))
            best_cost, best = cost.min(-1)
            row_matched = torch.isfinite(best_cost)
            selected = candidates[row_matched]
            if len(selected):
                matched[row, selected] = True
                target[row, selected] = labels["width_m"][
                    row, targets[best[row_matched]]
                ].to(target.dtype)

        if matched.any():
            error = (width[matched] - target[matched]).abs()
            loss = torch.nn.functional.smooth_l1_loss(width[matched], target[matched])
            p90 = torch.quantile(error.detach().float(), 0.9).to(width.dtype)
            mae = error.detach().mean()
            within_5 = (error.detach() <= 0.005).float().mean()
            within_10 = (error.detach() <= 0.010).float().mean()
        else:
            loss = width.sum() * 0.0
            mae = width.new_zeros(())
            p90 = width.new_zeros(())
            within_5 = width.new_zeros(())
            within_10 = width.new_zeros(())
        return {
            "loss": loss,
            "ag_width_targets": matched.float().sum().detach(),
            "ag_width_mae": mae,
            "ag_width_p90_error": p90,
            "ag_width_within_5mm": within_5,
            "ag_width_within_10mm": within_10,
        }
