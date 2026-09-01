"""Finite-horizon Stage-C value, ranking, safety and physical-effect objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PushEffectivenessLoss(nn.Module):
    """New Stage-C objective; the historical class name remains an import alias."""

    def __init__(
        self,
        *,
        q_weight: float = 1.0,
        rank_weight: float = 0.25,
        safety_weight: float = 0.5,
        auxiliary_weight: float = 0.1,
        rank_margin: float = 0.02,
        delta_scales: tuple[float, ...] = (10.0, 1.0, 1.0, 5.0, 1.0),
    ) -> None:
        super().__init__()
        if len(delta_scales) != 5 or any(value <= 0 for value in delta_scales):
            raise ValueError("PUSH delta scales must contain five positive values")
        if min(q_weight, rank_weight, safety_weight, auxiliary_weight, rank_margin) < 0:
            raise ValueError("PUSH loss weights and rank margin must be nonnegative")
        self.q_weight = float(q_weight)
        self.rank_weight = float(rank_weight)
        self.safety_weight = float(safety_weight)
        self.auxiliary_weight = float(auxiliary_weight)
        self.rank_margin = float(rank_margin)
        self.register_buffer("delta_scales", torch.tensor(delta_scales, dtype=torch.float32))

    def _ranking(self, prediction: Tensor, target: Tensor, valid: Tensor, group: Tensor) -> Tensor:
        terms = []
        for group_id in torch.unique(group):
            members = torch.nonzero(group == group_id, as_tuple=False).flatten()
            if len(members) < 2:
                continue
            for horizon in range(prediction.shape[1]):
                ids = members[valid[members, horizon]]
                if len(ids) < 2:
                    continue
                truth = target[ids, horizon]
                difference = truth[:, None] - truth[None, :]
                left, right = torch.where(difference > self.rank_margin)
                if len(left):
                    predicted_difference = prediction[ids[left], horizon] - prediction[ids[right], horizon]
                    terms.append(torch.relu(self.rank_margin - predicted_difference).mean())
        return torch.stack(terms).mean() if terms else prediction.sum() * 0.0

    def forward(
        self,
        prediction: dict[str, Tensor],
        *,
        q_target: Tensor,
        q_valid: Tensor,
        safety_target: Tensor,
        safety_valid: Tensor,
        auxiliary_target: Tensor,
        auxiliary_valid: Tensor,
        group_index: Tensor,
    ) -> dict[str, Tensor]:
        q_prediction = prediction["q_value"]
        if q_prediction.shape != q_target.shape or q_valid.shape != q_target.shape:
            raise ValueError("PUSH Q prediction/target/mask shapes must align")
        if not bool(q_valid.any()):
            raise RuntimeError("Stage-C batch contains no valid offline Q targets")
        q_loss = torch.nn.functional.smooth_l1_loss(
            q_prediction[q_valid], q_target[q_valid], reduction="mean"
        )
        rank_loss = self._ranking(q_prediction, q_target, q_valid, group_index)
        if not bool(safety_valid.any()):
            raise RuntimeError("Stage-C batch contains no valid safety targets")
        safety_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction["safety_logit"][safety_valid], safety_target[safety_valid].float()
        )
        aux_mask = auxiliary_valid[:, None] & torch.isfinite(auxiliary_target)
        if bool(aux_mask.any()):
            scaled_target = auxiliary_target / self.delta_scales.to(auxiliary_target)
            scaled_prediction = prediction["potential_delta"] / self.delta_scales.to(
                prediction["potential_delta"]
            )
            auxiliary_loss = torch.nn.functional.smooth_l1_loss(
                scaled_prediction[aux_mask], scaled_target[aux_mask], reduction="mean"
            )
        else:
            auxiliary_loss = prediction["potential_delta"].sum() * 0.0
        total = (
            self.q_weight * q_loss
            + self.rank_weight * rank_loss
            + self.safety_weight * safety_loss
            + self.auxiliary_weight * auxiliary_loss
        )
        return {
            "push_effectiveness": total,
            "push_q_huber": q_loss.detach(),
            "push_rank": rank_loss.detach(),
            "push_safety_bce": safety_loss.detach(),
            "push_auxiliary_huber": auxiliary_loss.detach(),
            "push_q_supervised_count": q_valid.sum().detach().float(),
            "push_safety_supervised_count": safety_valid.sum().detach().float(),
        }
