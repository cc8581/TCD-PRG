"""Single-head Stage-C PUSH-value objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def lexicographic_order_matrix(keys: Tensor, eps: float = 1e-6) -> Tensor:
    """Return [N,N] order: +1 row better, -1 worse, 0 tied."""
    if keys.ndim != 2:
        raise ValueError("rank keys must be [N,K]")
    count = keys.shape[0]
    order = torch.zeros((count, count), dtype=torch.int8, device=keys.device)
    undecided = torch.ones((count, count), dtype=torch.bool, device=keys.device)
    undecided.fill_diagonal_(False)
    difference = keys[:, None, :] - keys[None, :, :]
    for column in range(keys.shape[1]):
        delta = difference[..., column]
        decisive = undecided & (delta.abs() > float(eps))
        order[decisive & (delta > 0)] = 1
        order[decisive & (delta < 0)] = -1
        undecided &= ~decisive
    return order


class PushEffectivenessLoss(nn.Module):
    """Supervise only one scalar ``push_value``; no safety or auxiliary heads."""

    def __init__(
        self,
        *,
        value_weight: float = 1.0,
        rank_weight: float = 1.0,
        score_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if min(value_weight, rank_weight) < 0 or score_temperature <= 0:
            raise ValueError("PUSH value/rank weights must be nonnegative and temperature positive")
        self.value_weight = float(value_weight)
        self.rank_weight = float(rank_weight)
        self.score_temperature = float(score_temperature)

    def _ordinal(self, score: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        if not bool(valid.any()):
            return score.sum() * 0.0
        score = score[valid]
        target = target[valid]
        positive = target > 0.5
        negative = target < -0.5
        neutral = ~(positive | negative)
        terms: list[Tensor] = []
        normalizer = score.new_tensor(2.0).log()
        if bool(positive.any()):
            terms.append(
                torch.nn.functional.softplus(-score[positive] / self.score_temperature)
                / normalizer
            )
        if bool(negative.any()):
            terms.append(
                torch.nn.functional.softplus(score[negative] / self.score_temperature)
                / normalizer
            )
        if bool(neutral.any()):
            terms.append(
                torch.nn.functional.smooth_l1_loss(
                    score[neutral], torch.zeros_like(score[neutral]), reduction="none"
                )
            )
        return torch.cat(terms).mean() if terms else score.sum() * 0.0

    def _ranking(
        self,
        score: Tensor,
        rank_key: Tensor,
        rank_valid: Tensor,
        group_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        terms: list[Tensor] = []
        pair_count = score.new_zeros(())
        normalizer = score.new_tensor(2.0).log()
        for group_id in torch.unique(group_index):
            members = torch.nonzero(group_index == group_id, as_tuple=False).flatten()
            ids = members[rank_valid[members]]
            if len(ids) < 2:
                continue
            order = lexicographic_order_matrix(rank_key[ids])
            left, right = torch.where(order > 0)
            if len(left):
                difference = score[ids[left]] - score[ids[right]]
                terms.append(
                    (
                        torch.nn.functional.softplus(
                            -difference / self.score_temperature
                        )
                        / normalizer
                    ).mean()
                )
                pair_count += float(len(left))
        loss = torch.stack(terms).mean() if terms else score.sum() * 0.0
        return loss, pair_count

    def forward(
        self,
        prediction: dict[str, Tensor],
        *,
        value_target: Tensor,
        value_valid: Tensor,
        rank_key: Tensor,
        rank_valid: Tensor,
        group_index: Tensor,
    ) -> dict[str, Tensor]:
        score = prediction["push_value"]
        if score.shape != value_target.shape or value_valid.shape != value_target.shape:
            raise ValueError("PUSH value score/target/mask shapes must align")
        if rank_valid.shape != value_target.shape:
            raise ValueError("PUSH rank-valid mask must align with value target")
        if rank_key.shape[:1] != score.shape or rank_key.ndim != 2:
            raise ValueError("PUSH rank keys must be [A,K]")
        if group_index.shape != score.shape:
            raise ValueError("PUSH group index must align with action score")
        if bool(value_valid.any()):
            if not bool(torch.isfinite(value_target[value_valid]).all()):
                raise ValueError("valid PUSH value targets must be finite")
            if not bool(torch.isfinite(rank_key[rank_valid]).all()):
                raise ValueError("valid PUSH rank keys must be finite")

        value_loss = self._ordinal(score, value_target, value_valid)
        rank_loss, pair_count = self._ranking(score, rank_key, rank_valid, group_index)
        total = self.value_weight * value_loss + self.rank_weight * rank_loss
        return {
            "push_effectiveness": total,
            "push_value_ordinal": value_loss.detach(),
            "push_rank": rank_loss.detach(),
            "push_value_supervised_count": value_valid.sum().detach().float(),
            "push_rank_pair_count": pair_count.detach(),
        }
