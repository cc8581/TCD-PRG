"""Class-balanced final-executability grasp verifier loss."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GraspVerifierLoss(nn.Module):
    @staticmethod
    def _ranking_metrics(
        logits: Tensor, positive: Tensor, negative: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute exact batch AUROC/AP and mark single-class batches invalid."""

        positive_scores = logits[positive].detach()
        negative_scores = logits[negative].detach()
        valid = logits.new_tensor(bool(positive_scores.numel() and negative_scores.numel()))
        if not bool(valid):
            zero = logits.new_zeros(())
            return zero, zero, valid
        comparisons = positive_scores[:, None] - negative_scores[None, :]
        auroc = (
            (comparisons > 0).float().mean()
            + 0.5 * (comparisons == 0).float().mean()
        )
        scores = logits[positive | negative].detach()
        targets = positive[positive | negative]
        order = torch.argsort(scores, descending=True, stable=True)
        sorted_targets = targets[order].float()
        precision = sorted_targets.cumsum(0) / torch.arange(
            1, sorted_targets.numel() + 1, device=logits.device, dtype=logits.dtype
        )
        average_precision = (precision * sorted_targets).sum() / sorted_targets.sum()
        return auroc, average_precision, valid

    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        logits = output["overall_logit"]
        target = labels["overall_target"].float()
        valid = labels["overall_valid"].bool() & torch.isfinite(target)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        raw = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none")
        positive = valid & (safe_target > 0.5)
        negative = valid & ~positive
        class_losses = []
        if positive.any():
            class_losses.append(raw[positive].mean())
        if negative.any():
            class_losses.append(raw[negative].mean())
        loss = torch.stack(class_losses).mean() if class_losses else logits.sum() * 0.0
        predicted = logits >= 0
        class_accuracies = []
        if positive.any():
            class_accuracies.append(predicted[positive].float().mean())
        if negative.any():
            class_accuracies.append((~predicted[negative]).float().mean())
        balanced_accuracy = (
            torch.stack(class_accuracies).mean()
            if class_accuracies else logits.new_zeros(())
        )
        valid_count = valid.sum().float()
        positive_fraction = positive.sum().float() / valid_count.clamp_min(1.0)
        probability = positive_fraction.clamp(1e-7, 1.0 - 1e-7)
        prior_bce = -(
            positive_fraction * probability.log()
            + (1.0 - positive_fraction) * (1.0 - probability).log()
        )
        auroc, average_precision, ranking_metrics_valid = self._ranking_metrics(
            logits, positive, negative
        )
        head_losses = [loss] if valid.any() else []
        auxiliary_metrics: dict[str, Tensor] = {}
        for head in ("collision", "approach"):
            head_logits = output.get(f"{head}_logit", logits.new_zeros(logits.shape))
            head_target = labels.get(
                f"{head}_target", logits.new_zeros(logits.shape)
            ).float()
            head_valid = labels.get(
                f"{head}_valid", torch.zeros_like(valid)
            ).bool() & torch.isfinite(head_target)
            if head_valid.any():
                head_loss = F.binary_cross_entropy_with_logits(
                    head_logits[head_valid], head_target[head_valid]
                )
                head_losses.append(head_loss)
            else:
                head_loss = head_logits.sum() * 0.0
            auxiliary_metrics[f"verifier_{head}_loss"] = head_loss.detach()
            auxiliary_metrics[f"verifier_{head}_valid_candidates"] = (
                head_valid.sum().float().detach()
            )
        joint_loss = (
            torch.stack(head_losses).mean() if head_losses else logits.sum() * 0.0
        )
        return {
            "loss": joint_loss,
            "verifier_valid_candidates": valid_count,
            "verifier_supervised_rows": valid.reshape(valid.shape[0], -1).any(-1).sum().float(),
            "verifier_positive_candidates": positive.sum().float(),
            "verifier_negative_candidates": negative.sum().float(),
            "verifier_positive_fraction": positive_fraction,
            "verifier_predicted_positive_fraction": (
                (predicted & valid).sum().float() / valid_count.clamp_min(1.0)
            ),
            "verifier_balanced_accuracy": balanced_accuracy,
            "verifier_prior_bce": prior_bce,
            "verifier_auroc": auroc,
            "verifier_average_precision": average_precision,
            "verifier_ranking_metrics_valid": ranking_metrics_valid,
            **auxiliary_metrics,
        }
