"""Final-executability grasp verifier loss."""

from torch import Tensor, nn

from .masked import safe_bce_with_logits


class GraspVerifierLoss(nn.Module):
    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> Tensor:
        return safe_bce_with_logits(
            output["overall_logit"], labels["overall_target"].float(), labels["overall_valid"]
        )
