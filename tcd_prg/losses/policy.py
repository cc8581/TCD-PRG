"""Set-valued final candidate policy supervision."""

from torch import Tensor, nn

from .masked import multi_positive_listwise_loss


class HierarchicalSetPolicyLoss(nn.Module):
    def forward(self, output: object, successful: Tensor, evaluated: Tensor) -> Tensor:
        valid = output.candidate_valid_mask & evaluated
        return multi_positive_listwise_loss(output.candidate_logits, successful, valid)
