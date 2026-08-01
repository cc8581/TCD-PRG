"""Physical and task dependency edge losses."""

from torch import Tensor, nn

from .masked import safe_bce_with_logits


class DependencyGraphLoss(nn.Module):
    def forward(self, output: object, labels: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            "physical_edge": safe_bce_with_logits(
                output.physical_edge_logits, labels["physical_edge_target"].float(),
                labels["physical_edge_valid"],
            ),
            "task_edge": safe_bce_with_logits(
                output.task_edge_logits, labels["task_edge_target"].float(),
                labels["task_edge_valid"],
            ),
        }
