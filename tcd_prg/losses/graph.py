"""Physical/task edges, blockers and topology-masked graph losses."""

from torch import Tensor, nn

from .masked import safe_bce_with_logits


class DependencyGraphLoss(nn.Module):
    def forward(self, output: object, labels: dict[str, Tensor]) -> dict[str, Tensor]:
        losses = {
            "graph_physical_edge": safe_bce_with_logits(
                output.physical_edge_logits, labels["physical_edge_target"].float(), labels["physical_edge_valid"]
            ),
            "graph_task_edge": safe_bce_with_logits(
                output.task_edge_logits, labels["task_edge_target"].float(), labels["task_edge_valid"]
            ),
            "graph_direct_blocker": safe_bce_with_logits(
                output.direct_blocker_logits, labels["direct_blocker_target"].float(), labels["blocker_valid"]
            ),
            "graph_indirect_blocker": safe_bce_with_logits(
                output.indirect_blocker_logits, labels["indirect_blocker_target"].float(), labels["blocker_valid"]
            ),
            "graph_actionable": safe_bce_with_logits(
                output.actionable_blocker_logits, labels["actionable_target"].float(), labels["blocker_valid"]
            ),
        }
        topology_valid = labels["topology_edge_valid"] & labels[
            "sequence_topology_valid"
        ][:, None, None]
        losses["graph_topology_order"] = safe_bce_with_logits(
            output.topology_order_logits,
            labels["topology_target"].float(),
            topology_valid,
        )
        return losses
