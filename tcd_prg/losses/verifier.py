"""Independent multi-head grasp verification loss."""

from torch import Tensor, nn

from .masked import safe_bce_with_logits


class GraspVerifierLoss(nn.Module):
    HEADS = ("stability", "task_compatibility", "collision", "clearance", "approach", "overall")

    def forward(self, output: dict[str, Tensor], labels: dict[str, Tensor]) -> dict[str, Tensor]:
        result = {}
        for head in self.HEADS:
            result[f"verify_{head}"] = safe_bce_with_logits(
                output[f"{head}_logit"], labels[f"{head}_target"].float(), labels[f"{head}_valid"]
            )
        return result

