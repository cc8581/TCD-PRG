"""Object pointer and removal-grasp ranking for PICK_REMOVE."""

import torch
from torch import Tensor, nn


class PickRemoveHead(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.object_pointer = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.candidate_rank = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 1))

    def forward(
        self,
        object_tokens: Tensor,
        object_mask: Tensor,
        task_token: Tensor,
        graph_context: Tensor,
        candidate_tokens: Tensor | None = None,
        candidate_object: Tensor | None = None,
        candidate_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        task = task_token[:, None].expand_as(object_tokens)
        object_logits = self.object_pointer(torch.cat((object_tokens, graph_context, task), -1)).squeeze(-1)
        object_logits = object_logits.masked_fill(~object_mask, -30.0)
        result = {"object_logits": object_logits}
        if candidate_tokens is not None and candidate_object is not None:
            row = torch.arange(object_tokens.shape[0], device=object_tokens.device)[:, None]
            acted = object_tokens[row, candidate_object.clamp(0, object_tokens.shape[1] - 1)]
            graph = graph_context[row, candidate_object.clamp(0, graph_context.shape[1] - 1)]
            context = torch.cat((candidate_tokens, acted, graph, task_token[:, None].expand_as(candidate_tokens)), -1)
            score = self.candidate_rank(context).squeeze(-1)
            if candidate_mask is not None:
                score = score.masked_fill(~candidate_mask, -30.0)
            result.update(candidate_logits=score, candidate_tokens=context[..., : candidate_tokens.shape[-1]])
        return result
