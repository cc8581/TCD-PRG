"""Candidate-only policy router."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tcd_prg.constants import ActionType


@dataclass(slots=True)
class RouterOutput:
    candidate_logits: Tensor
    candidate_valid_mask: Tensor


class MaskedHierarchicalCandidateRouter(nn.Module):
    """Rank the final valid candidate union with one learned policy head."""

    def __init__(self, dim: int = 256, action_types: int = 3, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.action_types = action_types
        self.action_embedding = nn.Embedding(action_types + 1, dim)
        self.remaining_embedding = nn.Embedding(6, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.score = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 1))

    def forward(
        self, task_token: Tensor, global_token: Tensor, object_tokens: Tensor,
        object_mask: Tensor, candidate_tokens: Tensor, candidate_type: Tensor,
        candidate_object: Tensor, candidate_valid: Tensor, remaining_steps: Tensor,
        previous_action: Tensor | None = None,
    ) -> RouterOutput:
        del object_mask
        b, k, _ = candidate_tokens.shape
        safe_type = candidate_type.clamp(0, self.action_types - 1)
        safe_object = candidate_object.clamp(0, object_tokens.shape[1] - 1)
        row = torch.arange(b, device=candidate_tokens.device)[:, None]
        # Sequence-history supervision is not available for cached generated
        # state groups.  Keep this context neutral until a history-keyed cache
        # and labels exist, so deployment cannot introduce an unseen feature.
        del previous_action
        previous = torch.zeros_like(task_token)
        remaining = self.remaining_embedding(remaining_steps.clamp(0, 5))
        enriched = candidate_tokens + self.action_embedding(safe_type) + object_tokens[row, safe_object]
        padding = ~candidate_valid
        all_invalid = padding.all(-1)
        if all_invalid.any():
            padding = padding.clone()
            enriched = enriched.clone()
            padding[all_invalid, 0] = False
            enriched[all_invalid, 0] = 0.0
        encoded = self.transformer(enriched, src_key_padding_mask=padding)
        context = torch.cat((
            encoded,
            task_token[:, None].expand(-1, k, -1),
            global_token[:, None].expand(-1, k, -1),
            remaining[:, None].expand(-1, k, -1),
            previous[:, None].expand(-1, k, -1),
        ), -1)
        logits = self.score(context).squeeze(-1).masked_fill(~candidate_valid, -30.0)
        return RouterOutput(logits, candidate_valid)

    @staticmethod
    def fixed_priority(candidate_type: Tensor, candidate_valid: Tensor) -> Tensor:
        selected = torch.full((candidate_type.shape[0],), -1, dtype=torch.long, device=candidate_type.device)
        for row in range(candidate_type.shape[0]):
            for kind in (ActionType.TASK_GRASP, ActionType.PICK_REMOVE, ActionType.PUSH):
                valid = torch.nonzero(candidate_valid[row] & (candidate_type[row] == int(kind)), as_tuple=False)
                if len(valid):
                    selected[row] = valid[0, 0]
                    break
        return selected

    @staticmethod
    def select(output: RouterOutput, candidate_type: Tensor, candidate_object: Tensor) -> Tensor:
        del candidate_type, candidate_object
        selected = output.candidate_logits.argmax(-1)
        return torch.where(
            output.candidate_valid_mask.any(-1), selected, torch.full_like(selected, -1)
        )


class FlatCandidateClassifier(MaskedHierarchicalCandidateRouter):
    """Compatibility ablation; the primary router is already candidate-flat."""


def fixed_priority_output(
    candidate_type: Tensor, candidate_object: Tensor, candidate_valid: Tensor,
    object_mask: Tensor,
) -> RouterOutput:
    del candidate_object, object_mask
    priorities = torch.tensor([1.0, 2.0, 3.0], device=candidate_type.device)
    logits = priorities[candidate_type.clamp(0, 2)].masked_fill(~candidate_valid, -30.0)
    return RouterOutput(logits, candidate_valid)
