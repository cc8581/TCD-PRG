"""Masked hierarchical action-type, acted-object and candidate router."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tcd_prg.constants import ActionType


@dataclass(slots=True)
class RouterOutput:
    action_type_logits: Tensor
    object_logits: Tensor
    candidate_logits: Tensor
    type_valid_mask: Tensor
    object_valid_mask: Tensor
    candidate_valid_mask: Tensor
    remaining_steps_prediction: Tensor


class MaskedHierarchicalCandidateRouter(nn.Module):
    def __init__(self, dim: int = 256, action_types: int = 3, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.action_types = action_types
        self.action_embedding = nn.Embedding(action_types + 1, dim)
        self.remaining_embedding = nn.Embedding(6, dim)
        encoder_layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True, norm_first=True)
        self.candidate_transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.type_head = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, action_types))
        self.remaining_head = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.object_head = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, action_types))
        self.candidate_head = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 1))

    def forward(
        self,
        task_token: Tensor,
        global_token: Tensor,
        object_tokens: Tensor,
        object_mask: Tensor,
        candidate_tokens: Tensor,
        candidate_type: Tensor,
        candidate_object: Tensor,
        candidate_valid: Tensor,
        remaining_steps: Tensor,
        previous_action: Tensor | None = None,
    ) -> RouterOutput:
        b, k, dim = candidate_tokens.shape
        previous = (
            self.action_embedding(previous_action.clamp(0, self.action_types))
            if previous_action is not None
            else torch.zeros_like(task_token)
        )
        remaining = self.remaining_embedding(remaining_steps.clamp(0, 5))
        type_valid = torch.stack(
            [candidate_valid & (candidate_type == action_type) for action_type in range(self.action_types)], -1
        ).any(1)
        type_context = torch.cat((task_token, global_token, remaining, previous), -1)
        type_logits = self.type_head(type_context).masked_fill(~type_valid, -30.0)
        object_context = torch.cat(
            (
                object_tokens,
                task_token[:, None].expand_as(object_tokens),
                global_token[:, None].expand_as(object_tokens),
                remaining[:, None].expand_as(object_tokens),
                previous[:, None].expand_as(object_tokens),
            ),
            -1,
        )
        object_logits = self.object_head(object_context).transpose(1, 2)
        object_valid = torch.zeros(
            (b, self.action_types, object_tokens.shape[1]), dtype=torch.bool, device=object_tokens.device
        )
        for action_type in range(self.action_types):
            for object_index in range(object_tokens.shape[1]):
                object_valid[:, action_type, object_index] = (
                    candidate_valid & (candidate_type == action_type) & (candidate_object == object_index)
                ).any(1)
        object_valid &= object_mask[:, None]
        object_logits = object_logits.masked_fill(~object_valid, -30.0)
        safe_type = candidate_type.clamp(0, self.action_types - 1)
        safe_object = candidate_object.clamp(0, object_tokens.shape[1] - 1)
        row = torch.arange(b, device=candidate_tokens.device)[:, None]
        enriched = candidate_tokens + self.action_embedding(safe_type) + object_tokens[row, safe_object]
        encoded = self.candidate_transformer(enriched, src_key_padding_mask=~candidate_valid)
        candidate_context = torch.cat(
            (
                encoded,
                task_token[:, None].expand(-1, k, -1),
                global_token[:, None].expand(-1, k, -1),
                remaining[:, None].expand(-1, k, -1),
                previous[:, None].expand(-1, k, -1),
            ),
            -1,
        )
        candidate_logits = self.candidate_head(candidate_context).squeeze(-1).masked_fill(~candidate_valid, -30.0)
        return RouterOutput(
            type_logits, object_logits, candidate_logits, type_valid, object_valid,
            candidate_valid, self.remaining_head(type_context).squeeze(-1)
        )

    @staticmethod
    def fixed_priority(candidate_type: Tensor, candidate_valid: Tensor) -> Tensor:
        """TASK_GRASP > PICK_REMOVE > PUSH baseline, returning candidate indices."""

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
        """Select the best valid candidate under all three hierarchical scores."""

        batch_size = output.candidate_logits.shape[0]
        selected = torch.full((batch_size,), -1, dtype=torch.long, device=candidate_type.device)
        for row in range(batch_size):
            type_index = output.action_type_logits[row].argmax()
            object_index = output.object_logits[row, type_index].argmax()
            mask = (
                output.candidate_valid_mask[row]
                & (candidate_type[row] == type_index)
                & (candidate_object[row] == object_index)
            )
            score = output.candidate_logits[row].masked_fill(~mask, -30.0)
            if mask.any():
                selected[row] = score.argmax()
        return selected


class FlatCandidateClassifier(nn.Module):
    """Ablation that ranks the union of all candidate types in one classifier."""

    def __init__(self, dim: int = 256, action_types: int = 3, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.action_types = action_types
        self.action_embedding = nn.Embedding(action_types + 1, dim)
        self.remaining_embedding = nn.Embedding(6, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.score = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, 1))
        self.remaining_head = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(
        self,
        task_token: Tensor,
        global_token: Tensor,
        object_tokens: Tensor,
        object_mask: Tensor,
        candidate_tokens: Tensor,
        candidate_type: Tensor,
        candidate_object: Tensor,
        candidate_valid: Tensor,
        remaining_steps: Tensor,
        previous_action: Tensor | None = None,
    ) -> RouterOutput:
        b, k, _ = candidate_tokens.shape
        safe_type = candidate_type.clamp(0, self.action_types - 1)
        safe_object = candidate_object.clamp(0, object_tokens.shape[1] - 1)
        row = torch.arange(b, device=candidate_tokens.device)[:, None]
        previous = (
            self.action_embedding(previous_action.clamp(0, self.action_types))
            if previous_action is not None
            else torch.zeros_like(task_token)
        )
        remaining = self.remaining_embedding(remaining_steps.clamp(0, 5))
        encoded = self.transformer(
            candidate_tokens + self.action_embedding(safe_type) + object_tokens[row, safe_object],
            src_key_padding_mask=~candidate_valid,
        )
        context = torch.cat(
            (
                encoded,
                task_token[:, None].expand(-1, k, -1),
                global_token[:, None].expand(-1, k, -1),
                remaining[:, None].expand(-1, k, -1),
                previous[:, None].expand(-1, k, -1),
            ),
            -1,
        )
        candidate_logits = self.score(context).squeeze(-1).masked_fill(~candidate_valid, -30.0)
        type_valid = torch.stack(
            [candidate_valid & (candidate_type == kind) for kind in range(self.action_types)], -1
        ).any(1)
        object_valid = torch.zeros(
            (b, self.action_types, object_tokens.shape[1]), dtype=torch.bool, device=object_tokens.device
        )
        type_logits = torch.full((b, self.action_types), -30.0, device=object_tokens.device)
        object_logits = torch.full_like(object_valid, -30.0, dtype=candidate_logits.dtype)
        for kind in range(self.action_types):
            kind_mask = candidate_valid & (candidate_type == kind)
            type_logits[:, kind] = candidate_logits.masked_fill(~kind_mask, -30.0).max(-1).values
            for object_index in range(object_tokens.shape[1]):
                mask = kind_mask & (candidate_object == object_index)
                object_valid[:, kind, object_index] = mask.any(-1) & object_mask[:, object_index]
                object_logits[:, kind, object_index] = candidate_logits.masked_fill(~mask, -30.0).max(-1).values
        type_logits = type_logits.masked_fill(~type_valid, -30.0)
        object_logits = object_logits.masked_fill(~object_valid, -30.0)
        remaining_context = torch.cat((task_token, global_token, remaining, previous), -1)
        return RouterOutput(
            type_logits, object_logits, candidate_logits, type_valid, object_valid,
            candidate_valid, self.remaining_head(remaining_context).squeeze(-1)
        )


def fixed_priority_output(
    candidate_type: Tensor,
    candidate_object: Tensor,
    candidate_valid: Tensor,
    object_mask: Tensor,
) -> RouterOutput:
    """Return deterministic TASK_GRASP > PICK_REMOVE > PUSH scores."""

    b, _ = candidate_type.shape
    priorities = torch.tensor([1.0, 2.0, 3.0], device=candidate_type.device)
    safe_type = candidate_type.clamp(0, 2)
    candidate_logits = priorities[safe_type].masked_fill(~candidate_valid, -30.0)
    type_valid = torch.stack([candidate_valid & (candidate_type == kind) for kind in range(3)], -1).any(1)
    type_logits = priorities[None].expand(b, -1).masked_fill(~type_valid, -30.0)
    object_valid = torch.zeros((b, 3, object_mask.shape[1]), dtype=torch.bool, device=object_mask.device)
    object_logits = torch.full(object_valid.shape, -30.0, device=object_mask.device)
    for kind in range(3):
        for object_index in range(object_mask.shape[1]):
            mask = candidate_valid & (candidate_type == kind) & (candidate_object == object_index)
            object_valid[:, kind, object_index] = mask.any(-1) & object_mask[:, object_index]
            object_logits[:, kind, object_index] = priorities[kind]
    object_logits = object_logits.masked_fill(~object_valid, -30.0)
    return RouterOutput(
        type_logits, object_logits, candidate_logits, type_valid, object_valid,
        candidate_valid, torch.zeros(b, device=candidate_type.device)
    )
