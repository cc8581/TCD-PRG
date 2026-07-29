"""Encode heterogeneous action parameters without using local IDs as numeric features."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ActionCandidateEncoder(nn.Module):
    """Candidate token from action type, acted-object token and physical parameters.

    IDs select object tokens only; their integer magnitude never enters an MLP.
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(3, dim)
        self.geometry = nn.Sequential(nn.Linear(21, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.fusion = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(
        self,
        object_tokens: Tensor,
        action_type: Tensor,
        acted_object: Tensor,
        contact_world: Tensor,
        direction_world: Tensor,
        pose_world: Tensor,
        destination_world: Tensor,
        parameter_valid: Tensor,
        task_token: Tensor,
    ) -> Tensor:
        row = torch.arange(object_tokens.shape[0], device=object_tokens.device)[:, None]
        selected_object = object_tokens[row, acted_object.clamp(0, object_tokens.shape[1] - 1)]
        numeric = torch.cat((contact_world, direction_world, pose_world, destination_world, parameter_valid.float()), -1)
        numeric = torch.nan_to_num(numeric)
        geometry = self.geometry(numeric)
        task = task_token[:, None].expand_as(selected_object)
        return self.fusion(
            torch.cat((selected_object, self.type_embedding(action_type.clamp(0, 2)), geometry, task), -1)
        )

