"""Joint local-scene/AG-gripper/task overall grasp verifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GripperSceneTaskVerifier(nn.Module):
    """Verify K candidates without recomputing the global scene backbone.

    Shapes: scene/gripper xyz ``[B,K,L|G,3]``; point context
    ``[B,K,L,C]``; output heads ``[B,K]``.
    """

    HEADS = ("overall", "collision", "approach")

    def __init__(
        self,
        scene_feature_dim: int = 256,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.scene_projection = nn.Sequential(
            nn.Linear(scene_feature_dim + 7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gripper_projection = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.task_projection = nn.Linear(hidden_dim, hidden_dim)
        self.token_type = nn.Embedding(3, hidden_dim)
        self.cls_token = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.overall = nn.Linear(hidden_dim, 1)
        self.collision = nn.Linear(hidden_dim, 1)
        self.approach = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        scene_xyz_grasp: Tensor,
        gripper_xyz_grasp: Tensor,
        scene_features: Tensor,
        target_mask: Tensor,
        region_probability: Tensor,
        task_token: Tensor,
        scene_valid: Tensor | None = None,
        gripper_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if scene_valid is None:
            scene_valid = torch.ones(scene_xyz_grasp.shape[:-1], dtype=torch.bool, device=scene_xyz_grasp.device)
        if gripper_valid is None:
            gripper_valid = torch.ones(gripper_xyz_grasp.shape[:-1], dtype=torch.bool, device=gripper_xyz_grasp.device)
        scene_input = torch.cat(
            (
                scene_xyz_grasp,
                scene_features,
                target_mask.unsqueeze(-1).float(),
                region_probability.unsqueeze(-1),
                scene_valid.unsqueeze(-1).float(),
                torch.linalg.norm(scene_xyz_grasp, dim=-1, keepdim=True),
            ),
            -1,
        )
        scene_encoded = self.scene_projection(scene_input)
        gripper_input = torch.cat((gripper_xyz_grasp, gripper_valid.unsqueeze(-1).float()), -1)
        gripper_encoded = self.gripper_projection(gripper_input)
        batch_size, candidates = scene_xyz_grasp.shape[:2]
        scene_encoded = scene_encoded + self.token_type.weight[1]
        gripper_encoded = gripper_encoded + self.token_type.weight[2]
        task = self.task_projection(task_token)[:, None].expand(-1, candidates, -1)
        cls = self.cls_token + self.token_type.weight[0]
        cls = cls[None, None].expand(batch_size, candidates, -1) + task
        # 每个候选独立拼接 CLS、局部场景点和精确夹爪几何点，不重复运行全局点云骨干。
        tokens = torch.cat((cls[:, :, None], scene_encoded, gripper_encoded), 2)
        padding = torch.cat((
            torch.zeros((batch_size, candidates, 1), dtype=torch.bool, device=tokens.device),
            ~scene_valid,
            ~gripper_valid,
        ), 2)
        flat_tokens = tokens.flatten(0, 1)
        flat_padding = padding.flatten(0, 1)
        encoded = self.transformer(flat_tokens, src_key_padding_mask=flat_padding)
        fused = encoded[:, 0].reshape(batch_size, candidates, -1)
        return {
            f"{head}_logit": getattr(self, head)(fused).squeeze(-1)
            for head in self.HEADS
        }
