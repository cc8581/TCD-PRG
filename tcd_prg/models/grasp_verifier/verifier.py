"""Joint local-scene/AG-gripper/task multi-head grasp verifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GripperSceneTaskVerifier(nn.Module):
    """Verify K candidates without recomputing the global scene backbone.

    Shapes: scene/gripper xyz ``[B,K,L|G,3]``; point context
    ``[B,K,L,C]``; output heads ``[B,K]``.
    """

    HEADS = ("stability", "task_compatibility", "collision", "clearance", "approach", "overall")

    def __init__(self, scene_feature_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_feature_dim + 7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gripper_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion = nn.Sequential(nn.Linear(4 * hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Linear(2 * hidden_dim, hidden_dim))
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in self.HEADS})

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
        scene_encoded = self.scene_encoder(scene_input)
        scene_encoded = scene_encoded.masked_fill(~scene_valid.unsqueeze(-1), torch.finfo(scene_encoded.dtype).min)
        scene_max = scene_encoded.max(dim=2).values
        scene_max = torch.where(scene_valid.any(2, keepdim=True), scene_max, 0.0)
        scene_mean = (scene_encoded.masked_fill(~scene_valid.unsqueeze(-1), 0).sum(2) / scene_valid.sum(2, keepdim=True).clamp_min(1))
        gripper_input = torch.cat((gripper_xyz_grasp, gripper_valid.unsqueeze(-1).float()), -1)
        gripper_encoded = self.gripper_encoder(gripper_input)
        gripper_encoded = gripper_encoded.masked_fill(~gripper_valid.unsqueeze(-1), torch.finfo(gripper_encoded.dtype).min)
        gripper_max = gripper_encoded.max(dim=2).values
        gripper_max = torch.where(gripper_valid.any(2, keepdim=True), gripper_max, 0.0)
        task = task_token[:, None].expand(-1, scene_max.shape[1], -1)
        fused = self.fusion(torch.cat((scene_max, scene_mean, gripper_max, task), -1))
        return {f"{name}_logit": head(fused).squeeze(-1) for name, head in self.heads.items()}
