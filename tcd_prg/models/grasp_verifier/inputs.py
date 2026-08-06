"""Prepare exact local scene/gripper tensors for grasp verification."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from tcd_prg.constants import ActionType
from tcd_prg.geometry.gripper import GripperGeometry
from tcd_prg.geometry.se3 import quaternion_xyzw_to_matrix


class GripperGeometryProvider(Protocol):
    def get(self, width_m: float) -> GripperGeometry: ...


@torch.no_grad()
def build_verifier_inputs(
    batch: dict[str, Tensor],
    gripper_provider: GripperGeometryProvider,
    local_scene_points: int = 128,
    local_radius_m: float = 0.25,
) -> dict[str, Tensor]:
    """Build candidate-frame verifier tensors without rerunning the backbone.

    The returned candidate axis is identical to the state group's candidate
    axis. PUSH rows are masked. Scene points are selected by deterministic
    nearest-neighbour distance to each candidate TCP.
    """

    if batch["xyz"].device.type != "cpu":
        raise ValueError("Verifier preprocessing must run on CPU before device transfer")
    if local_scene_points <= 0:
        raise ValueError("local_scene_points must be positive")
    parameters = batch["action_parameters"]
    action_type = batch["action_type"]
    grasp_candidate = batch["candidate_mask"] & (
        (action_type == int(ActionType.TASK_GRASP))
        | (action_type == int(ActionType.PICK_REMOVE))
    )
    # TASK_GRASP 与 PICK_REMOVE 共用 verifier 坐标协议，PUSH 不进入抓取验证器。
    pose = torch.where(
        (action_type == int(ActionType.PICK_REMOVE)).unsqueeze(-1),
        parameters["removal_grasp_pose_world"],
        parameters["task_grasp_pose_world"],
    )
    widths = parameters["grasp_width_m"]
    grasp_candidate &= torch.isfinite(pose).all(-1) & torch.isfinite(widths)
    batch_size, candidates = action_type.shape
    gripper_points = gripper_provider.point_count  # type: ignore[attr-defined]
    scene_index = torch.zeros(
        batch_size, candidates, local_scene_points, dtype=torch.long
    )
    scene_xyz_grasp = torch.zeros(
        batch_size, candidates, local_scene_points, 3, dtype=batch["xyz"].dtype
    )
    scene_valid = torch.zeros(batch_size, candidates, local_scene_points, dtype=torch.bool)
    gripper_xyz_grasp = torch.zeros(
        batch_size, candidates, gripper_points, 3, dtype=batch["xyz"].dtype
    )
    gripper_valid = torch.zeros(batch_size, candidates, gripper_points, dtype=torch.bool)
    rotation = quaternion_xyzw_to_matrix(torch.nan_to_num(pose[..., 3:], nan=0.0))
    valid_widths = widths[grasp_candidate].numpy()
    prefetched = (
        gripper_provider.get_many(valid_widths)  # type: ignore[attr-defined]
        if hasattr(gripper_provider, "get_many")
        else tuple(gripper_provider.get(float(width)) for width in valid_widths)
    )
    geometry_iterator = iter(prefetched)
    radius_squared = float(local_radius_m) ** 2
    for row in range(batch_size):
        valid_scene_indices = torch.nonzero(batch["point_mask"][row], as_tuple=False).flatten()
        valid_scene_xyz = batch["xyz"][row, valid_scene_indices]
        for candidate in torch.nonzero(grasp_candidate[row], as_tuple=False).flatten().tolist():
            origin = pose[row, candidate, :3]
            distance_squared = (valid_scene_xyz - origin).square().sum(-1)
            within = distance_squared <= radius_squared
            count = min(local_scene_points, int(within.sum()))
            if count == 0:
                grasp_candidate[row, candidate] = False
                continue
            # Only the nearest L points are needed; a full O(N log N) sort per
            # candidate is unnecessary. sorted=True preserves deterministic order.
            chosen = torch.topk(
                distance_squared.masked_fill(~within, float("inf")),
                k=count, largest=False, sorted=True,
            ).indices
            indices = valid_scene_indices[chosen]
            scene_index[row, candidate, :count] = indices
            # 场景点和夹爪点统一变换到候选 TCP 坐标系，网络无需学习绝对世界位姿。
            local = batch["xyz"][row, indices] - origin
            scene_xyz_grasp[row, candidate, :count] = (
                local @ rotation[row, candidate]
            )
            scene_valid[row, candidate, :count] = True
            geometry = next(geometry_iterator)
            points = torch.from_numpy(np.asarray(geometry.points_tcp, dtype=np.float32))
            count_g = min(gripper_points, len(points))
            gripper_xyz_grasp[row, candidate, :count_g] = points[:count_g]
            gripper_valid[row, candidate, :count_g] = True
    return {
        "candidate_valid": grasp_candidate,
        "scene_point_index": scene_index,
        "scene_xyz_grasp": scene_xyz_grasp,
        "scene_valid": scene_valid,
        "gripper_xyz_grasp": gripper_xyz_grasp,
        "gripper_valid": gripper_valid,
    }
