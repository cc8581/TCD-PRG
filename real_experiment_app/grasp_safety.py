"""Non-negotiable orientation gate for executable parallel-jaw grasps."""

from __future__ import annotations

import numpy as np

from .transforms import pose7_to_matrix


def grasp_local_z_is_downward_or_level(candidate: dict) -> bool:
    """Accept only local-Z directions whose world-Z component is non-positive.

    The application world frame uses +Z as the table normal.  The candidate
    pose's local +Z is the gripper approach/tool axis, so an upward component
    would require an approach from below the object and is not executable by
    the configured tabletop cell.
    """

    if int(candidate.get("action_type", -1)) != 2:
        return True
    try:
        rotation = pose7_to_matrix(candidate["grasp_pose_world"])[:3, :3]
    except (KeyError, TypeError, ValueError):
        return False
    local_z_in_world = rotation[:, 2]
    return bool(np.isfinite(local_z_in_world).all() and local_z_in_world[2] <= 1e-6)
