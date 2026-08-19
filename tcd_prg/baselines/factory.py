"""Construct every comparison method behind the common policy interface."""

from __future__ import annotations

from tcd_prg.config import TCDPRGConfig

from .base import ManipulationPolicy
from .gapg_wrapper import GAPGPolicyWrapper
from .one_shot import OneShotSequencePolicy
from .rules import (
    DirectGraspOnlyPolicy,
    FixedPriorityPolicy,
    PickRemoveOnlyPolicy,
    PushOnlyPolicy,
)


def create_baseline(
    config: TCDPRGConfig,
    candidate_policy: ManipulationPolicy | None = None,
) -> ManipulationPolicy:
    """Create a policy; learned-candidate rules require ``candidate_policy``."""

    kind = config.baseline.type
    if kind == "original_gapg_wrapper":
        return GAPGPolicyWrapper(
            config.baseline.gapg_root,
            config.baseline.grasp_checkpoint,
            config.baseline.push_checkpoint,
            config.baseline.graspnet_checkpoint,
            python=config.observation.pybullet_python,
            seed=config.training.seed,
        )
    if candidate_policy is None:
        raise ValueError(f"baseline.type={kind} requires a learned candidate policy")
    constructors = {
        "direct_grasp_only": DirectGraspOnlyPolicy,
        "push_only": PushOnlyPolicy,
        "pick_remove_only": PickRemoveOnlyPolicy,
        "fixed_priority": FixedPriorityPolicy,
    }
    if kind == "one_shot_sequence_prediction":
        return OneShotSequencePolicy(candidate_policy)
    if kind == "tcd_prg":
        return candidate_policy
    if kind not in constructors:
        raise ValueError(f"Unknown baseline.type={kind}")
    return constructors[kind](candidate_policy)
