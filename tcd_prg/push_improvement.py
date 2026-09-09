"""Shared Stage-C binary improvement-label protocol."""

from __future__ import annotations


PUSH_IMPROVEMENT_COMPONENT_NAMES = (
    "goal",
    "dependency_blockers",
    "direct_blockers",
    "task_region_pressed",
    "target_pressed",
    "target_visibility",
    "verified_grasp_progress",
)

# Discrete structural components are exact. Visibility changes below one
# percentage point are not meaningful supervision. Grasp progress is a ratio of
# integer verified/required counts, so only a numerical tolerance is needed.
PUSH_IMPROVEMENT_COMPONENT_EPS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 1e-6)
PUSH_IMPROVEMENT_DEFINITION = "binary_conflict_is_not_improvement_v2"


def improvement_event(before, after):
    """Return binary improvement plus auditable positive/regression masks.

    A valid transition is positive only when at least one component improves and
    no component meaningfully regresses.  Mixed-direction evidence is therefore
    supervised as not improved rather than allowing one component to override
    another.
    """
    import numpy as np
    before = np.asarray(before, np.float64)
    after = np.asarray(after, np.float64)
    if before.shape != (7,) or after.shape != (7,):
        raise ValueError("Stage-C improvement keys must contain seven components")
    delta = after - before
    eps = np.asarray(PUSH_IMPROVEMENT_COMPONENT_EPS, np.float64)
    positive = delta > eps
    regression = delta < -eps
    improved = bool(positive.any() and not regression.any())
    return improved, positive, regression, delta
