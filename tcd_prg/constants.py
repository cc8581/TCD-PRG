"""Stable semantic constants shared by data, models, losses and execution."""

from enum import IntEnum


class ActionType(IntEnum):
    """Dataset action encoding. Do not reorder."""

    PUSH = 0
    PICK_REMOVE = 1
    TASK_GRASP = 2


class OutcomeCode(IntEnum):
    """Observed action outcome encoding from action HDF5 v2."""

    SUCCESS = 0
    IMPROVED = 1
    NO_IMPROVEMENT = 2
    UNSTABLE = 3
    OUT_OF_WORKSPACE = 4
    OTHER_INVALID = 5
    TERMINAL_POSITIVE = 6


class CandidateStatus(IntEnum):
    """Three-way candidate supervision; UNKNOWN is never a negative."""

    UNKNOWN_UNTESTED = -1
    NEGATIVE = 0
    POSITIVE = 1


PUSH_DISTANCE_M = 0.15
MAX_PREPARATION_ACTIONS = 5
QUATERNION_ORDER = "xyzw"

# Bump this whenever the exact scene-certification protocol changes in a way
# that can alter cached outcomes.  Version 2 makes the current physical object
# mask part of the cache contract, so caches produced with removed objects
# still present cannot supervise a later state.
GLOBAL_GRASP_CERTIFICATION_FORMAT = "global_grasp_scene_certification_v2"
GLOBAL_GRASP_CERTIFIER_VERSION = "fr5_ag16095_exact_v2_physical_active"
