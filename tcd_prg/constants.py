"""Stable semantic constants shared by data, models, losses and execution."""

from enum import IntEnum


class ActionType(IntEnum):
    """Dataset action encoding. Do not reorder."""

    # 数值写入 HDF5 与缓存协议，禁止因代码可读性调整枚举顺序。
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

    # UNKNOWN 是 ignore 状态，不得在 BCE/listwise loss 中隐式转换成 0。
    UNKNOWN_UNTESTED = -1
    NEGATIVE = 0
    POSITIVE = 1


PUSH_DISTANCE_M = 0.15        # 主实验固定推动距离，修改会改变动作定义而非普通超参数
MAX_PREPARATION_ACTIONS = 5   # 准备动作上限 H，不包含最终 TASK_GRASP
QUATERNION_ORDER = "xyzw"

# 认证代码、碰撞资产或物理活跃物体语义变化时必须升级版本，使旧 cache 自动失效。
GLOBAL_GRASP_CERTIFICATION_FORMAT = "global_grasp_scene_certification_v2"
GLOBAL_GRASP_CERTIFIER_VERSION = "fr5_ag16095_exact_v2_physical_active"
