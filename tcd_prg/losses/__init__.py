from .instance import InstanceSetLoss
from .objective import TCDPRGObjective
from .region import TaskRegionLoss
from .task_grasp_binary import TaskGraspBinaryLoss
from .total import MultiTaskLoss

__all__ = [
    "InstanceSetLoss",
    "TCDPRGObjective",
    "TaskRegionLoss",
    "TaskGraspBinaryLoss",
    "MultiTaskLoss",
]
