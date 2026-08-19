from .actions import PushLoss
from .instance import InstanceSetLoss
from .objective import TCDPRGObjective
from .region import TaskRegionLoss
from .task_grasp_score import TaskGraspScoringLoss
from .total import MultiTaskLoss

__all__ = [
    "PushLoss",
    "InstanceSetLoss",
    "TCDPRGObjective",
    "TaskRegionLoss",
    "TaskGraspScoringLoss",
    "MultiTaskLoss",
]
