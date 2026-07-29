"""Machine-readable dataset label and execution capabilities."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetCapabilities:
    has_instance_masks: bool = False
    has_task_regions: bool = False
    has_task_grasps: bool = False
    has_push_actions: bool = False
    has_pick_remove_actions: bool = False
    has_sequences: bool = False
    has_relation_graph: bool = False
    has_exact_ik: bool = False
    has_intermediate_observations: bool = False
    supports_closed_loop: bool = False

    def loss_available(self, name: str) -> bool:
        requirements = {
            "region": self.has_task_regions,
            "proposal": self.has_task_grasps,
            "verify": self.has_task_grasps,
            "graph": self.has_relation_graph,
            "remove": self.has_pick_remove_actions,
            "push": self.has_push_actions,
            "policy": self.has_sequences and (self.has_push_actions or self.has_pick_remove_actions),
        }
        if name not in requirements:
            raise KeyError(name)
        return requirements[name]

