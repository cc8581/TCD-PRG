"""Reobserve-after-every-action closed-loop planner with H=5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from tcd_prg.baselines.base import ManipulationPolicy
from tcd_prg.constants import ActionType, MAX_PREPARATION_ACTIONS
from tcd_prg.datasets.types import SceneObservation


class ObservationSource(Protocol):
    def observe(self) -> SceneObservation: ...


class Executor(Protocol):
    def certify(self, action: Any) -> tuple[bool, str]: ...

    def execute(self, action: Any) -> bool: ...


@dataclass(slots=True)
class PlanResult:
    success: bool
    preparation_actions: int
    actions: list[Any] = field(default_factory=list)
    failure_reason: str = ""


class ClosedLoopPlanner:
    def __init__(
        self,
        policy: ManipulationPolicy,
        observations: ObservationSource,
        executor: Executor,
        max_preparation_actions: int = MAX_PREPARATION_ACTIONS,
    ) -> None:
        if max_preparation_actions != MAX_PREPARATION_ACTIONS:
            raise ValueError("Main TCD-PRG experiment requires H=5")
        self.policy, self.observations, self.executor = policy, observations, executor
        self.max_preparation_actions = max_preparation_actions

    def run(self) -> PlanResult:
        self.policy.reset()
        actions: list[Any] = []
        # 每次执行后重新观测和生成候选；H=5 只限制准备动作，不包含最终 Task Grasp。
        for preparation_step in range(self.max_preparation_actions + 1):
            observation = self.observations.observe()
            encoded = self.policy.encode_observation(observation)
            candidates = self.policy.generate_candidates(encoded)
            action, rejection = self._select_certified(candidates)
            if action is None:
                reason = "no_valid_action" if rejection is None else f"certification:{rejection}"
                return PlanResult(False, preparation_step, actions, reason)
            if not self.executor.execute(action):
                return PlanResult(False, preparation_step, actions, "execution_failed")
            actions.append(action)
            self.policy.update_after_action(action, observation)
            action_type = int(action["action_type"] if isinstance(action, dict) else action.action_type)
            if action_type == int(ActionType.TASK_GRASP):
                return PlanResult(True, preparation_step, actions)
            if preparation_step == self.max_preparation_actions:
                break
        return PlanResult(False, self.max_preparation_actions, actions, "horizon_exhausted")

    def _select_certified(self, candidates: Any) -> tuple[Any | None, str | None]:
        """Try the next ranked candidate after deterministic safety rejection."""

        last_reason = None
        while True:
            action = self.policy.select_action(candidates)
            if action is None:
                return None, last_reason
            # 确定性认证只放在最终执行边界；拒绝后直接尝试同一观测下的次高分候选。
            certified, reason = self.executor.certify(action)
            if certified:
                return action, None
            last_reason = reason
            if not isinstance(candidates, dict) or "candidates" not in candidates:
                return None, reason
            index = action.get("candidate_index") if isinstance(action, dict) else None
            if index is None:
                return None, reason
            tensor_group = candidates["candidates"]
            if "valid" not in tensor_group or not bool(tensor_group["valid"][0, int(index)]):
                return None, reason
            tensor_group["valid"][0, int(index)] = False
            # 仅更新候选有效 mask 并重跑 Router，不重复编码未变化的点云。
            encoded = candidates.get("encoded")
            if encoded is not None and hasattr(self.policy, "model"):
                candidates["router"] = self.policy.model.route_cached(  # type: ignore[attr-defined]
                    encoded.device_batch, encoded.output, tensor_group
                )
