"""Staged task decision engine independent of Qt, model and PyBullet details."""

from __future__ import annotations

import math
import time
from typing import Protocol

from tcd_prg.constants import ActionType

from .grasp_safety import grasp_local_z_is_downward_or_level
from .types import FusedScene, Prediction
from .workflow import (
    obstruction_graph,
    ranked_actions,
    reachable_obstructors,
    top_obstructor,
)


class PhysicsEvaluator(Protocol):
    def collision_free_grasps(
        self, scene: FusedScene, candidates: tuple[dict, ...], target: int
    ) -> list[dict]: ...


class DecisionEngine:
    """Turns model candidates into one task action through explicit stages."""

    def __init__(self, settings: dict, physics: PhysicsEvaluator):
        self.settings = settings
        self.physics = physics

    def decide(self, scene: FusedScene, analysis: Prediction, score_pushes=None) -> Prediction:
        session = DecisionSession(self.settings, self.physics, scene, analysis)
        session.check_target_grasp()
        if session.result is None:
            session.infer_obstruction()
        if session.result is None and session.status == "running":
            candidates = score_pushes(session.pushable_objects) if score_pushes else None
            session.rank_pushes(candidates)
        return session.finish()


class DecisionSession:
    """A resumable decision transaction used by both manual and auto modes."""

    def __init__(
        self, settings: dict, physics: PhysicsEvaluator, scene: FusedScene, analysis: Prediction
    ):
        self.settings, self.physics, self.scene, self.analysis = settings, physics, scene, analysis
        started = time.perf_counter()
        self.started = started
        self.timings = dict(analysis.timings or {})
        self.path: list[str] = ["LOAD_OR_CAPTURE", "PERCEPTION", "SELECT_TARGET"]
        self.target = int(analysis.target_query)
        self.obstruction = None
        self.obstructions: tuple[int, ...] = ()
        self.adjacent_blockers: tuple[int, ...] = ()
        self.pushable_objects: tuple[int, ...] = ()
        self.grasp_diagnostics: tuple[dict, ...] = ()
        self.grasp_generation_status = "not_checked"
        self.pushes: tuple[dict, ...] = ()
        self.all_candidates = tuple(analysis.candidates)
        self.result: dict | None = None
        self.acted: int | None = None
        self.status = "running"
        self.reason = ""

    def check_target_grasp(self) -> dict:
        top_n = int(self.settings.get("grasp_top_n", 36))
        prediction_stage = time.perf_counter()
        generated = ranked_actions(
            self.analysis.candidates, int(ActionType.TASK_GRASP), self.target
        )
        ranked = generated[:top_n]
        task = tuple(candidate for candidate in ranked if grasp_local_z_is_downward_or_level(candidate))
        self.timings["target_grasp_candidate_selection_s"] = time.perf_counter() - prediction_stage
        stage = time.perf_counter()
        checks = ()
        if task and callable(getattr(self.physics, "assess_grasps", None)):
            checks = tuple(self.physics.assess_grasps(self.scene, task, self.target))
            by_index = {int(item["candidate_index"]): item for item in task}
            free = [by_index[int(row["candidate_index"])] for row in checks if row["collision_free"]]
        else:
            free = self.physics.collision_free_grasps(self.scene, task, self.target) if task else []
        checked = {int(row["candidate_index"]): row for row in checks}
        ranked_ids = {int(item["candidate_index"]) for item in ranked}
        free_ids = {int(item["candidate_index"]) for item in free}
        diagnostics = []
        for candidate in generated:
            index = int(candidate["candidate_index"])
            if index not in ranked_ids:
                status = "not_checked_top_n"
            elif not grasp_local_z_is_downward_or_level(candidate):
                status = "direction_filtered"
            elif index in checked:
                status = "collision_free" if checked[index]["collision_free"] else "collision"
            elif index in free_ids:
                status = "collision_free"
            else:
                status = "collision_unreported"
            diagnostics.append({"candidate_index": index, "status": status,
                                **checked.get(index, {})})
        self.grasp_diagnostics = tuple(diagnostics)
        self.grasp_generation_status = "model_not_generated" if not generated else "generated"
        self.adjacent_blockers = tuple(sorted({
            int(neighbor)
            for row in checks
            if not row["collision_free"] and not row.get("table_collision", False)
            and not row.get("target_non_finger_collision", False)
            for neighbor in row.get("neighbor_ids", ())
        }))
        self.timings["target_grasp_collision_s"] = time.perf_counter() - stage
        self.path.append("TARGET_GRASP_AND_COLLISION")
        if free:
            selection_started = time.perf_counter()
            self.result, self.acted, self.status = dict(free[0]), self.target, "selected"
            self.timings["final_action_decision_s"] = time.perf_counter() - selection_started
            self.path.append("TASK_GRASP")
        return {
            "stage": self.path[-1],
            "candidates": task,
            "collision_free": tuple(free),
            "selected": self.result,
            "grasp_diagnostics": self.grasp_diagnostics,
            "grasp_generated_count": len(generated),
            "grasp_generation_status": self.grasp_generation_status,
            "adjacent_blockers": self.adjacent_blockers,
        }

    def infer_obstruction(self) -> dict:
        stage = time.perf_counter()
        diagnostics = {}
        relations = obstruction_graph(
            self.scene.xyz_m,
            self.scene.instance_id,
            overlap_margin_scale=float(self.settings.get("overlap_margin_scale", 0.15)),
            minimum_xy_overlap=float(self.settings.get("minimum_xy_overlap", 0.05)),
            minimum_height_gap_scale=float(self.settings.get("minimum_height_gap_scale", 0.08)),
            diagnostics=diagnostics,
        )
        self.obstruction = top_obstructor(self.target, relations)
        self.obstructions = reachable_obstructors(self.target, relations)
        self.pushable_objects = tuple(sorted(set(self.obstructions) | set(self.adjacent_blockers)))
        self.timings["obstruction_inference_s"] = time.perf_counter() - stage
        self.path.append("OBSTRUCTION_INFERENCE")
        if self.target in diagnostics.get("ambiguous_instances", ()):
            self.status = "operator_attention"
            self.reason = "目标实例包含多个明显分离点簇，请重新分割或选择目标后再推断压覆"
            self.pushable_objects = ()
        elif not self.pushable_objects:
            self.status, self.reason = "operator_attention", "未找到可解除目标压覆的物体"
            if self.grasp_generation_status == "model_not_generated":
                self.reason = "模型没有生成目标抓取候选，也未找到压覆物体"
        return {"stage": self.path[-1], "relations": relations,
                "obstruction": self.obstruction, "obstructions": self.obstructions,
                "adjacent_blockers": self.adjacent_blockers,
                "pushable_objects": self.pushable_objects, **diagnostics}

    def _with_blocker_evidence(self, candidate: dict) -> dict:
        item = dict(candidate)
        object_id = int(item.get("acted_object", -1))
        if object_id not in self.pushable_objects:
            return item
        evidence = []
        for row in self.grasp_diagnostics:
            neighbors = tuple(int(value) for value in row.get("neighbor_ids", ()))
            if object_id in neighbors and not row.get("table_collision", False) and not row.get("target_non_finger_collision", False):
                evidence.append({
                    "candidate_index": int(row["candidate_index"]),
                    "other_neighbor_ids": tuple(value for value in neighbors if value != object_id),
                })
        item["blocked_grasps"] = tuple(evidence)
        item["push_reason"] = (
            "overhead_and_adjacent" if object_id in self.obstructions and object_id in self.adjacent_blockers
            else "overhead" if object_id in self.obstructions else "adjacent"
        )
        return item

    def rank_pushes(self, candidates: tuple[dict, ...] | None = None) -> dict:
        if not self.pushable_objects:
            raise RuntimeError("请先执行阻挡物体推断")
        stage = time.perf_counter()
        if candidates is not None:
            self.all_candidates += tuple(self._with_blocker_evidence(item) for item in candidates)
        self.pushes = tuple(sorted((
            self._with_blocker_evidence(item) for item in self.all_candidates
            if int(item.get("action_type", -1)) == int(ActionType.PUSH)
            and int(item.get("acted_object", -1)) in self.pushable_objects
            and math.isfinite(float(item.get("improvement_probability", float("nan"))))
            and 0.0 <= float(item["improvement_probability"]) <= 1.0
        ), key=lambda item: float(item["improvement_probability"]), reverse=True))
        self.timings["push_candidate_ranking_s"] = time.perf_counter() - stage
        if "PUSH_RULE_GENERATION" not in self.path:
            self.path.append("PUSH_RULE_GENERATION")
        self.path.append("PUSH_EVALUATOR_SCORING")
        if not self.pushes:
            self.status, self.reason = "operator_attention", "没有通过PUSH evaluator评分的有效候选"
        else:
            self.result = dict(self.pushes[0])
            self.acted, self.status = int(self.result["acted_object"]), "selected"
            self.path.append("PUSH")
        return {
            "stage": "PUSH_EVALUATOR_SCORING",
            "candidates": self.pushes,
            "selected": self.result,
        }

    def record_push_rules(self, candidates: tuple[dict, ...]) -> dict:
        if not self.pushable_objects:
            raise RuntimeError("请先执行阻挡物体推断")
        self.path.append("PUSH_RULE_GENERATION")
        if not candidates:
            self.status, self.reason = "operator_attention", "没有规则生成的PUSH候选"
        return {"stage": "PUSH_RULE_GENERATION", "candidates": tuple(
            self._with_blocker_evidence(item) for item in candidates
        )}

    def finish(self) -> Prediction:
        self.timings["decision_total_s"] = time.perf_counter() - self.started
        if self.result is None:
            return Prediction(
                {"reason": self.reason},
                self.analysis.inference_seconds,
                self.all_candidates,
                self.timings,
                tuple(self.path),
                self.target,
                None,
                "operator_attention",
            )
        return Prediction(
            dict(self.result),
            self.analysis.inference_seconds,
            self.all_candidates,
            self.timings,
            tuple(self.path),
            self.target,
            self.acted,
        )
