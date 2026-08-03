"""Unified component metrics used by both training validation and offline evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from tcd_prg.config import EvaluationConfig, GraphConfig
from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.geometry.se3 import parallel_jaw_rotation_distance
from tcd_prg.losses.labels import (
    build_global_grasp_labels,
    build_graph_labels,
    build_grasp_proposal_labels,
    build_push_supervision,
    build_verifier_labels,
)
from tcd_prg.models.policy.router import MaskedHierarchicalCandidateRouter

from .evaluator import Evaluator
from .metrics import (
    binary_confusion, confusion_metrics, direction_angle_error_deg, ndcg, recall_at_k,
)


class OfflineModelEvaluator:
    """Accumulate auditable per-state records without changing model predictions."""

    def __init__(
        self, model_config: Any, bootstrap_samples: int = 1_000,
        confidence: float = 0.95, graph_config: GraphConfig | None = None,
        evaluation_config: EvaluationConfig | None = None,
    ) -> None:
        self.model_config = model_config
        self.graph_config = graph_config or GraphConfig()
        self.evaluation_config = evaluation_config or EvaluationConfig()
        self.evaluator = Evaluator(
            bootstrap_samples, confidence, self.evaluation_config.calibration_bins
        )
        self.decisions: dict[tuple[int, int, int], dict[str, Any]] = {}

    @staticmethod
    def _numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    @staticmethod
    def _probability(logit: torch.Tensor) -> np.ndarray:
        return torch.sigmoid(logit.detach().float()).cpu().numpy()

    @staticmethod
    def _safe_relation_names(names: tuple[str, ...], count: int, prefix: str) -> list[str]:
        return [names[index] if index < len(names) else f"{prefix}_{index}" for index in range(count)]

    def _add_grasp_metrics(
        self, record: dict[str, Any], prefix: str, output: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor] | None, row: int,
    ) -> None:
        if labels is None or not bool(labels["sample_valid"][row]):
            return
        target_index = torch.nonzero(labels["target_valid"][row], as_tuple=False).flatten()
        if not len(target_index):
            return
        score = torch.sigmoid(output["quality_logit"][row].detach().float())
        ranked = torch.argsort(score, descending=True, stable=True)
        prediction_t = output["translation_world"][row, ranked]
        prediction_r = output["rotation_matrix"][row, ranked]
        prediction_w = output["width_m"][row, ranked]
        target_t = labels["translation_world"][row, target_index]
        target_r = labels["rotation_matrix"][row, target_index]
        target_w = labels["width_m"][row, target_index]
        translation = torch.cdist(prediction_t.float(), target_t.float())
        rotation = parallel_jaw_rotation_distance(
            prediction_r[:, None].expand(-1, len(target_index), -1, -1),
            target_r[None].expand(len(ranked), -1, -1, -1),
        )
        width = torch.abs(prediction_w[:, None] - target_w[None])
        if prefix == "global_grasp":
            translation_threshold = self.evaluation_config.global_translation_threshold_m
            rotation_threshold = math.radians(self.evaluation_config.global_rotation_threshold_deg)
            width_threshold = self.evaluation_config.global_width_threshold_m
        else:
            translation_threshold = self.model_config.grasp_nms_translation_m
            rotation_threshold = math.radians(self.model_config.grasp_nms_rotation_deg)
            width_threshold = self.model_config.grasp_nms_width_m
        compatible = (
            (translation <= translation_threshold)
            & (rotation <= rotation_threshold)
            & (width <= width_threshold)
        )
        object_target = labels.get("object_index")
        if object_target is not None and "object_logits" in output:
            predicted_object = output["object_logits"][row, ranked].argmax(-1)
            compatible &= predicted_object[:, None] == object_target[row, target_index][None]
        matched: set[int] = set()
        true_positive: list[bool] = []
        errors: list[tuple[float, float, float]] = []
        for prediction_index in range(len(ranked)):
            available = [
                index for index in torch.nonzero(compatible[prediction_index], as_tuple=False)
                .flatten().tolist() if index not in matched
            ]
            success = bool(available)
            true_positive.append(success)
            if success:
                chosen = min(available, key=lambda index: float(translation[prediction_index, index]))
                matched.add(chosen)
                errors.append((
                    float(translation[prediction_index, chosen]),
                    math.degrees(float(rotation[prediction_index, chosen])),
                    float(width[prediction_index, chosen]),
                ))
        tp = np.asarray(true_positive, dtype=bool)
        total_positive = len(target_index)
        for k in self.evaluation_config.ranking_topk:
            used = min(k, len(tp))
            matched_k = int(tp[:used].sum())
            record[f"{prefix}_known_hit_at_{k}"] = float(matched_k > 0)
            record[f"{prefix}_known_recall_at_{k}"] = float(matched_k / total_positive)
            if bool(labels.get("label_set_complete", torch.zeros_like(labels["sample_valid"]))[row]):
                record[f"{prefix}_precision_at_{k}"] = float(matched_k / max(1, used))
        complete = bool(labels.get(
            "label_set_complete", torch.zeros_like(labels["sample_valid"])
        )[row])
        if complete:
            precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
            record[f"{prefix}_average_precision"] = float(
                np.sum(precision * tp) / max(1, total_positive)
            )
        record[f"{prefix}_known_positive_count"] = float(total_positive)
        record[f"{prefix}_predicted_query_count"] = float(len(ranked))
        if errors:
            array = np.asarray(errors, np.float64)
            record[f"{prefix}_matched_translation_error_m"] = float(array[:, 0].mean())
            record[f"{prefix}_matched_rotation_error_deg"] = float(array[:, 1].mean())
            record[f"{prefix}_matched_width_error_m"] = float(array[:, 2].mean())

    def update(
        self, batch: dict[str, Any], output: dict[str, Any],
        loss_terms: dict[str, torch.Tensor] | None = None,
    ) -> None:
        batch_size = batch["xyz"].shape[0]
        selected = None
        if "router" in output:
            selected = MaskedHierarchicalCandidateRouter.select(
                output["router"], batch["action_type"], batch["acted_object"]
            )
        graph_labels = build_graph_labels(batch) if output.get("graph") is not None else None
        verifier_labels = build_verifier_labels(batch) if output.get("verifier") is not None else None
        task_labels = build_grasp_proposal_labels(batch, self.model_config) \
            if "task_grasp" in output else None
        global_labels = build_global_grasp_labels(batch, self.model_config) \
            if "global_grasp" in output else None
        if output.get("push") is not None:
            push_prediction, push_labels = build_push_supervision(
                output["push"], batch, self.model_config
            )
        else:
            empty = torch.zeros_like(batch["candidate_mask"], dtype=torch.bool)
            push_prediction, push_labels = {}, {
                "direction_valid": empty, "utility_valid": empty,
            }
        physical_names = self._safe_relation_names(
            self.graph_config.physical_relations,
            output["graph"].physical_edge_logits.shape[-1] if output.get("graph") is not None else 0,
            "physical",
        )
        task_names = self._safe_relation_names(
            self.graph_config.task_relations,
            output["graph"].task_edge_logits.shape[-1] if output.get("graph") is not None else 0,
            "task",
        )

        for row in range(batch_size):
            sample = batch["samples"][row]
            record: dict[str, Any] = {
                "scene_id": sample.observation.scene_id,
                "state_id": sample.observation.state_id,
                "task_index": sample.observation.task_index,
                "category": int(sample.observation.object_category_id[sample.observation.target_object]),
                "task_region": sample.observation.task_region_id,
                "sequence_length": sample.state_labels.sequence_depth,
                "occlusion_level": int(min(4, (1.0 - sample.state_labels.target_visible_ratio) * 5)),
            }
            if loss_terms:
                for key, value in loss_terms.items():
                    if key.startswith("loss_"):
                        record[key] = float(value.detach())

            if "region_target" in batch and "region" in output:
                valid = self._numpy(batch["region_valid"][row]).astype(bool)
                target = self._numpy(batch["region_target"][row]).astype(bool)
                probability = self._numpy(output["region"]["region_probability"][row])
                prediction = probability >= self.evaluation_config.region_probability_threshold
                record["_confusion_region_foreground"] = binary_confusion(
                    prediction, target, valid
                )
                if bool(batch["visibility_valid"][row]):
                    visibility_probability = float(torch.sigmoid(
                        output["region"]["visibility_logit"][row].detach().float()
                    ))
                    visibility_target = float(batch["visibility_target"][row])
                    record["region_visibility_mae"] = abs(
                        visibility_probability - visibility_target
                    )
                    record["_binary_region_visibility"] = (
                        [visibility_probability], [visibility_target > 0.0]
                    )

            if output.get("graph") is not None and graph_labels is not None:
                physical_confusions: list[tuple[int, int, int, int]] = []
                for index, name in enumerate(physical_names):
                    valid = self._numpy(graph_labels["physical_edge_valid"][row, ..., index]).astype(bool)
                    target = self._numpy(graph_labels["physical_edge_target"][row, ..., index]).astype(bool)
                    probability = self._probability(
                        output["graph"].physical_edge_logits[row, ..., index]
                    )
                    confusion = binary_confusion(
                        probability >= self.model_config.graph_edge_threshold, target, valid
                    )
                    physical_confusions.append(confusion)
                    record[f"_confusion_physical_{name}"] = confusion
                    record[f"_binary_physical_{name}"] = (probability[valid], target[valid])
                if physical_confusions:
                    record["_confusion_physical_edge_micro"] = tuple(
                        np.asarray(physical_confusions, np.int64).sum(0).tolist()
                    )
                    f1_values = [confusion_metrics(*value)["f1"] for value in physical_confusions]
                    f1_values = [value for value in f1_values if np.isfinite(value)]
                    if f1_values:
                        record["physical_edge_macro_f1"] = float(np.mean(f1_values))
                task_confusions: list[tuple[int, int, int, int]] = []
                for index, name in enumerate(task_names):
                    valid = self._numpy(graph_labels["task_edge_valid"][row, ..., index]).astype(bool)
                    target = self._numpy(graph_labels["task_edge_target"][row, ..., index]).astype(bool)
                    probability = self._probability(
                        output["graph"].task_edge_logits[row, ..., index]
                    )
                    confusion = binary_confusion(
                        probability >= self.model_config.graph_edge_threshold, target, valid
                    )
                    task_confusions.append(confusion)
                    record[f"_confusion_task_edge_{name}"] = confusion
                    record[f"_binary_task_edge_{name}"] = (probability[valid], target[valid])
                if task_confusions:
                    record["_confusion_task_edge_micro"] = tuple(
                        np.asarray(task_confusions, np.int64).sum(0).tolist()
                    )
                    f1_values = [confusion_metrics(*value)["f1"] for value in task_confusions]
                    f1_values = [value for value in f1_values if np.isfinite(value)]
                    if f1_values:
                        record["task_edge_macro_f1"] = float(np.mean(f1_values))
                object_valid = self._numpy(
                    batch["object_mask"][row] & batch["object_active"][row]
                ).astype(bool)
                for name, prediction_key, target_key in (
                    ("direct_blocker", "derived_direct_mask", "direct_blocker_target"),
                    ("indirect_blocker", "derived_indirect_mask", "indirect_blocker_target"),
                    ("actionable_blocker", "derived_actionable_mask", "actionable_blocker_target"),
                ):
                    record[f"_confusion_{name}"] = binary_confusion(
                        self._numpy(getattr(output["graph"], prediction_key)[row]).astype(bool),
                        self._numpy(batch[target_key][row]).astype(bool), object_valid,
                    )
                direct_score = self._numpy(output["graph"].task_edge_logits[row]).max(-1)
                direct_target = self._numpy(batch["direct_blocker_target"][row]).astype(bool)
                for k in (1, 3):
                    record[f"direct_blocker_recall_at_{k}"] = recall_at_k(
                        direct_score, direct_target, object_valid, k
                    )

            candidate_valid = self._numpy(batch["candidate_mask"][row]).astype(bool)
            success = self._numpy(batch["policy_success_mask"][row]).astype(bool)
            evaluated = success | self._numpy(
                batch["evaluation_status"][row] == int(CandidateStatus.NEGATIVE)
            ).astype(bool)
            action_type = self._numpy(batch["action_type"][row]).astype(int)
            acted_object = self._numpy(batch["acted_object"][row]).astype(int)
            if "router" in output:
                router_score = self._numpy(output["router"].candidate_logits[row])
                record["candidate_ndcg"] = ndcg(
                    router_score, success.astype(float), candidate_valid & evaluated
                )
            if selected is not None and int(selected[row]) >= 0:
                index = int(selected[row])
                known = bool(evaluated[index])
                record["selected_candidate_known"] = float(known)
                record["selected_action_type"] = int(action_type[index])
                if known:
                    record["selected_candidate_success"] = float(success[index])
                    positive = success & evaluated
                    if positive.any():
                        record["selected_action_type_correct"] = float(
                            action_type[index] in set(action_type[positive])
                        )
                        record["selected_object_correct"] = float(
                            acted_object[index] in set(acted_object[positive])
                        )
                self.decisions[(sample.observation.scene_id, sample.observation.state_id,
                                sample.observation.task_index)] = {
                    "index": index, "action_type": action_type, "evaluated": evaluated,
                    "success": success, "to_state": sample.candidates.to_state.copy(),
                    "after_state_valid": sample.candidates.after_state_valid.copy(),
                    "depth": sample.state_labels.sequence_depth,
                    "graspable": sample.state_labels.graspable, "record": record,
                }

            generated = batch.get("generated_policy_candidates")
            if generated is not None and "generated_router" in output:
                valid = self._numpy(generated["valid"][row]).astype(bool)
                known = self._numpy(
                    generated["label_status"][row] != int(CandidateStatus.UNKNOWN_UNTESTED)
                ).astype(bool)
                generated_success = self._numpy(generated["policy_success"][row]).astype(bool)
                logits = self._numpy(output["generated_router"].candidate_logits[row])
                record["generated_candidate_ndcg"] = ndcg(
                    logits, generated_success.astype(float), valid & known
                )
                record["generated_positive_coverage"] = float(generated_success[valid & known].any())
                record["generated_effective_policy_row"] = float(
                    generated_success[valid & known].any() and (~generated_success & valid & known).any()
                )
                if valid.any():
                    chosen = np.flatnonzero(valid)[np.argmax(logits[valid])]
                    record["generated_selected_candidate_known"] = float(known[chosen])
                    if known[chosen]:
                        record["generated_selected_candidate_success"] = float(
                            generated_success[chosen]
                        )

            if output.get("verifier") is not None and verifier_labels is not None:
                valid = self._numpy(verifier_labels["overall_valid"][row]).astype(bool)
                if valid.any():
                    target = self._numpy(verifier_labels["overall_target"][row]).astype(bool)
                    probability = self._probability(output["verifier"]["overall_logit"][row])
                    record["_confusion_verifier_overall"] = binary_confusion(
                        probability >= self.evaluation_config.verifier_probability_threshold,
                        target, valid
                    )
                    record["_binary_verifier_overall"] = (probability[valid], target[valid])

            positive_push_objects = set(acted_object[
                success & (action_type == int(ActionType.PUSH))
            ]) if output.get("push") is not None else set()
            if positive_push_objects:
                ranked_objects = torch.argsort(
                    output["push"]["object_logits"][row], descending=True
                ).detach().cpu().tolist()
                record["push_object_top1"] = float(ranked_objects[0] in positive_push_objects)
                record["push_object_top3"] = float(
                    bool(set(ranked_objects[:3]) & positive_push_objects)
                )
            if "router" in output:
                push_candidates = candidate_valid & evaluated & (
                    action_type == int(ActionType.PUSH)
                )
                record["push_candidate_ndcg"] = ndcg(
                    self._numpy(output["router"].candidate_logits[row]),
                    success.astype(float), push_candidates,
                )
            parameter_valid = self._numpy(push_labels["direction_valid"][row]).astype(bool)
            if parameter_valid.any():
                label_direction = self._numpy(
                    batch["action_parameters"]["push_direction_world"][row]
                )
                predicted_bin = self._numpy(
                    push_prediction["direction_logits"][row]
                ).argmax(-1)
                target_bin = self._numpy(push_labels["direction_bin"][row]).astype(int)
                valid_indices = np.flatnonzero(parameter_valid)
                record["push_direction_bin_top1"] = float(np.mean(
                    predicted_bin[valid_indices] == target_bin[valid_indices]
                ))
                top2 = torch.topk(
                    push_prediction["direction_logits"][row], k=min(
                        2, push_prediction["direction_logits"].shape[-1]
                    ), dim=-1,
                ).indices.detach().cpu().numpy()
                record["push_direction_bin_top2"] = float(np.mean([
                    target_bin[index] in top2[index] for index in valid_indices
                ]))
                point_index = push_prediction["point_index"][row]
                residual = self._numpy(output["push"]["direction_residual"][
                    row, point_index, torch.from_numpy(predicted_bin).to(point_index.device)
                ])
                angles = (predicted_bin + 0.5) * 2.0 * np.pi / self.model_config.num_direction_bins
                direction = np.stack((np.cos(angles), np.sin(angles)), -1) + residual
                record["push_direction_angle_error_deg"] = float(np.mean([
                    direction_angle_error_deg(direction[index], label_direction[index, :2])
                    for index in np.flatnonzero(parameter_valid)
                ]))
                valid_points = self._numpy(batch["point_mask"][row]).astype(bool)
                contact_score = self._numpy(output["push"]["contact_logits"][row])
                predicted_contact = np.flatnonzero(valid_points)[np.argmax(contact_score[valid_points])]
                contacts = self._numpy(
                    batch["action_parameters"]["push_contact_world"][row]
                )[parameter_valid]
                if len(contacts):
                    point = self._numpy(batch["xyz"][row, predicted_contact])
                    record["push_contact_distance_error_m"] = float(
                        np.linalg.norm(contacts - point[None], axis=1).min()
                    )
            utility_valid = self._numpy(push_labels["utility_valid"][row]).astype(bool)
            if utility_valid.any():
                error = np.abs(
                    self._numpy(push_prediction["utility_delta"][row])
                    - self._numpy(push_labels["utility_delta"][row])
                )
                record["push_utility_delta_mae"] = float(np.mean(error[utility_valid]))

            if "task_grasp" in output:
                self._add_grasp_metrics(
                    record, "task_grasp", output["task_grasp"], task_labels, row
                )
            if "global_grasp" in output:
                self._add_grasp_metrics(
                    record, "global_grasp", output["global_grasp"], global_labels, row
                )
            if (
                "region_target" in batch and "task_grasp" in output
                and "attention_point_index" in output["task_grasp"]
            ):
                ranked = torch.argsort(
                    output["task_grasp"]["quality_logit"][row], descending=True, stable=True
                )
                anchor = output["task_grasp"]["attention_point_index"][row, ranked]
                valid_anchor = batch["region_valid"][row, anchor]
                region_anchor = batch["region_target"][row, anchor]
                for k in self.evaluation_config.ranking_topk:
                    used = min(k, len(anchor))
                    selected_valid = valid_anchor[:used]
                    if bool(selected_valid.any()):
                        record[f"task_grasp_anchor_region_precision_at_{k}"] = float(
                            region_anchor[:used][selected_valid].float().mean()
                        )
            if "_inference_time_s_per_sample" in output:
                record["planning_time_s"] = float(output["_inference_time_s_per_sample"])
            self.evaluator.add(**record)
            key = (sample.observation.scene_id, sample.observation.state_id,
                   sample.observation.task_index)
            if key in self.decisions:
                self.decisions[key]["record"] = self.evaluator.records[-1]

    def finalize_closed_loop_replay(self, horizons: tuple[int, ...] = (0, 1, 3, 5)) -> None:
        """Replay known dataset transitions; this is not an online execution success rate."""

        roots = [(key, value) for key, value in self.decisions.items() if value["depth"] == 0]
        for (scene_id, state_id, task_index), root in roots:
            record = root["record"]
            for horizon in sorted(set(horizons)):
                key = (scene_id, state_id, task_index)
                preparations, evaluable, success = 0, True, False
                visited = set()
                while preparations <= horizon:
                    if key in visited or key not in self.decisions:
                        evaluable = False
                        break
                    visited.add(key)
                    decision = self.decisions[key]
                    index = int(decision["index"])
                    if not bool(decision["evaluated"][index]):
                        evaluable = False
                        break
                    kind = int(decision["action_type"][index])
                    positive = bool(decision["success"][index])
                    if kind == int(ActionType.TASK_GRASP):
                        success = positive
                        break
                    if not positive or not bool(decision["after_state_valid"][index]):
                        break
                    preparations += 1
                    if preparations > horizon:
                        break
                    key = (scene_id, int(decision["to_state"][index]), task_index)
                record[f"labelled_replay_evaluable_h{horizon}"] = float(evaluable)
                if evaluable:
                    record[f"labelled_replay_task_success_h{horizon}"] = float(success)
                    if horizon == max(horizons):
                        record["labelled_replay_preparation_actions"] = float(preparations)
                        if not root["graspable"]:
                            record["labelled_replay_recovery_success"] = float(success)

    def summarize(self) -> dict[str, Any]:
        return self.evaluator.summarize()

    def export(self, output_dir: str, config: dict[str, Any]) -> None:
        self.evaluator.export(output_dir, config)
