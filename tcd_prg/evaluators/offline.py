"""Unified state-group metrics for TCD-PRG and its configured ablations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from tcd_prg.constants import ActionType, CandidateStatus
from tcd_prg.losses.labels import (
    build_grasp_proposal_labels,
    build_push_supervision,
    build_remove_labels,
    build_verifier_labels,
)
from tcd_prg.models.policy.router import MaskedHierarchicalCandidateRouter

from .evaluator import Evaluator
from .metrics import (
    binary_auroc,
    binary_f1,
    dice_score,
    direction_angle_error_deg,
    intersection_over_union,
    macro_f1,
    ndcg,
)


class OfflineModelEvaluator:
    def __init__(self, model_config: Any, bootstrap_samples: int = 1_000,
                 confidence: float = 0.95) -> None:
        self.model_config = model_config
        self.evaluator = Evaluator(bootstrap_samples, confidence)
        self.decisions: dict[tuple[int, int, int], dict[str, Any]] = {}

    @staticmethod
    def _numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    def update(self, batch: dict[str, Any], output: dict[str, Any]) -> None:
        batch_size = batch["xyz"].shape[0]
        selected = None
        if "router" in output:
            selected = MaskedHierarchicalCandidateRouter.select(
                output["router"], batch["action_type"], batch["acted_object"]
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
            if "region_target" in batch:
                valid = self._numpy(batch["region_valid"][row]).astype(bool)
                target = self._numpy(batch["region_target"][row]).astype(bool)
                prediction = self._numpy(output["region"]["region_probability"][row]) >= 0.5
                record["region_miou"] = intersection_over_union(prediction, target, valid)
                record["region_dice"] = dice_score(prediction, target, valid)
                if not target[valid].any():
                    record["invisible_region_false_positive"] = float(prediction[valid].any())
            if output["graph"] is not None:
                object_valid = self._numpy(batch["object_mask"][row] & batch["object_active"][row]).astype(bool)
                pair_valid = object_valid[:, None, None] & object_valid[None, :, None]
                physical_prediction = self._numpy(
                    output["graph"].physical_edge_logits[row]
                ) >= 0
                physical_target = self._numpy(batch["relation_graph"][row]).astype(bool)
                record["physical_edge_f1"] = binary_f1(
                    physical_prediction, physical_target,
                    np.broadcast_to(pair_valid, physical_target.shape),
                )
                task_prediction = self._numpy(output["graph"].task_edge_logits[row]) >= 0
                task_target = self._numpy(batch["task_block_graph"][row]).astype(bool)
                record["task_blocking_edge_f1"] = binary_f1(
                    task_prediction, task_target,
                    np.broadcast_to(object_valid[:, None], task_target.shape),
                )
                record["direct_blocker_f1"] = binary_f1(
                    self._numpy(output["graph"].direct_blocker_logits[row]) >= 0,
                    self._numpy(batch["direct_blocker_target"][row]).astype(bool), object_valid,
                )
                record["indirect_blocker_f1"] = binary_f1(
                    self._numpy(output["graph"].indirect_blocker_logits[row]) >= 0,
                    self._numpy(batch["indirect_blocker_target"][row]).astype(bool), object_valid,
                )
                record["actionable_blocker_accuracy"] = float(np.mean(
                    (self._numpy(output["graph"].actionable_blocker_logits[row]) >= 0)[object_valid]
                    == self._numpy(batch["actionable_blocker_target"][row]).astype(bool)[object_valid]
                )) if object_valid.any() else float("nan")
                direct_target = self._numpy(batch["direct_blocker_target"][row]).astype(bool)
                direct_score = self._numpy(output["graph"].direct_blocker_logits[row])
                valid_indices = np.flatnonzero(object_valid)
                ranked = valid_indices[np.argsort(-direct_score[valid_indices])]
                positives = set(np.flatnonzero(direct_target & object_valid).tolist())
                if positives:
                    for k in (1, 3):
                        record[f"direct_blocker_recall_at_{k}"] = len(
                            positives.intersection(ranked[:k].tolist())
                        ) / len(positives)
            candidate_valid = self._numpy(batch["candidate_mask"][row]).astype(bool)
            evaluated = self._numpy(
                batch["evaluation_status"][row] != int(CandidateStatus.UNKNOWN_UNTESTED)
            ).astype(bool)
            success = self._numpy(batch["success_mask"][row]).astype(bool)
            action_type = self._numpy(batch["action_type"][row]).astype(int)
            acted_object = self._numpy(batch["acted_object"][row]).astype(int)
            if selected is not None and int(selected[row]) >= 0:
                index = int(selected[row])
                record["action_type_accuracy"] = float(
                    action_type[index] in set(action_type[success & evaluated])
                )
                record["acted_object_accuracy"] = float(
                    acted_object[index] in set(acted_object[success & evaluated])
                )
                record["successful_action_set_recall"] = float(success[index] & evaluated[index])
                record["selected_candidate_evaluated"] = float(evaluated[index])
                record["selected_action_type"] = action_type[index]
                self.decisions[(sample.observation.scene_id, sample.observation.state_id,
                                sample.observation.task_index)] = {
                    "index": index,
                    "action_type": action_type,
                    "evaluated": evaluated,
                    "success": success,
                    "to_state": sample.candidates.to_state.copy(),
                    "after_state_valid": sample.candidates.after_state_valid.copy(),
                    "depth": sample.state_labels.sequence_depth,
                    "graspable": sample.state_labels.graspable,
                    "record": record,
                }
            if "router" in output:
                record["candidate_ndcg"] = ndcg(
                    self._numpy(output["router"].candidate_logits[row]), success.astype(float),
                    candidate_valid & evaluated,
                )
                if bool(batch["remaining_steps_valid"][row]):
                    record["remaining_steps_mae"] = abs(
                        float(output["router"].remaining_steps_prediction[row])
                        - float(batch["remaining_steps_target"][row])
                    )
            proposal = build_grasp_proposal_labels(batch, self.model_config, generic_remove=False)
            proposal_valid = self._numpy(proposal["proposal_valid"][row]).astype(bool)
            contact_target = self._numpy(proposal["contact_target"][row]).astype(bool)
            proposal_score = self._numpy(
                output["task_grasp"]["contact_logits"][row]
                + output["task_grasp"]["proposal_confidence_logit"][row]
                + output["task_grasp"]["task_compatibility_logit"][row]
            )
            proposal_indices = np.flatnonzero(proposal_valid)
            if contact_target.any() and len(proposal_indices):
                order = proposal_indices[np.argsort(-proposal_score[proposal_indices])]
                for k in (1, 5, 10):
                    record[f"task_grasp_recall_at_{k}"] = float(contact_target[order[:k]].any())
                if "region_target" in batch:
                    region = self._numpy(batch["region_target"][row]).astype(bool)
                    top = order[: min(10, len(order))]
                    record["correct_region_contact_rate_at_10"] = float(np.mean(region[top]))
                    record["wrong_region_grasp_rate_at_10"] = float(np.mean(~region[top]))
            graspable = (
                sample.state_labels.verified_positive_grasp_count
                >= sample.state_labels.required_grasp_count
            )
            if output["verifier"] is not None:
                task_grasp = candidate_valid & (action_type == int(ActionType.TASK_GRASP))
                overall = self._numpy(output["verifier"]["overall_logit"][row])
                predicted_direct = bool(np.any((overall >= 0) & task_grasp))
                record["state_graspability_accuracy"] = float(predicted_direct == graspable)
                record["direct_grasp_false_positive"] = float(predicted_direct and not graspable)
                verifier_labels = build_verifier_labels(batch)
                for head in ("stability", "task_compatibility", "collision", "clearance",
                             "approach", "overall"):
                    valid_head = self._numpy(verifier_labels[f"{head}_valid"][row]).astype(bool)
                    if valid_head.any():
                        target_head = self._numpy(
                            verifier_labels[f"{head}_target"][row]
                        ).astype(bool)
                        score_head = self._numpy(output["verifier"][f"{head}_logit"][row])
                        record[f"verifier_{head}_f1"] = binary_f1(
                            score_head >= 0, target_head, valid_head
                        )
                        record[f"verifier_{head}_auroc"] = binary_auroc(
                            score_head, target_head, valid_head
                        )
                record["collision_prediction_accuracy"] = self._masked_accuracy(
                    self._numpy(output["verifier"]["collision_logit"][row]) >= 0,
                    self._numpy(verifier_labels["collision_target"][row]).astype(bool),
                    self._numpy(verifier_labels["collision_valid"][row]).astype(bool),
                )
            positive_push_objects = set(acted_object[success & (action_type == int(ActionType.PUSH))])
            if "router" in output:
                push_candidates = candidate_valid & evaluated & (
                    action_type == int(ActionType.PUSH)
                )
                record["push_candidate_ndcg"] = ndcg(
                    self._numpy(output["router"].candidate_logits[row]),
                    success.astype(float), push_candidates,
                )
            if positive_push_objects:
                predicted = int(output["push"]["object_logits"][row].argmax())
                record["push_acted_object_top1"] = float(predicted in positive_push_objects)
            push_prediction, push_labels = build_push_supervision(
                output["push"], batch, use_potential=True, use_risk=True
            )
            push_parameter_valid = self._numpy(push_labels["direction_valid"][row]).astype(bool)
            if push_parameter_valid.any():
                label_direction = self._numpy(
                    batch["action_parameters"]["push_direction_world"][row]
                )
                predicted_bin = self._numpy(
                    push_prediction["direction_logits"][row]
                ).argmax(-1)
                residual = self._numpy(push_prediction["direction_residual"][row])
                angles = (predicted_bin + 0.5) * 2.0 * np.pi / self.model_config.num_direction_bins
                predicted_direction = np.stack((np.cos(angles), np.sin(angles)), axis=-1) + residual
                errors = [direction_angle_error_deg(predicted_direction[index], label_direction[index, :2])
                          for index in np.flatnonzero(push_parameter_valid)]
                record["push_direction_angle_error_deg"] = float(np.mean(errors))
                approach_valid = self._numpy(push_labels["approach_valid"][row]).astype(bool)
                record["push_approach_accuracy"] = self._masked_accuracy(
                    self._numpy(push_prediction["approach_logits"][row]).argmax(-1),
                    self._numpy(push_labels["approach_mode"][row]).astype(int), approach_valid,
                )
                predicted_contact = int(self._numpy(output["push"]["contact_logits"][row]).argmax())
                contacts = self._numpy(batch["action_parameters"]["push_contact_world"][row])
                valid_contacts = contacts[push_parameter_valid]
                if len(valid_contacts):
                    point = self._numpy(batch["xyz"][row, predicted_contact])
                    record["push_contact_distance_error_m"] = float(
                        np.linalg.norm(valid_contacts - point[None], axis=1).min()
                    )
            outcome_valid = self._numpy(push_labels["outcome_valid"][row]).astype(bool)
            record["push_outcome_macro_f1"] = macro_f1(
                self._numpy(push_prediction["outcome_logits"][row]).argmax(-1),
                self._numpy(push_labels["outcome_code"][row]).astype(int), outcome_valid, 7,
            )
            potential_valid = self._numpy(push_labels["potential_after_valid"][row]).astype(bool)
            if potential_valid.any():
                error = np.abs(
                    self._numpy(push_prediction["potential_delta"][row])
                    - self._numpy(push_labels["potential_delta"][row])
                )
                record["potential_delta_mae"] = float(np.mean(error[potential_valid]))
            risk_valid = self._numpy(push_labels["risk_valid"][row]).astype(bool)
            risk_target = self._numpy(push_labels["risk_target"][row]).astype(bool)
            risk_score = self._numpy(push_prediction["risk_logits"][row])
            for risk_index, risk_name in enumerate(("unstable", "out_of_workspace", "other_invalid")):
                record[f"push_{risk_name}_auroc"] = binary_auroc(
                    risk_score[:, risk_index], risk_target[:, risk_index], risk_valid[:, risk_index]
                )
            positive_remove_objects = set(
                acted_object[success & (action_type == int(ActionType.PICK_REMOVE))]
            )
            if positive_remove_objects:
                predicted = int(output["pick_remove"]["object_logits"][row].argmax())
                record["remove_acted_object_top1"] = float(predicted in positive_remove_objects)
            remove_labels = build_remove_labels(batch)
            if "candidate_logits" in output["pick_remove"]:
                remove_valid = self._numpy(remove_labels["candidate_valid"][row]).astype(bool)
                remove_positive = self._numpy(remove_labels["candidate_positive"][row]).astype(float)
                remove_score = self._numpy(output["pick_remove"]["candidate_logits"][row])
                record["remove_candidate_ndcg"] = ndcg(
                    remove_score, remove_positive, remove_valid
                )
                record["remove_grasp_recall_at_5"] = self._recall_at_k(
                    remove_score, remove_positive.astype(bool), remove_valid, 5
                )
                remove_outcome_valid = self._numpy(
                    remove_labels["outcome_valid"][row]
                ).astype(bool)
                record["remove_outcome_macro_f1"] = macro_f1(
                    self._numpy(output["pick_remove"]["outcome_logits"][row]).argmax(-1),
                    self._numpy(remove_labels["outcome_code"][row]).astype(int),
                    remove_outcome_valid, 7,
                )
            if "_inference_time_s_per_sample" in output:
                record["planning_time_s"] = float(output["_inference_time_s_per_sample"])
            self.evaluator.add(**record)
            decision_key = (
                sample.observation.scene_id,
                sample.observation.state_id,
                sample.observation.task_index,
            )
            if decision_key in self.decisions:
                self.decisions[decision_key]["record"] = self.evaluator.records[-1]

    @staticmethod
    def _masked_accuracy(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
        valid = np.asarray(valid, bool)
        return float(np.mean(np.asarray(prediction)[valid] == np.asarray(target)[valid])) \
            if valid.any() else float("nan")

    @staticmethod
    def _recall_at_k(scores: np.ndarray, target: np.ndarray, valid: np.ndarray, k: int) -> float:
        positive = np.asarray(target, bool) & np.asarray(valid, bool)
        if not positive.any():
            return float("nan")
        indices = np.flatnonzero(valid)
        ranked = indices[np.argsort(-np.asarray(scores)[indices])][:k]
        return float(np.count_nonzero(positive[ranked]) / np.count_nonzero(positive))

    def finalize_closed_loop_replay(self, horizons: tuple[int, ...] = (0, 1, 3, 5)) -> None:
        """Replay selected labelled transitions without treating UNKNOWN as failure."""

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
                record[f"closed_loop_evaluable_h{horizon}"] = float(evaluable)
                if evaluable:
                    record[f"task_success_rate_h{horizon}"] = float(success)
                if horizon == max(horizons) and evaluable:
                    record["correct_functional_region_grasp_rate"] = float(success)
                    record["average_preparation_steps"] = float(preparations)
                    if not root["graspable"]:
                        record["closed_loop_recovery_rate"] = float(success)

    def export(self, output_dir: str, config: dict[str, Any]) -> None:
        self.evaluator.export(output_dir, config)
