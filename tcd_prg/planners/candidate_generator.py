"""Decode executable candidates from predicted object-centric scene features."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from tcd_prg.config import ModelConfig
from tcd_prg.constants import PUSH_DISTANCE_M, ActionType
from tcd_prg.geometry.grasp_nms import task_grasp_nms
from tcd_prg.geometry.se3 import matrix_to_quaternion_xyzw

from .push_decoder import decode_push_candidates, push_nms_mask


class DenseCandidateGenerator:
    def __init__(self, config: ModelConfig, *, use_push_potential: bool = True) -> None:
        self.config = config
        self.use_push_potential = bool(use_push_potential)

    @staticmethod
    def _empty_row(device: torch.device) -> dict[str, Tensor]:
        return {
            "type": torch.empty(0, dtype=torch.long, device=device),
            "object": torch.empty(0, dtype=torch.long, device=device),
            "contact_world": torch.empty(0, 3, device=device),
            "direction_world": torch.empty(0, 3, device=device),
            "pose_world": torch.empty(0, 7, device=device),
            "destination_world": torch.empty(0, 3, device=device),
            "width_m": torch.empty(0, device=device),
            "proposal_score": torch.empty(0, device=device),
            "point_index": torch.empty(0, dtype=torch.long, device=device),
            "direction_bin": torch.empty(0, dtype=torch.long, device=device),
            "direction_score": torch.empty(0, device=device),
            "object_score": torch.empty(0, device=device),
            "contact_score": torch.empty(0, device=device),
            "utility": torch.empty(0, device=device),
            "direction_residual": torch.empty(0, 2, device=device),
            "effective_logit": torch.empty(0, device=device),
            "effective_probability": torch.empty(0, device=device),
        }

    @staticmethod
    def _nearest_scene_point(translation: Tensor, xyz: Tensor, point_mask: Tensor) -> Tensor:
        points = torch.nonzero(point_mask, as_tuple=False).flatten()
        if not len(points):
            return torch.zeros(len(translation), dtype=torch.long, device=translation.device)
        distance = torch.cdist(translation, xyz[points])
        return points[distance.argmin(-1)]

    @staticmethod
    def _target_local_object_mask(
        xyz: Tensor,
        point_mask: Tensor,
        instance_probability: Tensor,
        object_mask: Tensor,
        target_object: int,
        margin_m: float,
    ) -> Tensor:
        """Approximate GAPG-style expanded target neighborhood in XY.

        This is candidate-domain pruning only, not a learned dependency graph.
        A predicted object is relevant when its hard mask AABB intersects the
        target AABB expanded by ``margin_m``.
        """
        owner = instance_probability.argmax(0)
        owner_domain = (
            (owner[None] == torch.arange(instance_probability.shape[0], device=xyz.device)[:, None])
            & point_mask[None]
            & object_mask[:, None]
        )
        confident = owner_domain & (instance_probability >= 0.5)
        has_confident = confident.any(-1)
        hard = torch.where(has_confident[:, None], confident, owner_domain)
        xy = xyz[:, :2]
        inf = torch.full_like(xy, float("inf"))
        ninf = torch.full_like(xy, float("-inf"))
        minimum = torch.where(hard[..., None], xy[None], inf[None]).amin(1)
        maximum = torch.where(hard[..., None], xy[None], ninf[None]).amax(1)
        has_points = hard.any(-1) & object_mask
        target_min = minimum[target_object] - float(margin_m)
        target_max = maximum[target_object] + float(margin_m)
        overlap = (maximum >= target_min).all(-1) & (minimum <= target_max).all(-1)
        return has_points & overlap & has_points[target_object]

    def _nms_indices(
        self,
        pose: Tensor,
        width: Tensor,
        score: Tensor,
        objects: Tensor,
        amount: int,
        *,
        global_grasp: bool,
    ) -> Tensor:
        if not len(score):
            return torch.empty(0, dtype=torch.long, device=score.device)
        if global_grasp:
            thresholds = (
                self.config.global_grasp_nms_translation_m,
                self.config.global_grasp_nms_rotation_deg,
                self.config.global_grasp_nms_width_m,
                self.config.global_grasp_nms_approach_deg,
            )
        else:
            thresholds = (
                self.config.grasp_nms_translation_m,
                self.config.grasp_nms_rotation_deg,
                self.config.grasp_nms_width_m,
                self.config.grasp_nms_approach_deg,
            )
        keep = task_grasp_nms(
            pose[None],
            width[None],
            score[None],
            objects[None],
            torch.ones((1, len(score)), dtype=torch.bool, device=score.device),
            translation_threshold_m=thresholds[0],
            rotation_threshold_deg=thresholds[1],
            width_threshold_m=thresholds[2],
            approach_threshold_deg=thresholds[3],
        )[0]
        selected = torch.nonzero(keep, as_tuple=False).flatten()
        return selected[score[selected].argsort(descending=True, stable=True)[:amount]]

    def select_task_grasp_indices(
        self,
        translation: Tensor,
        rotation: Tensor,
        width: Tensor,
        score: Tensor,
        valid: Tensor,
    ) -> Tensor:
        """Deployment selection: valid candidates, then NMS and top-k."""
        eligible = torch.nonzero(valid.bool(), as_tuple=False).flatten()
        if not len(eligible):
            return eligible
        pose = torch.cat((translation, matrix_to_quaternion_xyzw(rotation)), -1)
        objects = torch.zeros_like(score, dtype=torch.long)
        selected_local = self._nms_indices(
            pose[eligible],
            width[eligible],
            score[eligible],
            objects[eligible],
            self.config.task_grasp_candidates,
            global_grasp=False,
        )
        return eligible[selected_local]

    def apply_push_nms(self, candidates: dict[str, Tensor], candidate_scores: Tensor) -> Tensor:
        keep = candidates["valid"].clone()
        for row in range(keep.shape[0]):
            push = torch.nonzero(
                keep[row] & (candidates["type"][row] == int(ActionType.PUSH)),
                as_tuple=False,
            ).flatten()
            decoded = {
                "object": candidates["object"][row, push],
                "contact_world": candidates["contact_world"][row, push],
                "direction_world": candidates["direction_world"][row, push],
                "proposal_score": candidate_scores[row, push],
            }
            keep[row, push] = push_nms_mask(decoded, self.config)
        return keep

    def global_predictions(
        self,
        batch: dict[str, Tensor],
        output: dict[str, Any],
        topk: int | None = None,
        score_kind: str = "scene",
    ) -> list[dict[str, Tensor]]:
        if score_kind not in {"raw", "scene"}:
            raise ValueError("score_kind must be scene (raw is a compatibility alias)")
        head = output["global_grasp"]
        amount = int(topk or self.config.candidate_topk)
        sensor = output.get("sensor", batch.get("model_inputs", batch))
        rows = []
        for row in range(sensor["xyz"].shape[0]):
            translation = head["translation_world"][row]
            rotation = head["rotation_matrix"][row]
            pose = torch.cat((translation, matrix_to_quaternion_xyzw(rotation)), -1)
            width = head["width_m"][row]
            score = torch.sigmoid(head["quality_logit"][row])
            point = head["attention_point_index"][row]
            objects = head["object_logits"][row].argmax(-1)
            selected = self._nms_indices(pose, width, score, objects, amount, global_grasp=True)
            if "valid" in head:
                selected = selected[head["valid"][row, selected]]
            rows.append(
                {
                    "object": objects[selected],
                    "contact_world": translation[selected],
                    "pose_world": pose[selected],
                    "width_m": width[selected],
                    "raw_score": score[selected],
                    "scene_score": score[selected],
                    "point_index": point[selected],
                    "mode_index": selected,
                }
            )
        return rows

    def generate(
        self,
        model: Any,
        batch: dict[str, Tensor],
        output: dict[str, Any],
    ) -> dict[str, Tensor]:
        rows: list[dict[str, Tensor]] = []
        sensor = output.get("sensor", batch.get("model_inputs", batch))
        _, decoded_push_rows = decode_push_candidates(
            sensor,
            output["push_condition"],
            output["push"],
            self.config,
        )
        if not bool(getattr(model, "push_evaluator_ready", False)):
            # Unloaded evaluation weights must never authorize a PUSH selection.
            for decoded in decoded_push_rows:
                decoded["effective_logit"] = torch.full_like(decoded["effective_logit"], float("nan"))
                decoded["effective_probability"] = torch.full_like(decoded["effective_probability"], float("nan"))
        for batch_row in range(sensor["xyz"].shape[0]):
            xyz = sensor["xyz"][batch_row]
            point_mask = sensor["point_mask"][batch_row]
            push_condition = output["push_condition"]
            instance_probability = push_condition.object_probability[batch_row]
            active = push_condition.object_valid[batch_row]
            target_object = int(
                (instance_probability * push_condition.target_probability[batch_row][None])
                .sum(-1)
                .argmax()
            )
            if not bool(push_condition.target_valid[batch_row]):
                rows.append(self._empty_row(xyz.device))
                continue
            type_parts: list[Tensor] = []
            object_parts: list[Tensor] = []
            contact_parts: list[Tensor] = []
            direction_parts: list[Tensor] = []
            point_parts: list[Tensor] = []
            pose_parts: list[Tensor] = []
            destination_parts: list[Tensor] = []
            width_parts: list[Tensor] = []
            score_parts: list[Tensor] = []
            direction_bin_parts: list[Tensor] = []
            direction_score_parts: list[Tensor] = []
            object_score_parts: list[Tensor] = []
            contact_score_parts: list[Tensor] = []
            utility_parts: list[Tensor] = []
            direction_residual_parts: list[Tensor] = []
            effective_logit_parts: list[Tensor] = []
            effective_probability_parts: list[Tensor] = []

            # Terminal task grasp candidates belong to the predicted target query.
            task = output["task_grasp"]
            task_score = task["task_valid_probability"][batch_row]
            task_pose = torch.cat(
                (
                    task["translation_world"][batch_row],
                    matrix_to_quaternion_xyzw(task["rotation_matrix"][batch_row]),
                ),
                -1,
            )
            task_objects = torch.full_like(task_score, target_object, dtype=torch.long)
            task_valid = (
                task["valid"][batch_row]
                if "valid" in task
                else torch.ones_like(task_score, dtype=torch.bool)
            )
            selected = self.select_task_grasp_indices(
                task["translation_world"][batch_row],
                task["rotation_matrix"][batch_row],
                task["width_m"][batch_row],
                task_score,
                task_valid,
            )
            selected = selected[
                task_score[selected] >= self.config.task_grasp_probability_threshold
            ]
            if not bool(active[target_object]):
                selected = selected[:0]
            if len(selected):
                points = task["attention_point_index"][batch_row, selected]
                type_parts.append(torch.full_like(selected, int(ActionType.TASK_GRASP)))
                object_parts.append(task_objects[selected])
                contact_parts.append(
                    torch.full((len(selected), 3), float("nan"), device=xyz.device)
                )
                direction_parts.append(
                    torch.full((len(selected), 3), float("nan"), device=xyz.device)
                )
                pose_parts.append(task_pose[selected])
                destination_parts.append(
                    torch.full((len(selected), 3), float("nan"), device=xyz.device)
                )
                width_parts.append(task["width_m"][batch_row, selected])
                score_parts.append(task_score[selected])
                point_parts.append(points)
                direction_bin_parts.append(torch.full_like(selected, -1))
                direction_score_parts.append(torch.full_like(task_score[selected], float("nan")))
                object_score_parts.append(torch.full_like(task_score[selected], float("nan")))
                contact_score_parts.append(torch.full_like(task_score[selected], float("nan")))
                utility_parts.append(torch.full_like(task_score[selected], float("nan")))
                direction_residual_parts.append(
                    torch.full((len(selected), 2), float("nan"), device=xyz.device)
                )
                effective_logit_parts.append(torch.full_like(task_score[selected], float("nan")))
                effective_probability_parts.append(
                    torch.full_like(task_score[selected], float("nan"))
                )

            # Generic remove grasps are assigned to predicted object queries.
            global_head = output["global_grasp"]
            global_pose = torch.cat(
                (
                    global_head["translation_world"][batch_row],
                    matrix_to_quaternion_xyzw(global_head["rotation_matrix"][batch_row]),
                ),
                -1,
            )
            global_score = torch.sigmoid(global_head["quality_logit"][batch_row])
            global_point = global_head["attention_point_index"][batch_row]
            global_object = global_head["object_logits"][batch_row].argmax(-1)

            remove_domain = active.clone()
            remove_domain[target_object] = False
            target_local = self._target_local_object_mask(
                xyz,
                point_mask,
                instance_probability,
                active,
                target_object,
                self.config.pick_remove_target_margin_m,
            )
            remove_eligible = remove_domain & target_local
            valid_remove = (global_object >= 0) & remove_eligible[
                global_object.clamp(0, len(active) - 1)
            ]
            if "valid" in global_head:
                valid_remove &= global_head["valid"][batch_row]
            valid_remove &= global_score >= self.config.pick_remove_probability_threshold
            candidates = torch.nonzero(valid_remove, as_tuple=False).flatten()
            if len(candidates):
                candidate_score = global_score[candidates]
                local = self._nms_indices(
                    global_pose[candidates],
                    global_head["width_m"][batch_row, candidates],
                    candidate_score,
                    global_object[candidates],
                    self.config.pick_remove_candidates,
                    global_grasp=True,
                )
                selected = candidates[local]
                type_parts.append(torch.full_like(selected, int(ActionType.PICK_REMOVE)))
                object_parts.append(global_object[selected])
                contact_parts.append(
                    torch.full(
                        (len(selected), 3),
                        float("nan"),
                        device=xyz.device,
                    )
                )
                direction_parts.append(
                    torch.full(
                        (len(selected), 3),
                        float("nan"),
                        device=xyz.device,
                    )
                )
                pose_parts.append(global_pose[selected])
                destination_parts.append(
                    torch.full(
                        (len(selected), 3),
                        float("nan"),
                        device=xyz.device,
                    )
                )
                width_parts.append(global_head["width_m"][batch_row, selected])
                score_parts.append(candidate_score[local])
                point_parts.append(global_point[selected])
                direction_bin_parts.append(torch.full_like(selected, -1))
                direction_score_parts.append(torch.full_like(candidate_score[local], float("nan")))
                object_score_parts.append(torch.full_like(candidate_score[local], float("nan")))
                contact_score_parts.append(torch.full_like(candidate_score[local], float("nan")))
                utility_parts.append(torch.full_like(candidate_score[local], float("nan")))
                direction_residual_parts.append(
                    torch.full((len(selected), 2), float("nan"), device=xyz.device)
                )
                effective_logit_parts.append(torch.full_like(candidate_score[local], float("nan")))
                effective_probability_parts.append(
                    torch.full_like(candidate_score[local], float("nan"))
                )

            decoded_push = decoded_push_rows[batch_row]
            if len(decoded_push["point_index"]):
                expanded_point = decoded_push["point_index"]
                type_parts.append(torch.full_like(expanded_point, int(ActionType.PUSH)))
                object_parts.append(decoded_push["object"])
                contact_parts.append(decoded_push["contact_world"])
                direction_parts.append(decoded_push["direction_world"])
                pose_parts.append(
                    torch.full(
                        (len(expanded_point), 7),
                        float("nan"),
                        device=xyz.device,
                    )
                )
                destination_parts.append(
                    torch.full(
                        (len(expanded_point), 3),
                        float("nan"),
                        device=xyz.device,
                    )
                )
                width_parts.append(
                    torch.full(
                        (len(expanded_point),),
                        float("nan"),
                        device=xyz.device,
                    )
                )
                score_parts.append(decoded_push["proposal_score"])
                point_parts.append(expanded_point)
                direction_bin_parts.append(decoded_push["direction_bin"])
                direction_score_parts.append(decoded_push["direction_score"])
                object_score_parts.append(decoded_push["object_score"])
                contact_score_parts.append(decoded_push["contact_score"])
                utility_parts.append(decoded_push["utility"])
                direction_residual_parts.append(decoded_push["direction_residual"])
                effective_logit_parts.append(decoded_push["effective_logit"])
                effective_probability_parts.append(decoded_push["effective_probability"])

            def joined(
                parts: list[Tensor],
                shape: tuple[int, ...],
                dtype: torch.dtype,
                fill: float = float("nan"),
            ) -> Tensor:
                return (
                    torch.cat(parts)
                    if parts
                    else torch.full(shape, fill, dtype=dtype, device=xyz.device)
                )

            rows.append(
                {
                    "type": joined(type_parts, (0,), torch.long, -1),
                    "object": joined(object_parts, (0,), torch.long, -1),
                    "contact_world": joined(contact_parts, (0, 3), xyz.dtype),
                    "direction_world": joined(direction_parts, (0, 3), xyz.dtype),
                    "pose_world": joined(pose_parts, (0, 7), xyz.dtype),
                    "destination_world": joined(destination_parts, (0, 3), xyz.dtype),
                    "width_m": joined(width_parts, (0,), xyz.dtype),
                    "proposal_score": joined(score_parts, (0,), xyz.dtype, -1.0),
                    "point_index": joined(point_parts, (0,), torch.long, -1),
                    "direction_bin": joined(direction_bin_parts, (0,), torch.long, -1),
                    "direction_score": joined(direction_score_parts, (0,), xyz.dtype),
                    "object_score": joined(object_score_parts, (0,), xyz.dtype),
                    "contact_score": joined(contact_score_parts, (0,), xyz.dtype),
                    "utility": joined(utility_parts, (0,), xyz.dtype),
                    "direction_residual": joined(direction_residual_parts, (0, 2), xyz.dtype),
                    "effective_logit": joined(effective_logit_parts, (0,), xyz.dtype),
                    "effective_probability": joined(effective_probability_parts, (0,), xyz.dtype),
                }
            )

        max_candidates = max(1, max(len(row["type"]) for row in rows))
        fill = {
            "type": -1,
            "object": -1,
            "point_index": -1,
            "direction_bin": -1,
            "width_m": float("nan"),
            "direction_score": float("nan"),
            "object_score": float("nan"),
            "contact_score": float("nan"),
            "utility": float("nan"),
            "direction_residual": float("nan"),
            "effective_logit": float("nan"),
            "effective_probability": float("nan"),
            "proposal_score": -1.0,
            "contact_world": float("nan"),
            "direction_world": float("nan"),
            "pose_world": float("nan"),
            "destination_world": float("nan"),
        }
        result: dict[str, Tensor] = {}
        for key in rows[0]:
            value = torch.full(
                (len(rows), max_candidates) + rows[0][key].shape[1:],
                fill[key],
                dtype=rows[0][key].dtype,
                device=rows[0][key].device,
            )
            for row_index, row in enumerate(rows):
                value[row_index, : len(row[key])] = row[key]
            result[key] = value

        result["valid"] = result["type"] >= 0
        result["push_distance_m"] = torch.where(
            result["type"] == int(ActionType.PUSH),
            torch.full_like(result["proposal_score"], PUSH_DISTANCE_M),
            torch.full_like(result["proposal_score"], float("nan")),
        )
        return result
