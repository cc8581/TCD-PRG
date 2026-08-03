"""Filter complete grasp sets and decode heterogeneous executable candidates."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType, PUSH_DISTANCE_M
from tcd_prg.geometry.grasp_nms import task_grasp_nms
from tcd_prg.geometry.se3 import matrix_to_quaternion_xyzw


class DenseCandidateGenerator:
    """Apply quality filtering, Top-K and SE(3) NMS to complete grasps."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @staticmethod
    def _top_per_object(score: Tensor, instance_id: Tensor, object_ids: Tensor, total: int) -> Tensor:
        if not len(object_ids) or total <= 0:
            return torch.empty(0, dtype=torch.long, device=score.device)
        per_object = max(1, math.ceil(total / len(object_ids)))
        selected = []
        for object_id in object_ids.tolist():
            points = torch.nonzero(instance_id == object_id, as_tuple=False).flatten()
            if len(points):
                selected.append(points[score[points].topk(min(per_object, len(points))).indices])
        if not selected:
            return torch.empty(0, dtype=torch.long, device=score.device)
        candidates = torch.cat(selected)
        return candidates[score[candidates].topk(min(total, len(candidates))).indices]

    @staticmethod
    def _nearest_scene_point(
        translation: Tensor, xyz: Tensor, point_mask: Tensor
    ) -> Tensor:
        points = torch.nonzero(point_mask, as_tuple=False).flatten()
        if not len(points):
            return torch.zeros(len(translation), dtype=torch.long, device=translation.device)
        distance = torch.cdist(translation, xyz[points])
        return points[distance.argmin(-1)]

    def _with_graph_fallback(self, eligible: Tensor, domain: Tensor, score: Tensor) -> Tensor:
        amount = self.config.graph_candidate_fallback_objects
        fallback_domain = domain & ~eligible
        if amount and fallback_domain.any():
            indices = torch.nonzero(fallback_domain, as_tuple=False).flatten()
            selected = indices[score[indices].topk(min(amount, len(indices))).indices]
            eligible = eligible.clone()
            eligible[selected] = True
        return eligible

    def _nms_indices(
        self, pose: Tensor, width: Tensor, score: Tensor, objects: Tensor,
        amount: int, *, global_grasp: bool,
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
            pose[None], width[None], score[None], objects[None],
            torch.ones((1, len(score)), dtype=torch.bool, device=score.device),
            translation_threshold_m=thresholds[0], rotation_threshold_deg=thresholds[1],
            width_threshold_m=thresholds[2], approach_threshold_deg=thresholds[3],
        )[0]
        selected = torch.nonzero(keep, as_tuple=False).flatten()
        return selected[score[selected].argsort(descending=True, stable=True)[:amount]]

    def apply_push_nms(
        self, candidates: dict[str, Tensor], router_logits: Tensor
    ) -> Tensor:
        """按 Router 分数保留同物体上接触点/方向相近的唯一 PUSH 动作。"""

        keep = candidates["valid"].clone()
        cosine_threshold = math.cos(math.radians(self.config.push_nms_direction_deg))
        for row in range(keep.shape[0]):
            push = torch.nonzero(
                keep[row] & (candidates["type"][row] == int(ActionType.PUSH)),
                as_tuple=False,
            ).flatten()
            ordered = push[router_logits[row, push].argsort(descending=True, stable=True)]
            accepted: list[int] = []
            for index in ordered.tolist():
                duplicate = False
                for prior in accepted:
                    same_object = bool(
                        candidates["object"][row, index] == candidates["object"][row, prior]
                    )
                    contact_distance = torch.linalg.vector_norm(
                        candidates["contact_world"][row, index]
                        - candidates["contact_world"][row, prior]
                    )
                    first = torch.nn.functional.normalize(
                        candidates["direction_world"][row, index, :2], dim=-1
                    )
                    second = torch.nn.functional.normalize(
                        candidates["direction_world"][row, prior, :2], dim=-1
                    )
                    similar_direction = bool((first * second).sum() >= cosine_threshold)
                    if (
                        same_object
                        and bool(contact_distance < self.config.push_nms_contact_m)
                        and similar_direction
                    ):
                        duplicate = True
                        break
                if duplicate:
                    keep[row, index] = False
                else:
                    accepted.append(index)
        return keep

    def global_predictions(
        self, batch: dict[str, Tensor], output: dict[str, Any], topk: int | None = None,
        score_kind: str = "scene",
    ) -> list[dict[str, Tensor]]:
        """Return task-free complete grasp queries without task-dependent ranking."""

        if score_kind not in {"raw", "scene"}:
            raise ValueError("score_kind must be scene (raw is a compatibility alias)")
        del score_kind
        head = output["global_grasp"]
        amount = int(topk or self.config.candidate_topk)
        rows = []
        for row in range(batch["xyz"].shape[0]):
            translation = head["translation_world"][row]
            rotation = head["rotation_matrix"][row]
            pose = torch.cat((translation, matrix_to_quaternion_xyzw(rotation)), -1)
            width = head["width_m"][row]
            score = torch.sigmoid(head["quality_logit"][row])
            point = head["attention_point_index"][row]
            objects = head["object_logits"][row].argmax(-1)
            selected = self._nms_indices(pose, width, score, objects, amount, global_grasp=True)
            rows.append({
                "object": objects[selected],
                "contact_world": translation[selected],
                "pose_world": pose[selected],
                "width_m": width[selected],
                "raw_score": score[selected],
                "scene_score": score[selected],
                "point_index": point[selected],
                "mode_index": selected,
            })
        return rows

    def generate(self, model: Any, batch: dict[str, Tensor], output: dict[str, Any]) -> dict[str, Tensor]:
        rows: list[dict[str, Tensor]] = []
        encoded = output["encoded"]
        for batch_row in range(batch["xyz"].shape[0]):
            xyz, instance = batch["xyz"][batch_row], batch["instance_id"][batch_row]
            active = batch["object_mask"][batch_row] & batch["object_active"][batch_row]
            graph_output = output["graph"]
            if graph_output is None or self.config.graph_candidate_mode == "none":
                actionable = active
                graph_prior = active.to(xyz.dtype)
            elif self.config.graph_candidate_mode == "hard":
                actionable = graph_output.derived_actionable_mask[batch_row]
                graph_prior = actionable.to(xyz.dtype)
            else:
                # soft 图只调整候选证据，不因单条边预测错误而提前删除可行动物体。
                actionable = active
                graph_prior = getattr(
                    graph_output, "dependency_prior",
                    graph_output.derived_actionable_mask.to(xyz.dtype),
                )[batch_row]
            target_object = int(batch["target_object"][batch_row])
            type_parts, object_parts, contact_parts, direction_parts, point_parts = [], [], [], [], []
            pose_parts, destination_parts, width_parts, score_parts = [], [], [], []
            direction_bin_parts, direction_score_parts = [], []

            task = output["task_grasp"]
            task_pose = torch.cat((
                task["translation_world"][batch_row],
                matrix_to_quaternion_xyzw(task["rotation_matrix"][batch_row]),
            ), -1)
            task_score = torch.sigmoid(task["quality_logit"][batch_row])
            task_objects = torch.full_like(task_score, target_object, dtype=torch.long)
            selected = self._nms_indices(
                task_pose, task["width_m"][batch_row], task_score, task_objects,
                self.config.task_grasp_candidates, global_grasp=False,
            )
            if not bool(active[target_object] & batch["target_mask"][batch_row].any()):
                selected = selected[:0]
            if len(selected):
                points = task["attention_point_index"][batch_row, selected]
                type_parts.append(torch.full_like(selected, int(ActionType.TASK_GRASP)))
                object_parts.append(task_objects[selected])
                contact_parts.append(torch.full((len(selected), 3), float("nan"), device=xyz.device))
                direction_parts.append(torch.full((len(selected), 3), float("nan"), device=xyz.device))
                pose_parts.append(task_pose[selected])
                destination_parts.append(torch.full((len(selected), 3), float("nan"), device=xyz.device))
                width_parts.append(task["width_m"][batch_row, selected])
                score_parts.append(task_score[selected])
                point_parts.append(points)
                direction_bin_parts.append(torch.full_like(selected, -1))
                direction_score_parts.append(torch.full_like(task_score[selected], float("nan")))

            global_head = output["global_grasp"]
            global_pose = torch.cat((
                global_head["translation_world"][batch_row],
                matrix_to_quaternion_xyzw(global_head["rotation_matrix"][batch_row]),
            ), -1)
            global_score = torch.sigmoid(global_head["quality_logit"][batch_row])
            global_point = global_head["attention_point_index"][batch_row]
            global_object = global_head["object_logits"][batch_row].argmax(-1)
            remove_domain = active.clone()
            remove_domain[target_object] = False
            remove_eligible = active & actionable & remove_domain
            per_object_score = global_score.new_full(active.shape, -1.0)
            for object_index in torch.nonzero(active, as_tuple=False).flatten().tolist():
                mask = global_object == object_index
                if mask.any():
                    per_object_score[object_index] = global_score[mask].max()
            if graph_output is not None and self.config.graph_candidate_mode == "hard":
                remove_eligible = self._with_graph_fallback(remove_eligible, remove_domain, per_object_score)
            valid_remove = remove_eligible[global_object.clamp(0, len(active) - 1)] & (global_object >= 0)
            candidates = torch.nonzero(valid_remove, as_tuple=False).flatten()
            if len(candidates):
                candidate_score = global_score[candidates]
                if self.config.graph_candidate_mode == "soft":
                    candidate_score = candidate_score * (
                        0.25 + 0.75 * graph_prior[global_object[candidates]]
                    )
                local = self._nms_indices(
                    global_pose[candidates], global_head["width_m"][batch_row, candidates],
                    candidate_score, global_object[candidates],
                    self.config.pick_remove_candidates, global_grasp=True,
                )
                selected = candidates[local]
                type_parts.append(torch.full_like(selected, int(ActionType.PICK_REMOVE)))
                object_parts.append(global_object[selected])
                contact_parts.append(torch.full((len(selected), 3), float("nan"), device=xyz.device))
                direction_parts.append(torch.full((len(selected), 3), float("nan"), device=xyz.device))
                pose_parts.append(global_pose[selected])
                destination_parts.append(torch.full((len(selected), 3), float("nan"), device=xyz.device))
                width_parts.append(global_head["width_m"][batch_row, selected])
                score_parts.append(global_score[selected])
                point_parts.append(global_point[selected])
                direction_bin_parts.append(torch.full_like(selected, -1))
                direction_score_parts.append(torch.full_like(global_score[selected], float("nan")))

            push = output["push"]
            push_eligible = active & actionable
            if graph_output is not None and self.config.graph_candidate_mode == "hard":
                if self.config.allow_target_push_recovery:
                    push_eligible[target_object] = active[target_object]
                push_eligible = self._with_graph_fallback(push_eligible, active, push["object_logits"][batch_row])
            push_objects = torch.nonzero(push_eligible, as_tuple=False).flatten()
            if len(push_objects):
                object_score = torch.sigmoid(push["object_logits"][batch_row, push_objects])
                if self.config.graph_candidate_mode == "soft":
                    object_score = object_score * (0.25 + 0.75 * graph_prior[push_objects])
                push_objects = push_objects[object_score.argsort(descending=True)[
                    :self.config.graph_candidate_topk_objects
                ]]
            push_index = self._top_per_object(
                torch.sigmoid(push["contact_logits"][batch_row]), instance,
                push_objects, self.config.push_candidates,
            )
            if len(push_index):
                direction_probability = torch.softmax(
                    push["direction_logits"][batch_row, push_index], dim=-1
                )
                # 每个接触点展开 Top-M 方向，让 utility 和 Router 决定最终动作，
                # 而不是由 direction head 的 argmax 在候选生成阶段一票否决。
                directions_per_contact = min(
                    self.config.push_directions_per_contact, direction_probability.shape[-1]
                )
                direction_score, direction_bin = direction_probability.topk(
                    directions_per_contact, dim=-1
                )
                expanded_point = push_index[:, None].expand(-1, directions_per_contact).reshape(-1)
                direction_bin = direction_bin.reshape(-1)
                direction_score = direction_score.reshape(-1)
                contact_score = torch.sigmoid(
                    push["contact_logits"][batch_row, expanded_point]
                )
                if len(expanded_point) > self.config.max_push_candidates:
                    keep = (contact_score * direction_score).topk(
                        self.config.max_push_candidates
                    ).indices
                    expanded_point = expanded_point[keep]
                    direction_bin = direction_bin[keep]
                    direction_score = direction_score[keep]
                    contact_score = contact_score[keep]
                angle = (direction_bin.float() + 0.5) * 2.0 * math.pi / self.config.num_direction_bins
                center = torch.stack((torch.cos(angle), torch.sin(angle)), -1)
                residual = push["direction_residual"][
                    batch_row, expanded_point, direction_bin
                ]
                planar = torch.nn.functional.normalize(center + residual, dim=-1)
                direction = torch.cat((planar, torch.zeros(len(planar), 1, device=xyz.device)), -1)
                type_parts.append(torch.full_like(expanded_point, int(ActionType.PUSH)))
                object_parts.append(instance[expanded_point])
                contact_parts.append(xyz[expanded_point])
                direction_parts.append(direction)
                pose_parts.append(torch.full((len(expanded_point), 7), float("nan"), device=xyz.device))
                destination_parts.append(torch.full((len(expanded_point), 3), float("nan"), device=xyz.device))
                width_parts.append(torch.full((len(expanded_point),), float("nan"), device=xyz.device))
                score_parts.append(contact_score)
                point_parts.append(expanded_point)
                direction_bin_parts.append(direction_bin)
                direction_score_parts.append(direction_score)

            def joined(parts: list[Tensor], shape: tuple[int, ...], dtype: torch.dtype, fill: float = float("nan")) -> Tensor:
                return torch.cat(parts) if parts else torch.full(shape, fill, dtype=dtype, device=xyz.device)

            rows.append({
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
            })

        max_candidates = max(1, max(len(row["type"]) for row in rows))
        fill = {"type": -1, "object": -1, "point_index": -1, "direction_bin": -1,
                "width_m": float("nan"), "direction_score": float("nan"),
                "proposal_score": -1.0, "contact_world": float("nan"),
                "direction_world": float("nan"), "pose_world": float("nan"),
                "destination_world": float("nan")}
        result: dict[str, Tensor] = {}
        for key in rows[0]:
            value = torch.full(
                (len(rows), max_candidates) + rows[0][key].shape[1:], fill[key],
                dtype=rows[0][key].dtype, device=rows[0][key].device,
            )
            for row_index, row in enumerate(rows):
                value[row_index, :len(row[key])] = row[key]
            result[key] = value
        result["valid"] = result["type"] >= 0
        result["push_distance_m"] = torch.where(
            result["type"] == int(ActionType.PUSH),
            torch.full_like(result["proposal_score"], PUSH_DISTANCE_M),
            torch.full_like(result["proposal_score"], float("nan")),
        )
        flags = torch.stack((
            torch.isfinite(result["contact_world"]).all(-1),
            torch.isfinite(result["direction_world"]).all(-1),
            torch.isfinite(result["pose_world"]).all(-1),
            torch.isfinite(result["destination_world"]).all(-1),
            torch.isfinite(result["width_m"]),
        ), -1)
        result["tokens"] = model.candidate_encoder(
            encoded.object_tokens, result["type"], result["object"], result["contact_world"],
            result["direction_world"], result["pose_world"], result["destination_world"],
            flags, encoded.task_token,
        )
        evidence = torch.zeros(result["type"].shape + (7,), device=result["type"].device)
        evidence[..., 1] = torch.where(result["valid"], result["proposal_score"], 0.0)
        for row in range(result["type"].shape[0]):
            push_candidates = torch.nonzero(
                result["valid"][row] & (result["type"][row] == int(ActionType.PUSH)),
                as_tuple=False,
            ).flatten()
            if len(push_candidates):
                points = result["point_index"][row, push_candidates]
                direction_bin = result["direction_bin"][row, push_candidates]
                evidence[row, push_candidates, 0] = output["push"]["utility_delta"][
                    row, points, direction_bin
                ]
                evidence[row, push_candidates, 3] = result["direction_score"][
                    row, push_candidates
                ]
        if output["graph"] is not None and self.config.graph_candidate_mode == "soft":
            for row in range(result["type"].shape[0]):
                prior = getattr(
                    output["graph"], "dependency_prior",
                    output["graph"].derived_actionable_mask.to(evidence.dtype),
                )[row]
                safe_object = result["object"][row].clamp(0, len(prior) - 1)
                evidence[row, :, 4] = torch.where(
                    result["valid"][row], prior[safe_object], 0.0
                )
        result["evidence"] = evidence
        return result

    @staticmethod
    def verifier_batch(base_batch: dict[str, Tensor], candidates: dict[str, Tensor]) -> dict[str, Any]:
        kind = candidates["type"].cpu()
        pose = candidates["pose_world"].cpu()
        nan_pose = torch.full_like(pose, float("nan"))
        return {
            "xyz": base_batch["xyz"].cpu(), "point_mask": base_batch["point_mask"].cpu(),
            "instance_id": base_batch["instance_id"].cpu(),
            "candidate_mask": candidates["valid"].cpu(), "action_type": kind,
            "action_parameters": {
                "removal_grasp_pose_world": torch.where(
                    (kind == int(ActionType.PICK_REMOVE)).unsqueeze(-1), pose, nan_pose
                ),
                "task_grasp_pose_world": torch.where(
                    (kind == int(ActionType.TASK_GRASP)).unsqueeze(-1), pose, nan_pose
                ),
                "grasp_width_m": candidates["width_m"].cpu(),
            },
        }
