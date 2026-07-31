"""Decode dense TCD-PRG heads into heterogeneous executable candidates."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from tcd_prg.config import ModelConfig
from tcd_prg.constants import ActionType, PUSH_DISTANCE_M
from tcd_prg.geometry.se3 import matrix_to_quaternion_xyzw
from tcd_prg.geometry.grasp_nms import task_grasp_nms


def _rotation_from_approach(approach: Tensor, rotation_bin: Tensor, bins: int) -> Tensor:
    z = torch.nn.functional.normalize(approach, dim=-1)
    world_z = torch.tensor([0.0, 0.0, 1.0], dtype=z.dtype, device=z.device).expand_as(z)
    world_y = torch.tensor([0.0, 1.0, 0.0], dtype=z.dtype, device=z.device).expand_as(z)
    reference = torch.where((z[..., 2].abs() > 0.9).unsqueeze(-1), world_y, world_z)
    x0 = torch.nn.functional.normalize(torch.cross(reference, z, dim=-1), dim=-1)
    y0 = torch.cross(z, x0, dim=-1)
    angle = (rotation_bin.float() + 0.5) * (2.0 * math.pi / bins) - math.pi
    x = torch.cos(angle).unsqueeze(-1) * x0 + torch.sin(angle).unsqueeze(-1) * y0
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


class DenseCandidateGenerator:
    """Generate task grasp, global-grasp-based removal, and push candidates."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @staticmethod
    def _top_per_object(score: Tensor, instance_id: Tensor, object_ids: Tensor,
                        total: int) -> Tensor:
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

    def _with_graph_fallback(self, eligible: Tensor, domain: Tensor, score: Tensor) -> Tensor:
        """Keep graph actionability hard, plus a bounded recovery frontier.

        Published labels omit some task edges even for successful sequences.
        One high-scoring non-graph object is therefore retained so a missing
        predicted edge cannot make the closed-loop policy irrecoverable.
        """

        amount = self.config.graph_candidate_fallback_objects
        fallback_domain = domain & ~eligible
        if amount and fallback_domain.any():
            indices = torch.nonzero(fallback_domain, as_tuple=False).flatten()
            selected = indices[score[indices].topk(min(amount, len(indices))).indices]
            eligible = eligible.clone()
            eligible[selected] = True
        return eligible

    def global_predictions(
        self, batch: dict[str, Tensor], output: dict[str, Any], topk: int | None = None,
        score_kind: str = "raw",
    ) -> list[dict[str, Tensor]]:
        """Decode task-free grasps for every physically present visible object.

        This path deliberately does not inspect target, actionable, graph, task,
        or router tensors. It is therefore suitable for standalone comparison.
        """

        head = output["global_grasp"]
        rows: list[dict[str, Tensor]] = []
        if score_kind not in {"raw", "scene"}:
            raise ValueError("score_kind must be raw or scene")
        amount = int(topk or self.config.candidate_topk)
        for row in range(batch["xyz"].shape[0]):
            xyz = batch["xyz"][row]
            instance = batch["instance_id"][row]
            present = batch.get("object_present", batch["object_mask"])[row]
            valid_instance = (instance >= 0) & (instance < present.shape[0])
            if self.config.global_grasp_input_mode == "scene_only":
                domain = batch["point_mask"][row]
            else:
                domain = (
                    batch["point_mask"][row] & valid_instance
                    & present[instance.clamp(0, present.shape[0] - 1)]
                )
            raw_score = (
                torch.sigmoid(head["contact_logits"][row, :, None])
                * torch.sigmoid(head["intrinsic_confidence_logit"][row])
            ).masked_fill(~domain[:, None], -1.0)
            scene_score = raw_score * torch.sigmoid(head["scene_confidence_logit"][row])
            ranking_score = raw_score if score_kind == "raw" else scene_score
            count = min(amount * 8, int(domain.sum()) * ranking_score.shape[1])
            flat = ranking_score.flatten().topk(count).indices if count else torch.empty(
                0, dtype=torch.long, device=xyz.device
            )
            point = torch.div(flat, ranking_score.shape[1], rounding_mode="floor")
            mode = flat.remainder(ranking_score.shape[1])
            rotation_bin = head["rotation_logits"][row, point, mode].argmax(-1)
            rotation = _rotation_from_approach(
                head["approach_direction"][row, point, mode], rotation_bin,
                self.config.num_grasp_rotation_bins,
            )
            pose_world = torch.cat((
                xyz[point] + head["center_offset_m"][row, point, mode],
                matrix_to_quaternion_xyzw(rotation),
            ), -1)
            object_index = instance[point]
            # Strict scene-only comparison may annotate selected grasps after
            # decoding, but instance IDs must not influence ranking or NMS.
            nms_object_index = (
                torch.zeros_like(object_index)
                if self.config.global_grasp_input_mode == "scene_only"
                else object_index
            )
            width_m = head["width_m"][row, point, mode]
            ranked_score = ranking_score[point, mode]
            if len(point):
                keep = task_grasp_nms(
                    pose_world[None], width_m[None], ranked_score[None], nms_object_index[None],
                    torch.ones((1, len(point)), dtype=torch.bool, device=point.device),
                    translation_threshold_m=self.config.global_grasp_nms_translation_m,
                    rotation_threshold_deg=self.config.global_grasp_nms_rotation_deg,
                    width_threshold_m=self.config.global_grasp_nms_width_m,
                    approach_threshold_deg=self.config.global_grasp_nms_approach_deg,
                )[0]
                if self.config.global_grasp_input_mode != "scene_only":
                    keep |= object_index < 0
                selected = torch.nonzero(keep, as_tuple=False).flatten()
                selected = selected[
                    ranked_score[selected].argsort(descending=True, stable=True)
                ][:amount]
                point, mode, rotation = point[selected], mode[selected], rotation[selected]
                pose_world, object_index, width_m = (
                    pose_world[selected], object_index[selected], width_m[selected]
                )
            rows.append({
                "object": object_index,
                "contact_world": xyz[point],
                "pose_world": pose_world,
                "width_m": width_m,
                "raw_score": raw_score[point, mode],
                "scene_score": scene_score[point, mode],
                "intrinsic_score": torch.sigmoid(
                    head["intrinsic_confidence_logit"][row, point, mode]
                ),
                "point_index": point,
                "mode_index": mode,
            })
        return rows

    def generate(self, model: Any, batch: dict[str, Tensor], output: dict[str, Any]) -> dict[str, Tensor]:
        """Return padded tensors and learned candidate tokens for one encoded batch."""

        rows: list[dict[str, Tensor]] = []
        encoded = output["encoded"]
        for batch_row in range(batch["xyz"].shape[0]):
            xyz, instance = batch["xyz"][batch_row], batch["instance_id"][batch_row]
            active = batch["object_mask"][batch_row] & batch["object_active"][batch_row]
            if output["graph"] is not None:
                actionable = output["graph"].derived_actionable_mask[batch_row]
            else:
                # No-graph ablation intentionally falls back to all active
                # objects; the full method always uses the derived hard mask.
                actionable = active
            target_object = int(batch["target_object"][batch_row])
            type_parts, object_parts, contact_parts, direction_parts, point_index_parts = [], [], [], [], []
            pose_parts, destination_parts, width_parts, score_parts, approach_parts = [], [], [], [], []

            task_head = output["task_grasp"]
            task_score = (
                torch.sigmoid(task_head["contact_logits"][batch_row])
                * torch.sigmoid(task_head["proposal_confidence_logit"][batch_row])
                * torch.sigmoid(task_head["task_compatibility_logit"][batch_row])
            ).masked_fill(~batch["target_mask"][batch_row], -1.0)
            task_count = min(self.config.task_grasp_candidates, int(batch["target_mask"][batch_row].sum()))
            if task_count:
                index = task_score.topk(task_count).indices
                rotation_bin = task_head["rotation_logits"][batch_row, index].argmax(-1)
                rotation = _rotation_from_approach(
                    task_head["approach_direction"][batch_row, index], rotation_bin,
                    self.config.num_grasp_rotation_bins,
                )
                pose = torch.cat((
                    xyz[index] + task_head["center_offset_m"][batch_row, index],
                    matrix_to_quaternion_xyzw(rotation),
                ), -1)
                type_parts.append(torch.full_like(index, int(ActionType.TASK_GRASP)))
                object_parts.append(torch.full_like(index, target_object))
                contact_parts.append(torch.full((len(index), 3), float("nan"), device=xyz.device))
                direction_parts.append(torch.full((len(index), 3), float("nan"), device=xyz.device))
                pose_parts.append(pose)
                destination_parts.append(torch.full((len(index), 3), float("nan"), device=xyz.device))
                width_parts.append(task_head["width_m"][batch_row, index])
                score_parts.append(task_score[index])
                approach_parts.append(torch.full_like(index, -1))
                point_index_parts.append(index)

            global_grasp = output["global_grasp"]
            global_raw_score = (
                torch.sigmoid(global_grasp["contact_logits"][batch_row, :, None])
                * torch.sigmoid(global_grasp["intrinsic_confidence_logit"][batch_row])
            )
            global_score = global_raw_score * torch.sigmoid(
                global_grasp["scene_confidence_logit"][batch_row]
            )
            remove_domain = active.clone()
            remove_domain[target_object] = False
            remove_eligible = active & actionable & remove_domain
            if output["graph"] is not None:
                remove_eligible = self._with_graph_fallback(
                    remove_eligible, remove_domain,
                    output["pick_remove"]["object_logits"][batch_row],
                )
            remove_objects = torch.nonzero(remove_eligible, as_tuple=False).flatten()
            if output["graph"] is not None and len(remove_objects):
                priority = output["graph"].actionable_blocker_logits[batch_row, remove_objects]
                remove_objects = remove_objects[priority.argsort(descending=True)]
            # Select point-mode pairs only after the dependency graph has
            # restricted objects; the global head itself predicts all present objects.
            flat_score = global_score.flatten()
            flat_instance = instance[:, None].expand(-1, global_score.shape[1]).reshape(-1)
            remove_flat = self._top_per_object(
                flat_score, flat_instance, remove_objects[:4], self.config.pick_remove_candidates
            )
            if len(remove_flat):
                remove_index = torch.div(remove_flat, global_score.shape[1], rounding_mode="floor")
                remove_mode = remove_flat.remainder(global_score.shape[1])
                rotation_bin = global_grasp["rotation_logits"][
                    batch_row, remove_index, remove_mode
                ].argmax(-1)
                rotation = _rotation_from_approach(
                    global_grasp["approach_direction"][batch_row, remove_index, remove_mode], rotation_bin,
                    self.config.num_grasp_rotation_bins,
                )
                pose = torch.cat((
                    xyz[remove_index] + global_grasp["center_offset_m"][
                        batch_row, remove_index, remove_mode
                    ],
                    matrix_to_quaternion_xyzw(rotation),
                ), -1)
                type_parts.append(torch.full_like(remove_index, int(ActionType.PICK_REMOVE)))
                object_parts.append(instance[remove_index])
                contact_parts.append(torch.full((len(remove_index), 3), float("nan"), device=xyz.device))
                direction_parts.append(torch.full((len(remove_index), 3), float("nan"), device=xyz.device))
                pose_parts.append(pose)
                destination_parts.append(torch.full((len(remove_index), 3), float("nan"), device=xyz.device))
                width_parts.append(global_grasp["width_m"][batch_row, remove_index, remove_mode])
                score_parts.append(global_score[remove_index, remove_mode])
                approach_parts.append(torch.full_like(remove_index, -1))
                point_index_parts.append(remove_index)

            push = output["push"]
            # Target self-push is an explicit recovery primitive in the current
            # dataset, including successful actions whose target is not on the
            # graph frontier. The configurable override records that semantics
            # directly instead of implying graph gating.
            push_eligible = active & actionable
            if output["graph"] is not None:
                if self.config.allow_target_push_recovery:
                    push_eligible[target_object] = active[target_object]
                push_eligible = self._with_graph_fallback(
                    push_eligible, active, push["object_logits"][batch_row]
                )
            push_objects = torch.nonzero(push_eligible, as_tuple=False).flatten()
            if len(push_objects):
                push_objects = push_objects[
                    push["object_logits"][batch_row, push_objects].argsort(descending=True)
                ][:4]
            push_index = self._top_per_object(
                torch.sigmoid(push["contact_logits"][batch_row]), instance,
                push_objects, self.config.push_candidates,
            )
            if len(push_index):
                direction_bin = push["direction_logits"][batch_row, push_index].argmax(-1)
                angle = (direction_bin.float() + 0.5) * 2.0 * math.pi / self.config.num_direction_bins
                center = torch.stack((torch.cos(angle), torch.sin(angle)), -1)
                planar = torch.nn.functional.normalize(
                    center + push["direction_residual"][batch_row, push_index], dim=-1
                )
                direction = torch.cat((planar, torch.zeros(len(planar), 1, device=xyz.device)), -1)
                type_parts.append(torch.full_like(push_index, int(ActionType.PUSH)))
                object_parts.append(instance[push_index])
                contact_parts.append(xyz[push_index])
                direction_parts.append(direction)
                pose_parts.append(torch.full((len(push_index), 7), float("nan"), device=xyz.device))
                destination_parts.append(torch.full((len(push_index), 3), float("nan"), device=xyz.device))
                width_parts.append(torch.full((len(push_index),), float("nan"), device=xyz.device))
                score_parts.append(torch.sigmoid(push["contact_logits"][batch_row, push_index]))
                approach_parts.append(torch.full_like(push_index, -1))
                point_index_parts.append(push_index)

            if type_parts:
                row = {
                    "type": torch.cat(type_parts), "object": torch.cat(object_parts),
                    "contact_world": torch.cat(contact_parts),
                    "direction_world": torch.cat(direction_parts),
                    "pose_world": torch.cat(pose_parts),
                    "destination_world": torch.cat(destination_parts),
                    "width_m": torch.cat(width_parts),
                    "proposal_score": torch.cat(score_parts),
                    "push_approach_mode": torch.cat(approach_parts),
                    "point_index": torch.cat(point_index_parts),
                }
            else:
                # Keep a single padded slot so downstream Transformer modules do
                # not receive a zero-length sequence.  Its valid mask remains false.
                row = {
                    "type": torch.empty(0, dtype=torch.long, device=xyz.device),
                    "object": torch.empty(0, dtype=torch.long, device=xyz.device),
                    "contact_world": xyz.new_empty((0, 3)),
                    "direction_world": xyz.new_empty((0, 3)),
                    "pose_world": xyz.new_empty((0, 7)),
                    "destination_world": xyz.new_empty((0, 3)),
                    "width_m": xyz.new_empty((0,)),
                    "proposal_score": xyz.new_empty((0,)),
                    "push_approach_mode": torch.empty(0, dtype=torch.long, device=xyz.device),
                    "point_index": torch.empty(0, dtype=torch.long, device=xyz.device),
                }
            rows.append(row)
        max_candidates = max(1, max(len(row["type"]) for row in rows))
        result: dict[str, Tensor] = {}
        fill = {"type": -1, "object": -1, "push_approach_mode": -1, "point_index": -1,
                "width_m": float("nan"), "proposal_score": -1.0,
                "contact_world": float("nan"), "direction_world": float("nan"),
                "pose_world": float("nan"), "destination_world": float("nan")}
        for key in rows[0]:
            tail = rows[0][key].shape[1:]
            value = torch.full((len(rows), max_candidates) + tail, fill[key],
                               dtype=rows[0][key].dtype, device=rows[0][key].device)
            for row_index, row in enumerate(rows):
                value[row_index, : len(row[key])] = row[key]
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
            encoded.object_tokens, result["type"], result["object"],
            result["contact_world"], result["direction_world"], result["pose_world"],
            result["destination_world"], flags, encoded.task_token,
        )
        evidence = torch.zeros(
            result["type"].shape + (22,), device=result["type"].device
        )
        for row in range(result["type"].shape[0]):
            valid = torch.nonzero(result["valid"][row], as_tuple=False).flatten()
            for candidate in valid.tolist():
                point = int(result["point_index"][row, candidate])
                kind = int(result["type"][row, candidate])
                evidence[row, candidate, 15] = result["proposal_score"][row, candidate]
                if kind == int(ActionType.PUSH):
                    evidence[row, candidate, 7:12] = output["push"]["potential_delta"][row, point]
                    evidence[row, candidate, 12:15] = torch.sigmoid(
                        output["push"]["risk_logits"][row, point]
                    )
        result["evidence"] = evidence
        return result

    @staticmethod
    def verifier_batch(base_batch: dict[str, Tensor], candidates: dict[str, Tensor]) -> dict[str, Any]:
        """Create the minimal CPU batch contract consumed by verifier preprocessing."""

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
