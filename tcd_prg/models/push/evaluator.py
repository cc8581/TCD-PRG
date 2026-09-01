"""Instance-aware finite-horizon PUSH value evaluator."""

import torch
from torch import nn

from .actions import PushActions
from .pointnet2 import PushPointNet2


class PushEffectivenessEvaluator(nn.Module):
    """Predict task value, safety and physical effects for complete PUSH actions."""

    horizons = 5

    def __init__(self, feature_dim=256, num_categories=64, num_task_regions=64, *, initialize_backbone=True):
        super().__init__()
        d = feature_dim
        self.point_encoder = nn.Sequential(nn.Linear(d + 3, d), nn.LayerNorm(d), nn.GELU())
        self.category = nn.Embedding(num_categories, d)
        self.region = nn.Embedding(num_task_regions, d)
        self.instance_relation = nn.Sequential(
            nn.Linear(d + 12, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
        )
        self.relation_attention = nn.Sequential(
            nn.Linear(d + 12, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.trunk = nn.Sequential(
            nn.Linear(7 * d + 10, 2 * d), nn.GELU(), nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d), nn.GELU(),
        )
        self.q_head = nn.Linear(d, self.horizons)
        with torch.no_grad():
            self.q_head.bias[1:].fill_(-4.0)
        self.safety_head = nn.Linear(d, 1)
        self.auxiliary_delta_head = nn.Linear(d, 5)
        self.feature_dim = d
        self.backbone = None
        if initialize_backbone:
            self.initialize_backbone()

    def initialize_backbone(self):
        if self.backbone is None:
            self.backbone = PushPointNet2(self.feature_dim).to(self.trunk[0].weight.device)
            self.backbone.train(self.training)

    @staticmethod
    def _pool(value, weight):
        return (value * weight[:, None]).sum(0) / weight.sum().clamp_min(1e-6)

    def _instance_context(
        self, points, encoded, object_probability, object_valid, contact, direction,
        distance, acted_object, target_center,
    ):
        tokens, scores = [], []
        for slot in range(object_probability.shape[0]):
            if not bool(object_valid[slot]):
                continue
            weight = object_probability[slot]
            if not bool(weight.sum() > 0):
                continue
            center = self._pool(points, weight)
            extent = self._pool((points - center).square(), weight).sqrt()
            relative = center - contact
            along = torch.minimum((relative * direction).sum().clamp_min(0), distance)
            path_offset = torch.linalg.vector_norm(relative - along * direction)
            geometry = torch.cat((
                center - target_center, relative, extent, along.reshape(1),
                path_offset.reshape(1),
                points.new_tensor([
                    float(slot == int(acted_object)),
                ]),
            ))
            relation_input = torch.cat((self._pool(encoded, weight), geometry))
            tokens.append(self.instance_relation(relation_input))
            scores.append(self.relation_attention(relation_input).squeeze(-1))
        if not tokens:
            return encoded.new_zeros(self.feature_dim)
        attention = torch.softmax(torch.stack(scores), 0)
        return (torch.stack(tokens) * attention[:, None]).sum(0)

    @staticmethod
    def _monotonic_q(raw):
        current = torch.sigmoid(raw[:, :1])
        values = [current]
        for index in range(1, raw.shape[1]):
            current = current + (1.0 - current) * torch.sigmoid(raw[:, index:index + 1])
            values.append(current)
        return torch.cat(values, -1)

    def forward(self, sensor, condition, actions: PushActions):
        xyz = sensor["xyz"]
        condition.validate(xyz.shape[1])
        actions.validate(len(xyz), condition.object_valid.shape[1])
        if not len(actions.batch_index):
            empty = self.q_head.weight.sum().expand(0)
            return {
                "q_value": empty.reshape(0, self.horizons), "safety_logit": empty,
                "safety_probability": empty, "potential_delta": empty.reshape(0, 5),
                "effective_logit": empty, "effective_probability": empty,
            }
        if not bool(condition.target_valid[actions.batch_index].all()):
            raise ValueError("PUSH requires a visible target")
        if not bool(condition.object_valid[actions.batch_index, actions.object].all()):
            raise ValueError("PUSH object is not represented")
        self.initialize_backbone()
        mask = sensor["point_mask"].bool()
        groups = {}
        for b in torch.unique(actions.batch_index).tolist():
            valid = mask[b]
            points = xyz[b, valid]
            if not len(points):
                raise ValueError("PUSH action scene requires visible points")
            groups.setdefault(len(points), []).append((b, points, sensor["rgb"][b, valid]))
        context = {}
        for scenes in groups.values():
            features = self.backbone(
                torch.stack([item[1] for item in scenes]),
                torch.stack([item[2] for item in scenes]),
            )
            for (b, points, _), scene_features in zip(scenes, features):
                target_weight = condition.target_probability[b, mask[b]]
                target_center = self._pool(points, target_weight)
                encoded = self.point_encoder(torch.cat((scene_features, points - target_center), -1))
                context[b] = (points, encoded, target_weight, target_center)
        rows = []
        for i in range(len(actions.batch_index)):
            b, obj = int(actions.batch_index[i]), int(actions.object[i])
            valid = mask[b]
            points, encoded, target_weight, target_center = context[b]
            contact, direction = actions.contact_world[i], actions.direction_world[i]
            relative = points - contact
            object_weight = condition.object_probability[b, obj, valid]
            object_center = self._pool(points, object_weight)
            local_ids = relative.square().sum(-1).topk(min(64, len(points)), largest=False).indices
            along = (relative * direction).sum(-1).clamp(min=0)
            along = torch.minimum(along, actions.push_distance[i])
            corridor = (relative - along[:, None] * direction).square().sum(-1)
            path_ids = corridor.topk(min(128, len(points)), largest=False).indices
            task = self.category(condition.task_category_id[b]) + self.region(condition.task_region_id[b])
            region_weight = condition.region_probability[b, valid]
            relation = self._instance_context(
                points, encoded, condition.object_probability[b, :, valid],
                condition.object_valid[b], contact, direction, actions.push_distance[i],
                obj, target_center,
            )
            geometry = torch.cat((
                contact - target_center, contact - object_center, direction,
                actions.push_distance[i:i + 1],
            ))
            rows.append(torch.cat((
                self._pool(encoded, object_weight), self._pool(encoded, target_weight),
                encoded[local_ids].mean(0), encoded[path_ids].mean(0),
                self._pool(encoded, region_weight), task, relation, geometry,
            )))
        shared = self.trunk(torch.stack(rows))
        q_value = self._monotonic_q(self.q_head(shared))
        safety_logit = self.safety_head(shared).squeeze(-1)
        safety_probability = torch.sigmoid(safety_logit)
        probability = q_value[:, -1]
        return {
            "q_value": q_value,
            "safety_logit": safety_logit,
            "safety_probability": safety_probability,
            "potential_delta": self.auxiliary_delta_head(shared),
            # Compatibility alias for downstream candidate containers. It is
            # the full-budget Q, not the obsolete potential-improved target.
            "effective_logit": torch.logit(probability.clamp(1e-6, 1 - 1e-6)),
            "effective_probability": probability,
        }
