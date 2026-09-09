"""Instance-aware single-head PUSH value evaluator."""

import torch
from torch import nn

from .actions import PushActions
from .pointnet2 import PushPointNet2


class PushImprovementEvaluator(nn.Module):
    """Predict one scalar task-environment value for each complete PUSH action."""

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
        self.value_head = nn.Linear(d, 1)
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

    def forward(self, sensor, condition, actions: PushActions):
        xyz = sensor["xyz"]
        condition.validate(xyz.shape[1])
        actions.validate(len(xyz), condition.object_valid.shape[1])
        if not len(actions.batch_index):
            empty = self.value_head.weight.sum().expand(0)
            return {"improvement_logit": empty}
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
        rows, row_ids = [], []
        for b, (points, encoded, target_weight, target_center) in context.items():
            ids = torch.where(actions.batch_index == b)[0]
            obj = actions.object[ids]
            valid = mask[b]
            weights = condition.object_probability[b, :, valid]
            # Preserve the original weighted reduction, but build it once per
            # scene, rather than once per action. No parameter/layout changes.
            centers = torch.stack([self._pool(points, w) for w in weights])
            extents = torch.stack([
                self._pool((points - center).square(), w).sqrt()
                for w, center in zip(weights, centers)
            ])
            pooled = torch.stack([self._pool(encoded, w) for w in weights])
            slots = torch.where(condition.object_valid[b] & (weights.sum(-1) > 0))[0]
            contact, direction = actions.contact_world[ids], actions.direction_world[ids]
            distance = actions.push_distance[ids]
            relative = points[None] - contact[:, None]
            object_center = centers[obj]
            local_ids = relative.square().sum(-1).topk(min(64, len(points)), largest=False).indices
            along = (relative * direction[:, None]).sum(-1).clamp(min=0)
            along = torch.minimum(along, distance[:, None])
            corridor = (relative - along[..., None] * direction[:, None]).square().sum(-1)
            path_ids = corridor.topk(min(128, len(points)), largest=False).indices
            task = self.category(condition.task_category_id[b]) + self.region(
                condition.task_region_id[b]
            )
            region_weight = condition.region_probability[b, valid]
            if len(slots):
                relative_center = centers[slots][None] - contact[:, None]
                center_along = torch.minimum(
                    (relative_center * direction[:, None]).sum(-1).clamp_min(0), distance[:, None])
                offset = torch.linalg.vector_norm(
                    relative_center - center_along[..., None] * direction[:, None], dim=-1)
                relation_geometry = torch.cat((
                    (centers[slots] - target_center)[None].expand(len(ids), -1, -1),
                    relative_center, extents[slots][None].expand(len(ids), -1, -1),
                    center_along[..., None], offset[..., None],
                    (slots[None] == obj[:, None]).to(points.dtype)[..., None],
                ), -1)
                relation_input = torch.cat((
                    pooled[slots][None].expand(len(ids), -1, -1), relation_geometry), -1)
                tokens = self.instance_relation(relation_input)
                attention = self.relation_attention(relation_input).softmax(dim=1)
                relation = (tokens * attention).sum(1)
            else:
                relation = encoded.new_zeros(len(ids), self.feature_dim)
            geometry = torch.cat((
                contact - target_center, contact - object_center, direction,
                distance[:, None],
            ), -1)
            rows.append(torch.cat((
                pooled[obj], self._pool(encoded, target_weight)[None].expand(len(ids), -1),
                encoded[local_ids].mean(1), encoded[path_ids].mean(1),
                self._pool(encoded, region_weight)[None].expand(len(ids), -1),
                task[None].expand(len(ids), -1), relation, geometry,
            ), -1))
            row_ids.append(ids)
        # Callers align supervision with the original (possibly interleaved)
        # action order, not with the scene grouping used above.
        shared = self.trunk(torch.cat(rows)[torch.cat(row_ids).argsort()])
        improvement_logit = self.value_head(shared).squeeze(-1)
        return {"improvement_logit": improvement_logit}
