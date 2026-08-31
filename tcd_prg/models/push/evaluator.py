"""Independent scene/action evaluator; no proposal features or forcing."""
import torch
from torch import nn
from .actions import PushActions
from .pointnet2 import PushPointNet2


class PushEffectivenessEvaluator(nn.Module):
    def __init__(self, feature_dim=256, num_categories=64, num_task_regions=64, *, initialize_backbone=True):
        super().__init__()
        d = feature_dim
        self.point_encoder = nn.Sequential(nn.Linear(d + 3, d), nn.LayerNorm(d), nn.GELU())
        self.category = nn.Embedding(num_categories, d)
        self.region = nn.Embedding(num_task_regions, d)
        self.network = nn.Sequential(nn.Linear(6 * d + 10, 2 * d), nn.GELU(),
                                     nn.LayerNorm(2 * d), nn.Linear(2 * d, d),
                                     nn.GELU(), nn.Linear(d, 1))
        self.feature_dim = d
        self.backbone = None
        if initialize_backbone:
            self.initialize_backbone()

    def initialize_backbone(self):
        if self.backbone is None:
            self.backbone = PushPointNet2(self.feature_dim).to(self.network[0].weight.device)
            self.backbone.train(self.training)

    def forward(self, sensor, condition, actions: PushActions):
        xyz = sensor["xyz"]
        condition.validate(xyz.shape[1])
        actions.validate(len(xyz), condition.object_valid.shape[1])
        if not len(actions.batch_index):
            return self.network[-1].weight.sum().expand(0)
        if not bool(condition.target_valid[actions.batch_index].all()):
            raise ValueError("PUSH requires a visible target")
        if not bool(condition.object_valid[actions.batch_index, actions.object].all()):
            raise ValueError("PUSH object is not represented")
        self.initialize_backbone()
        mask = sensor["point_mask"].bool()
        def pool(value, weight):
            return (value * weight[:, None]).sum(0) / weight.sum().clamp_min(1e-6)
        # Upstream PointNet++ has no padding mask. Batch scenes with the same
        # visible point count, so invalid padding never enters SA/FP or BatchNorm.
        # Fixed-size training clouds consequently use one backbone call.
        groups = {}
        for b in torch.unique(actions.batch_index).tolist():
            valid = mask[b]
            points = xyz[b, valid]
            if not len(points):
                raise ValueError("PUSH action scene requires visible points")
            groups.setdefault(len(points), []).append((b, points, sensor['rgb'][b, valid]))
        context = {}
        for scenes in groups.values():
            features = self.backbone(torch.stack([item[1] for item in scenes]),
                                     torch.stack([item[2] for item in scenes]))
            for (b, points, _), scene_features in zip(scenes, features):
                target_weight = condition.target_probability[b, mask[b]]
                target_center = pool(points, target_weight)
                encoded = self.point_encoder(torch.cat((scene_features, points-target_center), -1))
                context[b] = (points, encoded, target_weight, target_center)
        rows = []
        # Encode each scene once; candidate pooling never retains K x N x D graphs.
        for i in range(len(actions.batch_index)):
            b, obj = int(actions.batch_index[i]), int(actions.object[i])
            valid = mask[b]
            points, encoded, target_weight, target_center = context[b]
            contact, direction = actions.contact_world[i], actions.direction_world[i]
            relative = points - contact
            object_weight = condition.object_probability[b, obj, valid]
            object_center = pool(points, object_weight)
            local_ids = relative.square().sum(-1).topk(min(64, len(points)), largest=False).indices
            along = (relative * direction).sum(-1).clamp(min=0)
            along = torch.minimum(along, actions.push_distance[i])
            corridor = (relative - along[:, None] * direction).square().sum(-1)
            path_ids = corridor.topk(min(128, len(points)), largest=False).indices
            task = self.category(condition.task_category_id[b]) + self.region(condition.task_region_id[b])
            region_weight = condition.region_probability[b, valid]
            geometry = torch.cat((contact - target_center, contact - object_center,
                                  direction, actions.push_distance[i:i+1]))
            rows.append(torch.cat((pool(encoded, object_weight), pool(encoded, target_weight),
                                   encoded[local_ids].mean(0), encoded[path_ids].mean(0),
                                   pool(encoded, region_weight), task, geometry)))
        return self.network(torch.stack(rows)).squeeze(-1)
