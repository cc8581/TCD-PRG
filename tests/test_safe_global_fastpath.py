from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from tcd_prg.datasets.collate import collate_global_grasp
from tcd_prg.datasets.types import GlobalGraspLabels, GlobalGraspSample, SceneObservation
from tcd_prg.models.backbones.task_point_transformer import (
    SceneGeometryOutput,
    TaskConditionedPointTransformer,
)


class TinyNeutral(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.proj = nn.Linear(6, dim)

    def forward(self, xyz, rgb, instance_id, point_mask, object_mask, grid_coord=None):
        del grid_coord
        feat = self.proj(torch.cat((xyz, rgb), -1)) * point_mask.unsqueeze(-1)
        object_count = object_mask.shape[1]
        tokens = []
        for object_index in range(object_count):
            mask = point_mask & (instance_id == object_index)
            denom = mask.sum(-1, keepdim=True).clamp_min(1)
            tokens.append((feat * mask.unsqueeze(-1)).sum(1) / denom)
        object_tokens = torch.stack(tokens, 1) * object_mask.unsqueeze(-1)
        global_token = feat.sum(1) / point_mask.sum(1, keepdim=True).clamp_min(1)
        return SceneGeometryOutput(feat, object_tokens, object_mask, global_token)


def test_neutral_fastpath_matches_scene_fields_from_full_encoder():
    torch.manual_seed(7)
    encoder = TaskConditionedPointTransformer(
        dim=8,
        task_dim=4,
        num_categories=8,
        num_regions=8,
        scene_backbone=TinyNeutral(8),
    )
    xyz = torch.randn(2, 9, 3)
    rgb = torch.randn(2, 9, 3)
    instance = torch.tensor([[0,0,0,0,1,1,1,1,1],[0,0,0,1,1,1,1,1,1]])
    point_mask = torch.ones(2, 9, dtype=torch.bool)
    target_mask = instance == 0
    object_mask = torch.ones(2, 2, dtype=torch.bool)
    category = torch.tensor([1, 2])
    region = torch.tensor([3, 4])
    target_object = torch.tensor([0, 0])

    neutral = encoder.forward_scene_geometry(xyz, rgb, instance, point_mask, object_mask)
    full = encoder(
        xyz, rgb, instance, point_mask, target_mask, object_mask,
        category, region, True, target_object=target_object,
    )
    assert torch.equal(neutral.point_features, full.scene_point_features)
    assert torch.equal(neutral.object_tokens, full.scene_object_tokens)
    assert torch.equal(neutral.global_scene_token, full.scene_global_token)


def _observation(scene_id: int) -> SceneObservation:
    return SceneObservation(
        scene_id=scene_id,
        state_id=0,
        task_index=0,
        xyz=np.asarray([[0,0,0],[0.01,0,0],[0,0.01,0]], np.float32),
        rgb=np.asarray([[1,0,0],[0,1,0],[0,0,1]], np.float32),
        instance_id=np.asarray([0,0,0], np.int64),
        target_mask=np.asarray([True,True,True]),
        target_object=0,
        task_region_id=1,
        object_uuid=("obj",),
        object_pose=np.asarray([[0,0,0,0,0,0,1]], np.float32),
        object_category_id=np.asarray([2], np.int64),
        object_present=np.asarray([True]),
        object_active=np.asarray([True]),
        camera_parameters=(),
    )


def _global_label() -> GlobalGraspLabels:
    return GlobalGraspLabels(
        object_index=np.asarray([0], np.int64),
        source_grasp_index=np.asarray([5], np.int64),
        contact_point_world=np.asarray([[0,0,0]], np.float32),
        grasp_pose_world=np.asarray([[0,0,0,0,0,0,1]], np.float32),
        approach_direction_world=np.asarray([[0,0,1]], np.float32),
        width_m=np.asarray([0.04], np.float32),
        intrinsic_stable=np.asarray([True]),
        scene_executable=np.asarray([1], np.int8),
        anchor_visible_distance_m=np.asarray([0.0], np.float32),
        valid_mask=np.asarray([True]),
        conversion_version="test",
        label_set_complete=False,
    )


def test_global_collator_contains_only_required_contract():
    samples = [
        GlobalGraspSample(_observation(0), _global_label()),
        GlobalGraspSample(_observation(1), _global_label()),
    ]
    batch = collate_global_grasp(samples, grid_size_m=None, training=False)
    required = {
        "xyz", "rgb", "point_mask", "instance_id", "object_mask",
        "object_present", "object_active", "global_loss_sample_valid",
        "global_grasp_labels", "samples",
    }
    assert required.issubset(batch)
    forbidden = {
        "action_type", "policy_success_mask", "relation_graph",
        "action_parameters", "verifier_inputs",
    }
    assert forbidden.isdisjoint(batch)
    assert batch["global_grasp_labels"]["scene_executable"].tolist() == [[1], [1]]


def test_dataset_adapter_lightweight_global_sample_skips_unrelated_loaders():
    from tcd_prg.datasets.base import DatasetAdapter
    from tcd_prg.datasets.capabilities import DatasetCapabilities

    class Adapter(DatasetAdapter):
        capabilities = DatasetCapabilities()

        def __init__(self):
            self.calls = []

        def iter_action_groups(self, split=None):
            del split
            return iter(())

        def load_observation(self, scene_id, state_id, task_index):
            self.calls.append("observation")
            return _observation(scene_id)

        def load_state_labels(self, scene_id, state_id):
            self.calls.append("state_labels")
            raise AssertionError("Global-only load must not request state labels")

        def load_action_group(self, scene_id, group_index):
            self.calls.append("action_group")
            raise AssertionError("Global-only load must not request action groups")

        def load_sequences(self, scene_id, task_index=None):
            self.calls.append("sequences")
            raise AssertionError("Global-only load must not request sequences")

        def load_global_grasps(self, scene_id, state_id, observation, training=True):
            self.calls.append("global_grasps")
            assert training is True
            return _global_label()

    adapter = Adapter()
    sample = adapter.load_global_sample(0, 0, 0)
    assert sample.global_grasps.scene_executable.tolist() == [1]
    assert adapter.calls == ["observation", "global_grasps"]
