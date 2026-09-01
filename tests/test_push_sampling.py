import torch
import pytest
from tcd_prg.models import push_condition_from_gt
from tcd_prg.trainers.push_evaluator import logged_push_actions
from tcd_prg.trainers.push_sampling import sample_push_training_input, compiled_fps
from test_push_pointnet2 import _scene_batch


def test_compiled_fps_alignment_short_clouds_and_action_remapping():
    batch = _scene_batch(3)
    batch['candidate_mask'][0].zero_()
    batch['point_mask'][1, ::3] = False
    batch['xyz'][1, ::3] = float('nan')
    condition = push_condition_from_gt(batch, 4)
    actions, _ = logged_push_actions(batch, condition)
    sensor, sampled, remapped = sample_push_training_input(batch, condition, actions, 96)
    assert sensor['xyz'].shape == (2, 96, 3)
    assert sensor['point_mask'].sum(1).tolist() == [48, 72]
    assert torch.isfinite(sensor['xyz']).all()
    assert remapped.batch_index.tolist() == [0, 0, 1, 1]
    assert torch.equal(remapped.contact_world, actions.contact_world)
    for row, original in enumerate((1, 2)):
        valid = torch.where(batch['point_mask'][original])[0]
        _, idx = compiled_fps()(batch['xyz'][original, valid][None], K=len(valid))
        expected = valid[idx[0]]
        mask = sensor['point_mask'][row]
        assert torch.equal(sensor['xyz'][row, mask], batch['xyz'][original, expected])
        assert torch.equal(sensor['rgb'][row, mask], batch['rgb'][original, expected])
        assert torch.equal(sampled.object_probability[row][:, mask], condition.object_probability[original][:, expected])
        assert torch.equal(sampled.target_probability[row, mask], condition.target_probability[original, expected])
        assert torch.equal(sampled.region_probability[row, mask], condition.region_probability[original, expected])
        assert not sampled.object_probability[row][:, ~mask].any()
    sampled.validate(96)


@pytest.mark.parametrize('count', [1024, 4096])
def test_training_fps_uses_requested_count(count):
    batch = _scene_batch(1)
    n = 4200
    batch['xyz'] = torch.rand(1, n, 3)
    batch['rgb'] = torch.rand(1, n, 3)
    batch['point_mask'] = torch.ones(1, n, dtype=torch.bool)
    # Extend each GT point field with repeated aligned labels.
    for key in ('instance_id', 'target_mask', 'region_valid', 'region_target'):
        batch[key] = batch[key][:, torch.arange(n) % batch[key].shape[1]]
    condition = push_condition_from_gt(batch, 4)
    actions, _ = logged_push_actions(batch, condition)
    sensor, sampled, _ = sample_push_training_input(batch, condition, actions, count)
    assert sensor['xyz'].shape == (1, count, 3)
    assert sensor['xyz'][0].unique(dim=0).shape[0] == count
    sampled.validate(count)


def test_training_uses_configured_count_but_default_validation_does_not():
    from torch import nn
    from test_independent_push import model
    from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
    from tcd_prg.trainers.push_evaluator import push_effectiveness_batch_loss
    class FeatureProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 16)
            self.shapes = []
        def forward(self, xyz, rgb):
            self.shapes.append(tuple(xyz.shape))
            return self.linear(xyz)
    m = model()
    probe = FeatureProbe()
    m.push_evaluator.backbone = probe
    batch = _scene_batch(2)
    for key in ('xyz', 'rgb', 'point_mask', 'instance_id', 'target_mask', 'region_valid', 'region_target'):
        batch[key] = batch[key][:, torch.arange(2048) % batch[key].shape[1]]
    original = batch['xyz'].clone()
    from types import SimpleNamespace
    from tcd_prg.scripts.train_push_evaluator import accumulated_batches
    config = SimpleNamespace(training=SimpleNamespace(gradient_accumulation_steps=1, push_fps_points=1024),
                             model=SimpleNamespace(instance_queries=4))
    opt = torch.optim.SGD(m.parameters(), lr=.001)
    updates = list(accumulated_batches(m, [batch], device=torch.device('cpu'), config=config,
                                      loss_function=PushEffectivenessLoss(), optimizer=opt))
    assert len(updates) == 1 and updates[0][1:3] == (4, 2)
    assert probe.shapes == [(2, 1024, 3)]
    assert torch.equal(batch['xyz'], original)
    with torch.no_grad():
        push_effectiveness_batch_loss(m.eval(), batch, instance_queries=4, loss_function=PushEffectivenessLoss())
    assert probe.shapes[-1] == (2, original.shape[1], 3)
