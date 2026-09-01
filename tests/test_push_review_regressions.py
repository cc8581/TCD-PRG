"""Counterexamples from the PUSH review, not only happy-path fixtures."""
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch

from tcd_prg.config import BackboneConfig, ModelConfig
from tcd_prg.models import StandalonePushModel, push_condition_from_gt
from tcd_prg.models.push.rules import RulePushGenerator, polygon, projection_overlap
from tcd_prg.models.staged_checkpoint import load_push_evaluator
from tcd_prg.trainers.push_checkpoint import PushTrainingCheckpoint, atomic_save


def covering_scene(height=.05, offset=0.):
    target = torch.tensor([[x, y, 0.] for x in [-.02, .02] for y in [-.02, .02]])
    upper = torch.tensor([[x+offset, y, height+z] for z in [-.01, 0., .01]
                          for x,y in [(-.05,-.05),(-.05,0),(-.05,.05),(0,.05),
                                      (.05,.05),(.05,0),(.05,-.05),(0,-.05)]])
    xyz = torch.cat((target, upper))[None]
    ids = torch.cat((torch.zeros(len(target)), torch.ones(len(upper)))).long()[None]
    return dict(xyz=xyz, point_mask=torch.ones_like(ids, dtype=torch.bool), instance_id=ids,
                object_mask=torch.ones(1,2,dtype=torch.bool), target_mask=ids==0,
                region_valid=ids==0, region_target=ids==0,
                task_category_id=torch.tensor([0]), task_region_id=torch.tensor([0]))


@pytest.mark.parametrize('height,offset,expected', [(.05,0.,True),(-.05,0.,False),(.05,.3,False)])
def test_covering_upper_object_needs_no_interior_samples(height, offset, expected):
    batch = covering_scene(height, offset)
    actions = RulePushGenerator(ModelConfig())(batch, push_condition_from_gt(batch,2))
    assert bool(len(actions.object)) == expected
    if expected:
        assert set(actions.object.tolist()) == {1}


def test_projection_overlap_handles_crossing_edges_and_reverse_containment():
    a = polygon(np.array([[-3.,-1.], [3.,-1.], [3.,1.], [-3.,1.]]))
    b = polygon(np.array([[-1.,-3.], [1.,-3.], [1.,3.], [-1.,3.]]))
    intersection = projection_overlap(a,b)
    assert len(intersection)==4
    assert np.allclose(np.abs(intersection), 1., atol=.002)
    assert len(projection_overlap(a, a/2)) == 4
    assert len(projection_overlap(a/2, a)) == 4
    assert len(projection_overlap(a, b+20)) == 0


def tiny_training():
    # Exercise real AdamW moments, independently of expensive point encoding.
    model = StandalonePushModel(ModelConfig(feature_dim=16))
    optimizer = torch.optim.AdamW(model.push_evaluator.parameters(), lr=.001)
    return model, optimizer


def update(model, optimizer):
    optimizer.zero_grad()
    sum(p.square().sum() for p in model.push_evaluator.parameters()).backward()
    optimizer.step()


def test_best_is_immediate_and_resume_preserves_optimizer_next_update(tmp_path):
    model, optimizer = tiny_training()
    update(model, optimizer)
    check = PushTrainingCheckpoint(tmp_path/'best.pt', model, {}, {'population':3})
    check.consider_best({'push_evaluator_pairwise_ranking_accuracy':.8},1)
    assert check.output.is_file()  # Before final validation, or any later training.
    check.save_latest(optimizer,1)
    stored = torch.load(check.output,weights_only=False)
    assert stored['optimizer_steps']==1
    assert not any('geometry_encoder' in k for k in stored['model'])
    restored, new_optimizer = tiny_training()
    other = PushTrainingCheckpoint(tmp_path/'resumed.pt',restored,{}, {'population':3})
    assert other.restore(check.latest,new_optimizer)==1
    update(model,optimizer)
    update(restored,new_optimizer)
    for a,b in zip(model.push_evaluator.parameters(),restored.push_evaluator.parameters()):
        assert torch.equal(a,b)
    check.consider_best({'push_evaluator_pairwise_ranking_accuracy':.5},2)
    assert torch.load(check.output,weights_only=False)['optimizer_steps']==1
    load_push_evaluator(restored,check.output)
    assert restored.push_evaluator_ready
    other.resume_signature = {'population':4}
    with pytest.raises(RuntimeError, match='matching training'):
        other.restore(check.latest,new_optimizer)


def test_atomic_failure_keeps_published_checkpoint(tmp_path,monkeypatch):
    path = tmp_path/'best.pt'
    atomic_save({'value':1},path)
    def fail(payload, stream):
        stream.write(b'incomplete')
        raise OSError('simulated disk failure')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError,match='disk failure'):
        atomic_save({'value':2},path)
    assert torch.load(path,weights_only=False)=={'value':1}
    assert not list(tmp_path.glob('*.tmp'))


def test_resume_retains_best_published_after_last_snapshot(tmp_path):
    model, optimizer = tiny_training()
    check = PushTrainingCheckpoint(tmp_path/'best.pt',model,{}, {})
    check.consider_best({'push_evaluator_pairwise_ranking_accuracy':.6},1)
    check.save_latest(optimizer,1)
    update(model,optimizer)
    check.consider_best({'push_evaluator_pairwise_ranking_accuracy':.9},2)
    expected = {k:v.clone() for k,v in check.best_state.items()}
    restored, new_optimizer = tiny_training()
    other = PushTrainingCheckpoint(check.output,restored,{}, {})
    assert other.restore(check.latest,new_optimizer)==1
    assert other.best_step==2
    assert all(torch.equal(v,other.best_state[k]) for k,v in expected.items())


def test_training_entry_survives_final_validation_failure_and_resumes(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from test_independent_push import scene
    from tcd_prg.config import TCDPRGConfig, TrainingConfig
    from tcd_prg.scripts import train_push_evaluator as entry
    from tcd_prg.models.push.pointnet2 import PushPointNet2
    pretrained_calls=[]
    original_pretrained=PushPointNet2.load_pretrained
    def load_pretrained(module):
        pretrained_calls.append(1)
        return original_pretrained(module)
    monkeypatch.setattr(PushPointNet2,'load_pretrained',load_pretrained)
    config = TCDPRGConfig(model=ModelConfig(feature_dim=16,instance_queries=4),
                         backbone=BackboneConfig(backend='legacy',attention_points=8),
                         training=TrainingConfig(device='cpu',amp=False,batch_size=1,num_workers=0,
                                                 push_fps_points=32,
                                                 max_optimizer_steps=2,
                                                 validation_interval=1,pretrain_checkpoint=None))
    monkeypatch.setattr(entry,'load_config',lambda *args:config)
    monkeypatch.setattr(entry,'create_adapter',lambda *args,**kwargs:SimpleNamespace(scene_splits={'val':(1,)}))
    ready = {'loss':False}
    class GuardedSamples(list):
        def __getitem__(self,index):
            assert ready['loss'], 'Startup must not read samples to compute class weights'
            return super().__getitem__(index)
    monkeypatch.setattr(entry,'ActionStateGroupDataset',lambda *args,**kwargs:GuardedSamples([scene()]))
    def collator(cfg, **kwargs):
        assert Path(cfg.output_dir) == tmp_path.resolve()
        return lambda samples:samples[0]
    monkeypatch.setattr(entry,'PushValueBatchCollator',collator)
    real_loss = entry.PushEffectivenessLoss
    def fixed_loss(*args,**kwargs):
        assert not args and set(kwargs) == {
            'q_weight', 'rank_weight', 'safety_weight', 'auxiliary_weight',
            'rank_margin', 'delta_scales'
        }
        ready['loss'] = True
        return real_loss(*args,**kwargs)
    monkeypatch.setattr(entry,'PushEffectivenessLoss',fixed_loss)
    real_evaluate = entry._evaluate
    calls = []
    def evaluate(*args,**kwargs):
        calls.append(1)
        if len(calls)==3:
            raise RuntimeError('simulated final validation failure')
        return real_evaluate(*args,**kwargs)
    monkeypatch.setattr(entry,'_evaluate',evaluate)
    path = tmp_path/'best.pt'
    argv = ['train_push_evaluator.py','--output',str(path)]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(RuntimeError,match='simulated final validation'):
        entry.main()
    assert path.is_file()
    latest = tmp_path/'best_last.pt'
    payload = torch.load(latest,weights_only=False)
    assert payload['optimizer_steps']==2
    assert payload['scheduler']['last_epoch']==2
    assert (tmp_path/'resolved_config.yaml').is_file()
    assert payload['selection_metric']=='push_evaluator_pairwise_ranking_accuracy'
    assert 'positive_count' not in payload and 'negative_count' not in payload
    assert not hasattr(entry,'_counts') and not hasattr(entry,'_count_loader')
    monkeypatch.setattr(entry,'_evaluate',real_evaluate)
    config.training.max_optimizer_steps = 3
    ready['loss'] = False
    monkeypatch.setattr(sys,'argv',argv+['--resume',str(latest)])
    entry.main()
    assert torch.load(latest,weights_only=False)['optimizer_steps']==3
    assert 'final_validation_metrics' in torch.load(path,weights_only=False)
    assert len(pretrained_calls)==1  # Fresh start only, never overwrite resume.


def test_new_objective_uses_only_explicit_valid_masks():
    from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
    q = torch.zeros(2, 5, requires_grad=True)
    safety = torch.zeros(2, requires_grad=True)
    delta = torch.zeros(2, 5, requires_grad=True)
    loss = PushEffectivenessLoss()(
        {'q_value':q, 'safety_logit':safety, 'potential_delta':delta},
        q_target=torch.ones(2,5), q_valid=torch.tensor([[True]*5,[False]*5]),
        safety_target=torch.tensor([True,False]), safety_valid=torch.tensor([True,False]),
        auxiliary_target=torch.ones(2,5), auxiliary_valid=torch.tensor([True,False]),
        group_index=torch.tensor([0,0]))['push_effectiveness']
    loss.backward()
    assert q.grad[1].abs().sum() == 0 and safety.grad[1] == 0 and delta.grad[1].abs().sum() == 0
