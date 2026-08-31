import copy
from types import SimpleNamespace
import torch
from tcd_prg.models.push.pointnet2 import PushPointNet2
from tcd_prg.trainers.push_evaluator import push_effectiveness_batch_loss
from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
from tcd_prg.scripts.train_push_evaluator import accumulated_batches
from test_independent_push import scene, model


def test_pointnet_backbone_receives_effect_gradients_and_updates():
    torch.manual_seed(7)
    m=model(); b=scene(); opt=torch.optim.AdamW(m.parameters(),lr=.001)
    before={k:v.clone() for k,v in m.push_evaluator.backbone.named_parameters()}
    loss,_=push_effectiveness_batch_loss(m,b,instance_queries=4,loss_function=PushEffectivenessLoss())
    loss.backward()
    for part in (m.push_evaluator.backbone.network.sa1,m.push_evaluator.backbone.network.sa4,
                 m.push_evaluator.backbone.network.fp1,m.push_evaluator.network):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in part.parameters())
    opt.step()
    assert any(not torch.equal(v,before[k]) for k,v in m.push_evaluator.backbone.named_parameters())


def test_fixed_input_repeats_and_ignores_external_perception_features():
    m=model().eval();b=scene()
    with torch.no_grad():
        a=push_effectiveness_batch_loss(m,b,instance_queries=4,loss_function=PushEffectivenessLoss())[1]['effective_logit']
        b['geometry_feature'].fill_(float('nan'))
        c=push_effectiveness_batch_loss(m,b,instance_queries=4,loss_function=PushEffectivenessLoss())[1]['effective_logit']
    assert torch.equal(a,c)


def test_accumulation_matches_action_weighted_batch_and_flushes_tail():
    torch.manual_seed(17)
    # Isolate weighted gradient algebra from upstream random FPS and the
    # ill-conditioned BatchNorm statistics of this tiny, repeated-point fixture.
    # eval keeps all parameter gradients enabled; training is covered separately.
    m=model().eval();reference=copy.deepcopy(m)
    first=scene();second=copy.deepcopy(first)
    second['candidate_mask'][0,1]=False  # Two actions then one; not equal batch means.
    empty=copy.deepcopy(first);empty['candidate_mask'].zero_()
    config=SimpleNamespace(training=SimpleNamespace(gradient_accumulation_steps=2),model=SimpleNamespace(instance_queries=4))
    opt=torch.optim.SGD(m.parameters(),lr=.01)
    batches=accumulated_batches(m,[first,empty,second,first],device=torch.device('cpu'),config=config,
                                loss_function=PushEffectivenessLoss(),optimizer=opt)
    rng = torch.get_rng_state()
    _,count,_,_=next(batches);assert count==3
    torch.set_rng_state(rng)  # Match upstream random FPS choices for the reference.
    l1,_=push_effectiveness_batch_loss(reference,first,instance_queries=4,loss_function=PushEffectivenessLoss())
    l2,_=push_effectiveness_batch_loss(reference,second,instance_queries=4,loss_function=PushEffectivenessLoss())
    ((2*l1+l2)/3).backward()
    for p,q in zip(m.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)
    opt.step()
    _,count,_,_=next(batches);assert count==2
    assert list(batches)==[]


def test_encoder_runs_once_for_multiple_actions():
    m=model();calls=[]
    hook=m.push_evaluator.backbone.register_forward_hook(lambda *args:calls.append(1))
    push_effectiveness_batch_loss(m,scene(),instance_queries=4,loss_function=PushEffectivenessLoss())
    hook.remove();assert len(calls)==1


def _scene_batch(count):
    def repeat(value):
        if isinstance(value, dict):
            return {key: repeat(item) for key, item in value.items()}
        return value.repeat(count, *([1] * (value.ndim - 1)))
    return repeat(scene())


def test_multiple_scenes_share_backbone_call_and_receive_gradients():
    m=model(); batch=_scene_batch(2)
    batch['xyz'][1] *= 1.2
    calls=[]
    hook=m.push_evaluator.backbone.register_forward_pre_hook(
        lambda module,args:calls.append(tuple(args[0].shape)))
    batch['rgb'].requires_grad_()
    try:
        loss,details=push_effectiveness_batch_loss(m,batch,instance_queries=4,
                                                 loss_function=PushEffectivenessLoss())
        loss.backward()
    finally:
        hook.remove()
    assert calls==[(2,batch['xyz'].shape[1],3)]
    assert details['effective_logit'].shape==(4,)
    assert torch.isfinite(batch['rgb'].grad).all()
    assert (batch['rgb'].grad.abs().flatten(1).sum(1)>0).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in m.push_evaluator.backbone.network.sa4.parameters())


def test_grouped_encoding_excludes_padding_and_preserves_action_mapping():
    from torch import nn
    from tcd_prg.models import push_condition_from_gt
    from tcd_prg.trainers.push_evaluator import logged_push_actions
    class PointFeatures(nn.Module):
        def __init__(self):
            super().__init__();self.calls=[]
        def forward(self,xyz,rgb):
            self.calls.append(tuple(xyz.shape))
            assert torch.isfinite(xyz).all() and torch.isfinite(rgb).all()
            return torch.cat((xyz,rgb),-1).repeat(1,1,3)[...,:16]
    m=model().eval();batch=_scene_batch(4)
    batch['xyz'][1] *= 1.2
    batch['xyz'][2] *= .8
    batch['candidate_mask'][3].zero_()  # No action: never encode this scene.
    batch['point_mask'][1,-5:]=False
    batch['xyz'][1,-5:]=float('nan');batch['rgb'][1,-5:]=float('nan')
    batch['xyz'][3]=float('nan');batch['rgb'][3]=float('nan')
    condition=push_condition_from_gt(batch,4)
    actions,_=logged_push_actions(batch,condition)
    backbone=PointFeatures();m.push_evaluator.backbone=backbone
    with torch.no_grad():
        actual=m.score_actions(batch,condition,actions)
        n=batch['xyz'].shape[1]
        assert backbone.calls==[(2,n,3),(1,n-5,3)]
        # Independent scene calls provide a mapping reference without FPS/BN
        # randomness; interleaved output must retain the original action order.
        expected=torch.empty_like(actual)
        for b in (0,1,2):
            ids=torch.where(actions.batch_index==b)[0]
            expected[ids]=m.score_actions(batch,condition,actions.select(ids))
    torch.testing.assert_close(actual,expected)


def test_pointnet_tiny_or_duplicate_cloud_is_finite():
    m=PushPointNet2(16)
    for n in (1,5,40):
        out=m(torch.zeros(1,n,3),torch.zeros(1,n,3))
        assert out.shape==(1,n,16) and torch.isfinite(out).all()


def test_upstream_pretrained_weights_load_completely_and_stay_trainable():
    from tcd_prg.models.push.pointnet2 import SOURCE_ROOT, PRETRAINED_RELATIVE
    m=PushPointNet2(16)
    provenance=m.load_pretrained()
    source=torch.load(SOURCE_ROOT/PRETRAINED_RELATIVE,map_location='cpu',weights_only=False)['model_state_dict']
    assert m.network.state_dict().keys()==source.keys()
    assert all(torch.equal(value,source[key]) for key,value in m.network.state_dict().items())
    assert all(p.requires_grad for part in (m.network.sa1,m.network.sa4,m.network.fp1,m.network.bn1) for p in part.parameters())
    assert provenance['task']=='S3DIS semantic segmentation'


def test_upstream_feature_adapter_matches_original_and_preserves_rng():
    m=PushPointNet2(16).eval();m.load_pretrained()
    xyz=torch.rand(1,64,3);rgb=torch.rand_like(xyz)
    state=torch.get_rng_state()
    with torch.no_grad():
        actual=m(xyz,rgb)
        assert torch.equal(state,torch.get_rng_state())
        values=[]
        hook=m.network.drop1.register_forward_pre_hook(lambda module,args:values.append(args[0]))
        try:
            with torch.random.fork_rng():
                torch.manual_seed(0)
                m.network(m.prepare_input(xyz,rgb))
            expected=m.projection(values[0].transpose(1,2))
        finally:
            hook.remove()
    assert torch.equal(actual,expected)


def test_upstream_import_does_not_replace_other_models_namespace():
    import sys
    from tcd_prg.models.push.pointnet2 import upstream_model_class
    before={k:v for k,v in sys.modules.items() if k=='models' or k.startswith('models.')}
    upstream_model_class()
    after={k:v for k,v in sys.modules.items() if k=='models' or k.startswith('models.')}
    assert before==after


def test_source_checksum_accepts_git_line_endings_but_rejects_edits(tmp_path):
    import hashlib
    import pytest
    from tcd_prg.models.push.pointnet2 import _verified
    path=tmp_path/'source.py'
    expected=hashlib.sha256(b'x = 1\n').hexdigest()
    for content in (b'x = 1\n',b'x = 1\r\n'):
        path.write_bytes(content)
        assert _verified(path,expected,source=True)==path
    path.write_bytes(b'x = 2\n')
    with pytest.raises(RuntimeError,match='checksum mismatch'):
        _verified(path,expected,source=True)


def test_inactive_push_keeps_ab_parameter_layout_and_rng():
    # A/B construction must retain its previous inactive C parameter layout and
    # RNG stream. The new backbone is materialized only by C execution/loading.
    from torch import nn
    from tcd_prg.models.push import PushEffectivenessEvaluator
    class ReferenceInactive(nn.Module):
        def __init__(self):
            super().__init__();d=16
            self.point_encoder=nn.Sequential(nn.Linear(d+3,d),nn.LayerNorm(d),nn.GELU())
            self.category=nn.Embedding(64,d);self.region=nn.Embedding(64,d)
            self.network=nn.Sequential(nn.Linear(6*d+10,2*d),nn.GELU(),nn.LayerNorm(2*d),
                                       nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1))
    torch.manual_seed(91);reference=ReferenceInactive();state=torch.get_rng_state()
    torch.manual_seed(91);current=PushEffectivenessEvaluator(16,initialize_backbone=False)
    assert current.backbone is None and torch.equal(state,torch.get_rng_state())
    assert current.state_dict().keys()==reference.state_dict().keys()
    assert all(torch.equal(v,reference.state_dict()[k]) for k,v in current.state_dict().items())


def test_combined_deployment_loads_same_pointnet_without_using_a_encoder(tmp_path):
    from tcd_prg.models.push import PushEffectivenessEvaluator
    from tcd_prg.models import push_condition_from_gt
    from tcd_prg.models.tcd_prg import TCDPRGModel
    from tcd_prg.models.staged_checkpoint import load_push_evaluator
    from tcd_prg.trainers.push_checkpoint import PushTrainingCheckpoint
    m=model().eval();b=scene();b['push_condition']=push_condition_from_gt(b,4)
    check=PushTrainingCheckpoint(tmp_path/'new.pt',m,{}, {})
    check.consider_best({'push_evaluator_ap':.5},1)
    # No encoder is present: the public combined C boundary must not touch A.
    combined=SimpleNamespace(push=m.push,push_evaluator=PushEffectivenessEvaluator(16,initialize_backbone=False).eval())
    load_push_evaluator(combined,check.output)
    with torch.no_grad():
        expected=m(b)['push']
        actual=TCDPRGModel.forward_push_from_condition(combined,m._sensor(b),b['push_condition'])
    assert torch.equal(expected['actions'].contact_world,actual['actions'].contact_world)
    assert torch.equal(expected['effective_logit'],actual['effective_logit'])
