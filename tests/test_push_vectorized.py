import copy
from dataclasses import replace
import torch
from test_independent_push import model, scene
from tcd_prg.models import push_condition_from_gt
from tcd_prg.trainers.push_evaluator import logged_push_actions


def test_interleaved_scenes_match_single_action_outputs_and_gradients():
    torch.manual_seed(19)
    def repeat(x):
        return {k: repeat(v) for k,v in x.items()} if isinstance(x,dict) else x.repeat(2,*([1]*(x.ndim-1)))
    batch=repeat(scene())
    batch['xyz'][1] += .02
    condition=push_condition_from_gt(batch,4)
    actions,_=logged_push_actions(batch,condition)
    order=torch.tensor([3,0,2,1])
    def subset(a,ids):
        return replace(a, **{k:getattr(a,k)[ids] for k in a.__dataclass_fields__})
    actions=subset(actions,order)
    network=model().eval()
    reference=copy.deepcopy(network)
    torch.manual_seed(5)
    actual=network.score_actions(batch,condition,actions)
    # The backbone's eval FPS is deterministic for each scene.
    expected=[]
    for i in range(4):
        torch.manual_seed(5)
        expected.append(reference.score_actions(batch,condition,subset(actions,slice(i,i+1))))
    for key in ('improvement_logit',):
        torch.testing.assert_close(actual[key],torch.cat([v[key] for v in expected]),atol=2e-5,rtol=2e-4)
    actual['improvement_logit'].sum().backward()
    sum(v['improvement_logit'].sum() for v in expected).backward()
    for (name,p),(_,q) in zip(network.named_parameters(),reference.named_parameters()):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(),name
            torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=3e-3)
