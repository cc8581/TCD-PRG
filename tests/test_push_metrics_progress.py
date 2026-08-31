"""Metric semantics, tie handling, progress denominators and resumable selection."""
import io
import itertools
import json
import math

import pytest
import torch

from tcd_prg.evaluators.push_effectiveness import push_effectiveness_metrics
from tcd_prg.trainers.push_progress import PushTrainingProgress


def test_ap_groups_equal_scores_independent_of_input_order():
    scores = torch.tensor([.8,.8,.1])
    labels = torch.tensor([1,0,1],dtype=torch.bool)
    for order in itertools.permutations(range(3)):
        ids = torch.tensor(order)
        metrics = push_effectiveness_metrics(scores[ids],labels[ids])
        assert metrics['push_evaluator_ap'].item() == pytest.approx(7/12)
    tied = push_effectiveness_metrics(torch.full((4,),.5),torch.tensor([1,0,1,0]))
    assert tied['push_evaluator_ap']==tied['push_evaluator_positive_fraction']==.5
    assert tied['push_evaluator_auroc']==.5


def test_hit5_is_not_recall_and_population_is_explicit():
    # Group 0: 6 positives -> Hit@5=1, NOT Recall@5=5/6.
    # Group 1: all negative, must count in global AP/positive fraction but not conditional Hit.
    metrics = push_effectiveness_metrics(torch.linspace(.9,.1,8),
                                        torch.tensor([1,1,1,1,1,1,0,0]),
                                        torch.tensor([0,0,0,0,0,0,1,1]))
    assert metrics['push_evaluator_positive_fraction']==.75
    assert metrics['push_evaluator_logged_hit_at_5_given_positive']==1
    assert metrics['push_evaluator_logged_group_count']==2
    assert metrics['push_evaluator_logged_positive_group_count']==1
    assert metrics['push_evaluator_logged_negative_group_count']==1
    assert not any('precision_at_1' in key or 'recall_at_5' in key for key in metrics)


@pytest.mark.parametrize('labels', [[],[0,0],[1,1]])
def test_degenerate_metric_populations_are_explicit(labels):
    metrics = push_effectiveness_metrics(torch.full((len(labels),),.5),torch.tensor(labels),
                                        torch.zeros(len(labels),dtype=torch.long))
    if not labels or not any(labels):
        assert math.isnan(metrics['push_evaluator_ap'])
        assert math.isnan(metrics['push_evaluator_logged_hit_at_1_given_positive'])
    else:
        assert metrics['push_evaluator_ap']==1
    assert math.isnan(metrics['push_evaluator_auroc'])


def test_window_mean_is_action_weighted_and_eta_excludes_validation(tmp_path,capsys):
    clock = [100.]
    logger = PushTrainingProgress(tmp_path,maximum=20,initial_step=10,clock=lambda:clock[0])
    logger.add(.2,2,1)
    clock[0] += 2
    logger.pause()
    clock[0] += 100  # Deliberately long validation must not inflate s/step or ETA.
    logger.resume()
    logger.add(.8,6,2)
    clock[0] += 2
    record = logger.log(12,1e-4)
    assert record['loss']==pytest.approx(.65)
    assert record['positive_fraction']==3/8
    assert record['window_steps']==2
    assert record['seconds_per_step']==2
    assert record['eta_train_seconds']==16
    assert record['elapsed_seconds']==104
    output = capsys.readouterr().out
    assert 'eta: 00:00:16' in output
    assert '[0000012/0000020]' in output and 'lr: 1.000e-04' in output
    assert json.loads((tmp_path/'train_metrics.jsonl').read_text())==record
    logger.add(.4,1,1)
    clock[0] += 1
    record = logger.log(13,1e-4)
    assert record['loss']==.4 and record['actions']==1 and record['window_steps']==1


def test_validation_progress_finishes_and_counts_empty_groups(monkeypatch):
    from test_independent_push import scene,model
    from tcd_prg.config import TCDPRGConfig,ModelConfig
    from tcd_prg.losses.push_effectiveness import PushEffectivenessLoss
    from tcd_prg.scripts import train_push_evaluator as entry
    from tqdm import tqdm
    output=io.StringIO()
    bars=[]
    def progress(**kwargs):
        bar=tqdm(file=output,**kwargs)
        bars.append(bar)
        return bar
    monkeypatch.setattr(entry,'tqdm',progress)
    empty=scene()
    empty['candidate_mask'].zero_()
    result=entry._evaluate(model(),[scene(),empty],device=torch.device('cpu'),
                           config=TCDPRGConfig(model=ModelConfig(instance_queries=4)),
                           loss_function=PushEffectivenessLoss(1.0),phase='test')
    assert bars[0].n==bars[0].total==2 and bars[0].disable
    assert '100%' in output.getvalue()
    assert result['push_evaluator_evaluated_count']==2
    assert result['push_evaluator_logged_empty_group_count']==1
    assert result['push_evaluator_positive_fraction']==.5


def test_resume_preserves_optimizer_but_discards_old_metric_best(tmp_path,capsys):
    from test_push_review_regressions import tiny_training, update
    from tcd_prg.trainers.push_checkpoint import PushTrainingCheckpoint
    model,optimizer=tiny_training()
    update(model,optimizer)
    checkpoint=PushTrainingCheckpoint(tmp_path/'best.pt',model,{}, {})
    checkpoint.consider_best({'push_evaluator_ap':.9},1)
    checkpoint.save_latest(optimizer,1)
    payload=torch.load(checkpoint.latest,weights_only=False)
    payload.pop('push_metric_protocol_version')
    payload['best_metrics']={'push_evaluator_auprc':.99}
    torch.save(payload,checkpoint.latest)
    other,opt=tiny_training()
    restored=PushTrainingCheckpoint(tmp_path/'new-best.pt',other,{}, {})
    with pytest.raises(RuntimeError, match='metric protocol mismatch'):
        restored.restore(checkpoint.latest,opt)
