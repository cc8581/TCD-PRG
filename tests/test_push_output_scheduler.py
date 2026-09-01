import math

import pytest
import torch

from tcd_prg.trainers.push_scheduler import PushLRScheduler
from tcd_prg.trainers.push_checkpoint import PushTrainingCheckpoint, resume_compatibility
from tcd_prg.trainers.push_progress import print_validation_summary


def test_schedule_matches_other_stages():
    a = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    b = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-4)
    ours = PushLRScheduler(a, 2, 10)
    reference = torch.optim.lr_scheduler.LambdaLR(
        b, lambda step: max(1e-8, step / 2) if step < 2 else
        .5 * (1 + math.cos(math.pi * min(1., (step - 2) / 8))))
    for _ in range(12):
        assert a.param_groups[0]['lr'] == pytest.approx(b.param_groups[0]['lr'])
        a.step()
        b.step()
        ours.step()
        reference.step()


@pytest.mark.parametrize('scheduled', [True])
def test_checkpoint_restores_schedule_and_next_update(tmp_path, scheduled):
    from test_push_review_regressions import tiny_training, update
    model, optimizer = tiny_training()
    scheduler = PushLRScheduler(optimizer, 2, 10) if scheduled else None
    for _ in range(4):
        update(model, optimizer)
        if scheduler:
            scheduler.step()
    checkpoint = PushTrainingCheckpoint(tmp_path/'best.pt', model, {}, {}, scheduler)
    checkpoint.save_latest(optimizer, 4)
    other, opt = tiny_training()
    restored_scheduler = PushLRScheduler(opt, 2, 10)
    restored = PushTrainingCheckpoint(tmp_path/'other.pt', other, {}, {}, restored_scheduler)
    assert restored.restore(checkpoint.latest, opt) == 4
    if scheduler is None:
        scheduler = PushLRScheduler(optimizer, 2, 10)
        scheduler.set_step(4)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert opt.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    update(model, optimizer)
    update(other, opt)
    for left, right in zip(model.push_evaluator.parameters(), other.push_evaluator.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_resume_output_changes_do_not_hide_objective_changes():
    old = {'config': {'output_dir': 'outputs/full', 'logging': {'log_interval': 20},
                      'training': {'batch_size': 64, 'gradient_accumulation_steps': 1},
                      'optimizer': {'learning_rate': .001}}}
    new = {'config': {'output_dir': 'outputs/run', 'logging': {'log_interval': 10},
                      'training': {'batch_size': 16, 'gradient_accumulation_steps': 1},
                      'optimizer': {'learning_rate': .001}}}
    assert resume_compatibility(old) == resume_compatibility(new)
    assert old['config']['output_dir'] == 'outputs/full'
    assert old['config']['training']['batch_size'] == 64
    new['config']['training']['gradient_accumulation_steps'] = 2
    assert resume_compatibility(old) != resume_compatibility(new)
    new['config']['training']['gradient_accumulation_steps'] = 1
    new['config']['optimizer']['learning_rate'] = .002
    assert resume_compatibility(old) != resume_compatibility(new)


def test_push_validation_transaction_controls_resume_order(tmp_path):
    from test_push_review_regressions import tiny_training

    model, optimizer = tiny_training()
    checkpoint = PushTrainingCheckpoint(tmp_path / 'best.pt', model, {}, {})
    checkpoint.begin_validation(optimizer, 2000)
    pending = torch.load(checkpoint.latest, weights_only=False)
    assert pending['optimizer_steps'] == 2000
    assert pending['pending_validation_step'] == 2000
    assert pending['last_completed_validation_step'] == 0

    restored_model, restored_optimizer = tiny_training()
    restored = PushTrainingCheckpoint(tmp_path / 'best.pt', restored_model, {}, {})
    assert restored.restore(checkpoint.latest, restored_optimizer) == 2000
    assert restored.validation_due(2000, 1000)

    checkpoint.complete_validation(optimizer, 2000)
    completed = PushTrainingCheckpoint(tmp_path / 'best.pt', restored_model, {}, {})
    assert completed.restore(checkpoint.latest, restored_optimizer) == 2000
    assert not completed.validation_due(2000, 1000)


def test_legacy_push_checkpoint_phase_is_recovered_from_boundary_and_log(tmp_path):
    from test_push_review_regressions import tiny_training

    model, optimizer = tiny_training()
    checkpoint = PushTrainingCheckpoint(tmp_path / 'best.pt', model, {}, {})
    checkpoint.save_latest(optimizer, 2000)
    payload = torch.load(checkpoint.latest, weights_only=False)
    payload.pop('last_completed_validation_step')
    payload.pop('pending_validation_step')
    torch.save(payload, checkpoint.latest)

    restored_model, restored_optimizer = tiny_training()
    pending = PushTrainingCheckpoint(tmp_path / 'best.pt', restored_model, {}, {})
    pending.restore(checkpoint.latest, restored_optimizer)
    assert pending.validation_due(2000, 1000)

    (tmp_path / 'validation_metrics.jsonl').write_text(
        '{"phase":"periodic","optimizer_step":2000}\n', encoding='utf-8'
    )
    completed = PushTrainingCheckpoint(tmp_path / 'best.pt', restored_model, {}, {})
    completed.restore(checkpoint.latest, restored_optimizer)
    assert not completed.validation_due(2000, 1000)

    payload['optimizer_steps'] = 2200
    torch.save(payload, checkpoint.latest)
    middle = PushTrainingCheckpoint(tmp_path / 'best.pt', restored_model, {}, {})
    middle.restore(checkpoint.latest, restored_optimizer)
    assert not middle.validation_due(2200, 1000)


def test_validation_short_labels(capsys):
    metrics = {'push_evaluator_' + key: value for key, value in {
        'loss': .6783, 'pairwise_ranking_accuracy': .5906, 'q_mae': .072,
        'top1_regret': .031, 'safety_accuracy': .91}.items()}
    print_validation_summary(metrics, 5000, .60)
    output = capsys.readouterr().out
    assert output.startswith('Val [push_evaluator] [0005000]  loss: 0.6783')
    assert 'rank: 59.1%' in output and 'Q-MAE: 0.0720' in output
    assert '{' not in output and 'push_evaluator_' not in output
    print_validation_summary(metrics, 10000, .60, 'final')
    assert 'best rank(subset)' in capsys.readouterr().out
