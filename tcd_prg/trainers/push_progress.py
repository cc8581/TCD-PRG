"""PUSH-only window summaries; validation time never contaminates training ETA."""
import json
import time
from pathlib import Path


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class PushTrainingProgress:
    def __init__(self, output_dir, maximum, initial_step=0, clock=time.monotonic):
        self.path = Path(output_dir) / "train_metrics.jsonl"
        self.maximum = maximum
        self.clock = clock
        self.started = clock()
        self.paused_at = None
        self.excluded = 0.0
        self.last_active = 0.0
        self.last_step = initial_step
        self.initial_step = initial_step
        self.loss_sum = 0.0
        self.actions = self.positive = 0
        self.diagnostics = dict(gradient_norm=0., gradient_norm_after_clip=0., gradient_clip_scale=0., data_seconds=0.)
        self.max_memory_mb = 0.

    def pause(self):
        if self.paused_at is None:
            self.paused_at = self.clock()

    def resume(self):
        if self.paused_at is not None:
            self.excluded += self.clock() - self.paused_at
            self.paused_at = None

    def add(self, loss, actions, positive, *, gradient_norm=0., clip_scale=1., data_seconds=0., max_memory_mb=0.):
        self.loss_sum += float(loss) * int(actions)
        self.actions += int(actions)
        self.positive += int(positive)
        self.diagnostics['gradient_norm'] += gradient_norm
        self.diagnostics['gradient_norm_after_clip'] += gradient_norm * clip_scale
        self.diagnostics['gradient_clip_scale'] += clip_scale
        self.diagnostics['data_seconds'] += data_seconds
        self.max_memory_mb = max(self.max_memory_mb, max_memory_mb)

    def log(self, step, learning_rate):
        steps = step - self.last_step
        if not self.actions or steps <= 0:
            return None
        now = self.clock()
        active = (now if self.paused_at is None else self.paused_at) - self.started - self.excluded
        seconds_per_step = max(0.0, active - self.last_active) / steps
        record = dict(optimizer_step=step, max_optimizer_steps=self.maximum,
                      window_steps=steps, loss=self.loss_sum / self.actions,
                      safe_fraction=self.positive / self.actions, actions=self.actions,
                      learning_rate=float(learning_rate), seconds_per_step=seconds_per_step,
                      elapsed_seconds=now - self.started,
                      eta_train_seconds=active / max(1, step-self.initial_step) * max(0, self.maximum-step),
                      max_memory_mb=self.max_memory_mb,
                      **{key:value/steps for key,value in self.diagnostics.items()})
        fields = [f"Train [push_evaluator] [{step:07d}/{self.maximum:07d}]",
                  f"eta: {duration(record['eta_train_seconds'])}",
                  f"loss: {record['loss']:.4f}", f"safe: {record['safe_fraction']:.1%}",
                  f"lr: {learning_rate:.3e}",
                  f"grad: {record['gradient_norm']:.3f}->{record['gradient_norm_after_clip']:.3f}",
                  f"clip: {record['gradient_clip_scale']:.3f}",
                  f"time: {seconds_per_step:.3f}", f"data: {record['data_seconds']:.3f}"]
        if self.max_memory_mb > 0:
            fields.append(f"max mem: {self.max_memory_mb:.0f}M")
        print("  ".join(fields), flush=True)
        append_record(self.path, record)
        self.last_step, self.last_active = step, active
        self.loss_sum = 0.0
        self.actions = self.positive = 0
        self.diagnostics = dict.fromkeys(self.diagnostics, 0.)
        return record


def print_validation_summary(metrics, step, best_score, phase='periodic'):
    """Same short-label/colon/spacing conventions as the A/B terminal summaries."""
    fields = [f"Val [push_evaluator] [{step:07d}]",
              f"loss: {metrics['push_evaluator_loss']:.4f}",
              f"rank: {metrics['push_evaluator_pairwise_ranking_accuracy']:.1%}",
              f"best rank: {best_score:.1%}",
              f"Q-MAE: {metrics['push_evaluator_q_mae']:.4f}",
              f"top1 regret: {metrics['push_evaluator_top1_regret']:.4f}",
              f"safe acc: {metrics['push_evaluator_safety_accuracy']:.1%}"]
    if phase == 'final':
        fields.append('split: full validation')
        fields[3] = f"best rank(subset): {best_score:.1%}"
    print('  '.join(fields), flush=True)
