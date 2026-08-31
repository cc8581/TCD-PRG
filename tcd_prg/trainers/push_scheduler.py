"""PUSH optimizer-step schedule matching the A/B warmup + cosine formula."""
import math


class PushLRScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps):
        if warmup_steps < 0 or max_steps <= 0:
            raise ValueError("PUSH schedule requires nonnegative warmup and positive max steps")
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.set_step(0)

    def factor(self, step):
        if step < self.warmup_steps:
            return max(1e-8, step / max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return .5 * (1 + math.cos(math.pi * min(1., progress)))

    def set_step(self, step):
        self.last_epoch = int(step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base * self.factor(self.last_epoch)

    def step(self):
        self.set_step(self.last_epoch + 1)

    def state_dict(self):
        return dict(last_epoch=self.last_epoch, base_lrs=self.base_lrs,
                    warmup_steps=self.warmup_steps, max_steps=self.max_steps)

    def load_state_dict(self, state):
        if state['warmup_steps'] != self.warmup_steps or state['base_lrs'] != self.base_lrs:
            raise RuntimeError("PUSH scheduler configuration mismatch")
        # An explicitly extended run uses the newly configured cosine horizon.
        self.set_step(state['last_epoch'])
