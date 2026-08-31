"""Independent, end-to-end PointNet++ PUSH evaluator; no perception weights."""
from torch import nn
from .push import PushEffectivenessEvaluator, RulePushGenerator


class StandalonePushModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.push = RulePushGenerator(config)
        self.push_evaluator_ready = False
        self.push_evaluator = PushEffectivenessEvaluator(
            config.feature_dim, config.num_categories, config.num_task_regions)

    @staticmethod
    def _sensor(batch):
        source = batch.get('model_inputs', batch)
        return {key: source[key] for key in ('xyz', 'rgb', 'point_mask')}

    def score_actions(self, batch, condition, actions):
        return self.push_evaluator(self._sensor(batch), condition, actions)

    def forward(self, batch, *, forward_mode='push'):
        if forward_mode != 'push':
            raise ValueError('StandalonePushModel supports only push')
        if self.training:
            raise RuntimeError('Use score_actions with logged actions for training; rules are inference-only')
        sensor = self._sensor(batch)
        condition = batch['push_condition']
        actions = self.push(sensor, condition)
        logits = self.push_evaluator(sensor, condition, actions)
        return {'sensor': sensor, 'push_condition': condition,
                'push': {'actions': actions, 'effective_logit': logits}}
