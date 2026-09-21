"""Independent end-to-end PUSH evaluator with a selectable geometry backbone."""
from torch import nn

from tcd_prg.config import BackboneConfig

from .push import PushImprovementEvaluator, RulePushGenerator


class StandalonePushModel(nn.Module):
    def __init__(
        self,
        config,
        backbone_config=None,
        push_backbone="point_transformer_v3",
    ):
        super().__init__()
        self.push = RulePushGenerator(config)
        self.push_evaluator_ready = False
        self.push_evaluator = PushImprovementEvaluator(
            config.feature_dim,
            config.num_categories,
            config.num_task_regions,
            backbone_backend=push_backbone,
            backbone_config=backbone_config or BackboneConfig(),
            activation_checkpointing=config.activation_checkpointing,
        )

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
        evaluation = self.push_evaluator(sensor, condition, actions)
        return {'sensor': sensor, 'push_condition': condition,
                'push': {'actions': actions, **evaluation}}
