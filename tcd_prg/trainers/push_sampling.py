"""Stage-C scene FPS using PyTorch3D's compiled operator."""
from dataclasses import replace
from functools import lru_cache

import torch
from torch.nn.utils.rnn import pad_sequence

@lru_cache(maxsize=1)
def compiled_fps():
    try:
        from pytorch3d import _C
        from pytorch3d.ops import sample_farthest_points
        if not hasattr(_C, 'sample_farthest_points'):
            raise ImportError('PyTorch3D FPS extension is missing')
    except (ImportError, OSError) as error:
        raise RuntimeError('PUSH training requires PyTorch3D with compiled FPS; '
                           'install its CUDA extension. No Python fallback is allowed.') from error
    return sample_farthest_points


def sample_push_training_input(sensor, condition, actions, count):
    """Sample action-bearing scenes and gather all point conditions together.

    Preserve action eligibility/labels computed on the original observation.
    Short clouds retain only real FPS points; masked padding never enters FPS.
    """
    if count <= 0:
        raise ValueError('PUSH FPS count must be positive')
    scene_ids, batch_index = torch.unique(actions.batch_index, sorted=True, return_inverse=True)
    if not len(scene_ids):
        return sensor, condition, actions
    xyz = sensor['xyz']
    ids = [torch.where(sensor['point_mask'][b].bool())[0] for b in scene_ids.tolist()]
    lengths = torch.tensor([len(index) for index in ids], device=xyz.device, dtype=torch.long)
    if bool((lengths == 0).any()):
        raise ValueError('PUSH action scene requires visible points')
    packed = pad_sequence([xyz[b, index] for b, index in zip(scene_ids.tolist(), ids)], batch_first=True)
    if not torch.isfinite(packed).all():
        raise ValueError('PUSH visible XYZ must be finite')
    with torch.no_grad():
        _, selected = compiled_fps()(packed.float(), lengths=lengths, K=count, random_start_point=False)
        # PyTorch3D uses -1 for absent selections in short clouds. Preserve the
        # valid mask rather than duplicating points to manufacture density.
        point_mask = selected >= 0
        original = pad_sequence(ids, batch_first=True).gather(1, selected.clamp_min(0))
    rows = scene_ids[:, None]
    sampled = {'xyz': xyz[rows, original], 'rgb': sensor['rgb'][rows, original],
               'point_mask': point_mask}
    probabilities = condition.object_probability[scene_ids]
    sampled_condition = replace(
        condition,
        object_probability=probabilities.gather(2, original[:, None].expand(-1, probabilities.shape[1], -1)) * point_mask[:, None],
        object_valid=condition.object_valid[scene_ids],
        target_probability=condition.target_probability[rows, original] * point_mask,
        region_probability=condition.region_probability[rows, original] * point_mask,
        target_valid=condition.target_valid[scene_ids],
        task_category_id=condition.task_category_id[scene_ids],
        task_region_id=condition.task_region_id[scene_ids])
    return sampled, sampled_condition, replace(actions, batch_index=batch_index)
