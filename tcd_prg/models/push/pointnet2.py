"""Adapter for the unmodified yanx27 S3DIS PointNet++ network.

Only input preparation, feature extraction and output projection live here.
SA/FP layers and their forward implementation are imported from upstream.
"""
import builtins
import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

SOURCE_ROOT = Path(__file__).resolve().parents[3] / 'third_party' / 'pointnet2_yanx27'
SOURCE_REVISION = 'eb64fe0b4c24055559cea26299cb485dcb43d8dd'
PRETRAINED_RELATIVE = 'log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth'
PRETRAINED_SHA256 = '5ac6896d077ca7302fc263626f8239cffc9717785f067ea94221194dfcb79625'
SOURCE_HASHES = {
    'pointnet2_sem_seg.py': '3b73fd2a8b5f308a561449accc627396e9231903c13e6ff748aa31e71afbb129',
    'pointnet2_utils.py': '640891adff1c045bed111a4d461ce23b9ec815067cef447a9d04b3e64239f0dd',
}


def _verified(path, expected, *, source=False):
    if not path.is_file():
        raise FileNotFoundError(f'Missing yanx27 dependency: {path}; see docs/push_pipeline.md')
    content = path.read_bytes()
    if source:
        content = content.replace(b'\r\n', b'\n')  # Git autocrlf is not a source change.
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError(f'yanx27 dependency checksum mismatch: {path}')
    return path


@lru_cache(maxsize=1)
def upstream_model_class():
    """Resolve upstream's absolute import without polluting global `models`."""
    def load(name, filename, importer=None):
        path = _verified(SOURCE_ROOT / 'models' / filename, SOURCE_HASHES[filename], source=True)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        if importer is not None:
            module.__dict__['__builtins__'] = {**vars(builtins), '__import__': importer}
        spec.loader.exec_module(module)
        return module

    utils = load('_tcd_push_yanx27_utils', 'pointnet2_utils.py')

    def upstream_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'models.pointnet2_utils' and level == 0:
            return utils
        return builtins.__import__(name, globals, locals, fromlist, level)

    return load('_tcd_push_yanx27_sem_seg', 'pointnet2_sem_seg.py', upstream_import).get_model


class PushPointNet2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = upstream_model_class()(13)
        # Retain the original classifier for strict upstream checkpoint loading.
        # Its S3DIS logits are not the PUSH objective; only this unused head is frozen.
        self.network.conv2.requires_grad_(False)
        self.projection = nn.Linear(128, dim)

    def load_pretrained(self):
        path = _verified(SOURCE_ROOT / PRETRAINED_RELATIVE, PRETRAINED_SHA256)
        # Published checkpoint includes numpy scalar metadata, unsupported by
        # torch 2.2 weights_only. Unpickling is allowed only after the pinned hash.
        payload = torch.load(path, map_location='cpu', weights_only=False)
        self.network.load_state_dict(payload['model_state_dict'], strict=True)
        return {'source_revision': SOURCE_REVISION, 'checkpoint_sha256': PRETRAINED_SHA256,
                'task': 'S3DIS semantic segmentation', 'input_protocol': 'scene_xyz_rgb_normalized_xyz_v1'}

    @staticmethod
    def prepare_input(xyz, rgb):
        minimum = xyz.amin(1, keepdim=True)
        maximum = xyz.amax(1, keepdim=True)
        normalized = (xyz - minimum) / (maximum - minimum).clamp_min(1e-6)
        # Metric XY centred on the scene; Z above its lowest observed surface.
        origin = torch.cat(((minimum[..., :2] + maximum[..., :2]) * .5, minimum[..., 2:]), -1)
        return torch.cat((xyz - origin, rgb, normalized), -1).transpose(1, 2).contiguous()

    def forward(self, xyz, rgb):
        if xyz.ndim != 3 or xyz.shape != rgb.shape or xyz.shape[-1] != 3 or xyz.shape[1] == 0:
            raise ValueError('PUSH PointNet++ requires matching nonempty [B,N,3] XYZ/RGB')
        if not torch.isfinite(xyz).all() or not torch.isfinite(rgb).all():
            raise ValueError('PUSH visible XYZ/RGB must be finite')
        if bool(((rgb < 0) | (rgb > 1)).any()):
            raise ValueError('PUSH RGB must be in [0,1]')
        count = xyz.shape[1]
        values = self.prepare_input(xyz, rgb)
        # Upstream ball-query requires at least nsample=32 input points.
        if count < 32:
            indices = torch.arange(32, device=xyz.device) % count
            values = values[:, :, indices]
        features = []
        hook = self.network.drop1.register_forward_pre_hook(lambda module, args: features.append(args[0]))
        try:
            if self.training:
                self.network(values)
            else:
                # Upstream FPS draws a random initial index even in eval mode.
                # Isolate its seed without modifying upstream or caller RNG state.
                devices = [xyz.device.index] if xyz.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(0)
                    if xyz.is_cuda:
                        torch.cuda.default_generators[xyz.device.index].manual_seed(0)
                    self.network(values)
            return self.projection(features[0][:, :, :count].transpose(1, 2))
        finally:
            hook.remove()
