from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tcd_prg.config import load_config
from tcd_prg.models import TCDPRGModel
from tcd_prg.planners import TCDPRGPolicy
from tcd_prg.models.staged_checkpoint import load_staged_tcd_prg, load_push_evaluator
from tcd_prg.trainers.push_sampling import sample_push_training_input
from .types import FusedScene, Prediction


def _quoted(name: str, value: str | Path) -> str:
    return f"{name}={json.dumps(str(value), ensure_ascii=False)}"


class DeploymentModel(TCDPRGModel):
    """Keep C's scoring FPS aligned with training; retain raw candidate metadata."""

    def forward_push_from_condition(self, sensor, condition):
        condition.validate(sensor['xyz'].shape[1])
        actions = self.push(sensor, condition)
        sampled, selected_condition, selected_actions = sample_push_training_input(
            sensor, condition, actions, self.push_fps_points)
        return {'actions': actions, **self.push_evaluator(sampled, selected_condition, selected_actions)}


def checkpoint_paths(app_config):
    section = app_config.raw['tcd_prg']
    result = {}
    for key in ('perception_checkpoint', 'grasp_checkpoint', 'push_evaluator_checkpoint'):
        value = section.get(key)
        if not value:
            raise ValueError(f'tcd_prg.{key} must identify a trained stage checkpoint')
        path = app_config.resolve(value)
        if not path.is_file():
            raise FileNotFoundError(f'Missing {key}: {path}')
        result[key] = path
    return result


class TCDPRGPredictor:
    def __init__(self, app_config):
        section = app_config.raw["tcd_prg"]
        checkpoints = checkpoint_paths(app_config)
        path_cfg = app_config.resolve(section["paths_config"])
        paths = __import__("yaml").safe_load(
            path_cfg.read_text(encoding="utf-8")
        ) or {}
        overrides = [
            _quoted("dataset.root", paths["dataset_root"]),
            _quoted("dataset.acronym_root", paths["acronym_root"]),
            _quoted("dataset.functional_region_root", paths["functional_region_root"]),
            _quoted("observation.pybullet_python", paths["pybullet_python"]),
        ]
        if paths.get("observation_cache_dir"):
            overrides.append(
                _quoted("cache.directory", paths["observation_cache_dir"])
            )
        self.config = load_config(
            str(app_config.resolve(section["config"])), overrides
        )
        requested = str(section.get("device", "cuda"))
        self.device = torch.device(
            requested
            if requested.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        self.model = DeploymentModel(
            self.config.model,
            self.config.ablation,
            self.config.backbone,
            self.config.graspnet,
        ).to(self.device)
        self.config.model.task_grasp_probability_threshold = load_staged_tcd_prg(
            self.model, checkpoints['perception_checkpoint'], checkpoints['grasp_checkpoint'], self.config)
        payload = torch.load(checkpoints['push_evaluator_checkpoint'], map_location='cpu', weights_only=False)
        sampling = payload.get('training_scene_sampling', {})
        if sampling.get('operator') != 'pytorch3d.compiled_fps' or not isinstance(sampling.get('points'), int) or sampling['points'] <= 0:
            raise RuntimeError('Stage-C checkpoint is missing its validated FPS sampling protocol')
        self.model.push_fps_points = sampling['points']
        del payload
        load_push_evaluator(self.model, checkpoints['push_evaluator_checkpoint'])
        self.model.to(self.device).eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.policy = TCDPRGPolicy(self.model, self.config)
        self.loaded_checkpoints = {key: str(path) for key, path in checkpoints.items()}

    def _validate_scene(self, scene):
        n = len(scene.xyz_m)
        if n == 0 or scene.xyz_m.shape != (n, 3) or scene.rgb.shape != (n, 3):
            raise ValueError('Expected nonempty XYZRGB arrays with shape [N,3]')
        if not np.isfinite(scene.xyz_m).all() or not np.isfinite(scene.rgb).all():
            raise ValueError('Scene XYZRGB must be finite')
        if np.any(scene.rgb < 0) or np.any(scene.rgb > 1):
            raise ValueError('Scene RGB must be normalized to [0,1]')
        if scene.source_view.shape != (n,) or not np.issubdtype(scene.source_view.dtype, np.integer):
            raise ValueError('source_view must be integer [N]')
        cameras = self._camera_parameters(scene)
        if not cameras or np.any(scene.source_view < 0) or np.any(scene.source_view >= len(cameras)):
            raise ValueError('Each point needs a calibrated source camera')
        if not 0 <= self.config.graspnet.camera_view_index < len(cameras):
            raise ValueError('Configured GraspNet reference camera is missing from this capture')

    def reset(self) -> None:
        self.policy.reset()

    def perceive(self, scene: FusedScene) -> FusedScene:
        """Run the integrated class-agnostic instance head on fused XYZRGB."""
        self._validate_scene(scene)
        result = self.policy.segment_fused_scene(scene.xyz_m, scene.rgb)
        scene.instance_id = np.asarray(result["instance_id"], np.int64)
        scene.category_by_instance = dict(result["category_by_instance"])
        return scene

    @staticmethod
    def _camera_parameters(scene: FusedScene) -> tuple[SimpleNamespace, ...]:
        result = []
        for transform in scene.camera_to_world:
            value = np.asarray(transform, np.float32)
            if value.shape != (4, 4) or not np.isfinite(value).all():
                raise ValueError("camera_to_world must contain finite 4x4 transforms")
            if not np.allclose(value[3], [0, 0, 0, 1]) or not np.allclose(value[:3,:3].T @ value[:3,:3], np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(value[:3,:3]), 1, atol=1e-4):
                raise ValueError('camera_to_world must be a rigid transform in metres')
            eye = value[:3, 3]
            result.append(SimpleNamespace(
                eye_world=eye,
                target_world=eye + value[:3, 2],
                # RGB-D camera Y points down; calibration up is its negative.
                up_world=-value[:3, 1],
            ))
        return tuple(result)

    def predict(
        self,
        scene: FusedScene,
        target: int,
        category: int,
        region: int,
    ) -> Prediction:
        started = time.perf_counter()
        self._validate_scene(scene)
        if not 0 <= category < self.config.model.num_categories or not 0 <= region < self.config.model.num_task_regions:
            raise ValueError('Category/region ID is outside the trained model vocabulary')
        # V2: the UI-selected predicted query is converted to an observable 3D
        # point prompt. query ids are never treated as stable identities.
        if int(target) >= 0:
            prompt = self.policy.target_prompt_from_instance(
                scene.xyz_m, scene.instance_id, int(target)
            )
            encoded = self.policy.encode_fused_scene(
                scene.xyz_m,
                scene.rgb,
                int(category),
                int(region),
                source_view=scene.source_view,
                camera_parameters=self._camera_parameters(scene),
                target_prompt_xyz=prompt,
                continue_target=False,
                enforce_target_confidence=True,
            )
        else:
            # Closed-loop continuation after PUSH/PICK_REMOVE: re-identify the
            # previous physical target among newly predicted queries.
            encoded = self.policy.encode_fused_scene(
                scene.xyz_m,
                scene.rgb,
                int(category),
                int(region),
                source_view=scene.source_view,
                camera_parameters=self._camera_parameters(scene),
                continue_target=True,
                enforce_target_confidence=True,
            )
        candidates = self.policy.generate_candidates(encoded)
        action = self.policy.select_action(candidates)
        if action is None:
            raise RuntimeError("Model produced no valid action")
        action['target_query'] = int(encoded.output['encoded'].target_query_index[0].detach().cpu())
        action['remaining_preparation_actions'] = max(0, 5 - self.policy.preparation_actions)
        return Prediction(action, time.perf_counter() - started)

    def action_executed(
        self,
        prediction: Prediction,
        scene: FusedScene,
        target: int,
        category: int,
        region: int,
    ) -> None:
        del scene, target, category, region
        self.policy.update_after_action(prediction.action, None)
