# ruff: noqa: E402, E501
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tcd_prg.config import load_config
from tcd_prg.models import TCDPRGModel
from tcd_prg.models.staged_checkpoint import load_push_evaluator, load_staged_tcd_prg
from tcd_prg.planners import TCDPRGPolicy
from tcd_prg.trainers.push_sampling import sample_push_training_input

from .types import FusedScene, Prediction
from .workflow import PUSH_DISTANCE_M


def _quoted(name: str, value: str | Path) -> str:
    return f"{name}={json.dumps(str(value), ensure_ascii=False)}"


class DeploymentModel(TCDPRGModel):
    """Keep C's scoring FPS aligned with training; retain raw candidate metadata."""

    def forward_push_from_condition(self, sensor, condition):
        condition.validate(sensor['xyz'].shape[1])
        actions = self.push(sensor, condition)
        return {'actions': actions, **self.score_push_actions(sensor, condition, actions)}

    def score_push_actions(self, sensor, condition, actions):
        sampled, selected_condition, selected_actions = sample_push_training_input(
            sensor, condition, actions, self.push_fps_points)
        return self.push_evaluator(sampled, selected_condition, selected_actions)


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
        fusion = app_config.raw.get("fusion", {})
        if int(fusion.get("target_scene_points", 0)) != 0:
            raise RuntimeError(
                "fusion.target_scene_points must be 0: deployment sampling is owned "
                "by the validated Stage-A/B/C model protocols"
            )
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
        # load_staged_tcd_prg has now checked Stage-A/B architecture, grid,
        # point-count and camera/proposal fields against the saved checkpoints.
        self.input_protocols = {
            "stage_a_grid_size_m": float(self.config.backbone.grid_size_m),
            "stage_b_scene_points": int(self.config.graspnet.scene_input_points),
            "stage_b_target_points": int(self.config.graspnet.target_input_points),
        }
        payload = torch.load(checkpoints['push_evaluator_checkpoint'],
                             map_location='cpu', weights_only=False)
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
        self._encoded = None
        self._push_actions = None
        workflow = app_config.raw.get("workflow", {})
        # Runtime thresholds/counts are operator settings. Architecture and
        # per-stage point sampling remain exactly as recorded by training.
        self.config.model.task_grasp_probability_threshold = float(
            workflow.get("grasp_confidence_threshold", 0.5)
        )
        self.config.model.pick_remove_probability_threshold = float(
            workflow.get("grasp_confidence_threshold", 0.5)
        )
        self.config.model.pick_remove_candidates = max(
            int(self.config.model.pick_remove_candidates),
            int(workflow.get("grasp_top_n", 36)),
        )
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
        self._encoded = None
        self._push_actions = None

    def perceive(self, scene: FusedScene) -> FusedScene:
        """Run the integrated class-agnostic instance head on fused XYZRGB."""
        self._encoded = None
        self._push_actions = None
        self._validate_scene(scene)
        result = self.policy.segment_fused_scene(scene.xyz_m, scene.rgb)
        from .workflow import detach_minor_instance_fragments
        scene.instance_id = detach_minor_instance_fragments(
            scene.xyz_m, np.asarray(result["instance_id"], np.int64)
        )
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
        result = self.analyze(scene, target, category, region)
        if not result.candidates:
            raise RuntimeError("Model produced no valid action")
        action = self.policy.select_action(result.action)
        if action is None:
            raise RuntimeError("Model produced no valid action")
        action["target_query"] = result.target_query
        return Prediction(
            action,
            result.inference_seconds,
            result.candidates,
            result.timings,
            target_query=result.target_query,
            acted_query=int(action["acted_object"]),
            target_instance=result.target_instance,
        )

    def _profiled_encode(self, encode):
        """Measure model heads and always remove temporary forward hooks."""
        timings: dict[str, float] = {}
        starts: dict[str, float] = {}
        handles = []

        def synchronize():
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.synchronize()

        for module_name, timing_name in (
            ("task_grasp", "target_grasp_prediction_s"),
            ("graspnet", "obstructor_grasp_prediction_s"),
            ("push", "push_rule_generation_s"),
            ("push_evaluator", "push_evaluator_scoring_s"),
        ):
            module = getattr(self.policy.model, module_name, None)
            if module is None:
                continue
            def before(_module, _inputs, name=timing_name):
                synchronize()
                starts[name] = time.perf_counter()
            def after(_module, _inputs, _output, name=timing_name):
                synchronize()
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - starts[name]
            handles.extend((module.register_forward_pre_hook(
                before), module.register_forward_hook(after)))
        try:
            return encode(), timings
        finally:
            for handle in handles:
                handle.remove()

    def analyze(
        self,
        scene: FusedScene,
        target: int,
        category: int,
        region: int,
        progress: Callable[[dict], None] | None = None,
    ) -> Prediction:
        """Encode once and expose every valid candidate for staged planning."""
        self._encoded = None
        self._push_actions = None
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
            encoded, head_timings = self._profiled_encode(lambda: self.policy.encode_fused_scene(
                scene.xyz_m,
                scene.rgb,
                int(category),
                int(region),
                source_view=scene.source_view,
                camera_parameters=self._camera_parameters(scene),
                target_prompt_xyz=prompt,
                continue_target=False,
                enforce_target_confidence=True,
                include_push=False,
            ))
        else:
            # Closed-loop continuation after PUSH/PICK_REMOVE: re-identify the
            # previous physical target among newly predicted queries.
            encoded, head_timings = self._profiled_encode(lambda: self.policy.encode_fused_scene(
                scene.xyz_m,
                scene.rgb,
                int(category),
                int(region),
                source_view=scene.source_view,
                camera_parameters=self._camera_parameters(scene),
                continue_target=True,
                enforce_target_confidence=True,
                include_push=False,
            ))
        encoded_seconds = time.perf_counter() - started
        if progress:
            progress({
                "stage": "MODEL_SCENE_ENCODING",
                "message": f"场景编码完成，用时 {encoded_seconds:.3f} s",
                "elapsed_s": encoded_seconds,
            })
        candidate_started = time.perf_counter()
        # The real closed loop has semantic stop conditions instead of the
        # training horizon. Candidate generation must therefore keep the
        # preparation branches available on every observation.
        saved_actions = self.policy.preparation_actions
        self.policy.preparation_actions = 0
        try:
            generated = self.policy.generate_candidates(encoded, include_push=False)
        finally:
            self.policy.preparation_actions = saved_actions
        actions = self._candidate_actions(generated, action_type=2)
        self._encoded = encoded
        candidate_seconds = time.perf_counter() - candidate_started
        if progress:
            progress({
                "stage": "MODEL_CANDIDATE_GENERATION",
                "message": f"目标抓取候选生成完成，用时 {candidate_seconds:.3f} s，共 {len(actions)} 个",
                "elapsed_s": candidate_seconds,
                "total_candidate_count": len(actions),
            })
        target_query = int(encoded.output["encoded"].target_query_index[0].detach().cpu())
        return Prediction(
            action=generated,
            inference_seconds=time.perf_counter() - started,
            candidates=tuple(actions),
            timings={"model_encode_s": encoded_seconds,
                "candidate_generation_s": candidate_seconds, **head_timings},
            target_query=target_query,
            target_instance=(int(target) if int(target) >= 0 else None),
        )

    def _candidate_actions(self, generated, *, action_type: int) -> tuple[dict, ...]:
        tensors = generated["candidates"]
        valid_indices = torch.nonzero(tensors["valid"][0], as_tuple=False).flatten()
        actions = []
        for index_tensor in valid_indices:
            index = int(index_tensor)
            action = self.policy._action(tensors, index)
            if int(action.get("action_type", -1)) != action_type:
                continue
            if int(action.get("action_type", -1)) == 0:
                action["push_distance_m"] = PUSH_DISTANCE_M
            point_index = int(tensors["point_index"][0, index])
            if point_index >= 0:
                action["association_point_world"] = (
                    generated["encoded"].cpu_batch["model_inputs"]["xyz"][0, point_index]
                    .detach().cpu().numpy()
                )
            actions.append(action)
        return tuple(actions)

    def generate_push_rules(
        self, obstructions: tuple[int, ...] | int, *, adjacent_objects: tuple[int, ...] = (),
    ) -> tuple[dict, ...]:
        if self._encoded is None:
            raise RuntimeError("请先完成目标抓取预测")
        sensor = self._encoded.output["sensor"]
        condition = self._encoded.output["push_condition"]
        with torch.no_grad():
            condition.validate(sensor["xyz"].shape[1])
            actions = (
                self.model.push(sensor, condition, adjacent_objects=adjacent_objects)
                if adjacent_objects else self.model.push(sensor, condition)
            )
        wanted = {int(value) for value in (obstructions if isinstance(obstructions, (tuple, list, set)) else (obstructions,))}
        keep = torch.nonzero(torch.tensor(
            [int(value) in wanted for value in actions.object.tolist()],
            device=actions.object.device), as_tuple=False).flatten()
        self._push_actions = actions.select(keep)
        return tuple({
            "candidate_index": index,
            "action_type": 0,
            "acted_object": int(self._push_actions.object[index]),
            "push_contact_world": self._push_actions.contact_world[index].detach().cpu().numpy(),
            "push_direction_world": self._push_actions.direction_world[index].detach().cpu().numpy(),
            "push_distance_m": float(self._push_actions.push_distance[index]),
        } for index in range(len(self._push_actions.object)))

    def score_push_rules(self) -> tuple[dict, ...]:
        if self._encoded is None or self._push_actions is None:
            raise RuntimeError("请先生成PUSH规则候选")
        if not len(self._push_actions.object):
            return ()
        sensor = self._encoded.output["sensor"]
        condition = self._encoded.output["push_condition"]
        with torch.no_grad():
            scored = self.model.score_push_actions(sensor, condition, self._push_actions)
            self._encoded.output["push"] = {"actions": self._push_actions, **scored}
            saved_actions = self.policy.preparation_actions
            self.policy.preparation_actions = 0
            try:
                generated = self.policy.generate_candidates(self._encoded)
            finally:
                self.policy.preparation_actions = saved_actions
        return self._candidate_actions(generated, action_type=0)

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
