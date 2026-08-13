from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))

from tcd_prg.config import load_config
from tcd_prg.datasets.types import CameraParameters, SceneObservation
from tcd_prg.models import TCDPRGModel
from tcd_prg.planners import TCDPRGPolicy
from tcd_prg.runtime import create_gripper_provider
from .types import FusedScene, Prediction


def _quoted(name: str, value: str | Path) -> str:
    return f"{name}={json.dumps(str(value), ensure_ascii=False)}"


class TCDPRGPredictor:
    def __init__(self, app_config):
        section = app_config.raw["tcd_prg"]
        path_cfg = app_config.resolve(section["paths_config"])
        paths = __import__("yaml").safe_load(path_cfg.read_text(encoding="utf-8")) or {}
        overrides = [_quoted("dataset.root", paths["dataset_root"]),
                     _quoted("dataset.acronym_root", paths["acronym_root"]),
                     _quoted("dataset.functional_region_root", paths["functional_region_root"]),
                     _quoted("observation.pybullet_python", paths["pybullet_python"])]
        if paths.get("observation_cache_dir"):
            overrides.append(_quoted("cache.directory", paths["observation_cache_dir"]))
        self.config = load_config(str(app_config.resolve(section["config"])), overrides)
        requested = str(section.get("device", "cuda"))
        self.device = torch.device(requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.model = TCDPRGModel(self.config.model, self.config.ablation, self.config.graph,
                                 self.config.router, self.config.backbone).to(self.device)
        checkpoint_path = app_config.resolve(section["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
        gripper = (create_gripper_provider(self.config, bool(section.get("allow_gripper_generate", False)))
                   if self.config.ablation.use_gripper_scene_verifier else None)
        self.policy = TCDPRGPolicy(self.model, self.config, gripper, certifier=None)

    def reset(self) -> None: self.policy.reset()

    def build_observation(self, scene: FusedScene, target: int, category: int,
                          region: int, required: int) -> SceneObservation:
        object_count = max(scene.instance_ids, default=-1) + 1
        poses = np.zeros((object_count, 7), np.float32); poses[:, 6] = 1
        for instance in scene.instance_ids:
            poses[instance, :3] = scene.xyz_m[scene.instance_id == instance].mean(0)
        categories = np.zeros(object_count, np.int64)
        for instance, value in scene.category_by_instance.items():
            if 0 <= instance < object_count: categories[instance] = int(value)
        categories[target] = int(category)
        camera_count = max(1, int(scene.source_view.max(initial=0))+1)
        cameras = tuple(CameraParameters(f"real_camera_{i}", 1, 1,
            np.zeros(3,np.float32),np.zeros(3,np.float32),np.array([0,1,0],np.float32),
            1,1,0,0,.1,2.) for i in range(camera_count))
        observation = SceneObservation(
            scene_id=-1, state_id=-1, task_index=-1, xyz=scene.xyz_m.astype(np.float32),
            rgb=scene.rgb.astype(np.float32), instance_id=scene.instance_id.astype(np.int64),
            target_mask=scene.instance_id == target, target_object=target,
            task_region_id=int(region), object_uuid=tuple(f"real_{i}" for i in range(object_count)),
            object_pose=poses, object_category_id=categories,
            object_present=np.ones(object_count,bool), object_active=np.ones(object_count,bool),
            camera_parameters=cameras, source_view=scene.source_view,
            metadata={"quaternion_order":"xyzw", "length_unit":"m", "oracle_excluded":True,
                      "required_grasp_count":int(required), "real_experiment":True})
        observation.validate()
        return observation

    def predict(self, scene: FusedScene, target: int, category: int,
                region: int, required: int = 1) -> Prediction:
        started = time.perf_counter()
        observation = self.build_observation(scene,target,category,region,required)
        encoded = self.policy.encode_observation(observation)
        candidates = self.policy.generate_candidates(encoded)
        action = self.policy.select_action(candidates)
        if action is None: raise RuntimeError("Model produced no valid action")
        return Prediction(action, time.perf_counter()-started)

    def action_executed(self, prediction: Prediction, scene: FusedScene, target: int,
                        category: int, region: int) -> None:
        observation = self.build_observation(scene,target,category,region,1)
        self.policy.update_after_action(prediction.action, observation)

