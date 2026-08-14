from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tcd_prg.config import load_config
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
        self.model = TCDPRGModel(
            self.config.model,
            self.config.ablation,
            self.config.graph,
            self.config.router,
            self.config.backbone,
        ).to(self.device)
        checkpoint_path = app_config.resolve(section["checkpoint"])
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
        gripper = (
            create_gripper_provider(
                self.config, bool(section.get("allow_gripper_generate", False))
            )
            if self.config.ablation.use_gripper_scene_verifier
            else None
        )
        self.policy = TCDPRGPolicy(
            self.model, self.config, gripper, certifier=None
        )

    def reset(self) -> None:
        self.policy.reset()

    def perceive(self, scene: FusedScene) -> FusedScene:
        """Run the integrated class-agnostic instance head on fused XYZRGB."""
        result = self.policy.segment_fused_scene(scene.xyz_m, scene.rgb)
        scene.instance_id = np.asarray(result["instance_id"], np.int64)
        scene.category_by_instance = dict(result["category_by_instance"])
        return scene

    def predict(
        self,
        scene: FusedScene,
        target: int,
        category: int,
        region: int,
        required: int = 1,
    ) -> Prediction:
        started = time.perf_counter()
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
                int(required),
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
                int(required),
                continue_target=True,
                enforce_target_confidence=True,
            )
        candidates = self.policy.generate_candidates(encoded)
        action = self.policy.select_action(candidates)
        if action is None:
            raise RuntimeError("Model produced no valid action")
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
