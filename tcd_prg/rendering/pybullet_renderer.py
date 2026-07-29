"""Deterministic three-view Mech-Eye PRO S PyBullet state renderer."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from tcd_prg.observation.base import ObservationRequest, PointObservation
from tcd_prg.observation.saved import deterministic_stratified_sample, reconstruct_pinhole_world


class PyBulletMechEyeRenderer:
    """Rebuild intermediate states from real ACRONYM meshes and saved poses."""

    def __init__(
        self,
        scene_root: str | Path,
        acronym_root: str | Path,
        width: int = 320,
        height: int = 200,
        renderer_version: str = "tcd_prg_pybullet_v1",
    ) -> None:
        self.scene_root = Path(scene_root)
        self.acronym_root = Path(acronym_root)
        self.width, self.height = width, height
        self.renderer_version = renderer_version
        self.metadata = json.loads((self.scene_root / "metadata.json").read_text(encoding="utf-8"))
        self.cameras = [c for c in self.metadata["camera_parameters"] if c["sensor_type"] != "oracle"]
        if len(self.cameras) != 3:
            raise ValueError("Renderer requires exactly three PRO S cameras")

    def _asset(self, h5_name: str) -> tuple[Path, float]:
        grasp_file = self.acronym_root / "grasps" / h5_name
        if not grasp_file.exists():
            raise FileNotFoundError(grasp_file)
        with h5py.File(grasp_file, "r") as handle:
            mesh = handle["object/file"][()]
            mesh = mesh.decode("utf-8") if isinstance(mesh, bytes) else str(mesh)
            scale = float(handle["object/scale"][()])
        path = self.acronym_root / mesh
        if not path.exists():
            raise FileNotFoundError(path)
        return path, scale

    @staticmethod
    def _projection(camera: dict, width: int, height: int, source_width: int, source_height: int) -> list[float]:
        fx = float(camera["fx"]) * width / source_width
        fy = float(camera["fy"]) * height / source_height
        cx = (float(camera["cx"]) + 0.5) * width / source_width - 0.5
        cy = (float(camera["cy"]) + 0.5) * height / source_height - 0.5
        near, far = float(camera["z_near"]), float(camera["z_far"])
        matrix = np.array(
            [
                [2 * fx / width, 0, 1 - 2 * cx / width, 0],
                [0, 2 * fy / height, 2 * cy / height - 1, 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )
        return matrix.T.reshape(-1).tolist()

    def render(self, request: ObservationRequest) -> PointObservation:
        if request.renderer_version != self.renderer_version:
            raise ValueError("Renderer version mismatch would invalidate cache determinism")
        try:
            import pybullet as p
        except ImportError as error:
            raise RuntimeError("Install tcd-prg[render] before reconstructing intermediate states") from error
        if len(request.object_asset_ids) != len(request.object_pose):
            raise ValueError("Asset catalog and object poses disagree")
        client = p.connect(p.DIRECT)
        try:
            p.resetSimulation(physicsClientId=client)
            body_to_object: dict[int, int] = {}
            for index, (asset_id, pose, present) in enumerate(
                zip(request.object_asset_ids, request.object_pose, request.object_present, strict=True)
            ):
                if not present:
                    continue
                mesh, scale = self._asset(asset_id)
                collision = p.createCollisionShape(
                    p.GEOM_MESH, fileName=str(mesh), meshScale=[scale] * 3, physicsClientId=client
                )
                visual = p.createVisualShape(
                    p.GEOM_MESH, fileName=str(mesh), meshScale=[scale] * 3, physicsClientId=client
                )
                body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=pose[:3].tolist(),
                    baseOrientation=pose[3:].tolist(),
                    physicsClientId=client,
                )
                body_to_object[body] = index
            raw_path = self.scene_root / f"scene_{request.scene_id:04d}" / "scene.npz"
            with np.load(raw_path, allow_pickle=False) as raw:
                for present, pose, size, color in zip(
                    raw["thin_support_block_present"], raw["thin_support_block_pose"],
                    raw["thin_support_block_size"], raw["thin_support_block_rgba"], strict=True
                ):
                    if not present:
                        continue
                    half = (size / 2).tolist()
                    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client)
                    visual = p.createVisualShape(
                        p.GEOM_BOX, halfExtents=half, rgbaColor=color.tolist(), physicsClientId=client
                    )
                    p.createMultiBody(0, collision, visual, pose[:3].tolist(), pose[3:].tolist(), physicsClientId=client)
            source_width, source_height = self.metadata["image_size"]
            observations = []
            for view_index, camera in enumerate(self.cameras):
                view = p.computeViewMatrix(camera["eye"], camera["target"], camera["up"])
                projection = self._projection(camera, self.width, self.height, source_width, source_height)
                _, _, rgba, depth_buffer, segmentation = p.getCameraImage(
                    self.width, self.height, viewMatrix=view, projectionMatrix=projection,
                    renderer=p.ER_TINY_RENDERER, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                    physicsClientId=client,
                )
                rgba = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
                depth_buffer = np.asarray(depth_buffer, dtype=np.float32).reshape(self.height, self.width)
                segmentation = np.asarray(segmentation, dtype=np.int64).reshape(self.height, self.width)
                body_id = segmentation & ((1 << 24) - 1)
                instance = np.full_like(body_id, -1, dtype=np.int16)
                for body, object_index in body_to_object.items():
                    instance[body_id == body] = object_index
                near, far = float(camera["z_near"]), float(camera["z_far"])
                depth = far * near / (far - (far - near) * depth_buffer)
                scaled_camera = dict(camera)
                scaled_camera.update(
                    fx=float(camera["fx"]) * self.width / source_width,
                    fy=float(camera["fy"]) * self.height / source_height,
                    cx=(float(camera["cx"]) + 0.5) * self.width / source_width - 0.5,
                    cy=(float(camera["cy"]) + 0.5) * self.height / source_height - 0.5,
                )
                observations.append(
                    reconstruct_pinhole_world(depth, rgba[..., :3], instance, scaled_camera, view_index)
                )
            union = PointObservation(
                np.concatenate([x.xyz for x in observations]), np.concatenate([x.rgb for x in observations]),
                np.concatenate([x.instance_id for x in observations]),
                np.concatenate([x.source_view for x in observations]),
            )
            return deterministic_stratified_sample(union, request.point_count, request.render_seed)
        finally:
            p.disconnect(client)

