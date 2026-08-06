"""Persistent Python 3.8 PyBullet worker for TCD-PRG cache precomputation.

The worker keeps one PyBullet process alive and accepts JSONL commands on
stdin.  Geometry projection and deterministic sampling are imported from the
repository's canonical single-request worker so the observation contract does
not diverge.
"""

from __future__ import print_function

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pybullet as pb

from render_observation_worker_py38 import (
    asset,
    deterministic_sample,
    project_world,
    projection,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--runtime-mesh-root", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser.parse_args()


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class SceneContext(object):
    """Keep one scene's meshes/bodies alive across many state renders."""

    def __init__(self, client, scene_root, runtime_mesh_root):
        self.client = client
        self.scene_root = scene_root
        self.runtime_mesh_root = runtime_mesh_root
        self.scene_id = None
        self.model_ids = ()
        self.scales = ()
        self.object_bodies = []
        self.body_to_object = {}
        self.scene_reload_count = 0

    @staticmethod
    def _park_position(index):
        # Far outside every formal camera frustum. Keep parked objects separated.
        return [1000.0 + 2.0 * float(index), 1000.0, -1000.0]

    def ensure_scene(self, scene_id, model_ids, scales):
        model_signature = tuple(str(value) for value in model_ids)
        scale_signature = tuple(float(value) for value in scales)
        if (
            self.scene_id == int(scene_id)
            and self.model_ids == model_signature
            and self.scales == scale_signature
        ):
            return False

        pb.resetSimulation(physicsClientId=self.client)
        self.scene_id = int(scene_id)
        self.model_ids = model_signature
        self.scales = scale_signature
        self.object_bodies = []
        self.body_to_object = {}

        # All scene objects are created once. Per-state active masks only move
        # bodies between their labelled pose and an off-camera parking pose.
        for index, (model_id, scale) in enumerate(
            zip(self.model_ids, self.scales)
        ):
            collision_mesh, visual_mesh, resolved_scale = asset(
                self.runtime_mesh_root, model_id, scale
            )
            collision = pb.createCollisionShape(
                pb.GEOM_MESH,
                fileName=str(collision_mesh),
                meshScale=[resolved_scale] * 3,
                physicsClientId=self.client,
            )
            visual = pb.createVisualShape(
                pb.GEOM_MESH,
                fileName=str(visual_mesh),
                meshScale=[resolved_scale] * 3,
                physicsClientId=self.client,
            )
            body = pb.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=self._park_position(index),
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client,
            )
            self.object_bodies.append(body)
            self.body_to_object[body] = index

        with np.load(
            str(
                self.scene_root
                / ("scene_%04d" % self.scene_id)
                / "scene.npz"
            ),
            allow_pickle=False,
        ) as raw:
            for is_present, pose, size, color in zip(
                raw["thin_support_block_present"],
                raw["thin_support_block_pose"],
                raw["thin_support_block_size"],
                raw["thin_support_block_rgba"],
            ):
                if not bool(is_present):
                    continue
                half = (size / 2).tolist()
                collision = pb.createCollisionShape(
                    pb.GEOM_BOX,
                    halfExtents=half,
                    physicsClientId=self.client,
                )
                visual = pb.createVisualShape(
                    pb.GEOM_BOX,
                    halfExtents=half,
                    rgbaColor=color.tolist(),
                    physicsClientId=self.client,
                )
                pb.createMultiBody(
                    0,
                    collision,
                    visual,
                    pose[:3].tolist(),
                    pose[3:].tolist(),
                    physicsClientId=self.client,
                )
        self.scene_reload_count += 1
        return True

    def apply_state(self, poses, physical_active):
        if len(poses) != len(self.object_bodies):
            raise ValueError(
                "request object count differs from loaded scene: %d != %d"
                % (len(poses), len(self.object_bodies))
            )
        for index, (body, pose, is_active) in enumerate(
            zip(self.object_bodies, poses, physical_active)
        ):
            if bool(is_active):
                position = pose[:3].tolist()
                orientation = pose[3:].tolist()
            else:
                position = self._park_position(index)
                orientation = [0.0, 0.0, 0.0, 1.0]
            pb.resetBasePositionAndOrientation(
                body,
                position,
                orientation,
                physicsClientId=self.client,
            )


def render_request(
    context,
    request_path,
    output_path,
    metadata,
    cameras,
    width,
    height,
):
    with np.load(str(request_path), allow_pickle=False) as request:
        scene_id = int(request["scene_id"])
        poses = request["object_pose"].astype(np.float64)
        present = request["object_present"].astype(bool)
        active = request["object_active"].astype(bool)
        physical_active = present & active
        model_ids = request["object_model_ids"].astype(str)
        scales = request["object_scales"].astype(np.float64)
        point_count = int(request["point_count"])
        render_seed = int(request["render_seed"])

    scene_reloaded = context.ensure_scene(scene_id, model_ids, scales)
    context.apply_state(poses, physical_active)

    source_width, source_height = metadata["image_size"]
    outputs = []
    for view_index, camera in enumerate(cameras):
        view = pb.computeViewMatrix(camera["eye"], camera["target"], camera["up"])
        proj, fx, fy, cx, cy = projection(
            camera, width, height, source_width, source_height
        )
        _, _, rgba, depth_buffer, segmentation = pb.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=pb.ER_TINY_RENDERER,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=context.client,
        )
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
        depth_buffer = np.asarray(depth_buffer, dtype=np.float32).reshape(
            height, width
        )
        segmentation = np.asarray(segmentation, dtype=np.int64).reshape(
            height, width
        )
        body_id = segmentation & ((1 << 24) - 1)
        instance = np.full(body_id.shape, -1, dtype=np.int16)
        for body, object_index in context.body_to_object.items():
            instance[body_id == body] = object_index
        near, far = float(camera["z_near"]), float(camera["z_far"])
        depth = far * near / (far - (far - near) * depth_buffer)
        scaled = dict(camera)
        scaled.update(fx=fx, fy=fy, cx=cx, cy=cy)
        outputs.append(
            project_world(
                depth,
                rgba[..., :3],
                instance,
                scaled,
                view_index,
            )
        )

    xyz = np.concatenate([item[0] for item in outputs])
    rgb = np.concatenate([item[1] for item in outputs])
    instance = np.concatenate([item[2] for item in outputs])
    source_view = np.concatenate([item[3] for item in outputs])
    xyz, rgb, instance, source_view = deterministic_sample(
        xyz, rgb, instance, source_view, point_count, render_seed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(
        str(temporary),
        xyz=xyz,
        rgb=rgb,
        instance_id=instance,
        source_view=source_view,
    )
    temporary.replace(output_path)
    return (
        int(len(xyz)),
        int(output_path.stat().st_size),
        bool(scene_reloaded),
        int(context.scene_reload_count),
    )

def main():
    args = parse_args()
    scene_root = Path(args.scene_root)
    runtime_mesh_root = Path(args.runtime_mesh_root)
    metadata = json.loads(
        (scene_root / "metadata.json").read_text(encoding="utf-8")
    )
    cameras = [
        item
        for item in metadata["camera_parameters"]
        if item["sensor_type"].lower() != "oracle"
    ]
    if len(cameras) != 3:
        raise RuntimeError("exactly three non-Oracle cameras are required")

    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    context = SceneContext(client, scene_root, runtime_mesh_root)
    emit({"ready": True, "protocol": "tcd_prg_batch_renderer_v2_scene_reuse"})
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            command = json.loads(line)
            request_id = str(command.get("id", ""))
            if command.get("command") == "shutdown":
                emit({"id": request_id, "ok": True, "shutdown": True})
                return
            try:
                (
                    point_count,
                    size_bytes,
                    scene_reloaded,
                    scene_reload_count,
                ) = render_request(
                    context,
                    Path(command["request"]),
                    Path(command["output"]),
                    metadata,
                    cameras,
                    int(args.width),
                    int(args.height),
                )
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "point_count": point_count,
                        "size_bytes": size_bytes,
                        "scene_reloaded": scene_reloaded,
                        "scene_reload_count": scene_reload_count,
                    }
                )
            except BaseException as error:
                emit(
                    {
                        "id": request_id,
                        "ok": False,
                        "error": "%s: %s" % (type(error).__name__, error),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        pb.disconnect(client)


if __name__ == "__main__":
    main()
