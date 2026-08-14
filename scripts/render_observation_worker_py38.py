"""Python 3.8 compatible external PyBullet rendering worker.

This file intentionally does not import :mod:`tcd_prg`.  It runs in the
existing GAPG Conda environment and exchanges immutable NPZ requests/results
with the training environment.
"""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as pb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--runtime-mesh-root", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser.parse_args()


def camera_basis(eye, target, up):
    forward = target - eye
    forward /= np.sqrt(np.sum(forward * forward))
    right = np.cross(forward, up)
    right /= np.sqrt(np.sum(right * right))
    camera_up = np.cross(right, forward)
    camera_up /= np.sqrt(np.sum(camera_up * camera_up))
    return right, camera_up, forward


def project_world(depth, rgb, instance, camera, view_index):
    valid = (
        np.isfinite(depth)
        & (depth > float(camera["z_near"]))
        & (depth < float(camera["z_far"]))
    )
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float32)
    x = (u.astype(np.float32) - float(camera["cx"])) * z / float(camera["fx"])
    y = (v.astype(np.float32) - float(camera["cy"])) * z / float(camera["fy"])
    eye = np.asarray(camera["eye"], dtype=np.float32)
    right, camera_up, forward = camera_basis(
        eye,
        np.asarray(camera["target"], dtype=np.float32),
        np.asarray(camera["up"], dtype=np.float32),
    )
    xyz = eye + x[:, None] * right - y[:, None] * camera_up + z[:, None] * forward
    return (
        xyz.astype(np.float32),
        rgb[v, u].astype(np.float32) / 255.0,
        instance[v, u].astype(np.int64),
        np.full(len(v), view_index, dtype=np.int16),
    )


def deterministic_sample(xyz, rgb, instance, source_view, count, seed):
    """Sensor-only deterministic sampling; GT instance id never affects selection."""
    if not len(xyz):
        raise RuntimeError("render produced no valid sensor pixels")
    if count <= 0 or len(xyz) <= count:
        return xyz, rgb, instance, source_view
    rng = np.random.default_rng(int(seed))
    index = rng.choice(len(xyz), int(count), replace=False)
    rng.shuffle(index)
    return xyz[index], rgb[index], instance[index], source_view[index]


def asset(runtime_mesh_root, model_id, scale):
    """Resolve the generator's ASCII-only visual and VHACD meshes."""
    visual = runtime_mesh_root / ("model_%s.obj" % model_id)
    collision = runtime_mesh_root / ("vhacd_v3_%s.obj" % model_id)
    if not visual.is_file():
        visual = collision
    if not collision.is_file() or not visual.is_file():
        raise FileNotFoundError(
            "runtime mesh cache is incomplete for model_id=%s" % model_id
        )
    return collision, visual, float(scale)


def projection(camera, width, height, source_width, source_height):
    fx = float(camera["fx"]) * width / source_width
    fy = float(camera["fy"]) * height / source_height
    cx = (float(camera["cx"]) + 0.5) * width / source_width - 0.5
    cy = (float(camera["cy"]) + 0.5) * height / source_height - 0.5
    near, far = float(camera["z_near"]), float(camera["z_far"])
    matrix = np.asarray(
        [
            [2 * fx / width, 0, 1 - 2 * cx / width, 0],
            [0, 2 * fy / height, 2 * cy / height - 1, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
    return matrix.T.reshape(-1).tolist(), fx, fy, cx, cy


def main():
    args = parse_args()
    scene_root = Path(args.scene_root)
    runtime_mesh_root = Path(args.runtime_mesh_root)
    metadata = json.loads((scene_root / "metadata.json").read_text(encoding="utf-8"))
    cameras = [item for item in metadata["camera_parameters"] if item["sensor_type"].lower() != "oracle"]
    if len(cameras) != 3:
        raise RuntimeError("exactly three non-Oracle cameras are required")
    with np.load(args.request, allow_pickle=False) as request:
        scene_id = int(request["scene_id"])
        poses = request["object_pose"].astype(np.float64)
        present = request["object_present"].astype(bool)
        active = request["object_active"].astype(bool)
        physical_active = present & active
        model_ids = request["object_model_ids"].astype(str)
        scales = request["object_scales"].astype(np.float64)
        point_count = int(request["point_count"])
        render_seed = int(request["render_seed"])
        renderer_version = str(request["renderer_version"])
    if renderer_version == "tcd_prg_pybullet_v2_variable_grid":
        raise RuntimeError(
            "Legacy v2 observation caches are read-only; rendering new entries is prohibited"
        )
    client = pb.connect(pb.DIRECT)
    try:
        pb.resetSimulation(physicsClientId=client)
        body_to_object = {}
        for index, (model_id, scale, pose, is_active) in enumerate(
            zip(model_ids, scales, poses, physical_active)
        ):
            if not bool(is_active):
                continue
            collision_mesh, visual_mesh, scale = asset(runtime_mesh_root, model_id, scale)
            collision = pb.createCollisionShape(
                pb.GEOM_MESH,
                fileName=str(collision_mesh),
                meshScale=[scale] * 3,
                physicsClientId=client,
            )
            visual = pb.createVisualShape(
                pb.GEOM_MESH,
                fileName=str(visual_mesh),
                meshScale=[scale] * 3,
                physicsClientId=client,
            )
            body = pb.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=pose[:3].tolist(),
                baseOrientation=pose[3:].tolist(),
                physicsClientId=client,
            )
            body_to_object[body] = index
        with np.load(scene_root / ("scene_%04d" % scene_id) / "scene.npz", allow_pickle=False) as raw:
            for is_present, pose, size, color in zip(
                raw["thin_support_block_present"],
                raw["thin_support_block_pose"],
                raw["thin_support_block_size"],
                raw["thin_support_block_rgba"],
            ):
                if not bool(is_present):
                    continue
                half = (size / 2).tolist()
                collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=client)
                visual = pb.createVisualShape(
                    pb.GEOM_BOX, halfExtents=half, rgbaColor=color.tolist(), physicsClientId=client
                )
                pb.createMultiBody(
                    0, collision, visual, pose[:3].tolist(), pose[3:].tolist(), physicsClientId=client
                )
        source_width, source_height = metadata["image_size"]
        outputs = []
        for view_index, camera in enumerate(cameras):
            view = pb.computeViewMatrix(camera["eye"], camera["target"], camera["up"])
            proj, fx, fy, cx, cy = projection(
                camera, args.width, args.height, source_width, source_height
            )
            _, _, rgba, depth_buffer, segmentation = pb.getCameraImage(
                args.width,
                args.height,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=pb.ER_TINY_RENDERER,
                flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=client,
            )
            rgba = np.asarray(rgba, dtype=np.uint8).reshape(args.height, args.width, 4)
            depth_buffer = np.asarray(depth_buffer, dtype=np.float32).reshape(args.height, args.width)
            segmentation = np.asarray(segmentation, dtype=np.int64).reshape(args.height, args.width)
            body_id = segmentation & ((1 << 24) - 1)
            instance = np.full(body_id.shape, -1, dtype=np.int16)
            for body, object_index in body_to_object.items():
                instance[body_id == body] = object_index
            near, far = float(camera["z_near"]), float(camera["z_far"])
            depth = far * near / (far - (far - near) * depth_buffer)
            scaled = dict(camera)
            scaled.update(fx=fx, fy=fy, cx=cx, cy=cy)
            outputs.append(project_world(depth, rgba[..., :3], instance, scaled, view_index))
        xyz = np.concatenate([item[0] for item in outputs])
        rgb = np.concatenate([item[1] for item in outputs])
        instance = np.concatenate([item[2] for item in outputs])
        source_view = np.concatenate([item[3] for item in outputs])
        xyz, rgb, instance, source_view = deterministic_sample(
            xyz, rgb, instance, source_view, point_count, render_seed
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary, xyz=xyz, rgb=rgb, instance_id=instance, source_view=source_view
        )
        temporary.replace(output)
    finally:
        pb.disconnect(client)


if __name__ == "__main__":
    main()
