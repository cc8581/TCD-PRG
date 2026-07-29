"""Sample the exact AG-160-95 collision surface in the URDF TCP frame.

This Python 3.8 worker runs in the existing ``gapg`` environment and never
imports the training package.
"""

from __future__ import print_function

import argparse
from pathlib import Path

import numpy as np
import pybullet as pb
import trimesh


AG_LINKS = {
    "ag95_base_link", "ag95_body", "left_outer_knuckle", "left_inner_knuckle",
    "left_finger", "left_finger_pad", "right_outer_knuckle", "right_inner_knuckle",
    "right_finger", "right_finger_pad",
}
COUPLED_JOINTS = {
    "left_outer_knuckle_joint", "left_finger_joint", "left_inner_knuckle_joint",
    "right_outer_knuckle_joint", "right_finger_joint", "right_inner_knuckle_joint",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    widths = parser.add_mutually_exclusive_group(required=True)
    widths.add_argument("--width-m", type=float)
    widths.add_argument("--widths-npz")
    parser.add_argument("--point-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def matrix(position, quaternion):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(pb.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def main():
    args = parse_args()
    if args.widths_npz:
        with np.load(args.widths_npz, allow_pickle=False) as request:
            widths_m = request["widths_m"].astype(np.float64)
    else:
        widths_m = np.asarray([args.width_m], dtype=np.float64)
    if np.any((widths_m < 0.0) | (widths_m > 0.095)):
        raise ValueError("AG total opening must be in [0, 0.095] m")
    if args.point_count <= 0:
        raise ValueError("point-count must be positive")
    client = pb.connect(pb.DIRECT)
    try:
        body = pb.loadURDF(
            str(Path(args.urdf).resolve()), useFixedBase=True,
            flags=pb.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client,
        )
        joint_by_name, link_by_name = {}, {}
        for index in range(pb.getNumJoints(body, physicsClientId=client)):
            info = pb.getJointInfo(body, index, physicsClientId=client)
            joint_by_name[info[1].decode("utf-8")] = index
            link_by_name[info[12].decode("utf-8")] = index
        missing = (AG_LINKS | {"tcp_link"}) - set(link_by_name)
        if missing:
            raise RuntimeError("URDF is missing required AG links: %s" % sorted(missing))
        mesh_cache = {}
        all_points = []
        for width_m in widths_m:
            joint_value = ((0.095 - width_m) / 0.095) * 0.93
            for name in COUPLED_JOINTS:
                pb.resetJointState(
                    body, joint_by_name[name], joint_value, physicsClientId=client
                )
            pb.performCollisionDetection(physicsClientId=client)
            tcp_state = pb.getLinkState(
                body, link_by_name["tcp_link"], computeForwardKinematics=1,
                physicsClientId=client
            )
            tcp_from_world = np.linalg.inv(matrix(tcp_state[4], tcp_state[5]))
            meshes = []
            for link_name in sorted(AG_LINKS):
                link = link_by_name[link_name]
                state = pb.getLinkState(
                    body, link, computeForwardKinematics=1, physicsClientId=client
                )
                world_from_link = matrix(state[4], state[5])
                for shape in pb.getCollisionShapeData(body, link, physicsClientId=client):
                    if int(shape[2]) != int(pb.GEOM_MESH):
                        continue
                    filename = shape[4]
                    if isinstance(filename, bytes):
                        filename = filename.decode("utf-8")
                    scale_key = tuple(float(value) for value in shape[3])
                    key = (filename, scale_key)
                    if key not in mesh_cache:
                        source = trimesh.load_mesh(filename, force="mesh", process=False)
                        if isinstance(source, trimesh.Scene):
                            source = source.dump(concatenate=True)
                        source = source.copy()
                        source.vertices *= np.asarray(scale_key, dtype=np.float64)[None]
                        mesh_cache[key] = source
                    mesh = mesh_cache[key].copy()
                    link_from_mesh = matrix(shape[5], shape[6])
                    mesh.apply_transform(
                        tcp_from_world.dot(world_from_link).dot(link_from_mesh)
                    )
                    meshes.append(mesh)
            if not meshes:
                raise RuntimeError("No AG collision meshes were found")
            combined = trimesh.util.concatenate(meshes)
            np.random.seed(args.seed + int(round(float(width_m) * 10000000.0)))
            points, _ = trimesh.sample.sample_surface(combined, args.point_count)
            all_points.append(np.asarray(points, dtype=np.float32))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary, points_tcp=np.stack(all_points),
            widths_m=np.asarray(widths_m, dtype=np.float32),
            source_urdf=np.asarray(str(Path(args.urdf).resolve())),
        )
        temporary.replace(output)
    finally:
        pb.disconnect(client)


if __name__ == "__main__":
    main()
