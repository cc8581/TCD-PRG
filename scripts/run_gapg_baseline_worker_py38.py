"""Execute the original GAPG policy in an isolated Python 3.8 process.

This file intentionally contains only glue code.  GAPG, graspnet-baseline and
graspnetAPI remain external dependencies pinned by ``third_party.lock.yaml``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gapg-root", required=True)
    parser.add_argument("--graspnet-root", required=True)
    parser.add_argument("--graspnet-api-root", required=True)
    parser.add_argument("--grasp-checkpoint", required=True)
    parser.add_argument("--push-checkpoint", required=True)
    parser.add_argument("--graspnet-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grasp-threshold", type=float, default=0.75)
    parser.add_argument(
        "--mode", choices=("policy", "global_scene", "global_instance"), default="policy"
    )
    return parser.parse_args()


def configure_imports(args):
    gapg = Path(args.gapg_root).resolve()
    graspnet = Path(args.graspnet_root).resolve()
    api = Path(args.graspnet_api_root).resolve()
    # Load GAPG's ``models`` package normally; GraspNet source modules use bare
    # imports and are therefore exposed from their exact upstream directories.
    sys.path.insert(0, str(gapg))
    for path in (api, graspnet / "models", graspnet / "dataset",
                 graspnet / "utils", graspnet / "pointnet2", graspnet):
        sys.path.append(str(path))


def sample_indices(count, required, rng):
    if count <= 0:
        raise ValueError("Cannot sample an empty point cloud")
    if count >= required:
        return rng.choice(count, required, replace=False)
    return np.concatenate((np.arange(count), rng.choice(count, required - count, replace=True)))


def load_networks(args):
    import torch
    from graspnet import GraspNet, pred_decode
    from models.grasp_networks import Space_GraspFusion
    from models.push_networks import Push_model

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    grasp = Space_GraspFusion(device=str(device)).to(device)
    grasp_state = torch.load(args.grasp_checkpoint, map_location=device)
    grasp.load_state_dict(grasp_state["model"])
    grasp.eval()
    push = Push_model(additional_channel=3).to(device)
    push_state = torch.load(args.push_checkpoint, map_location=device)
    push.load_state_dict(push_state["model"])
    push.eval()
    proposal = GraspNet(
        input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).to(device)
    proposal_state = torch.load(args.graspnet_checkpoint, map_location=device)
    proposal.load_state_dict(proposal_state["model_state_dict"])
    proposal.eval()
    return device, grasp, push, proposal, pred_decode


def propose_grasps(points, scene, rng, device, proposal, pred_decode):
    import torch
    from collision_detector import ModelFreeCollisionDetector
    from graspnetAPI import GraspGroup

    # Preserve the coordinate convention in GAPG generate_grasp.py.
    input_points = -points
    indices = sample_indices(len(input_points), 20000, rng)
    endpoints = {"point_clouds": torch.from_numpy(
        input_points[indices][None].astype(np.float32)
    ).to(device)}
    with torch.no_grad():
        decoded = pred_decode(proposal(endpoints))[0].detach().cpu().numpy()
    group = GraspGroup(decoded)
    group.translations = -group.translations
    group.rotation_matrices = -group.rotation_matrices
    group.translations = group.translations + group.rotation_matrices[:, :, 0] * 0.05
    collision = ModelFreeCollisionDetector(scene, voxel_size=0.01).detect(
        group, approach_dist=0.05, collision_thresh=0.01
    )
    group = group[~collision]
    group.sort_by_score()
    return group[:20]


def evaluate_grasps(scene, groups, verifier, device):
    import torch
    import torch.nn.functional as functional
    from scipy.spatial.transform import Rotation
    import utils

    gripper_points, _ = utils.grasp_pcd_bluenoise_like(
        n_target=170, oversample=2000, seed=55926
    )
    gripper_points = gripper_points.to(device)
    inputs, metadata = [], []
    scene_tensor = torch.from_numpy(scene).float()
    for group in groups:
        for grasp in group:
            rotation = grasp.rotation_matrix.copy()
            _, rotation = utils.adjust_pose_z_axis_to_down(rotation)
            translation = grasp.translation.copy()
            translation[2] -= 0.01
            pose = torch.from_numpy(np.hstack((translation, Rotation.from_matrix(rotation).as_quat()))).float()
            local = utils.fuse_state_torch_v2(
                utils.TransformPCD2EndLink(scene_tensor, pose), gripper_points
            )
            if len(local) == 0:
                continue
            local = utils.furthest_point_sampling_nocuda(local, n_samples=175).to(device)
            fused = torch.cat((local, gripper_points), 0)
            fused, _, _ = utils.pc_normalize_grasp(fused)
            labels = torch.cat((torch.ones((len(local), 1), device=device),
                                torch.zeros((len(gripper_points), 1), device=device)), 0)
            inputs.append(torch.cat((fused, labels), 1).T.float())
            metadata.append((translation, rotation, float(getattr(grasp, "width", 0.085))))
    if not inputs:
        return []
    with torch.no_grad():
        probability = functional.softmax(verifier(torch.stack(inputs)), dim=1)[:, 1]
    order = probability.argsort(descending=True).detach().cpu().numpy()
    results = []
    for index in order:
        translation, rotation, width = metadata[int(index)]
        quaternion = Rotation.from_matrix(rotation).as_quat()
        results.append({
            "action_type": 2,
            "acted_object": -1,
            "grasp_pose_world": np.hstack((translation, quaternion)).tolist(),
            "grasp_width_m": width,
            "score": float(probability[int(index)]),
        })
    return results


def push_candidates(scene, instance, active, target, push_model, device):
    import open3d as o3d
    import torch
    import utils
    from scipy.spatial.transform import Rotation

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(scene[instance == target])
    object_clouds = []
    for object_index in np.flatnonzero(active):
        points = scene[instance == object_index]
        if len(points) < 8:
            continue
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        if object_index == target or utils.any_point_in_expanded_obb(target_cloud, cloud):
            object_clouds.append(cloud)
    if not object_clouds:
        return []
    try:
        matrices = utils.sample_push_action(scene, object_clouds)
    except (ValueError, RuntimeError):
        return []
    if matrices is None or not len(matrices):
        return []
    poses = np.stack([
        np.hstack((matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()))
        for matrix in matrices
    ]).astype(np.float32)
    target_label = (instance == target).astype(np.float32)[:, None]
    labelled = np.concatenate((scene, target_label, 1.0 - target_label), axis=1)
    fixed = torch.tensor([0.5, 0.0], dtype=torch.float32, device=device)
    global_points = torch.from_numpy(labelled).float().to(device)
    inputs = []
    with torch.no_grad():
        for pose in torch.from_numpy(poses).to(device):
            transformed = utils.Transform_Push2Fixed_point_onehot(global_points, fixed, pose)
            sampled = utils.furthest_point_sampling_onehot_p3d(transformed, n_samples=1024)
            sampled = torch.cat((sampled, torch.zeros((len(sampled), 1), device=device)), 1)
            seed = torch.cat((fixed, pose[2:3], torch.tensor([0.0, 0.0, 1.0], device=device)))
            inputs.append(torch.cat((sampled, seed[None]), 0).T.float())
        scores = push_model(torch.stack(inputs)).squeeze(-1)[:, -1]
    order = scores.argsort(descending=True).detach().cpu().numpy()
    results = []
    for index in order:
        matrix = matrices[int(index)]
        direction = matrix[:3, 0]
        contact = matrix[:3, 3]
        nearest = int(instance[np.linalg.norm(scene - contact[None], axis=1).argmin()])
        results.append({
            "action_type": 0, "acted_object": nearest,
            "push_contact_world": contact.tolist(),
            "push_direction_world": direction.tolist(),
            "push_distance_m": 0.15,
            "score": float(scores[int(index)]),
        })
    return results


def main():
    args = arguments()
    configure_imports(args)
    import torch

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = np.load(args.input, allow_pickle=False)
    valid = data["point_valid"].astype(bool)
    scene = data["xyz"].astype(np.float32)[valid]
    instance = data["instance_id"].astype(np.int32)[valid]
    active = data["object_active"].astype(bool)
    target = int(data["target_object"])
    device, verifier, push_model, proposal, decoder = load_networks(args)
    if args.mode != "policy":
        groups = []
        if args.mode == "global_scene":
            groups.append(propose_grasps(
                scene, scene, np.random.RandomState(args.seed), device, proposal, decoder
            ))
        else:
            for object_index in np.flatnonzero(active):
                points = scene[instance == object_index]
                if len(points):
                    groups.append(propose_grasps(
                        points, scene, np.random.RandomState(args.seed + int(object_index)),
                        device, proposal, decoder,
                    ))
        candidates = evaluate_grasps(scene, groups, verifier, device)
        for item in candidates:
            contact = np.asarray(item["grasp_pose_world"][:3])
            item["acted_object"] = int(instance[np.sum((scene - contact[None]) ** 2, axis=1).argmin()])
        Path(args.output).write_text(json.dumps({
            "selected_action": None, "candidates": candidates,
            "backend": "original_gapg", "mode": args.mode, "seed": args.seed,
        }), encoding="utf-8")
        return
    target_points = scene[instance == target]
    candidates = []
    if len(target_points):
        groups = [propose_grasps(target_points, scene, np.random.RandomState(args.seed + i),
                                 device, proposal, decoder) for i in range(2)]
        candidates = evaluate_grasps(scene, groups, verifier, device)
        for item in candidates:
            item["acted_object"] = target
    executable = [item for item in candidates if item["score"] >= args.grasp_threshold]
    if executable:
        selected = executable[0]
    else:
        pushes = push_candidates(scene, instance, active, target, push_model, device)
        candidates.extend(pushes)
        selected = pushes[0] if pushes else None
    Path(args.output).write_text(json.dumps({
        "selected_action": selected, "candidates": candidates,
        "backend": "original_gapg", "seed": args.seed,
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
