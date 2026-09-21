"""Inference-only geometric PUSH candidates, in the observed table/world frame."""
import math

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError, cKDTree
from torch import nn

from tcd_prg.constants import PUSH_DISTANCE_M

from .actions import PushActions


def polygon(points):
    xy = np.unique(np.rint(points[:, :2] / .0015).astype(np.int64), axis=0) * .0015
    if len(xy) < 3:
        return None
    try:
        return xy[ConvexHull(xy).vertices]
    except QhullError:
        return None


def projection_overlap(first, second):
    """Intersect CCW convex footprints, including containment and edge crossings."""
    output = first.copy()
    for start, end in zip(second, np.roll(second, -1, axis=0), strict=True):
        if not len(output):
            break
        edge = end - start
        clipped = []
        previous = output[-1]
        previous_distance = edge[0] * (previous[1] - start[1]) - edge[1] * (previous[0] - start[0])
        for current in output:
            distance = edge[0] * (current[1] - start[1]) - edge[1] * (current[0] - start[0])
            if (distance >= 0) != (previous_distance >= 0):
                clipped.append(previous + (current - previous) *
                               (previous_distance / (previous_distance - distance)))
            if distance >= 0:
                clipped.append(current)
            previous, previous_distance = current, distance
        output = np.asarray(clipped, dtype=float).reshape(-1, 2)
    return output


def observed_contact_boundary(section):
    """Measured exposed points and their local inward normals; never hull edges."""
    ordered = np.lexsort((section[:, 2], section[:, 1], section[:, 0]))
    keys = np.rint(section[ordered, :2] / .0015).astype(np.int64)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    samples = section[ordered[first + counts // 2]]
    if len(samples) < 3:
        return ()
    tree = cKDTree(samples[:, :2])
    nearest = tree.query(samples[:, :2], k=2)[0][:, 1]
    positive = nearest[nearest > 1e-6]
    if not len(positive):
        return ()
    radius = max(.004, 3.0 * float(np.median(positive)))
    exposed = []
    neighborhoods = tree.query_ball_point(samples[:, :2], radius)
    for point, neighbors in zip(samples, neighborhoods, strict=True):
        offset = samples[neighbors, :2] - point[:2]
        offset = offset[np.linalg.norm(offset, axis=1) > 1e-6]
        if len(offset) < 2:
            continue
        angles = np.sort(np.arctan2(offset[:, 1], offset[:, 0]))
        wrap = np.r_[angles, angles[0] + 2.0 * math.pi]
        gaps = np.diff(wrap)
        normals = []
        for index in np.flatnonzero(gaps >= math.radians(120.0)):
            empty_angle = wrap[index] + gaps[index] * .5
            normals.append(-np.array([math.cos(empty_angle), math.sin(empty_angle)]))
        if normals:
            exposed.append((point, normals))
    return tuple(exposed)


class RulePushGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.push_contact_spacing_m <= 0:
            raise ValueError("PUSH spacing must be positive")
        self.config = config

    @torch.no_grad()
    def forward(self, sensor, condition, *, adjacent_objects=()):
        xyz = sensor["xyz"]
        condition.validate(xyz.shape[1])
        result = []
        spacing = self.config.push_contact_spacing_m
        adjacent = {int(value) for value in adjacent_objects}
        for b in range(len(xyz)):
            if not bool(condition.target_valid[b]):
                continue
            valid = sensor["point_mask"][b].bool()
            cloud = xyz[b].detach().cpu().numpy()
            target_mask = valid & (condition.target_probability[b] >= .5)
            target = cloud[target_mask.cpu().numpy()]
            if len(target) < 3:
                continue
            footprint = polygon(target)
            if footprint is None:
                continue
            target_center = (target.min(0) + target.max(0)) * .5
            probabilities = condition.object_probability[b]
            owner = probabilities.argmax(0)
            for obj in torch.nonzero(condition.object_valid[b], as_tuple=False).flatten().tolist():
                member = valid & (owner == obj) & (probabilities[obj] >= .5)
                if bool((member & target_mask).sum() > member.sum() * .5):
                    continue
                points = cloud[member.cpu().numpy()]
                if len(points) < 8:
                    continue
                # Convex footprints are only a coarse overlap prefilter. No
                # contact is ever interpolated on their synthetic edges.
                object_footprint = polygon(points)
                if object_footprint is None:
                    continue
                if obj not in adjacent:
                    overlap = projection_overlap(object_footprint, footprint)
                    if not len(overlap):
                        continue
                    # Overhead blockers require measured local vertical evidence.
                    probes = np.concatenate((overlap, overlap.mean(0, keepdims=True)))
                    target_near = np.linalg.norm(target[:, None, :2] - probes[None], axis=2).argmin(0)
                    object_near = np.linalg.norm(points[:, None, :2] - probes[None], axis=2).argmin(0)
                    above = np.any(
                        points[object_near, 2] > target[target_near, 2]
                        + self.config.push_above_margin_m
                    )
                    if not above:
                        continue
                low, high = points.min(0), points.max(0)
                center = (low + high) * .5  # observable geometric centre, never simulator COM
                height = max(float(high[2] - low[2]), 1e-6)
                z = low[2] + .5 * height
                section = points[np.abs(points[:, 2] - z) <= max(.004, .1 * height)]
                if len(section) < 8:
                    section = points[np.argsort(np.abs(points[:, 2] - z))[:min(128, len(points))]]
                boundary = observed_contact_boundary(section)
                if not boundary:
                    continue
                main = center[:2] - target_center[:2]
                norm = np.linalg.norm(main)
                if norm < 1e-8:
                    continue  # target-to-object direction is undefined
                main /= norm
                accepted = []
                for point, normals in boundary:
                    if np.dot(point[:2] - center[:2], main) > 0:
                        continue
                    if accepted and np.min(
                        np.linalg.norm(np.asarray(accepted) - point[:2], axis=1)
                    ) < spacing:
                        continue
                    inward = center[:2] - point[:2]
                    inward_norm = np.linalg.norm(inward)
                    if inward_norm < 1e-8:
                        continue
                    inward /= inward_norm
                    signed = math.atan2(main[0] * inward[1] - main[1] * inward[0],
                                        float(np.clip(main @ inward, -1., 1.)))
                    u = float(np.clip((abs(math.degrees(signed)) - 20.) / 50., 0., 1.))
                    weight = .15 + .60 * u * u * (3. - 2. * u)
                    angle = weight * signed
                    direction = [main[0] * math.cos(angle) - main[1] * math.sin(angle),
                                 main[0] * math.sin(angle) + main[1] * math.cos(angle), 0.]
                    # A push must enter the observed contact surface, not slide
                    # along it. cos(60 degrees) = 0.5 for its inward XY normal.
                    if max(float(np.dot(direction[:2], normal)) for normal in normals) < .5:
                        continue
                    accepted.append(point[:2])
                    result.append((b, obj, point.tolist(), direction))
        if not result:
            return PushActions.empty(xyz)
        return PushActions(
            torch.tensor([r[0] for r in result], dtype=torch.long, device=xyz.device),
            torch.tensor([r[1] for r in result], dtype=torch.long, device=xyz.device),
            xyz.new_tensor(np.asarray([r[2] for r in result])),
            xyz.new_tensor(np.asarray([r[3] for r in result])),
            xyz.new_full((len(result),), PUSH_DISTANCE_M),
        ).validate(len(xyz), condition.object_valid.shape[1])
