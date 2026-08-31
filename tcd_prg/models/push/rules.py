"""Inference-only geometric PUSH candidates, in the observed table/world frame."""
import math
import numpy as np
import torch
from torch import nn
from scipy.spatial import ConvexHull, QhullError
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
    for start, end in zip(second, np.roll(second, -1, axis=0)):
        if not len(output):
            break
        edge = end - start
        signed = lambda p: edge[0] * (p[1] - start[1]) - edge[1] * (p[0] - start[0])
        clipped = []
        previous = output[-1]
        previous_distance = signed(previous)
        for current in output:
            distance = signed(current)
            if (distance >= 0) != (previous_distance >= 0):
                clipped.append(previous + (current - previous) *
                               (previous_distance / (previous_distance - distance)))
            if distance >= 0:
                clipped.append(current)
            previous, previous_distance = current, distance
        output = np.asarray(clipped, dtype=float).reshape(-1, 2)
    return output


class RulePushGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.push_contact_spacing_m <= 0:
            raise ValueError("PUSH spacing must be positive")
        self.config = config

    @torch.no_grad()
    def forward(self, sensor, condition):
        xyz = sensor["xyz"]
        condition.validate(xyz.shape[1])
        result = []
        spacing = self.config.push_contact_spacing_m
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
                object_footprint = polygon(points)
                if object_footprint is None:
                    continue
                overlap = projection_overlap(object_footprint, footprint)
                if not len(overlap):
                    continue
                # Heights are observable nearest-surface estimates at shared XY
                # locations, not a requirement for sampled points inside the target.
                probes = np.concatenate((overlap, overlap.mean(0, keepdims=True)))
                target_near = np.linalg.norm(target[:, None, :2] - probes[None], axis=2).argmin(0)
                object_near = np.linalg.norm(points[:, None, :2] - probes[None], axis=2).argmin(0)
                above = np.any(points[object_near, 2] > target[target_near, 2] + self.config.push_above_margin_m)
                if not above:
                    continue
                low, high = points.min(0), points.max(0)
                center = (low + high) * .5  # observable geometric centre, never simulator COM
                height = max(float(high[2] - low[2]), 1e-6)
                z = low[2] + .5 * height
                section = points[np.abs(points[:, 2] - z) <= max(.004, .1 * height)]
                if len(section) < 8:
                    section = points[np.argsort(np.abs(points[:, 2] - z))[:min(128, len(points))]]
                boundary = polygon(section)
                if boundary is None:
                    continue
                main = center[:2] - target_center[:2]
                norm = np.linalg.norm(main)
                if norm < 1e-8:
                    spans = np.ptp(section[:, :2], axis=0)
                    main = np.array([1., 0.]) if spans[1] >= spans[0] else np.array([0., 1.])
                else:
                    main /= norm
                edges = np.roll(boundary, -1, axis=0) - boundary
                lengths = np.linalg.norm(edges, axis=1)
                cumulative = np.r_[0., np.cumsum(lengths)]
                perimeter = cumulative[-1]
                for arc in np.arange(0., perimeter, spacing):
                    e = min(np.searchsorted(cumulative, arc, side="right") - 1, len(edges)-1)
                    xy = boundary[e] + edges[e] * ((arc-cumulative[e]) / max(lengths[e], 1e-9))
                    if np.dot(xy-center[:2], main) > 0:
                        continue
                    inward = center[:2] - xy
                    inward /= max(np.linalg.norm(inward), 1e-9)
                    signed = math.atan2(main[0]*inward[1]-main[1]*inward[0], np.clip(main@inward, -1., 1.))
                    u = np.clip((abs(math.degrees(signed))-20.)/50., 0., 1.)
                    angle = (.15 + .60*u*u*(3.-2.*u)) * signed
                    direction = np.array([main[0]*math.cos(angle)-main[1]*math.sin(angle),
                                          main[0]*math.sin(angle)+main[1]*math.cos(angle), 0.])
                    nearest = np.linalg.norm(section[:, :2]-xy, axis=1).argmin()
                    result.append((b, obj, [xy[0], xy[1], section[nearest, 2]], direction))
        if not result:
            return PushActions.empty(xyz)
        return PushActions(
            torch.tensor([r[0] for r in result], dtype=torch.long, device=xyz.device),
            torch.tensor([r[1] for r in result], dtype=torch.long, device=xyz.device),
            xyz.new_tensor(np.asarray([r[2] for r in result])),
            xyz.new_tensor(np.asarray([r[3] for r in result])),
            xyz.new_full((len(result),), PUSH_DISTANCE_M),
        ).validate(len(xyz), condition.object_valid.shape[1])
