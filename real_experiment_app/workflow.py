# ruff: noqa: E501
"""Pure decision rules for the real experiment workflow.

The public interface is deliberately small: infer the top obstruction and
choose a successful PUSH simulation. Model and physics adapters stay outside.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PUSH_DISTANCE_M = 0.15


@dataclass(frozen=True, slots=True)
class ObstructionRelation:
    lower: int
    upper: int
    xy_overlap: float
    height_gap_m: float


def _bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise ValueError("instance point cloud is empty")
    return np.quantile(points, 0.05, axis=0), np.quantile(points, 0.95, axis=0)


def _observed_xy_columns(points: np.ndarray, cell_m: float = 0.005) -> np.ndarray:
    """Observed XY cells with their local bottom and top; no filled-in hull area."""
    cells = np.rint(points[:, :2] / cell_m).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    bottom = np.full(len(unique), np.inf)
    top = np.full(len(unique), -np.inf)
    np.minimum.at(bottom, inverse, points[:, 2])
    np.maximum.at(top, inverse, points[:, 2])
    return np.column_stack((unique * cell_m, bottom, top))


def _ambiguous_projection(columns: np.ndarray, tree: cKDTree) -> bool:
    """Reject a query merging substantial, separated XY components.

    Top/bottom surfaces at the same XY remain one object. Tiny stray fragments
    do not invalidate a mask. Ambiguous queries must be reselected/resegmented;
    selecting a component silently would change the downstream action identity.
    """
    pairs = tree.query_pairs(.02, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(columns), len(columns)))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return bool(np.count_nonzero(sizes >= max(8, .1 * len(columns))) > 1)


def detach_minor_instance_fragments(
    xyz: np.ndarray, instance_id: np.ndarray,
    *, max_fraction: float = 0.10, minimum_separation_m: float = 0.05,
) -> np.ndarray:
    """Leave well-separated minor mask fragments in the scene as unassigned points.

    A mask with two substantial components is intentionally left intact for the
    later ambiguity check. Only a dominant component plus remote small islands
    is repaired; nearby disconnected surfaces can be valid parts of an object.
    """
    result = np.asarray(instance_id, np.int64).copy()
    for object_id in np.unique(result):
        if object_id < 0:
            continue
        indices = np.flatnonzero(result == object_id)
        if len(indices) < 16:
            continue
        cells = np.rint(np.asarray(xyz)[indices, :2] / 0.005).astype(np.int64)
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        if len(unique) < 2:
            continue
        centers = unique * 0.005
        tree = cKDTree(centers)
        pairs = tree.query_pairs(0.02, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(unique), len(unique)))
        count, component = connected_components(graph, directed=False)
        if count < 2:
            continue
        point_component = component[inverse]
        sizes = np.bincount(point_component, minlength=count)
        primary = int(np.argmax(sizes))
        if sizes[primary] < (1 - max_fraction) * len(indices):
            continue
        main_tree = cKDTree(centers[component == primary])
        for part in range(count):
            if part == primary or sizes[part] > max_fraction * len(indices):
                continue
            separation = main_tree.query(centers[component == part], k=1)[0].min()
            if separation >= minimum_separation_m:
                result[indices[point_component == part]] = -1
    return result


def obstruction_graph(
    xyz: np.ndarray,
    instance_id: np.ndarray,
    *,
    overlap_margin_scale: float,
    minimum_xy_overlap: float,
    minimum_height_gap_scale: float,
    diagnostics: dict | None = None,
) -> tuple[ObstructionRelation, ...]:
    """Infer local overhead obstruction, including separated visible surfaces.

    These are blocking relations, not physical-contact/support assertions.
    Hidden contact surfaces and an air gap must not reject an overhead object.
    """
    clouds = {
        int(value): np.asarray(xyz[instance_id == value], np.float64)
        for value in np.unique(instance_id)
        if int(value) >= 0 and int((instance_id == value).sum()) >= 8
    }
    boxes = {key: _bounds(value) for key, value in clouds.items()}
    columns = {key: _observed_xy_columns(value) for key, value in clouds.items()}
    trees = {key: cKDTree(value[:, :2]) for key, value in columns.items()}
    ambiguous = {key for key in columns if _ambiguous_projection(columns[key], trees[key])}
    if diagnostics is not None:
        diagnostics["ambiguous_instances"] = tuple(sorted(ambiguous))
    relations = []
    for lower, (lo0, hi0) in boxes.items():
        if lower in ambiguous:
            continue
        size0 = np.maximum(hi0 - lo0, 1e-4)
        for upper, (lo1, hi1) in boxes.items():
            if upper == lower or upper in ambiguous:
                continue
            size1 = np.maximum(hi1 - lo1, 1e-4)
            margin = overlap_margin_scale * \
                min(np.linalg.norm(size0[:2]), np.linalg.norm(size1[:2]))
            overlap_size = np.maximum(0.0, np.minimum(
                hi0[:2] + margin, hi1[:2]) - np.maximum(lo0[:2] - margin, lo1[:2]))
            if not np.all(overlap_size > 0):
                continue
            # AABB overlap only rejects distant pairs. Each accepted cell must
            # have measured points from both objects at the same XY location.
            lower_cells, upper_cells = columns[lower], columns[upper]
            nearby = trees[lower].query_ball_point(upper_cells[:, :2], r=0.0075)
            shared = [(i, j) for i, neighbors in enumerate(nearby) for j in neighbors]
            if len(shared) < 3:
                continue
            upper_idx, lower_idx = np.asarray(shared, dtype=np.int64).T
            overlap = min(len(np.unique(upper_idx)), len(np.unique(lower_idx))) / min(
                len(lower_cells), len(upper_cells))
            if overlap < minimum_xy_overlap:
                continue
            local_height = (
                (upper_cells[upper_idx, 2] + upper_cells[upper_idx, 3]
                 - lower_cells[lower_idx, 2] - lower_cells[lower_idx, 3]) * .5
            )
            height_gap = float(np.median(local_height))
            scale = max(1e-4, min(size0[2], size1[2]))
            above = local_height > minimum_height_gap_scale * scale
            if np.mean(above) >= .6:
                relations.append(ObstructionRelation(lower, upper, overlap, height_gap))
    return tuple(relations)


def top_obstructor(target: int, relations: Iterable[ObstructionRelation]) -> int | None:
    """Return the topmost reachable obstruction, preferring stronger overlap."""
    outgoing: dict[int, list[ObstructionRelation]] = {}
    for relation in relations:
        outgoing.setdefault(relation.lower, []).append(relation)
    reachable: dict[int, tuple[int, float, float]] = {}
    frontier = [(int(target), 0, 1.0, 0.0, frozenset((int(target),)))]
    while frontier:
        node, depth, overlap, height, path = frontier.pop()
        for edge in outgoing.get(node, ()):
            if edge.upper in path:
                continue
            candidate = (depth + 1, min(overlap, edge.xy_overlap), height + edge.height_gap_m)
            if candidate > reachable.get(edge.upper, (-1, -1.0, -1.0)):
                reachable[edge.upper] = candidate
                frontier.append((edge.upper, *candidate, path | {edge.upper}))
    if not reachable:
        return None
    leaves = [
        node
        for node in reachable
        if not any(edge.upper in reachable for edge in outgoing.get(node, ()))
    ]
    domain = leaves or list(reachable)
    return max(domain, key=lambda node: reachable[node])


def reachable_obstructors(target: int, relations: Iterable[ObstructionRelation]) -> tuple[int, ...]:
    """All inferred instances above the target, excluding unrelated graph branches."""
    outgoing: dict[int, set[int]] = {}
    for relation in relations:
        outgoing.setdefault(relation.lower, set()).add(relation.upper)
    seen = {int(target)}
    frontier = [int(target)]
    while frontier:
        for upper in outgoing.get(frontier.pop(), ()):
            if upper not in seen:
                seen.add(upper)
                frontier.append(upper)
    seen.discard(int(target))
    return tuple(sorted(seen))


def select_simulated_push(results: Iterable[dict]) -> dict | None:
    valid = [item for item in results if bool(item.get("effective", False))]
    return max(valid, key=lambda item: float(item["distance_gain_m"])) if valid else None


def ranked_actions(candidates: Iterable[dict], action_type: int, acted_object: int) -> tuple[dict, ...]:
    """Filter one branch and sort by the score used by that branch."""
    selected = [
        item for item in candidates
        if int(item["action_type"]) == int(action_type)
        and int(item["acted_object"]) == int(acted_object)
    ]
    score = "improvement_probability" if int(action_type) == 0 else "proposal_score"
    return tuple(sorted(selected, key=lambda item: float(item.get(score, float("-inf"))), reverse=True))


def expanding_top_k(total: int, initial: int, increment: int = 5) -> tuple[int, ...]:
    """Return cumulative limits beginning at configured K, then adding five."""
    if total <= 0:
        return ()
    if initial <= 0:
        raise ValueError("initial Top-K must be positive")
    if increment <= 0:
        raise ValueError("Top-K increment must be positive")
    limits = list(range(min(initial, total), total + 1, increment))
    if not limits or limits[-1] != total:
        limits.append(total)
    return tuple(limits)


def scene_change_m(previous_xyz: np.ndarray, current_xyz: np.ndarray, sample_limit: int = 4096) -> float:
    """Robust symmetric nearest-neighbour scene change in metres."""
    from scipy.spatial import cKDTree
    a = np.asarray(previous_xyz)[::max(1, len(previous_xyz) // sample_limit)]
    b = np.asarray(current_xyz)[::max(1, len(current_xyz) // sample_limit)]
    if not len(a) or not len(b):
        return float("inf")
    ab = cKDTree(b).query(a, k=1)[0]
    ba = cKDTree(a).query(b, k=1)[0]
    return float(max(np.median(ab), np.median(ba)))


def similar_push(previous: dict | None, current: dict, contact_threshold_m: float, direction_threshold_deg: float) -> bool:
    if not previous or int(previous.get("action_type", -1)) != 0 or int(current.get("action_type", -1)) != 0:
        return False
    if int(previous.get("acted_object", -1)) != int(current.get("acted_object", -2)):
        return False
    p0, p1 = np.asarray(previous["push_contact_world"]), np.asarray(current["push_contact_world"])
    d0, d1 = np.asarray(previous["push_direction_world"]), np.asarray(
        current["push_direction_world"])
    cosine = float(np.dot(d0, d1) / max(np.linalg.norm(d0) * np.linalg.norm(d1), 1e-9))
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return float(np.linalg.norm(p0 - p1)) <= contact_threshold_m and angle <= direction_threshold_deg
