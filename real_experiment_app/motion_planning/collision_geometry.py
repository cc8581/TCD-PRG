"""Collision-geometry builders for MoveIt planning scenes.

The planner can consume several geometric approximations of the segmented scene:

- ``hybrid`` (default): convex hull meshes for segmented environment objects,
  an alpha-shape target mesh when stable, and legacy voxels for
  unassigned/background or degenerate point sets. This is the recommended
  fidelity/robustness trade-off.
- ``convex_hull``: one closed convex mesh per instance, falling back to voxels.
- ``alpha_shape``: a tighter concave tetrahedral alpha-complex boundary for
  segmented objects, with convex-hull then voxel fallback.
- ``obb``: one PCA-oriented bounding box mesh per segmented instance.
- ``voxel``: legacy solidified collision voxels only.

All returned mesh vertices are expressed directly in ``base_link`` coordinates,
so the ROS bridge can attach them to identity mesh poses in the MoveIt scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError


SUPPORTED_COLLISION_GEOMETRIES = frozenset(
    {"hybrid", "convex_hull", "alpha_shape", "obb", "voxel"}
)


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, np.float64)
        triangles = np.asarray(self.triangles, np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("mesh vertices must have shape [V,3]")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("mesh triangles must have shape [F,3]")
        if not len(vertices) or not len(triangles):
            raise ValueError("mesh must contain vertices and triangles")
        if not np.isfinite(vertices).all():
            raise ValueError("mesh vertices contain non-finite values")
        if triangles.min(initial=0) < 0 or triangles.max(initial=-1) >= len(vertices):
            raise ValueError("mesh triangle index is out of range")
        object.__setattr__(self, "vertices", np.ascontiguousarray(vertices))
        object.__setattr__(self, "triangles", np.ascontiguousarray(triangles))


@dataclass(frozen=True, slots=True)
class CollisionGeometry:
    mode: str
    environment_voxels: np.ndarray
    target_voxels: np.ndarray
    environment_meshes: tuple[TriangleMesh, ...]
    target_meshes: tuple[TriangleMesh, ...]

    @property
    def mesh_count(self) -> int:
        return len(self.environment_meshes) + len(self.target_meshes)

    @property
    def triangle_count(self) -> int:
        return sum(len(mesh.triangles) for mesh in self.environment_meshes + self.target_meshes)


def _validate_scene(
    xyz_m: np.ndarray,
    instance_id: np.ndarray,
    target_instance: int,
    voxel_size_m: float,
    target_voxel_size_m: float,
    collision_padding_m: float,
    mesh_max_points: int,
    alpha_radius_m: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_m, np.float64)
    instance = np.asarray(instance_id, np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or instance.shape != (len(xyz),):
        raise ValueError("scene XYZ and instance_id must have shapes [N,3] and [N]")
    if not len(xyz):
        raise ValueError("scene point cloud is empty")
    if not np.isfinite(xyz).all():
        raise ValueError("scene points must be finite")
    if not 0.005 <= float(voxel_size_m) <= 0.10:
        raise ValueError("voxel_size_m must lie in [0.005,0.10]")
    if not 0.005 <= float(target_voxel_size_m) <= 0.10:
        raise ValueError("target_voxel_size_m must lie in [0.005,0.10]")
    if not 0.0 <= float(collision_padding_m) <= 0.05:
        raise ValueError("collision_padding_m must lie in [0,0.05]")
    if int(mesh_max_points) < 64 or int(mesh_max_points) > 20_000:
        raise ValueError("mesh_max_points must lie in [64,20000]")
    if not 0.005 <= float(alpha_radius_m) <= 0.25:
        raise ValueError("alpha_radius_m must lie in [0.005,0.25]")
    if mode not in SUPPORTED_COLLISION_GEOMETRIES:
        raise ValueError(
            f"unsupported collision geometry '{mode}'; expected one of "
            f"{sorted(SUPPORTED_COLLISION_GEOMETRIES)}"
        )
    if not np.any(instance == int(target_instance)):
        raise ValueError(f"target instance {target_instance} has no scene points")
    return xyz, instance


def _solid_voxels(points_xyz: np.ndarray, ids: np.ndarray, size: float) -> np.ndarray:
    """Legacy per-instance voxel solidification preserved exactly in spirit.

    Segmented instances (id >= 0) are filled between the observed minimum and
    maximum Z voxel in each observed XY column. Unassigned/background points
    (id < 0) remain surface voxels so disconnected unknown clutter is not joined
    into one artificial solid.
    """

    points_xyz = np.asarray(points_xyz, np.float64).reshape(-1, 3)
    ids = np.asarray(ids, np.int64).reshape(-1)
    if not len(points_xyz):
        return np.empty((0, 3), np.float64)
    blocks: list[np.ndarray] = []
    for object_id in np.unique(ids):
        points = points_xyz[ids == object_id]
        if not len(points):
            continue
        keys = np.floor(points / size).astype(np.int64)
        if int(object_id) < 0:
            blocks.append(np.unique(keys, axis=0))
            continue
        columns: list[np.ndarray] = []
        for column in np.unique(keys[:, :2], axis=0):
            observed = keys[np.all(keys[:, :2] == column, axis=1), 2]
            z = np.arange(int(observed.min()), int(observed.max()) + 1, dtype=np.int64)
            columns.append(
                np.column_stack(
                    (
                        np.full(len(z), column[0], dtype=np.int64),
                        np.full(len(z), column[1], dtype=np.int64),
                        z,
                    )
                )
            )
        if sum(len(column) for column in columns) > 50_000:
            raise ValueError(f"instance {object_id} produces too many collision voxels")
        if columns:
            blocks.append(np.concatenate(columns, axis=0))
    if not blocks:
        return np.empty((0, 3), np.float64)
    unique = np.unique(np.concatenate(blocks, axis=0), axis=0)
    return (unique.astype(np.float64) + 0.5) * size


def voxel_centers(
    xyz_m: np.ndarray,
    instance_id: np.ndarray,
    target_instance: int,
    voxel_size_m: float,
    target_voxel_size_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning legacy environment and target voxels."""

    xyz = np.asarray(xyz_m, np.float64)
    instance = np.asarray(instance_id, np.int64)
    target_size = voxel_size_m if target_voxel_size_m is None else target_voxel_size_m
    if xyz.ndim != 2 or xyz.shape[1] != 3 or instance.shape != (len(xyz),):
        raise ValueError("scene XYZ and instance_id must have shapes [N,3] and [N]")
    if (
        not np.isfinite(xyz).all()
        or not 0.005 <= voxel_size_m <= 0.10
        or not 0.005 <= target_size <= 0.10
    ):
        raise ValueError("scene points must be finite and voxel size must lie in [0.005,0.10]")
    target_mask = instance == int(target_instance)
    return _solid_voxels(xyz[~target_mask], instance[~target_mask], voxel_size_m), _solid_voxels(
        xyz[target_mask], instance[target_mask], target_size
    )


def _radial_padding(points: np.ndarray, padding_m: float) -> np.ndarray:
    if padding_m <= 0.0:
        return points
    center = points.mean(axis=0)
    delta = points - center
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    direction = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 1e-9)
    return points + direction * float(padding_m)


def _require_3d_extent(points: np.ndarray, minimum_m: float = 5e-4) -> None:
    """Reject nearly planar/linear clouds so they fall back to solid voxels."""
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    local = centered @ axes.T
    extent = np.ptp(local, axis=0)
    if len(extent) < 3 or float(np.min(extent)) < minimum_m:
        raise ValueError("point cloud has insufficient 3-D extent for a closed mesh")


def _compact_mesh(vertices: np.ndarray, triangles: np.ndarray) -> TriangleMesh:
    used = np.unique(triangles.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return TriangleMesh(vertices[used], remap[triangles].astype(np.int32, copy=False))


def _convex_hull_mesh(points: np.ndarray, padding_m: float) -> TriangleMesh:
    points = np.unique(np.asarray(points, np.float64).reshape(-1, 3), axis=0)
    if len(points) < 4:
        raise ValueError("convex hull requires at least four unique points")
    _require_3d_extent(points)
    padded = _radial_padding(points, padding_m)
    try:
        hull = ConvexHull(padded, qhull_options="QJ")
    except QhullError as error:
        raise ValueError(f"convex hull failed: {error}") from error
    if len(hull.simplices) < 4:
        raise ValueError("convex hull is degenerate")
    return _compact_mesh(padded, np.asarray(hull.simplices, np.int32))


def _obb_mesh(points: np.ndarray, padding_m: float) -> TriangleMesh:
    points = np.unique(np.asarray(points, np.float64).reshape(-1, 3), axis=0)
    if len(points) < 4:
        raise ValueError("oriented box requires at least four unique points")
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    values, axes = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    local = centered @ axes
    lower = local.min(axis=0) - float(padding_m)
    upper = local.max(axis=0) + float(padding_m)
    if np.any(upper - lower < 1e-4):
        raise ValueError("oriented box is degenerate")
    corners_local = np.asarray(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ],
        np.float64,
    )
    vertices = center + corners_local @ axes.T
    triangles = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        np.int32,
    )
    return TriangleMesh(vertices, triangles)


def _deterministic_subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    points = np.unique(np.asarray(points, np.float64).reshape(-1, 3), axis=0)
    if len(points) <= maximum:
        return points
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    positions = np.linspace(0, len(order) - 1, maximum, dtype=np.int64)
    sampled = points[order[positions]]
    anchors = np.asarray(
        [
            points[np.argmin(points[:, 0])], points[np.argmax(points[:, 0])],
            points[np.argmin(points[:, 1])], points[np.argmax(points[:, 1])],
            points[np.argmin(points[:, 2])], points[np.argmax(points[:, 2])],
        ]
    )
    return np.unique(np.vstack((sampled, anchors)), axis=0)


def _alpha_shape_mesh(
    points: np.ndarray,
    alpha_radius_m: float,
    padding_m: float,
    mesh_max_points: int,
) -> TriangleMesh:
    points = _deterministic_subsample(points, mesh_max_points)
    if len(points) < 5:
        raise ValueError("alpha shape requires at least five unique points")
    _require_3d_extent(points)
    points = _radial_padding(points, padding_m)
    try:
        tetra = Delaunay(points, qhull_options="QJ").simplices
    except QhullError as error:
        raise ValueError(f"alpha-shape Delaunay failed: {error}") from error
    if not len(tetra):
        raise ValueError("alpha-shape Delaunay produced no tetrahedra")

    p = points[tetra]
    matrices = 2.0 * (p[:, 1:] - p[:, :1])
    rhs = np.sum(p[:, 1:] ** 2, axis=2) - np.sum(p[:, :1] ** 2, axis=2)
    determinant = np.linalg.det(matrices)
    valid = np.abs(determinant) > 1e-12
    centers = np.zeros((len(tetra), 3), np.float64)
    if np.any(valid):
        centers[valid] = np.linalg.solve(matrices[valid], rhs[valid])
    radii = np.full(len(tetra), np.inf, np.float64)
    radii[valid] = np.linalg.norm(centers[valid] - p[valid, 0], axis=1)
    kept = tetra[radii <= float(alpha_radius_m)]
    if not len(kept):
        raise ValueError("alpha radius retained no tetrahedra")

    faces = np.vstack(
        (
            kept[:, [0, 1, 2]],
            kept[:, [0, 1, 3]],
            kept[:, [0, 2, 3]],
            kept[:, [1, 2, 3]],
        )
    )
    canonical = np.sort(faces, axis=1)
    unique_faces, counts = np.unique(canonical, axis=0, return_counts=True)
    boundary = unique_faces[counts == 1]
    if len(boundary) < 4:
        raise ValueError("alpha shape produced a degenerate boundary")
    return _compact_mesh(points, boundary.astype(np.int32, copy=False))


def _mesh_for_mode(
    points: np.ndarray,
    mode: str,
    *,
    collision_padding_m: float,
    mesh_max_points: int,
    alpha_radius_m: float,
) -> TriangleMesh:
    if mode in {"hybrid", "convex_hull"}:
        return _convex_hull_mesh(points, collision_padding_m)
    if mode == "obb":
        return _obb_mesh(points, collision_padding_m)
    if mode == "alpha_shape":
        try:
            return _alpha_shape_mesh(
                points,
                alpha_radius_m=alpha_radius_m,
                padding_m=collision_padding_m,
                mesh_max_points=mesh_max_points,
            )
        except ValueError:
            # Alpha complexes are parameter-sensitive for sparse/partial depth
            # data. Falling back to a closed convex hull is safer than silently
            # dropping an obstacle.
            return _convex_hull_mesh(points, collision_padding_m)
    raise ValueError(f"mode {mode!r} does not produce meshes")


def _build_side(
    points: np.ndarray,
    ids: np.ndarray,
    *,
    mode: str,
    voxel_size_m: float,
    collision_padding_m: float,
    mesh_max_points: int,
    alpha_radius_m: float,
    allow_unknown_mesh: bool,
) -> tuple[np.ndarray, tuple[TriangleMesh, ...]]:
    voxel_blocks: list[np.ndarray] = []
    meshes: list[TriangleMesh] = []
    for object_id in np.unique(ids):
        object_points = points[ids == object_id]
        if not len(object_points):
            continue
        if mode == "voxel" or (int(object_id) < 0 and not allow_unknown_mesh):
            voxel_blocks.append(
                _solid_voxels(
                    object_points,
                    np.full(len(object_points), int(object_id), np.int64),
                    voxel_size_m,
                )
            )
            continue
        try:
            meshes.append(
                _mesh_for_mode(
                    object_points,
                    mode,
                    collision_padding_m=collision_padding_m,
                    mesh_max_points=mesh_max_points,
                    alpha_radius_m=alpha_radius_m,
                )
            )
        except ValueError:
            # Never delete sparse/degenerate obstacles. Preserve them as the
            # proven legacy voxel representation instead.
            voxel_blocks.append(
                _solid_voxels(
                    object_points,
                    np.full(len(object_points), int(object_id), np.int64),
                    voxel_size_m,
                )
            )
    nonempty = [block for block in voxel_blocks if len(block)]
    voxels = (
        np.unique(np.concatenate(nonempty, axis=0), axis=0)
        if nonempty
        else np.empty((0, 3), np.float64)
    )
    return voxels, tuple(meshes)


def build_collision_geometry(
    xyz_m: np.ndarray,
    instance_id: np.ndarray,
    target_instance: int,
    *,
    mode: str = "hybrid",
    voxel_size_m: float = 0.02,
    target_voxel_size_m: float = 0.01,
    collision_padding_m: float = 0.0,
    mesh_max_points: int = 2500,
    alpha_radius_m: float = 0.025,
) -> CollisionGeometry:
    """Convert a segmented cloud into MoveIt-friendly collision geometry.

    ``hybrid`` is the recommended default: labeled environment objects become
    closed convex hulls, the target uses a tighter alpha-shape mesh when stable,
    and unknown/background points retain the robust voxel fallback. This removes
    most voxel stair-stepping without allowing a stray unknown cloud to be bridged
    into a giant hull.
    """

    mode = str(mode).strip().lower()
    xyz, instance = _validate_scene(
        xyz_m,
        instance_id,
        target_instance,
        voxel_size_m,
        target_voxel_size_m,
        collision_padding_m,
        mesh_max_points,
        alpha_radius_m,
        mode,
    )
    target_mask = instance == int(target_instance)
    env_points, env_ids = xyz[~target_mask], instance[~target_mask]
    target_points = xyz[target_mask]
    target_ids = np.full(len(target_points), int(target_instance), np.int64)

    if mode == "voxel":
        env_voxels, target_voxels = voxel_centers(
            xyz, instance, target_instance, voxel_size_m, target_voxel_size_m
        )
        return CollisionGeometry(mode, env_voxels, target_voxels, (), ())

    # In the recommended hybrid/alpha/OBB modes, negative IDs are intentionally
    # kept as voxels: they may contain multiple disconnected unknown surfaces.
    allow_unknown_mesh = mode == "convex_hull"
    env_voxels, env_meshes = _build_side(
        env_points,
        env_ids,
        mode=mode,
        voxel_size_m=voxel_size_m,
        collision_padding_m=collision_padding_m,
        mesh_max_points=mesh_max_points,
        alpha_radius_m=alpha_radius_m,
        allow_unknown_mesh=allow_unknown_mesh,
    )
    target_mode = "alpha_shape" if mode == "hybrid" else mode
    target_voxels, target_meshes = _build_side(
        target_points,
        target_ids,
        mode=target_mode,
        voxel_size_m=target_voxel_size_m,
        collision_padding_m=collision_padding_m,
        mesh_max_points=mesh_max_points,
        alpha_radius_m=alpha_radius_m,
        allow_unknown_mesh=True,
    )
    if not len(target_voxels) and not target_meshes:
        raise ValueError("target collision geometry is empty")
    return CollisionGeometry(mode, env_voxels, target_voxels, env_meshes, target_meshes)


def pack_meshes(prefix: str, meshes: tuple[TriangleMesh, ...]) -> dict[str, np.ndarray]:
    """Pack variable-length meshes into allow_pickle=False NPZ arrays."""

    vertex_offsets = [0]
    triangle_offsets = [0]
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    for mesh in meshes:
        vertices.append(mesh.vertices)
        triangles.append(mesh.triangles)
        vertex_offsets.append(vertex_offsets[-1] + len(mesh.vertices))
        triangle_offsets.append(triangle_offsets[-1] + len(mesh.triangles))
    return {
        f"{prefix}_mesh_vertices_m": (
            np.concatenate(vertices, axis=0) if vertices else np.empty((0, 3), np.float64)
        ),
        f"{prefix}_mesh_triangles": (
            np.concatenate(triangles, axis=0) if triangles else np.empty((0, 3), np.int32)
        ),
        f"{prefix}_mesh_vertex_offsets": np.asarray(vertex_offsets, np.int64),
        f"{prefix}_mesh_triangle_offsets": np.asarray(triangle_offsets, np.int64),
    }
