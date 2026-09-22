import numpy as np

from real_experiment_app.motion_planning.collision_geometry import (
    build_collision_geometry,
    pack_meshes,
    voxel_centers,
)


def cube(center, size=0.1):
    c = np.asarray(center, np.float64)
    h = size / 2.0
    return c + np.asarray([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        [0, 0, -h], [0, 0, h], [0, -h, 0], [0, h, 0],
    ])


def test_legacy_voxel_mode_matches_compatibility_function():
    xyz = np.vstack((cube([0.2, 0.0, 0.1], 0.04), cube([0.4, 0.0, 0.1], 0.04)))
    ids = np.asarray([3] * 12 + [7] * 12, np.int64)
    expected_env, expected_target = voxel_centers(xyz, ids, 3, 0.02, 0.01)
    geometry = build_collision_geometry(
        xyz, ids, 3, mode="voxel", voxel_size_m=0.02, target_voxel_size_m=0.01
    )
    np.testing.assert_allclose(geometry.environment_voxels, expected_env)
    np.testing.assert_allclose(geometry.target_voxels, expected_target)
    assert geometry.environment_meshes == ()
    assert geometry.target_meshes == ()


def test_hybrid_uses_hulls_for_segmented_instances_and_voxels_for_unknown():
    target = cube([0.3, 0.0, 0.1], 0.08)
    obstacle = cube([0.5, 0.1, 0.1], 0.10)
    unknown = np.asarray([[0.7, -0.1, 0.05], [0.72, -0.1, 0.05]])
    xyz = np.vstack((target, obstacle, unknown))
    ids = np.asarray([1] * len(target) + [2] * len(obstacle) + [-1] * len(unknown), np.int64)
    geometry = build_collision_geometry(
        xyz,
        ids,
        1,
        mode="hybrid",
        voxel_size_m=0.02,
        target_voxel_size_m=0.01,
    )
    assert len(geometry.target_meshes) == 1
    assert len(geometry.environment_meshes) == 1
    assert len(geometry.environment_voxels) > 0
    assert len(geometry.target_voxels) == 0
    assert geometry.triangle_count > 0


def test_obb_mode_produces_closed_box_meshes():
    target = cube([0.3, 0.0, 0.1], 0.08)
    obstacle = cube([0.5, 0.1, 0.1], 0.10)
    xyz = np.vstack((target, obstacle))
    ids = np.asarray([1] * len(target) + [2] * len(obstacle), np.int64)
    geometry = build_collision_geometry(xyz, ids, 1, mode="obb")
    assert len(geometry.target_meshes) == 1
    assert len(geometry.environment_meshes) == 1
    assert geometry.target_meshes[0].vertices.shape == (8, 3)
    assert geometry.target_meshes[0].triangles.shape == (12, 3)


def test_alpha_shape_has_safe_fallback_and_returns_closed_geometry():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(80, 3))
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    target = target * 0.04 + np.asarray([0.3, 0.0, 0.1])
    obstacle = cube([0.5, 0.0, 0.1], 0.08)
    xyz = np.vstack((target, obstacle))
    ids = np.asarray([1] * len(target) + [2] * len(obstacle), np.int64)
    geometry = build_collision_geometry(
        xyz,
        ids,
        1,
        mode="alpha_shape",
        alpha_radius_m=0.03,
        mesh_max_points=256,
    )
    assert len(geometry.target_meshes) == 1
    assert len(geometry.target_meshes[0].triangles) >= 4
    assert len(geometry.environment_meshes) == 1


def test_mesh_packing_uses_local_triangle_indices_and_offsets():
    target = cube([0.3, 0.0, 0.1], 0.08)
    obstacle = cube([0.5, 0.0, 0.1], 0.08)
    xyz = np.vstack((target, obstacle))
    ids = np.asarray([1] * len(target) + [2] * len(obstacle), np.int64)
    geometry = build_collision_geometry(xyz, ids, 1, mode="hybrid")
    packed = pack_meshes("environment", geometry.environment_meshes)
    offsets = packed["environment_mesh_vertex_offsets"]
    triangle_offsets = packed["environment_mesh_triangle_offsets"]
    assert offsets[0] == 0
    assert offsets[-1] == len(packed["environment_mesh_vertices_m"])
    assert triangle_offsets[-1] == len(packed["environment_mesh_triangles"])
    for index in range(len(offsets) - 1):
        triangles = packed["environment_mesh_triangles"][
            triangle_offsets[index]:triangle_offsets[index + 1]
        ]
        assert triangles.max(initial=-1) < offsets[index + 1] - offsets[index]


def test_nearly_planar_segment_falls_back_to_solid_voxels():
    x, y = np.meshgrid(np.linspace(0.2, 0.3, 6), np.linspace(-0.05, 0.05, 6))
    target = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.1)))
    obstacle = cube([0.5, 0.0, 0.1], 0.08)
    xyz = np.vstack((target, obstacle))
    ids = np.asarray([1] * len(target) + [2] * len(obstacle), np.int64)
    geometry = build_collision_geometry(xyz, ids, 1, mode="hybrid")
    assert len(geometry.target_meshes) == 0
    assert len(geometry.target_voxels) > 0
    assert len(geometry.environment_meshes) == 1
