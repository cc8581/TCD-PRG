import numpy as np
from pathlib import Path

from real_experiment_app.workflow import (
    detach_minor_instance_fragments,
    expanding_top_k,
    obstruction_graph,
    ranked_actions,
    scene_change_m,
    select_simulated_push,
    similar_push,
    top_obstructor,
)


def test_minor_remote_fragment_becomes_unassigned_without_losing_scene_points():
    main = _box((0, 0, .04), count=180, seed=21)
    foreign = _box((.2, 0, .04), size=.01, count=10, seed=22)
    xyz = np.vstack((main, foreign))
    labels = np.full(len(xyz), 5)
    cleaned = detach_minor_instance_fragments(xyz, labels)
    assert np.all(cleaned[:len(main)] == 5)
    assert np.all(cleaned[len(main):] == -1)
    assert len(cleaned) == len(xyz)


def test_single_remote_point_can_become_unassigned():
    main = _box((0, 0, .04), count=180, seed=25)
    xyz = np.vstack((main, [[.2, 0, .04]]))
    cleaned = detach_minor_instance_fragments(xyz, np.full(len(xyz), 5))
    assert np.all(cleaned[:-1] == 5)
    assert cleaned[-1] == -1


def test_two_substantial_components_remain_ambiguous():
    xyz = np.vstack((_box((0, 0, .04), count=100, seed=23),
                     _box((.2, 0, .04), count=100, seed=24)))
    labels = np.full(len(xyz), 5)
    assert np.array_equal(detach_minor_instance_fragments(xyz, labels), labels)


def _box(center, size=0.04, count=80, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.uniform(-size / 2, size / 2, (count, 3))


def test_obstruction_graph_selects_topmost_chain_member():
    xy = np.array([(x, y) for x in np.linspace(-.02, .02, 9)
                   for y in np.linspace(-.02, .02, 9)])
    clouds = [np.vstack((np.column_stack((xy, np.full(len(xy), low))),
                         np.column_stack((xy, np.full(len(xy), high)))))
              for low, high in ((.02, .06), (.07, .11), (.12, .16))]
    xyz = np.concatenate(clouds)
    labels = np.repeat([4, 8, 12], [len(cloud) for cloud in clouds])
    relations = obstruction_graph(
        xyz,
        labels,
        overlap_margin_scale=0.15,
        minimum_xy_overlap=0.05,
        minimum_height_gap_scale=0.08,
    )
    assert top_obstructor(4, relations) == 12


def test_disconnected_query_does_not_bridge_unoccupied_xy():
    target = np.concatenate((_box((-0.09, 0, 0.04), seed=10),
                             _box((0.09, 0, 0.04), seed=11)))
    middle = _box((0, 0, 0.09), seed=12)
    relations = obstruction_graph(
        np.concatenate((target, middle)),
        np.repeat([19, 28], [len(target), len(middle)]),
        overlap_margin_scale=0.15,
        minimum_xy_overlap=0.05,
        minimum_height_gap_scale=0.08,
    )
    assert not relations


def test_side_by_side_is_not_obstruction_but_overhead_air_gap_is():
    lower = _box((0, 0, 0.04), seed=13)
    side = _box((0.052, 0, 0.09), seed=14)
    floating = _box((0, 0, 0.16), seed=15)
    relations = obstruction_graph(
        np.concatenate((lower, side, floating)),
        np.repeat([1, 2, 3], 80),
        overlap_margin_scale=0.15,
        minimum_xy_overlap=0.05,
        minimum_height_gap_scale=0.08,
    )
    assert {(edge.lower, edge.upper) for edge in relations} == {(1, 3)}


def test_local_obstruction_cycles_do_not_loop_or_return_target():
    from real_experiment_app.workflow import ObstructionRelation
    relations = (ObstructionRelation(1, 2, .5, .02),
                 ObstructionRelation(2, 3, .5, .02),
                 ObstructionRelation(3, 1, .5, .02))
    assert top_obstructor(1, relations) in (2, 3)


def test_scene26_queries8_and20_keep_overhead_blockers():
    # Frozen deployment perception output from cached scene26/state0/task0.
    # Local visible-surface gaps are 4-9 cm; physical contact is not the label.
    with np.load(Path(__file__).parent / "fixtures" / "scene26_obstruction.npz") as data:
        relations = obstruction_graph(
            data["xyz"], data["instance_id"], overlap_margin_scale=.15,
            minimum_xy_overlap=.05, minimum_height_gap_scale=.08,
        )
    edges = {(edge.lower, edge.upper) for edge in relations}
    assert {(8, 3), (8, 17), (20, 19)} <= edges
    assert not {(3, 8), (17, 8), (19, 20)} & edges
    assert all(0 <= edge.xy_overlap <= 1 for edge in relations)


def test_segmentation_cleanup_keeps_scene26_primary_blockers():
    with np.load(Path(__file__).parent / "fixtures" / "scene26_obstruction.npz") as data:
        xyz, labels = data["xyz"], data["instance_id"]
    cleaned = detach_minor_instance_fragments(xyz, labels)
    relations = obstruction_graph(
        xyz, cleaned, overlap_margin_scale=.15,
        minimum_xy_overlap=.05, minimum_height_gap_scale=.08,
    )
    edges = {(edge.lower, edge.upper) for edge in relations}
    assert {(8, 3), (8, 17)} <= edges
    assert cleaned[labels == 19].tolist().count(-1) == 1


def test_scene13_merged_query_is_reported_instead_of_false_blocker():
    diagnostics = {}
    with np.load(Path(__file__).parent / "fixtures" / "scene13_merged_query.npz") as data:
        relations = obstruction_graph(
            data["xyz"], data["instance_id"], overlap_margin_scale=.15,
            minimum_xy_overlap=.05, minimum_height_gap_scale=.08,
            diagnostics=diagnostics,
        )
    assert 19 in diagnostics["ambiguous_instances"]
    assert not any(edge.lower == 19 or edge.upper == 19 for edge in relations)


def test_rank_and_expand_push_candidates():
    candidates = [
        {"action_type": 0, "acted_object": 7, "improvement_probability": score}
        for score in (0.2, 0.9, 0.4)
    ]
    assert [x["improvement_probability"] for x in ranked_actions(candidates, 0, 7)] == [
        0.9,
        0.4,
        0.2,
    ]
    assert expanding_top_k(12, 5) == (5, 10, 12)
    assert expanding_top_k(18, 7) == (7, 12, 17, 18)


def test_simulation_selection_uses_largest_gap_gain():
    chosen = select_simulated_push(
        [
            {"effective": True, "distance_gain_m": 0.01},
            {"effective": False, "distance_gain_m": 1.0},
            {"effective": True, "distance_gain_m": 0.03},
        ]
    )
    assert chosen["distance_gain_m"] == 0.03


def test_scene_change_and_repeated_push_detection():
    xyz = np.arange(90, dtype=float).reshape(30, 3) / 1000
    assert scene_change_m(xyz, xyz.copy()) == 0
    action = {
        "action_type": 0,
        "acted_object": 4,
        "push_contact_world": [0, 0, 0],
        "push_direction_world": [1, 0, 0],
    }
    near = dict(action, push_contact_world=[0.001, 0, 0], push_direction_world=[0.99, 0.01, 0])
    assert similar_push(action, near, 0.02, 15)
