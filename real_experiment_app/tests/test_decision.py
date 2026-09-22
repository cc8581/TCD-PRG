import numpy as np

from real_experiment_app.decision import DecisionEngine, DecisionSession
from real_experiment_app.types import FusedScene, Prediction
from tcd_prg.constants import ActionType


class Physics:
    def __init__(self, free=(), push_results=()):
        self.free = set(free)
        self.push_results = list(push_results)
        self.push_batches = []

    def collision_free_grasps(self, scene, candidates, target):
        return [x for x in candidates if x["candidate_index"] in self.free]

    def simulate_pushes(self, scene, candidates, target):
        self.push_batches.append(len(candidates))
        result, self.push_results = (
            self.push_results[: len(candidates)],
            self.push_results[len(candidates) :],
        )
        return result


def scene():
    xy = np.array([(x, y) for x in np.linspace(-.02, .02, 9)
                   for y in np.linspace(-.02, .02, 9)])
    clouds = [np.vstack((np.column_stack((xy, np.full(len(xy), low))),
                         np.column_stack((xy, np.full(len(xy), high)))))
              for low, high in ((.02, .06), (.07, .11))]
    count = sum(len(cloud) for cloud in clouds)
    return FusedScene(
        np.concatenate(clouds), np.zeros((count, 3)),
        np.repeat([1, 2], [len(cloud) for cloud in clouds]), np.zeros(count, int), {}
    )


def action(index, kind, obj, score):
    key = "improvement_probability" if kind == int(ActionType.PUSH) else "proposal_score"
    candidate = {"candidate_index": index, "action_type": kind, "acted_object": obj, key: score}
    if kind == int(ActionType.TASK_GRASP):
        # 180 degrees about X maps local +Z to the world/table -Z direction.
        candidate["grasp_pose_world"] = [0.0, 0.0, .1, 1.0, 0.0, 0.0, 0.0]
    return candidate


def test_target_grasp_wins_after_collision_check():
    analysis = Prediction({}, 0.1, (action(1, 2, 1, 0.7), action(2, 2, 1, 0.9)), target_query=1)
    result = DecisionEngine({}, Physics(free=(1,))).decide(scene(), analysis)
    assert result.action["candidate_index"] == 1
    assert result.decision_path[-1] == "TASK_GRASP"
    assert result.timings["final_action_decision_s"] >= 0


def test_motion_planner_tries_candidates_in_order_and_persists_saved_plan():
    class Planner:
        def plan_first(self, _scene, candidates, target):
            assert target == 7
            assert [item["candidate_index"] for item in candidates] == [2, 1]
            selected = dict(candidates[1])
            selected["moveit_plan_id"] = "saved-42"
            return selected, ({"candidate_index": 2, "success": False},
                              {"candidate_index": 1, "success": True})

    candidates = (action(1, 2, 1, .7), action(2, 2, 1, .9))
    session = DecisionSession(
        {}, Physics(free=(1, 2)), scene(),
        Prediction({}, .1, candidates, target_query=1, target_instance=7), Planner()
    )
    update = session.check_target_grasp()
    assert update["selected"]["moveit_plan_id"] == "saved-42"
    assert len(update["motion_planning"]) == 2


def test_all_motion_plans_failed_stops_instead_of_selecting_push():
    class Planner:
        def plan_first(self, _scene, _candidates, _target):
            return None, ({"candidate_index": 1, "success": False},)

    session = DecisionSession(
        {}, Physics(free=(1,)), scene(),
        Prediction({}, .1, (action(1, 2, 1, .9),), target_query=1), Planner()
    )
    session.check_target_grasp()
    assert session.status == "operator_attention"
    assert "MoveIt" in session.reason


def test_disconnected_target_explains_ambiguous_segmentation():
    from pathlib import Path
    with np.load(Path(__file__).parent / "fixtures" / "scene13_merged_query.npz") as data:
        points, labels = data["xyz"], data["instance_id"]
    source = FusedScene(points, np.zeros_like(points), labels, np.zeros(len(points), int), {})
    session = DecisionSession({}, Physics(), source, Prediction({}, .1, target_query=19))
    update = session.infer_obstruction()
    assert update["obstruction"] is None
    assert session.status == "operator_attention"
    assert "分离点簇" in session.reason


def test_ambiguous_target_cannot_enter_adjacent_push_branch():
    from pathlib import Path
    with np.load(Path(__file__).parent / "fixtures" / "scene13_merged_query.npz") as data:
        points, labels = data["xyz"], data["instance_id"]
    source = FusedScene(points, np.zeros_like(points), labels, np.zeros(len(points), int), {})
    session = DecisionSession({}, Physics(), source, Prediction({}, .1, target_query=19))
    session.adjacent_blockers = (7,)
    session.infer_obstruction()
    assert session.status == "operator_attention"
    assert session.pushable_objects == ()


def test_upward_local_z_grasp_is_filtered_before_collision_check():
    upward = action(1, 2, 1, .95)
    upward["grasp_pose_world"] = [0.0, 0.0, .1, 0.0, 0.0, 0.0, 1.0]
    downward = action(2, 2, 1, .70)
    physics = Physics(free=(1, 2))

    result = DecisionEngine({}, physics).decide(
        scene(), Prediction({}, .1, (upward, downward), target_query=1)
    )

    assert result.action["candidate_index"] == 2


def test_failed_target_grasp_uses_scored_push_without_pick_remove_or_simulation():
    candidates = (action(3, 1, 2, 0.8), action(4, 0, 2, 0.9))
    physics = Physics(free=(3,))
    result = DecisionEngine({}, physics).decide(
        scene(), Prediction({}, 0.1, candidates, target_query=1)
    )
    assert result.action["candidate_index"] == 4 and result.acted_query == 2
    assert "PICK_REMOVE" not in result.decision_path
    assert "PYBULLET" not in result.decision_path
    assert physics.push_batches == []


def test_push_evaluator_score_selects_highest_candidate():
    candidates = tuple(action(i, 0, 2, 1 - i / 20) for i in range(7))
    results = [{"batch_index": i, "effective": False, "distance_gain_m": 0} for i in range(5)]
    results += [
        {"batch_index": 0, "effective": True, "distance_gain_m": 0.02},
        {"batch_index": 1, "effective": True, "distance_gain_m": 0.04},
    ]
    physics = Physics(push_results=results)
    result = DecisionEngine({"push_top_k": 5}, physics).decide(
        scene(), Prediction({}, 0.1, candidates, target_query=1)
    )
    assert physics.push_batches == []
    assert result.action["candidate_index"] == 0


def test_manual_session_advances_without_running_downstream_stages():
    candidates = (action(3, 1, 2, 0.8), action(4, 0, 2, 0.9))
    session = DecisionSession(
        {}, Physics(free=(3,)), scene(), Prediction({}, 0.1, candidates, target_query=1)
    )
    first = session.check_target_grasp()
    assert first["selected"] is None and session.obstruction is None
    second = session.infer_obstruction()
    assert second["obstruction"] == 2 and session.result is None
    third = session.rank_pushes()
    assert third["selected"]["candidate_index"] == 4
    assert session.finish().action["action_type"] == int(ActionType.PUSH)


def test_all_reachable_obstructors_compete_for_push_selection():
    base = scene()
    third = base.xyz_m[base.instance_id == 2].copy()
    third[:, 2] += .06
    source = FusedScene(
        np.vstack((base.xyz_m, third)),
        np.vstack((base.rgb, np.zeros_like(third))),
        np.concatenate((base.instance_id, np.full(len(third), 3))),
        np.zeros(len(base.xyz_m) + len(third), int), {},
    )
    candidates = (action(10, 0, 2, .6), action(11, 0, 3, .9))
    session = DecisionSession({}, Physics(), source, Prediction({}, .1, candidates, target_query=1))
    update = session.infer_obstruction()
    assert update["obstructions"] == (2, 3)
    assert [item["acted_object"] for item in session.rank_pushes()["candidates"]] == [3, 2]
    assert session.finish().acted_query == 3


def test_grasp_diagnostics_identify_only_actionable_neighbor_collisions():
    class DetailedPhysics(Physics):
        def assess_grasps(self, _scene, candidates, _target):
            return [
                {"candidate_index": candidates[0]["candidate_index"], "collision_free": False,
                 "neighbor_ids": (7,), "table_collision": False,
                 "target_non_finger_collision": False},
                {"candidate_index": candidates[1]["candidate_index"], "collision_free": False,
                 "neighbor_ids": (8,), "table_collision": True,
                 "target_non_finger_collision": False},
            ]

    candidates = (action(1, 2, 1, .9), action(2, 2, 1, .8))
    session = DecisionSession({}, DetailedPhysics(), scene(), Prediction({}, .1, candidates, target_query=1))
    update = session.check_target_grasp()
    assert update["adjacent_blockers"] == (7,)
    assert update["grasp_diagnostics"][1]["neighbor_ids"] == (8,)
    assert update["grasp_diagnostics"][1]["table_collision"] is True


def test_no_model_grasp_is_reported_separately_from_collision_failure():
    session = DecisionSession({}, Physics(), scene(), Prediction({}, .1, (), target_query=1))
    update = session.check_target_grasp()
    assert update["grasp_generation_status"] == "model_not_generated"
    assert update["grasp_diagnostics"] == ()
    assert update["adjacent_blockers"] == ()


def test_adjacent_blockers_continue_to_push_with_per_grasp_co_blockers():
    class DetailedPhysics(Physics):
        def assess_grasps(self, _scene, candidates, _target):
            return [{"candidate_index": candidates[0]["candidate_index"],
                     "collision_free": False, "neighbor_ids": (7, 8),
                     "table_collision": False, "target_non_finger_collision": False}]

    source = scene()
    source.xyz_m[source.instance_id == 2, 0] += .12
    session = DecisionSession({}, DetailedPhysics(), source, Prediction(
        {}, .1, (action(1, 2, 1, .9),), target_query=1,
    ))
    session.check_target_grasp()
    update = session.infer_obstruction()
    assert update["obstructions"] == ()
    assert update["adjacent_blockers"] == (7, 8)
    assert session.status == "running"
    rules = session.record_push_rules((action(10, 0, 7, .4), action(11, 0, 8, .5)))
    assert rules["candidates"][0]["blocked_grasps"] == (
        {"candidate_index": 1, "other_neighbor_ids": (8,)},
    )
    scored = session.rank_pushes((action(10, 0, 7, .4), action(11, 0, 8, .5)))
    assert [item["acted_object"] for item in scored["candidates"]] == [8, 7]
    assert scored["selected"]["blocked_grasps"][0]["other_neighbor_ids"] == (7,)
