from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from real_experiment_app.controller import ExperimentController
from real_experiment_app.types import FusedScene, Prediction
from tcd_prg.constants import ActionType


class FakePredictor:
    def __init__(self):
        self.calls = []
        self.executed = []
        self.resets = 0

    def predict(self, scene, target, category, region):
        self.calls.append((scene, target, category, region))
        return Prediction({"action_type": int(ActionType.PUSH)}, 0.01)

    def action_executed(self, action):
        self.executed.append(action)

    def reset(self):
        self.resets += 1


class FakeRobot:
    def __init__(self):
        self.actions = []

    def execute(self, action):
        self.actions.append(action)

    def enable(self): self.enabled = True
    def disable(self): self.enabled = False
    def clear_errors(self): self.cleared = True
    def pause(self): self.paused = True
    def resume(self): self.paused = False
    def home(self): self.homed = True
    def stop(self): self.stopped = True
    def status(self):
        return {"enabled": int(getattr(self, "enabled", False)), "emergency_stop": 0,
                "motion_done": 1, "queue_len": 0, "main_code": 0, "sub_code": 0,
                "joints_deg": [0] * 6}


class FakeCamera:
    def __init__(self): self.disconnected = False
    def connect(self): return True
    def disconnect(self): self.disconnected = True


class OfflineRobot(FakeRobot):
    def connect(self): return False
    def disconnect(self): pass


def controller():
    value = ExperimentController.__new__(ExperimentController)
    value.config = SimpleNamespace(raw={"workflow": {}})
    value.cameras = []
    value.robot = FakeRobot()
    value.predictor = FakePredictor()
    value.physics = None
    value._worker_lock = Lock()
    value._worker_epoch = 0
    value._closed = False
    value.scene = SimpleNamespace(instance_ids=[7], xyz_m=np.zeros((8, 3)))
    value.prediction = None
    value.connected = True
    value.cameras_connected = True
    value.robot_connected = True
    value.robot_enabled = True
    value.robot_paused = False
    value.active_task = None
    value.task_finished = False
    value.last_scene_xyz = None
    value.no_change_count = 0
    value.last_executed_action = None
    value.stop_reason = None
    value.cycle_started = None
    value.cycle_timings = {}
    value.manual_session = None
    value.pending_task = None
    value.manual_stage = "capture"
    value.manual_config_signature = None
    value.pending_task_grasp = None
    return value


def test_tcp_compensation_is_persisted():
    value = controller()
    saved = []
    value.config = SimpleNamespace(
        raw={"robot": {"model_tcp_to_robot_tcp": {}}},
        save=lambda: saved.append(True),
    )
    value.set_tcp_compensation([1, 2, 3, 4, 5, 6])
    assert saved == [True]
    assert value.config.raw["robot"]["model_tcp_to_robot_tcp"]["xyz_mm_rpy_deg"] == [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_stop_cancels_workers_and_marks_operator_stop():
    value = controller()
    value.predictor = SimpleNamespace(close=lambda: setattr(value, "predictor_closed", True))
    value.physics = SimpleNamespace(close=lambda: setattr(value, "physics_closed", True))
    value.stop()
    assert value.stop_reason == "操作者点击停止"
    assert value.predictor is None and value.physics is None
    assert value.predictor_closed and value.physics_closed


def test_offline_stop_does_not_attempt_robot_rpc():
    value = controller()
    value.robot_connected = False
    value.stop()
    assert not hasattr(value.robot, "stopped")


def test_manual_push_branch_orders_relation_rules_then_evaluator():
    value = controller()
    rng = np.random.default_rng(2)
    lower = rng.normal((0, 0, 0.04), 0.006, (80, 3))
    upper = rng.normal((0, 0, 0.09), 0.006, (80, 3))
    value.scene = FusedScene(
        np.concatenate((lower, upper)), np.zeros((160, 3)),
        np.repeat([1, 2], 80), np.zeros(160, int), {},
    )
    calls = []
    class StagedPredictor(FakePredictor):
        def analyze(self, _scene, _target, _category, _region):
            calls.append("grasp")
            return Prediction({}, 0.1, (), target_query=1)

        def generate_push_rules(self, obstruction):
            calls.append("rules")
            assert obstruction == (2,)
            return ({"action_type": 0, "acted_object": 2},)

        def score_push_rules(self):
            calls.append("evaluator")
            return ({"candidate_index": 9, "action_type": 0, "acted_object": 2,
                     "improvement_probability": 0.8},)

    value.predictor = StagedPredictor()
    value.physics = SimpleNamespace(collision_free_grasps=lambda *_args: [])
    value.manual_stage = "select_target"
    value.select_manual_target(1, 3, 2)
    value.manual_target_grasp()
    assert value.manual_stage == "obstruction"
    value.manual_obstruction()
    assert calls == ["grasp"]
    assert value.manual_stage == "push_rules"
    value.manual_push_rules()
    assert calls == ["grasp", "rules"]
    value.manual_push_scoring()
    assert calls == ["grasp", "rules", "evaluator"]
    assert value.prediction.action["candidate_index"] == 9
    assert value.manual_stage == "execute"
    assert value.robot.actions == []


def test_manual_adjacent_blocker_reaches_push_scoring_without_overhead_relation():
    value = controller()
    rng = np.random.default_rng(15)
    target = rng.normal((0, 0, .04), .006, (80, 3))
    neighbor = rng.normal((.12, 0, .04), .006, (80, 3))
    value.scene = FusedScene(
        np.vstack((target, neighbor)), np.zeros((160, 3)),
        np.repeat([1, 7], 80), np.zeros(160, int), {},
    )
    calls = []
    class StagedPredictor(FakePredictor):
        def analyze(self, *_args):
            return Prediction({}, .1, ({"candidate_index": 1, "action_type": 2,
                                        "acted_object": 1, "proposal_score": .9,
                                        "grasp_pose_world": [0, 0, .1, 1, 0, 0, 0]},), target_query=1)

        def generate_push_rules(self, objects, *, adjacent_objects):
            calls.append((objects, adjacent_objects))
            return ({"candidate_index": 0, "action_type": 0, "acted_object": 7},)

        def score_push_rules(self):
            return ({"candidate_index": 0, "action_type": 0, "acted_object": 7,
                     "improvement_probability": .8},)

    class DetailedPhysics:
        def assess_grasps(self, _scene, candidates, _target):
            return ({"candidate_index": candidates[0]["candidate_index"],
                     "collision_free": False, "neighbor_ids": (7,),
                     "table_collision": False, "target_non_finger_collision": False},)

    value.predictor = StagedPredictor()
    value.physics = DetailedPhysics()
    value.manual_stage = "select_target"
    value.select_manual_target(1, 3, 2)
    value.manual_target_grasp()
    assert value.manual_obstruction()["obstructions"] == ()
    assert value.manual_stage == "push_rules"
    value.manual_push_rules()
    assert calls == [((7,), (7,))]
    value.manual_push_scoring()
    assert value.prediction.action["acted_object"] == 7


def test_preparation_action_forces_reobservation_and_reidentifies_target():
    value = controller()
    first_scene = value.scene
    value.predict(7, 3, 2)
    assert value.predictor.calls[-1] == (first_scene, 7, 3, 2)

    assert "重新采集" in value.execute(-1, 3, 2)
    assert value.scene is None and value.prediction is None
    assert len(value.robot.actions) == len(value.predictor.executed) == 1
    with pytest.raises(RuntimeError, match="重新采集"):
        value.predict(-1, 3, 2)

    second_scene = SimpleNamespace(instance_ids=[19], xyz_m=np.zeros((8, 3)))
    value.scene = second_scene
    value.predict(-1, 3, 2)
    assert value.predictor.calls[-1] == (second_scene, None, 3, 2)


def test_active_task_locks_semantics_and_task_grasp_requires_reset():
    value = controller()
    prediction = value.predict(7, 3, 2)
    with pytest.raises(RuntimeError, match="不能改变"):
        value.predict(-1, 4, 2)

    prediction.action["action_type"] = int(ActionType.TASK_GRASP)
    value.prediction = prediction
    assert "确认抓取结果" in value.execute(-1, 3, 2)
    assert not value.task_finished
    assert not value.predictor.executed
    assert "确认成功" in value.confirm_task_grasp(True)
    assert value.task_finished
    assert len(value.predictor.executed) == 1
    value.scene = SimpleNamespace(instance_ids=[7], xyz_m=np.zeros((8, 3)))
    with pytest.raises(RuntimeError, match="重置任务"):
        value.predict(7, 3, 2)

    value.reset_task()
    assert value.active_task is None and not value.task_finished
    assert value.predictor.resets == 1


def test_failed_task_grasp_continues_original_task():
    value = controller()
    prediction = value.predict(7, 3, 2)
    prediction.action["action_type"] = int(ActionType.TASK_GRASP)
    value.prediction = prediction
    value.execute(-1, 3, 2)
    assert "未成功" in value.confirm_task_grasp(False)
    assert not value.task_finished
    assert value.active_task == (3, 2)
    assert not value.predictor.executed


def test_servo_gate_and_recovery_controls():
    value = controller()
    value.robot_enabled = False
    value.predict(7, 3, 2)
    with pytest.raises(RuntimeError, match="未使能"):
        value.execute(-1, 3, 2)
    assert "使能" in value.enable_robot()
    assert value.robot_enabled
    assert "暂停" in value.pause_robot() and value.robot_paused
    assert "继续" in value.resume_robot() and not value.robot_paused
    value.scene = SimpleNamespace(instance_ids=[7], xyz_m=np.zeros((8, 3)))
    assert "安全原点" in value.home_robot()
    assert value.scene is None and value.prediction is None
    assert "下使能" in value.disable_robot()
    assert not value.robot_enabled


def test_disconnected_robot_marks_prediction_only():
    value = controller()
    value.connected = False
    value.robot_connected = False
    prediction = value.predict(7, 3, 2)
    message = value.execute(-1, 3, 2)
    assert "仅预测完成" in message
    assert prediction.status == "prediction_only"


def test_camera_connection_survives_missing_robot():
    value = ExperimentController.__new__(ExperimentController)
    camera = FakeCamera()
    value.cameras = [camera]
    value.robot = OfflineRobot()
    value.connected = False
    value.cameras_connected = False
    value.robot_connected = False
    value.robot_enabled = False
    message = value.connect()
    assert value.cameras_connected and value.connected
    assert not value.robot_connected and not camera.disconnected
    assert "机械臂未连接" in message


def test_camera_and_robot_can_be_connected_independently():
    class OnlineRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.connect_calls = 0

        def connect(self):
            self.connect_calls += 1
            return True

    value = ExperimentController.__new__(ExperimentController)
    value.cameras = [FakeCamera()]
    value.robot = OnlineRobot()
    value.connected = value.cameras_connected = value.robot_connected = False
    value.robot_enabled = False
    assert "相机" in value.connect_cameras()
    assert value.cameras_connected and not value.robot_connected
    assert value.robot.connect_calls == 0
    assert "机械臂" in value.connect_robot()
    assert value.robot_connected and value.robot.connect_calls == 1


def test_manual_stage_rejects_out_of_order_and_invalidates_changed_config():
    value = controller()
    value.manual_stage = "obstruction"
    value.manual_session = object()
    value.config.raw = {"workflow": {"push_top_k": 5}, "physics": {"object_mass_kg": .2}}
    value.manual_config_signature = value._decision_config_signature()
    value._require_manual_stage("obstruction")
    with pytest.raises(RuntimeError, match="不能执行"):
        value._require_manual_stage("pybullet")
    value.config.raw["workflow"]["push_top_k"] = 10
    with pytest.raises(RuntimeError, match="下游结果已失效"):
        value._require_manual_stage("obstruction")
    assert value.manual_stage == "target_grasp" and value.manual_session is None
