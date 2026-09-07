from types import SimpleNamespace

import pytest

from real_experiment_app.controller import ExperimentController
from real_experiment_app.types import Prediction
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


def controller():
    value = ExperimentController.__new__(ExperimentController)
    value.config = SimpleNamespace()
    value.cameras = []
    value.robot = FakeRobot()
    value.predictor = FakePredictor()
    value.scene = SimpleNamespace(instance_ids=[7])
    value.prediction = None
    value.connected = True
    value.robot_enabled = True
    value.robot_paused = False
    value.active_task = None
    value.task_finished = False
    return value


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

    second_scene = SimpleNamespace(instance_ids=[19])
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
    assert "请检查实物结果" in value.execute(-1, 3, 2)
    value.scene = SimpleNamespace(instance_ids=[7])
    with pytest.raises(RuntimeError, match="重置任务"):
        value.predict(7, 3, 2)

    value.reset_task()
    assert value.active_task is None and not value.task_finished
    assert value.predictor.resets == 1


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
    value.scene = SimpleNamespace(instance_ids=[7])
    assert "安全原点" in value.home_robot()
    assert value.scene is None and value.prediction is None
    assert "下使能" in value.disable_robot()
    assert not value.robot_enabled
