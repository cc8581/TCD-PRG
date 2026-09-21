import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDockWidget

from real_experiment_app.config import AppConfig
from real_experiment_app.main import MainWindow


def window():
    app = QApplication.instance() or QApplication([])
    config = AppConfig.load(Path(__file__).parents[1] / "configs" / "real_experiment.yaml")
    value = MainWindow(config)
    value._test_app = app
    return value


def test_stop_is_available_for_busy_offline_automatic_work():
    value = window()
    value.offline_mode = True
    value.busy = True
    value._enable()
    assert value.stop_button.isEnabled()
    value.controller.close()


def test_mode_switch_shows_the_correct_source_without_enabling_robot():
    value = window()
    assert value.run_mode.currentData() == "simulation"
    assert value.local_scene.isVisibleTo(value)
    assert not value.acquire_button.isVisibleTo(value)
    assert not value.execute_button.isEnabled()
    value.run_mode.setCurrentIndex(value.run_mode.findData("real"))
    assert "点云采集" in value.flow.nodes["source"].text()
    assert value.acquire_button.isVisibleTo(value)
    assert not value.local_scene.isVisibleTo(value)
    assert not value.execute_button.isEnabled()
    value.controller.close()


def test_top_mode_switch_shows_only_its_dedicated_button_area():
    value = window()
    value.show()
    value._test_app.processEvents()
    assert value.mode_switch.isVisibleTo(value)
    assert value.mode_switch.value() == 0
    assert value.simulation_controls.isVisibleTo(value)
    assert not value.real_controls.isVisibleTo(value)
    assert value.sim_segment_button.isVisibleTo(value)
    assert not value.camera_connect_button.isVisibleTo(value)
    value.mode_switch.setValue(1)
    value._test_app.processEvents()
    assert value.run_mode.currentData() == "real"
    assert value.real_controls.isVisibleTo(value)
    assert not value.simulation_controls.isVisibleTo(value)
    assert value.camera_connect_button.isVisibleTo(value)
    assert value.robot_connect_button.isVisibleTo(value)
    assert value.camera_connect_button.isEnabled()
    assert value.robot_connect_button.isEnabled()
    assert value.segment_button.isVisibleTo(value)
    assert not value.sim_segment_button.isVisibleTo(value)
    value.mode_switch.setValue(0)
    value._test_app.processEvents()
    assert value.run_mode.currentData() == "simulation"
    assert value.simulation_controls.isVisibleTo(value)
    assert not value.real_controls.isVisibleTo(value)
    value.controller.close()


def test_simulation_controls_keep_inputs_next_to_their_flow_steps():
    value = window()
    value.show()
    value._test_app.processEvents()
    assert value.simulation_layout.indexOf(value.local_refresh_button) == 0
    assert value.simulation_layout.indexOf(value.sim_action_card) == 1
    assert value.local_scene.parentWidget().parentWidget() is value.sim_action_card
    assert value.target_controls.parentWidget() is value.sim_target_row
    assert not value.local_state.isVisibleTo(value)
    assert not value.local_task.isVisibleTo(value)
    assert value.local_load_button.text().startswith("1")
    for button in value._sim_flow_buttons.values():
        assert button.maximumWidth() == 210
        assert button.property("workflow_state") == "pending"
    assert value.sim_grasp_result.text() == "抓取成功分支：等待预测"
    assert value.sim_push_branch_label.text() == "抓取不可行 → PUSH"
    value.controller.close()


def test_compact_simulation_inputs_and_right_bottom_floating_result_dock():
    value = window()
    assert value.local_scene.placeholderText() == ""
    assert value.local_scene.width() == 140
    target_layout = value.target_controls.layout()
    assert target_layout.count() == 5
    assert target_layout.itemAt(0).widget() is value.instance
    assert target_layout.itemAt(2).widget() is value.category
    assert target_layout.itemAt(4).widget() is value.region
    assert value.result.parentWidget() is value.result_dock
    assert value.dockWidgetArea(value.controls_dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert value.dockWidgetArea(value.result_dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert value.result_dock.features() & QDockWidget.DockWidgetFeature.DockWidgetMovable
    assert value.result_dock.features() & QDockWidget.DockWidgetFeature.DockWidgetFloatable
    value.controller.close()


def test_simulation_button_states_follow_workflow_not_candidate_data():
    value = window()
    value._set_step("perception")
    assert value.sim_segment_button.property("workflow_state") == "running"
    assert value.sim_predict_button.property("workflow_state") == "pending"
    value._complete_steps(2)
    assert value.sim_segment_button.property("workflow_state") == "success"
    value.controller.close()


def test_mode_switch_reverts_when_hardware_is_connected():
    value = window()
    value.controller.cameras_connected = True
    value.mode_switch.setValue(1)
    assert value.run_mode.currentData() == "simulation"
    assert value.mode_switch.value() == 0
    assert value.simulation_controls.isVisibleTo(value)
    assert not value.real_controls.isVisibleTo(value)
    value.controller.cameras_connected = False
    value.controller.close()


def test_simulation_rejects_execution_even_when_a_prediction_exists():
    from real_experiment_app.types import Prediction

    value = window()
    value.controller.prediction = Prediction({"action_type": 0}, 0.1)
    value.start_execution()
    assert "不发送机械臂动作" in value.status.text()
    assert not value.execute_button.isEnabled()
    value.controller.close()


def test_prediction_highlights_only_the_entered_branch(monkeypatch):
    from real_experiment_app.types import Prediction

    value = window()
    monkeypatch.setattr(value.canvas, "set_data", lambda *_args, **_kwargs: None)
    grasp = {
        "action_type": 2, "acted_object": 3,
        "grasp_pose_world": np.array([0.0, 0.0, .2, 1.0, 0.0, 0.0, 0.0]),
        "grasp_width_m": 0.04,
    }
    value.on_prediction(Prediction(grasp, 0.1, decision_path=("TARGET_GRASP",)))
    assert value.flow.active_branch == "grasp"
    assert value.flow.grasp_branch.property("active") is True
    assert value.flow.push_branch.property("active") is False
    assert value.flow.states["grasp_execute"] == "success"

    value.flow.reset()
    push = {
        "action_type": 0, "acted_object": 4,
        "push_contact_world": np.zeros(3),
        "push_direction_world": np.array([1.0, 0.0, 0.0]),
        "push_distance_m": 0.15,
        "improvement_probability": 0.8,
    }
    value.on_prediction(Prediction(
        push, 0.2, decision_path=("TARGET_GRASP", "OBSTRUCTION_INFERENCE",
                                  "PUSH_RULE_GENERATION", "PUSH_EVALUATOR_SCORING", "PUSH"),
    ))
    assert value.flow.active_branch == "push"
    assert value.flow.grasp_branch.property("active") is False
    assert value.flow.push_branch.property("active") is True
    assert all(value.flow.states[stage] == "success" for stage in
               ("obstruction", "push_rules", "push_score", "push_execute"))
    value.controller.close()


def test_grasp_branch_candidate_selector_changes_only_the_visualized_pose(monkeypatch):
    from real_experiment_app.types import Prediction

    value = window()
    value.controller.scene = object()
    rendered = []
    monkeypatch.setattr(
        value.canvas, "set_data", lambda _scene, _target, action, candidates, **_kwargs: rendered.append(
            (action, candidates)
        )
    )
    first = {
        "action_type": 2, "acted_object": 3, "proposal_score": .9,
        "grasp_pose_world": np.array([0.0, 0.0, .2, 1.0, 0.0, 0.0, 0.0]),
        "grasp_width_m": .04,
    }
    second_pose = np.array([.03, 0.0, .2, 1.0, 0.0, 0.0, 0.0])
    second = {
        "action_type": 2, "acted_object": 3, "proposal_score": .7,
        "grasp_pose_world": second_pose, "grasp_width_m": .05,
    }
    upward = {
        "action_type": 2, "acted_object": 3, "proposal_score": .8,
        "grasp_pose_world": np.array([0.0, 0.0, .2, 0.0, 0.0, 0.0, 1.0]),
        "grasp_width_m": .04,
    }
    prediction = Prediction(first, .1, (first, second, upward), target_query=3)

    value.on_prediction(prediction)
    assert value.sim_grasp_candidates.count() == 2
    assert rendered[-1][0] is first
    value.sim_grasp_candidates.setCurrentIndex(1)
    assert rendered[-1][0] is second
    assert prediction.action is first
    value.controller.close()


def test_failed_grasp_candidates_remain_previewable_during_push_branch(monkeypatch):
    from real_experiment_app.types import Prediction
    from real_experiment_app.workflow import ObstructionRelation
    value = window()
    value.controller.scene = object()
    value.controller.manual_session = SimpleNamespace(target=8)
    value.current_step = "grasp"
    rendered = []
    monkeypatch.setattr(value.canvas, "set_data", lambda *args, **kwargs: rendered.append((args, kwargs)))
    first = {"action_type": 2, "acted_object": 8, "proposal_score": .9,
             "grasp_pose_world": [0., 0., .2, 1., 0., 0., 0.], "grasp_width_m": .04}
    second = {**first, "grasp_pose_world": [.03, 0., .2, 1., 0., 0., 0.]}
    value.on_manual_update({"stage": "TARGET_GRASP_AND_COLLISION",
                            "candidates": (first, second), "collision_free": (), "selected": None})
    assert value.sim_grasp_candidates.count() == 2
    assert not value.sim_grasp_candidates.isHidden()
    assert value.controller.prediction is None
    value._stage_obstructions(value.controller.scene, 8, {"relations": (ObstructionRelation(8, 3, .4, .08),)})
    value.on_prediction(Prediction({"reason": "没有有效PUSH候选"}, .1,
                                    target_query=8, status="operator_attention",
                                    decision_path=("OBSTRUCTION_INFERENCE",)))
    value.sim_grasp_candidates.setCurrentIndex(1)
    assert np.allclose(rendered[-1][0][2]["grasp_pose_world"], second["grasp_pose_world"])
    assert rendered[-1][1]["obstructions"] == (3,)
    assert value.controller.prediction is None
    value.controller.close()


def test_adjacent_blockers_are_highlighted_alongside_overhead_blockers():
    from real_experiment_app.workflow import ObstructionRelation
    value = window()
    scene = object()
    ids = value._stage_obstructions(scene, 8, {
        "relations": (ObstructionRelation(8, 3, .4, .08),),
        "adjacent_blockers": (7, 9),
    })
    assert ids == (3, 7, 9)
    value.controller.close()


def test_stale_job_completion_cannot_run_callback():
    value = window()
    called = []
    value.job_generation = 4
    value._job_done_if_current(3, "late", called.append)
    assert called == []
    value.controller.close()


def test_completion_callback_error_uses_failure_path():
    value = window()
    failures = []
    value.job_failed = failures.append
    value.job_done("result", lambda _value: (_ for _ in ()).throw(ValueError("callback failed")))
    assert failures and "callback failed" in failures[0]
    value.controller.close()


def test_local_scene_controls_are_available_without_devices(monkeypatch):
    monkeypatch.setattr("real_experiment_app.offline_ui.local_scene_shape", lambda *_: (3, 2))
    value = window()
    value._on_local_scenes((7, 8))
    assert value.local_scene.count() == 2
    assert value.local_scene.currentIndex() == -1
    assert not value.local_load_button.isEnabled()
    assert not value.execute_button.isEnabled()
    value.load_selected_local_scene()
    assert not value.offline_mode
    value.local_scene.setCurrentIndex(1)
    assert value.local_load_button.isEnabled()
    value._on_local_scenes((7, 8))
    assert value.local_scene.currentIndex() == -1
    assert not value.local_load_button.isEnabled()
    value.controller.close()


def test_compact_local_scene_loader_uses_the_initial_observation(monkeypatch):
    value = window()
    value.local_scene.addItem("scene_0007", 7)
    value.local_scene.setCurrentIndex(0)
    requested = []
    monkeypatch.setattr(value, "start_offline_replay", lambda *args: requested.append(args))

    value.load_selected_local_scene()

    assert requested == [(7, 0, 0)]
    value.controller.close()


def test_loaded_scene_is_rendered_before_instance_segmentation(monkeypatch):
    value = window()
    rendered = []

    def record(scene, target=None, action=None, candidates=()):
        rendered.append((scene, target, action, candidates))

    monkeypatch.setattr(
        value.canvas, "set_data", record
    )
    scene = SimpleNamespace(
        instance_ids=(), xyz_m=np.zeros((4, 3)), rgb=np.zeros((4, 3)),
        instance_id=np.full(4, -1), category_by_instance={},
    )
    value.controller.manual_stage = "perceive"

    value.on_scene(scene)

    assert rendered == [(scene, None, None, ())]
    value.controller.close()


def test_offline_load_requires_target_confirmation_before_grasp(monkeypatch):
    from real_experiment_app import main

    value = window()
    jobs = []
    scheduled = []
    def fail_on_job_error(failure):
        raise AssertionError(failure)

    monkeypatch.setattr(value, "run_job", lambda work, callback: jobs.append((work, callback)))
    monkeypatch.setattr(value, "job_failed", fail_on_job_error)
    monkeypatch.setattr(
        value, "on_scene",
        lambda scene: value.instance.addItem("target", 3) if scene.instance_ids else None,
    )
    monkeypatch.setattr(value.canvas, "set_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main.QTimer, "singleShot", lambda *args: scheduled.append(args))
    value.start_offline_replay(7, 0, 0)
    scene = SimpleNamespace(
        instance_ids=(), xyz_m=np.zeros((1, 3)), instance_id=np.array([-1]),
    )
    value.instance.blockSignals(True)
    value.job_done((scene, np.zeros(3), 1, 2, {"input_points": 855}), jobs[0][1])
    value.instance.blockSignals(False)
    assert scheduled == []
    assert value.controller.manual_stage == "perceive"
    assert value.segment_button.isEnabled()
    assert not value.fuse_button.isEnabled()
    def perceive():
        scene.instance_ids = (3,)
        scene.instance_id = np.array([3])
        value.controller.manual_stage = "select_target"
        return scene

    monkeypatch.setattr(value.controller, "perceive_scene", perceive)
    value.start_second_stage()
    value.instance.blockSignals(True)
    value.job_done(jobs[1][0](), jobs[1][1])
    value.instance.blockSignals(False)
    assert value.controller.manual_stage == "select_target"
    assert value.fuse_button.isEnabled()
    assert not value.predict_button.isEnabled()
    value.start_third_stage()
    value.job_done(jobs[2][0](), jobs[2][1])
    assert value.controller.manual_stage == "target_grasp"
    assert value.predict_button.isEnabled()
    assert not value.execute_button.isEnabled()
    value.reset_task()
    assert value.controller.manual_stage == "select_target"
    assert value.fuse_button.isEnabled()
    assert not value.predict_button.isEnabled()
    value.controller.close()


def test_offline_stays_manual_even_when_device_workflow_is_auto(monkeypatch):
    value = window()
    value.config.raw["workflow"]["mode"] = "auto"
    value.config.raw["workflow"]["target_mode"] = "random"
    value.offline_mode = True
    value.controller.scene = SimpleNamespace(instance_ids=(3,))
    value.controller.manual_stage = "select_target"
    value.instance.addItem("target", 3)
    value.instance.blockSignals(True)
    value.instance.setCurrentIndex(0)
    value.instance.blockSignals(False)
    value._enable()
    assert value.fuse_button.isEnabled()
    assert not value.predict_button.isEnabled()
    assert not value.execute_button.isEnabled()
    jobs = []
    monkeypatch.setattr(value, "run_job", lambda work, _callback: jobs.append(work))
    value.start_third_stage()
    jobs[0]()
    assert value.controller.pending_task[0] == 3
    assert value.controller.manual_stage == "target_grasp"
    stages = []
    monkeypatch.setattr(
        value, "start_manual_decision_stage", lambda index, _work: stages.append(index)
    )
    value.start_fourth_stage()
    assert stages == ["grasp"]
    value.controller.close()


def test_local_scene_cannot_start_with_connected_robot():
    value = window()
    value.controller.robot_connected = True
    value.start_offline_replay(0, 0, 0)
    assert not value.offline_mode
    assert "未连接设备" in value.status.text()
    value.controller.robot_connected = False
    value.controller.close()
