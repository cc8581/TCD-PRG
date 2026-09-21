from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
from PySide6.QtWidgets import QApplication

from real_experiment_app.config import AppConfig
from real_experiment_app.main import MainWindow
from real_experiment_app.offline_ui import (
    cached_states_for_task,
    local_scene_ids,
    local_scene_shape,
)
from tcd_prg.observation.cached import ObservationCacheMissError


def test_local_selector_lists_only_published_scene_pairs(tmp_path: Path, monkeypatch):
    root = tmp_path / "dataset"
    scenes = root / "task_clutter_scenes_20_categories"
    labels = root / "task_positive_multistep_sequences" / "scene_labels"
    (scenes / "scene_0007").mkdir(parents=True)
    (scenes / "scene_0007" / "scene.npz").touch()
    (scenes / "scene_0008").mkdir()
    (scenes / "scene_0008" / "scene.npz").touch()
    labels.mkdir(parents=True)
    with h5py.File(labels / "scene_0007.h5", "w") as handle:
        scene = handle.create_group("scene_0007")
        states = scene.create_group("states")
        states.create_dataset("object_pose", shape=(3, 1, 7))
        catalog = scene.create_group("catalog")
        catalog.create_dataset("task_object_index", shape=(2,))
    monkeypatch.setattr(
        "real_experiment_app.offline_ui.dataset_root_from_app_config",
        lambda _config: str(root),
    )
    config = SimpleNamespace()
    assert local_scene_ids(config) == (7,)
    assert local_scene_shape(config, 7) == (3, 2)


def test_local_selector_reports_missing_dataset(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "real_experiment_app.offline_ui.dataset_root_from_app_config",
        lambda _config: str(tmp_path / "missing"),
    )
    with pytest.raises(FileNotFoundError, match="不完整"):
        local_scene_ids(SimpleNamespace())


def test_cached_state_choices_check_exact_scene_and_task_without_loading(monkeypatch):
    checked = []

    def available(keys):
        checked.extend(keys)
        return {key: key[1] in (0, 2) for key in keys}

    monkeypatch.setattr(
        "real_experiment_app.offline_ui._offline_adapter",
        lambda _: SimpleNamespace(observations_available=available),
    )
    monkeypatch.setattr("real_experiment_app.offline_ui.local_scene_shape", lambda *_: (4, 3))
    assert cached_states_for_task(SimpleNamespace(), 51, 22) == (0, 2)
    assert checked == [(51, state, 22) for state in range(4)]


def test_missing_selected_cache_lists_manual_alternatives_without_switching(monkeypatch):
    app = QApplication.instance() or QApplication([])
    config = AppConfig.load(Path(__file__).parents[1] / "configs" / "real_experiment.yaml")
    window = MainWindow(config)
    window._test_app = app
    window.local_scene.addItem("scene_0051", 51)
    window.local_scene.setCurrentIndex(window.local_scene.count() - 1)
    window.local_state.setRange(0, 162)
    window.local_state.setValue(10)
    window.local_task.setRange(0, 28)
    window.local_task.setValue(22)

    def cache_miss(*_args):
        raise ObservationCacheMissError("Observation ad20dd is not cached")

    monkeypatch.setattr("real_experiment_app.offline_ui.load_offline_scene", cache_miss)
    monkeypatch.setattr(
        "real_experiment_app.offline_ui.cached_states_for_task",
        lambda *_args: (0, 8, 13),
        raising=False,
    )
    monkeypatch.setattr(window, "run_job", lambda function, callback: callback(function()))
    window.start_offline_replay(51, 10, 22)
    assert "state 10" in window.result.toPlainText()
    assert "0、8、13" in window.result.toPlainText()
    assert window.local_scene.currentData() == 51
    assert window.local_state.value() == 10
    assert window.local_task.value() == 22
    assert window.controller.scene is None
    window.close()
