import subprocess
import threading
from types import SimpleNamespace

from real_experiment_app.controller import ExperimentController
from real_experiment_app.physics_client import PhysicsClient
from real_experiment_app.predictor_client import PredictorClient


class StubbornProcess:
    def __init__(self):
        self.calls = []
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")

    def wait(self, timeout):
        self.calls.append(("wait", timeout))
        if "kill" not in self.calls:
            raise subprocess.TimeoutExpired("worker", timeout)
        self.returncode = -9

    def kill(self):
        self.calls.append("kill")


def _assert_forced_shutdown(client_type):
    client = client_type.__new__(client_type)
    client.process = StubbornProcess()
    cleanup = []
    client.temp = SimpleNamespace(cleanup=lambda: cleanup.append(True))
    client.close()
    assert client.process.calls == ["terminate", ("wait", 5), "kill", ("wait", 5)]
    assert cleanup == [True]


def test_predictor_shutdown_terminates_then_kills_and_cleans_temp():
    _assert_forced_shutdown(PredictorClient)


def test_physics_shutdown_terminates_then_kills_and_cleans_temp():
    _assert_forced_shutdown(PhysicsClient)


def test_idle_worker_receives_close_protocol_before_forced_termination():
    class GracefulProcess:
        def __init__(self):
            self.commands = []
            self.stdin = SimpleNamespace(
                write=lambda value: self.commands.append(value),
                flush=lambda: None,
                closed=False,
            )
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if not self.commands:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = 0

        def terminate(self):
            self.terminated = True

    for client_type in (PredictorClient, PhysicsClient):
        client = client_type.__new__(client_type)
        client.process = GracefulProcess()
        client.temp = SimpleNamespace(cleanup=lambda: None)
        client.close()
        assert client.process.commands == ['{"command": "close"}\n']
        assert not client.process.terminated


def test_close_during_model_startup_does_not_leave_worker_running(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    created = []

    class DelayedPredictor:
        def __init__(self, _path):
            self.closed = False
            created.append(self)
            entered.set()
            assert release.wait(3)

        def close(self):
            self.closed = True

    monkeypatch.setattr("real_experiment_app.controller.PredictorClient", DelayedPredictor)
    controller = ExperimentController.__new__(ExperimentController)
    controller.config = SimpleNamespace(path="unused")
    controller.predictor = None
    controller.physics = None
    controller._worker_lock = threading.Lock()
    controller._worker_epoch = 0
    controller._closed = False
    controller.cameras = []
    controller.robot = SimpleNamespace(disconnect=lambda: None)
    errors = []

    def load_model():
        try:
            controller.load_model()
        except RuntimeError as error:
            errors.append(str(error))

    background = threading.Thread(target=load_model)
    background.start()
    assert entered.wait(3)
    controller.close()
    release.set()
    background.join(3)
    assert not background.is_alive()
    assert created[0].closed
    assert errors == ["后台任务已取消"]
