import io
import json
from types import SimpleNamespace

import pytest

from real_experiment_app.predictor_client import PredictorClient


def test_call_forwards_progress_events_before_final_response():
    client = PredictorClient.__new__(PredictorClient)
    client.process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO(
            "not-json diagnostic\n"
            '{"event":"progress","update":{"stage":"MODEL_SCENE_ENCODING"}}\n'
            '{"event":"progress","update":{"stage":"MODEL_CANDIDATE_GENERATION"}}\n'
            '{"ok":true,"result":{"value":7}}\n'
        ),
    )
    updates = []
    result = client._call("analyze", progress=updates.append, target=1)
    assert result == {"value": 7}
    assert [item["stage"] for item in updates] == [
        "MODEL_SCENE_ENCODING",
        "MODEL_CANDIDATE_GENERATION",
    ]
    assert '"command": "analyze"' in client.process.stdin.getvalue()


def test_call_ignores_progress_when_no_callback_is_registered():
    client = PredictorClient.__new__(PredictorClient)
    client.process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO(
            '{"event":"progress","update":{"stage":"MODEL_SCENE_ENCODING"}}\n'
            '{"ok":true,"result":true}\n'
        ),
    )
    assert client._call("analyze") is True


def test_call_transports_non_ascii_scene_path_without_console_codepage_loss():
    scene_path = r"D:\Codex运行垃圾\scene.npz"
    client = PredictorClient.__new__(PredictorClient)
    client.process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO('{"ok":true,"result":true}\n'),
    )
    assert client._call("analyze", scene=scene_path) is True
    serialized = client.process.stdin.getvalue()
    assert serialized.isascii()
    assert json.loads(serialized)["scene"] == scene_path


def test_adjacent_push_eligibility_is_sent_to_worker():
    client = PredictorClient.__new__(PredictorClient)
    sent = []
    client._call = lambda command, **payload: sent.append((command, payload)) or ()
    assert client.generate_push_rules((3, 7), adjacent_objects=(7,)) == ()
    assert sent == [("generate_push_rules", {
        "obstructions": [3, 7], "adjacent_objects": [7],
    })]


def test_call_reports_exited_worker_before_writing_to_closed_stdin():
    client = PredictorClient.__new__(PredictorClient)
    stream = io.StringIO()
    stream.close()
    client.process = SimpleNamespace(
        stdin=stream,
        stdout=io.StringIO("worker failed\n"),
        poll=lambda: 9,
    )
    with pytest.raises(RuntimeError, match="模型工作进程已退出.*exit code 9"):
        client._call("perceive", scene="scene.npz")
