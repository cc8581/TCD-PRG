#!/usr/bin/env python3
"""Batch-precompute TCD-PRG content-addressed observation point clouds.

The tool deliberately reuses the repository's ObservationRequest and
request_hash contracts, while bypassing the bounded LRU writer used during
read-through training.  Requests are scanned into a disk-backed SQLite
manifest, de-duplicated by the exact content hash, then rendered in parallel
through the configured external PyBullet worker.

Designed for Windows/Python 3.10 and the current TCD-PRG repository layout.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import queue
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import h5py
import numpy as np

SCRIPT_VERSION = "2026.08.06.2"
DEFAULT_ESTIMATED_BYTES = 403_066
STATUS_PENDING = "pending"
STATUS_CACHED = "cached"
STATUS_GENERATED = "generated"
STATUS_FAILED = "failed"
STATUS_MISSING = "missing"
STATUS_INVALID = "invalid"
FINAL_STATUSES = {STATUS_CACHED, STATUS_GENERATED}
REQUIRED_FIELDS = ("xyz", "rgb", "instance_id", "source_view")


def _discover_project() -> Path:
    explicit: list[Path] = []
    environment = os.environ.get("TCD_PRG_PROJECT_ROOT")
    if environment:
        explicit.append(Path(environment))
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--project-root":
            explicit.append(Path(sys.argv[index + 1]))
        elif value.startswith("--project-root="):
            explicit.append(Path(value.split("=", 1)[1]))
    candidates = [*explicit, Path.cwd(), Path(__file__).resolve().parent]
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        candidate = candidate.expanduser()
        if (candidate / "tcd_prg").is_dir() and (candidate / "configs" / "config.yaml").is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot locate the TCD-PRG project root. Copy this script to "
        "<TCD-PRG>\\scripts, set TCD_PRG_PROJECT_ROOT, or pass --project-root."
    )


PROJECT = _discover_project()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tcd_prg.config import TCDPRGConfig, load_config  # noqa: E402
from tcd_prg.constants import ActionType  # noqa: E402
from tcd_prg.datasets.task_oriented_clutter import TaskOrientedClutterAdapter  # noqa: E402
from tcd_prg.observation.base import (  # noqa: E402
    ObservationProvider,
    ObservationRequest,
    PointObservation,
)
from tcd_prg.observation.cached import (  # noqa: E402
    CachedObservationProvider,
    request_hash,
)


class DiskSpaceReserveError(OSError):
    """Raised when the configured free-space reserve can no longer be maintained."""


class NullObservationProvider(ObservationProvider):
    """Provider used while constructing exact requests without loading points."""

    def is_available(self, request: ObservationRequest) -> bool:
        return False

    def get(self, request: ObservationRequest) -> PointObservation:
        raise RuntimeError("NullObservationProvider cannot load observations")


@dataclass(frozen=True, slots=True)
class RequestPayload:
    key: str
    scene_id: int
    state_id: int
    task_index: int
    group_index: int
    object_count: int
    object_pose: np.ndarray
    object_active: np.ndarray
    object_present: np.ndarray
    object_asset_ids: tuple[str, ...]
    object_model_ids: tuple[str, ...]
    object_scales: np.ndarray
    render_seed: int
    camera_profile: str
    point_count: int
    renderer_version: str

    def to_request(self) -> ObservationRequest:
        return ObservationRequest(
            scene_id=self.scene_id,
            state_id=self.state_id,
            object_pose=self.object_pose,
            object_active=self.object_active,
            object_present=self.object_present,
            object_asset_ids=self.object_asset_ids,
            object_model_ids=self.object_model_ids,
            object_scales=self.object_scales,
            render_seed=self.render_seed,
            camera_profile=self.camera_profile,
            point_count=self.point_count,
            renderer_version=self.renderer_version,
        )


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    reason: str = ""
    size_bytes: int = 0
    point_count: int = 0


@dataclass(slots=True)
class RenderResult:
    key: str
    status: str
    size_bytes: int
    elapsed_seconds: float
    scene_id: int
    state_id: int
    task_index: int
    group_index: int
    error: str = ""


class PersistentRendererClient:
    """One long-lived external PyBullet process owned by one executor thread."""

    def __init__(
        self,
        *,
        python_executable: Path,
        batch_worker_script: Path,
        scene_root: Path,
        runtime_mesh_root: Path,
        width: int,
        height: int,
        startup_timeout: int,
        log_path: Path,
    ) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(python_executable),
            "-u",
            str(batch_worker_script),
            "--scene-root",
            str(scene_root),
            "--runtime-mesh-root",
            str(runtime_mesh_root),
            "--width",
            str(width),
            "--height",
            str(height),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(batch_worker_script.parent),
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Persistent renderer has no stdin/stdout pipe")
        self.responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self.reader = threading.Thread(
            target=self._reader_loop,
            name=f"renderer-reader-{self.process.pid}",
            daemon=True,
        )
        self.reader.start()
        ready = self._next_response(startup_timeout)
        if not ready.get("ready"):
            self.close(force=True)
            raise RuntimeError(f"Persistent renderer did not become ready: {ready}")

    def _reader_loop(self) -> None:
        assert self.process.stdout is not None
        with self.log_path.open("a", encoding="utf-8") as log:
            for line in self.process.stdout:
                text = line.rstrip("\r\n")
                if not text:
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    log.write(text + "\n")
                    log.flush()
                    continue
                if isinstance(value, dict):
                    self.responses.put(value)
                else:
                    log.write(text + "\n")
                    log.flush()
            self.responses.put(
                {
                    "process_closed": True,
                    "returncode": self.process.poll(),
                    "log": str(self.log_path),
                }
            )

    def _next_response(self, timeout_seconds: int) -> dict[str, Any]:
        try:
            response = self.responses.get(timeout=max(1, timeout_seconds))
        except queue.Empty as error:
            self.close(force=True)
            raise TimeoutError(f"Persistent renderer timed out; log={self.log_path}") from error
        if response.get("process_closed"):
            raise RuntimeError(
                f"Persistent renderer exited returncode={response.get('returncode')}; "
                f"log={response.get('log')}"
            )
        return response

    def render(
        self,
        request_path: Path,
        output_path: Path,
        request_id: str,
        timeout_seconds: int,
    ) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Persistent renderer already exited with {self.process.returncode}; "
                f"log={self.log_path}"
            )
        assert self.process.stdin is not None
        self.process.stdin.write(
            json.dumps(
                {
                    "id": request_id,
                    "request": str(request_path),
                    "output": str(output_path),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self.process.stdin.flush()
        response = self._next_response(timeout_seconds)
        if str(response.get("id", "")) != request_id:
            raise RuntimeError(f"Persistent renderer response id mismatch: {response}")
        if not response.get("ok"):
            raise RuntimeError(
                f"Persistent renderer failed: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )

    def close(self, force: bool = False) -> None:
        if self.process.poll() is not None:
            return
        if not force:
            try:
                assert self.process.stdin is not None
                request_id = f"shutdown-{self.process.pid}"
                self.process.stdin.write(
                    json.dumps({"id": request_id, "command": "shutdown"}) + "\n"
                )
                self.process.stdin.flush()
                self._next_response(30)
                self.process.wait(timeout=30)
                return
            except BaseException:
                pass
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


class PersistentRendererManager:
    """Lazily allocate one persistent renderer per ThreadPool executor thread."""

    def __init__(
        self,
        *,
        python_executable: Path,
        batch_worker_script: Path,
        scene_root: Path,
        runtime_mesh_root: Path,
        width: int,
        height: int,
        startup_timeout: int,
        log_dir: Path,
    ) -> None:
        self.python_executable = python_executable
        self.batch_worker_script = batch_worker_script
        self.scene_root = scene_root
        self.runtime_mesh_root = runtime_mesh_root
        self.width = int(width)
        self.height = int(height)
        self.startup_timeout = int(startup_timeout)
        self.log_dir = log_dir
        self.local = threading.local()
        self.clients: list[PersistentRendererClient] = []
        self.lock = threading.Lock()
        self.next_index = 0

    def _client(self) -> PersistentRendererClient:
        client = getattr(self.local, "client", None)
        if client is None or client.process.poll() is not None:
            with self.lock:
                index = self.next_index
                self.next_index += 1
            client = PersistentRendererClient(
                python_executable=self.python_executable,
                batch_worker_script=self.batch_worker_script,
                scene_root=self.scene_root,
                runtime_mesh_root=self.runtime_mesh_root,
                width=self.width,
                height=self.height,
                startup_timeout=self.startup_timeout,
                log_path=self.log_dir / f"renderer_{index:03d}.log",
            )
            with self.lock:
                self.clients.append(client)
            self.local.client = client
        return client

    def render(
        self,
        request_path: Path,
        output_path: Path,
        request_id: str,
        timeout_seconds: int,
    ) -> None:
        self._client().render(request_path, output_path, request_id, timeout_seconds)

    def close(self) -> None:
        with self.lock:
            clients = list(self.clients)
            self.clients.clear()
        for client in clients:
            client.close()


@dataclass(slots=True)
class Counters:
    completed: int = 0
    generated: int = 0
    cached: int = 0
    failed: int = 0
    missing: int = 0
    invalid: int = 0
    bytes_added: int = 0


class ManifestDB:
    """Disk-backed exact-request manifest suitable for hundreds of thousands of rows."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA cache_size=-65536")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests (
                hash TEXT PRIMARY KEY,
                scene_id INTEGER NOT NULL,
                state_id INTEGER NOT NULL,
                task_index INTEGER NOT NULL,
                group_index INTEGER NOT NULL,
                object_count INTEGER NOT NULL,
                object_pose BLOB NOT NULL,
                object_active BLOB NOT NULL,
                object_present BLOB NOT NULL,
                object_asset_ids TEXT NOT NULL,
                object_model_ids TEXT NOT NULL,
                object_scales BLOB NOT NULL,
                render_seed INTEGER NOT NULL,
                camera_profile TEXT NOT NULL,
                point_count INTEGER NOT NULL,
                renderer_version TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                file_size INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS scanned_scenes (
                scene_id INTEGER PRIMARY KEY,
                group_count INTEGER NOT NULL,
                pair_count INTEGER NOT NULL,
                completed_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
            CREATE INDEX IF NOT EXISTS idx_requests_scene ON requests(scene_id, state_id, task_index);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def set_metadata(self, key: str, value: Any) -> None:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        self.connection.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, serialized),
        )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def insert_request(self, payload: RequestPayload) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO requests(
                hash, scene_id, state_id, task_index, group_index, object_count,
                object_pose, object_active, object_present, object_asset_ids,
                object_model_ids, object_scales, render_seed, camera_profile,
                point_count, renderer_version, status, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                payload.key,
                payload.scene_id,
                payload.state_id,
                payload.task_index,
                payload.group_index,
                payload.object_count,
                sqlite3.Binary(
                    np.ascontiguousarray(payload.object_pose, dtype=np.float32).tobytes()
                ),
                sqlite3.Binary(
                    np.ascontiguousarray(payload.object_active, dtype=np.uint8).tobytes()
                ),
                sqlite3.Binary(
                    np.ascontiguousarray(payload.object_present, dtype=np.uint8).tobytes()
                ),
                json.dumps(payload.object_asset_ids, ensure_ascii=False),
                json.dumps(payload.object_model_ids, ensure_ascii=False),
                sqlite3.Binary(
                    np.ascontiguousarray(payload.object_scales, dtype=np.float32).tobytes()
                ),
                payload.render_seed,
                payload.camera_profile,
                payload.point_count,
                payload.renderer_version,
                STATUS_PENDING,
                time.time(),
            ),
        )
        return cursor.rowcount > 0

    @staticmethod
    def _decode_row(row: sqlite3.Row | Sequence[Any]) -> RequestPayload:
        (
            key,
            scene_id,
            state_id,
            task_index,
            group_index,
            object_count,
            object_pose,
            object_active,
            object_present,
            object_asset_ids,
            object_model_ids,
            object_scales,
            render_seed,
            camera_profile,
            point_count,
            renderer_version,
        ) = row
        count = int(object_count)
        return RequestPayload(
            key=str(key),
            scene_id=int(scene_id),
            state_id=int(state_id),
            task_index=int(task_index),
            group_index=int(group_index),
            object_count=count,
            object_pose=np.frombuffer(object_pose, dtype=np.float32).copy().reshape(count, 7),
            object_active=np.frombuffer(object_active, dtype=np.uint8).astype(bool, copy=True),
            object_present=np.frombuffer(object_present, dtype=np.uint8).astype(bool, copy=True),
            object_asset_ids=tuple(json.loads(object_asset_ids)),
            object_model_ids=tuple(json.loads(object_model_ids)),
            object_scales=np.frombuffer(object_scales, dtype=np.float32).copy(),
            render_seed=int(render_seed),
            camera_profile=str(camera_profile),
            point_count=int(point_count),
            renderer_version=str(renderer_version),
        )

    def iter_requests(
        self,
        statuses: Sequence[str] | None = None,
        *,
        max_attempts: int | None = None,
    ) -> Iterator[RequestPayload]:
        fields = (
            "hash,scene_id,state_id,task_index,group_index,object_count,"
            "object_pose,object_active,object_present,object_asset_ids,"
            "object_model_ids,object_scales,render_seed,camera_profile,"
            "point_count,renderer_version"
        )
        clauses: list[str] = []
        values: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            values.extend(statuses)
        if max_attempts is not None:
            clauses.append("attempts < ?")
            values.append(max_attempts)
        query = f"SELECT {fields} FROM requests"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY scene_id,state_id,task_index,hash"
        cursor = self.connection.execute(query, values)
        while True:
            rows = cursor.fetchmany(256)
            if not rows:
                break
            for row in rows:
                yield self._decode_row(row)

    def update_result(self, result: RenderResult, increment_attempt: bool = False) -> None:
        self.connection.execute(
            """
            UPDATE requests
            SET status=?, file_size=?, attempts=attempts+?, last_error=?, updated_at=?
            WHERE hash=?
            """,
            (
                result.status,
                int(result.size_bytes),
                1 if increment_attempt else 0,
                result.error,
                time.time(),
                result.key,
            ),
        )

    def mark_status(self, key: str, status: str, size: int = 0, error: str = "") -> None:
        self.connection.execute(
            "UPDATE requests SET status=?,file_size=?,last_error=?,updated_at=? WHERE hash=?",
            (status, int(size), error, time.time(), key),
        )

    def count(self, status: str | None = None) -> int:
        if status is None:
            return int(self.connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0])
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM requests WHERE status=?", (status,)
            ).fetchone()[0]
        )

    def status_counts(self) -> dict[str, int]:
        return {
            str(status): int(count)
            for status, count in self.connection.execute(
                "SELECT status,COUNT(*) FROM requests GROUP BY status"
            )
        }

    def total_file_size(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COALESCE(SUM(file_size),0) FROM requests WHERE status IN (?,?)",
                (STATUS_CACHED, STATUS_GENERATED),
            ).fetchone()[0]
        )

    def representative_units(self, limit: int) -> list[tuple[int, int, int, int]]:
        rows = self.connection.execute(
            """
            SELECT scene_id,state_id,task_index,group_index
            FROM requests WHERE status IN (?,?)
            ORDER BY hash LIMIT ?
            """,
            (STATUS_CACHED, STATUS_GENERATED, int(limit)),
        ).fetchall()
        return [tuple(map(int, row)) for row in rows]

    def commit(self) -> None:
        self.connection.commit()


class ActiveMaskResolver:
    """Exact batched equivalent of TaskOrientedClutterAdapter._object_active."""

    def __init__(self, scene: h5py.Group) -> None:
        self.object_count = len(scene["catalog/object_index"])
        actions = scene["actions"]
        self.from_state = actions["from_state"][:].astype(np.int64, copy=False)
        self.to_state = actions["to_state"][:].astype(np.int64, copy=False)
        self.action_type = actions["action_type"][:].astype(np.int64, copy=False)
        self.payload_index = actions["payload_index"][:].astype(np.int64, copy=False)
        self.after_state_valid = actions["after_state_valid"][:].astype(bool, copy=False)
        self.action_task = actions["task_index"][:].astype(np.int64, copy=False)
        self.sequence_depth = scene["states/sequence_depth"][:].astype(np.int64, copy=False)
        self.state_task = scene["states/task_index"][:].astype(np.int64, copy=False)
        self.acted_object = np.full(len(self.action_type), -1, dtype=np.int64)
        pick_rows = np.flatnonzero(self.action_type == int(ActionType.PICK_REMOVE))
        pick_payload = actions["pick_remove/acted_object"][:].astype(np.int64, copy=False)
        if len(pick_rows):
            self.acted_object[pick_rows] = pick_payload[self.payload_index[pick_rows]]

        valid = (
            self.after_state_valid
            & (self.to_state >= 0)
            & (self.sequence_depth[self.from_state] < self.sequence_depth[self.to_state])
        )
        self.predecessors: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index in np.flatnonzero(valid).tolist():
            self.predecessors[int(self.action_task[index])][int(self.to_state[index])].append(index)
        self.memo: dict[int, dict[int, frozenset[int]]] = {}
        self.visiting: dict[int, set[int]] = defaultdict(set)

    def removed(self, state_id: int, task_index: int) -> frozenset[int]:
        memo = self.memo.setdefault(
            int(task_index),
            {
                int(state): frozenset()
                for state in np.flatnonzero(self.sequence_depth == 0).tolist()
            },
        )
        state_id = int(state_id)
        task_index = int(task_index)
        if state_id in memo:
            return memo[state_id]
        visiting = self.visiting[task_index]
        if state_id in visiting:
            raise ValueError("Cycle in depth-monotone transition graph")
        visiting.add(state_id)
        histories: list[frozenset[int]] = []
        for transition_index in self.predecessors.get(task_index, {}).get(state_id, []):
            history = set(self.removed(int(self.from_state[transition_index]), task_index))
            if int(self.action_type[transition_index]) == int(ActionType.PICK_REMOVE):
                history.add(int(self.acted_object[transition_index]))
            histories.append(frozenset(history))
        visiting.remove(state_id)
        memo[state_id] = frozenset().union(*histories) if histories else frozenset()
        return memo[state_id]

    def active(self, state_id: int, task_index: int | None = None) -> np.ndarray:
        if task_index is None:
            task_index = int(self.state_task[int(state_id)])
        inactive = self.removed(int(state_id), int(task_index))
        active = np.ones(self.object_count, dtype=bool)
        if inactive:
            indices = np.fromiter(inactive, dtype=np.int64)
            if np.any((indices < 0) | (indices >= self.object_count)):
                raise ValueError(f"Invalid removed object index: {indices.tolist()}")
            active[indices] = False
        return active


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")


def format_bytes(value: int | float) -> str:
    amount = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if abs(amount) < 1000.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1000.0
    return f"{amount:.2f} TB"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_executable(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_file():
        return path.resolve()
    located = shutil.which(str(value))
    if located:
        return Path(located).resolve()
    raise FileNotFoundError(f"Executable not found: {value}")


def load_local_paths(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    import yaml

    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in values.items()
    ):
        raise ValueError(f"Local path config must be a string mapping: {path}")
    return values


def quoted_override(name: str, value: Any) -> str:
    return f"{name}={json.dumps(value, ensure_ascii=False)}"


def load_runtime_config(args: argparse.Namespace) -> TCDPRGConfig:
    local = load_local_paths(args.paths_config)
    overrides: list[str] = []
    path_values = {
        "dataset.root": args.dataset_root or local.get("dataset_root"),
        "dataset.acronym_root": args.acronym_root or local.get("acronym_root"),
        "dataset.functional_region_root": (
            args.functional_region_root or local.get("functional_region_root")
        ),
        "observation.pybullet_python": (args.pybullet_python or local.get("pybullet_python")),
        "observation.runtime_mesh_root": (args.runtime_mesh_root or local.get("runtime_mesh_root")),
        "cache.directory": args.cache_dir or local.get("observation_cache_dir"),
    }
    for key, value in path_values.items():
        if value:
            overrides.append(quoted_override(key, str(value)))
    if args.point_count is not None:
        overrides.append(f"dataset.scene_points={int(args.point_count)}")
    overrides.extend(args.override)
    return load_config(args.config, overrides)


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.npz"


def validate_cache_file(
    path: Path,
    *,
    expected_point_count: int,
    object_count: int,
    check_values: bool = True,
) -> ValidationResult:
    if not path.is_file():
        return ValidationResult(False, "missing")
    try:
        size = path.stat().st_size
        with np.load(path, allow_pickle=False) as data:
            missing = [field for field in REQUIRED_FIELDS if field not in data.files]
            if missing:
                return ValidationResult(False, f"missing fields: {missing}", size)
            xyz = data["xyz"]
            rgb = data["rgb"]
            instance = data["instance_id"]
            source = data["source_view"]
            if xyz.dtype != np.float32:
                return ValidationResult(False, f"xyz dtype={xyz.dtype}, expected float32", size)
            if rgb.dtype != np.float32:
                return ValidationResult(False, f"rgb dtype={rgb.dtype}, expected float32", size)
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                return ValidationResult(False, f"xyz shape={xyz.shape}", size)
            if rgb.shape != xyz.shape:
                return ValidationResult(False, f"rgb shape={rgb.shape}, xyz={xyz.shape}", size)
            if instance.ndim != 1 or source.ndim != 1:
                return ValidationResult(
                    False,
                    f"instance/source rank invalid: {instance.shape}/{source.shape}",
                    size,
                )
            count = len(xyz)
            if not (len(rgb) == len(instance) == len(source) == count):
                return ValidationResult(False, "first dimensions do not match", size, count)
            if expected_point_count > 0 and count != expected_point_count:
                return ValidationResult(
                    False,
                    f"point count={count}, expected={expected_point_count}",
                    size,
                    count,
                )
            if count <= 0:
                return ValidationResult(False, "empty point cloud", size, count)
            if not np.issubdtype(instance.dtype, np.integer):
                return ValidationResult(False, f"instance dtype={instance.dtype}", size, count)
            if not np.issubdtype(source.dtype, np.integer):
                return ValidationResult(False, f"source dtype={source.dtype}", size, count)
            if check_values:
                if not np.isfinite(xyz).all():
                    return ValidationResult(False, "xyz contains non-finite values", size, count)
                if not np.isfinite(rgb).all():
                    return ValidationResult(False, "rgb contains non-finite values", size, count)
                if float(rgb.min()) < -1e-6 or float(rgb.max()) > 1.0 + 1e-6:
                    return ValidationResult(
                        False,
                        f"rgb range=[{float(rgb.min()):.6g},{float(rgb.max()):.6g}]",
                        size,
                        count,
                    )
                if int(instance.min()) < 0 or int(instance.max()) >= object_count:
                    return ValidationResult(
                        False,
                        f"instance range=[{int(instance.min())},{int(instance.max())}], "
                        f"object_count={object_count}",
                        size,
                        count,
                    )
                if int(source.min()) < 0 or int(source.max()) > 2:
                    return ValidationResult(
                        False,
                        f"source_view range=[{int(source.min())},{int(source.max())}], oracle leak",
                        size,
                        count,
                    )
        return ValidationResult(True, size_bytes=size, point_count=count)
    except (OSError, ValueError, KeyError, EOFError) as error:
        return ValidationResult(False, f"{type(error).__name__}: {error}")


def training_index_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    signature = ""
    with h5py.File(path, "r", swmr=True) as handle:
        signature = str(handle.attrs.get("generation_signature", ""))
        shape = tuple(int(x) for x in handle["action_state_group"].shape)
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "generation_signature": signature,
        "shape": shape,
    }


def select_scene_ids(adapter: TaskOrientedClutterAdapter, args: argparse.Namespace) -> list[int]:
    available = set(adapter.snapshot_scene_ids)
    if args.scene_ids_file:
        values: list[int] = []
        for raw in args.scene_ids_file.read_text(encoding="utf-8-sig").splitlines():
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            values.append(int(text.split()[0].split(",")[0]))
        selected = sorted(set(values))
    else:
        start = int(args.scene_start)
        if args.scene_count is None:
            selected = sorted(scene for scene in available if scene >= start)
        else:
            stop = start + int(args.scene_count)
            selected = list(range(start, stop))
    missing = [scene for scene in selected if scene not in available]
    if missing:
        preview = missing[:20]
        raise FileNotFoundError(
            f"Selected scenes have no completed scene_*.h5: {preview}"
            + (f" ... ({len(missing)} missing)" if len(missing) > len(preview) else "")
        )
    if not selected:
        raise ValueError("Scene selection is empty")
    return selected


def build_run_signature(
    config: TCDPRGConfig,
    scene_ids: Sequence[int],
    split: str,
    index_signature: Mapping[str, Any],
) -> str:
    payload = {
        "tool_version": SCRIPT_VERSION,
        "dataset_root": str(Path(config.dataset.root).resolve()),
        "scene_subdir": config.dataset.scene_subdir,
        "step_labels_subdir": config.dataset.step_labels_subdir,
        "action_labels_subdir": config.dataset.action_labels_subdir,
        "scene_ids": list(map(int, scene_ids)),
        "split": split,
        "camera_profile": config.observation.camera_profile,
        "renderer_version": config.observation.renderer_version,
        "point_count": int(config.dataset.scene_points),
        "render_width": int(config.observation.render_width),
        "render_height": int(config.observation.render_height),
        "index": dict(index_signature),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_adapter(
    config: TCDPRGConfig, provider: ObservationProvider | None = None
) -> TaskOrientedClutterAdapter:
    return TaskOrientedClutterAdapter(
        config.dataset.root,
        observation_provider=provider or NullObservationProvider(),
        point_count=config.dataset.scene_points,
        renderer_version=config.observation.renderer_version,
        camera_profile=config.observation.camera_profile,
        functional_region_root=config.dataset.functional_region_root,
        verifier_wrong_region_negatives=config.sampling.wrong_region_grasps,
        verifier_collision_negatives=config.sampling.collision_or_approach_negative_grasps,
        verifier_approach_negatives=config.sampling.collision_or_approach_negative_grasps,
        sampling_seed=config.training.seed,
        scene_subdir=config.dataset.scene_subdir,
        step_labels_subdir=config.dataset.step_labels_subdir,
        action_labels_subdir=config.dataset.action_labels_subdir,
        global_positive_grasps_per_object=config.sampling.global_positive_grasps_per_object,
        global_negative_grasps_per_object=config.sampling.global_negative_grasps_per_object,
        grasp_width_bounds=(config.model.min_grasp_width_m, config.model.max_grasp_width_m),
        index_cache_dir=config.cache.index_directory,
        data_fraction=config.training.data_fraction,
        split_ratios=config.training.split_ratios,
        split_seed=config.training.seed,
    )


def selected_index_rows(
    adapter: TaskOrientedClutterAdapter,
    scene_ids: Sequence[int],
    split: str,
) -> np.ndarray:
    rows = adapter._load_action_group_index()  # exact repository index contract
    row_max = int(rows[:, 1].max()) if len(rows) else 0
    maximum = max(max(scene_ids), row_max)
    scene_lookup = np.zeros(maximum + 1, dtype=bool)
    scene_lookup[np.asarray(scene_ids, dtype=np.int64)] = True
    mask = scene_lookup[rows[:, 1]]
    if split != "all":
        split_scenes = np.asarray(adapter.scene_splits[split], dtype=np.int64)
        mask &= np.isin(rows[:, 1], split_scenes)
    selected = rows[mask]
    if not len(selected):
        raise RuntimeError(f"No action-state groups match scenes={len(scene_ids)} split={split}")
    order = np.lexsort((selected[:, 2], selected[:, 3], selected[:, 4], selected[:, 1]))
    return selected[order]


def iter_scene_rows(rows: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    scene_ids, starts = np.unique(rows[:, 1], return_index=True)
    stops = np.r_[starts[1:], len(rows)]
    for scene_id, start, stop in zip(scene_ids, starts, stops, strict=True):
        yield int(scene_id), rows[int(start) : int(stop)]


def payload_for(
    adapter: TaskOrientedClutterAdapter,
    resolver: ActiveMaskResolver,
    scene_id: int,
    state_id: int,
    task_index: int,
    group_index: int,
    state_object_pose: np.ndarray,
    state_render_seed: np.ndarray,
    h5_names: tuple[str, ...],
    model_ids: tuple[str, ...],
    scales: np.ndarray,
) -> RequestPayload:
    object_pose = np.asarray(state_object_pose[state_id], dtype=np.float32)
    object_count = len(object_pose)
    active = resolver.active(state_id, task_index)
    request = adapter._make_observation_request(
        scene_id=scene_id,
        state_id=state_id,
        object_pose=object_pose,
        active=active,
        h5_names=h5_names[:object_count],
        model_ids=model_ids[:object_count],
        object_scales=scales[:object_count],
        render_seed=int(state_render_seed[state_id]),
    )
    key = request_hash(request)
    return RequestPayload(
        key=key,
        scene_id=scene_id,
        state_id=state_id,
        task_index=task_index,
        group_index=group_index,
        object_count=object_count,
        object_pose=np.asarray(request.object_pose, dtype=np.float32),
        object_active=np.asarray(request.object_active, dtype=bool),
        object_present=np.asarray(request.object_present, dtype=bool),
        object_asset_ids=tuple(request.object_asset_ids),
        object_model_ids=tuple(request.object_model_ids),
        object_scales=np.asarray(request.object_scales, dtype=np.float32),
        render_seed=int(request.render_seed),
        camera_profile=request.camera_profile,
        point_count=int(request.point_count),
        renderer_version=request.renderer_version,
    )


def scan_manifest(
    adapter: TaskOrientedClutterAdapter,
    rows: np.ndarray,
    db: ManifestDB,
    work_dir: Path,
    progress_interval: int,
    force_rebuild: bool,
) -> dict[str, int]:
    if db.get_metadata("scan_complete", False) and not force_rebuild:
        print(
            f"[scan] resume existing manifest: unique_requests={db.count():,}",
            flush=True,
        )
        return {
            "groups": int(db.get_metadata("group_count", 0)),
            "unique_requests": db.count(),
            "duplicate_groups": int(db.get_metadata("duplicate_group_count", 0)),
            "scenes": int(db.get_metadata("scanned_scene_count", 0)),
        }
    if force_rebuild:
        db.connection.execute("DELETE FROM requests")
        db.connection.execute("DELETE FROM scanned_scenes")
        db.connection.commit()
    started = time.time()
    completed_rows = db.connection.execute(
        "SELECT scene_id,group_count,pair_count FROM scanned_scenes"
    ).fetchall()
    completed_scene_ids = {int(row[0]) for row in completed_rows}
    groups = sum(int(row[1]) for row in completed_rows)
    inserted = db.count()
    duplicate_pairs = sum(int(row[1]) - int(row[2]) for row in completed_rows)
    scanned_scenes = len(completed_rows)
    scene_root = adapter.scene_root
    total_scenes = len(np.unique(rows[:, 1]))
    if scanned_scenes:
        print(
            f"[scan] resume scenes={scanned_scenes:,}/{total_scenes:,} "
            f"groups={groups:,} unique={inserted:,}",
            flush=True,
        )
    for scene_id, scene_rows in iter_scene_rows(rows):
        if scene_id in completed_scene_ids:
            continue
        with h5py.File(adapter._h5_path(scene_id), "r", swmr=True) as handle:
            scene = adapter._scene_group(handle, scene_id)
            states = scene["states"]
            state_object_pose = states["object_pose"][:].astype(np.float32, copy=False)
            state_render_seed = states["observation_render_seed"][:].astype(np.int64, copy=False)
            resolver = ActiveMaskResolver(scene)
            with np.load(
                scene_root / f"scene_{scene_id:04d}" / "scene.npz", allow_pickle=False
            ) as raw:
                raw_count = int(raw["object_count"])
                h5_names = tuple(str(x) for x in raw["object_h5_name"][:raw_count])
                model_ids = tuple(str(x) for x in raw["object_model_id"][:raw_count])
                scales = raw["object_scale"][:raw_count].astype(np.float32)
            pairs: dict[tuple[int, int], int] = {}
            for row in scene_rows:
                state_id = int(row[4])
                task_index = int(row[3])
                group_index = int(row[2])
                groups += 1
                pairs.setdefault((state_id, task_index), group_index)
            duplicate_pairs += len(scene_rows) - len(pairs)
            for (state_id, task_index), group_index in sorted(pairs.items()):
                payload = payload_for(
                    adapter,
                    resolver,
                    scene_id,
                    state_id,
                    task_index,
                    group_index,
                    state_object_pose,
                    state_render_seed,
                    h5_names,
                    model_ids,
                    scales,
                )
                if db.insert_request(payload):
                    inserted += 1
            db.connection.execute(
                "INSERT OR REPLACE INTO scanned_scenes(scene_id,group_count,pair_count,completed_at) "
                "VALUES(?,?,?,?)",
                (scene_id, len(scene_rows), len(pairs), time.time()),
            )
            completed_scene_ids.add(scene_id)
            scanned_scenes += 1
            if scanned_scenes % max(1, progress_interval) == 0 or scanned_scenes == total_scenes:
                db.commit()
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[scan] scenes={scanned_scenes:,}/{total_scenes:,} "
                    f"groups={groups:,} unique={inserted:,} "
                    f"speed={scanned_scenes / elapsed * 60:.2f} scenes/min",
                    flush=True,
                )
                atomic_json(
                    work_dir / "progress.json",
                    {
                        "phase": "scan",
                        "timestamp_utc": utc_now(),
                        "scenes_completed": scanned_scenes,
                        "scenes_total": total_scenes,
                        "groups_seen": groups,
                        "unique_requests": inserted,
                        "elapsed_seconds": time.time() - started,
                    },
                )
    db.set_metadata("scan_complete", True)
    db.set_metadata("group_count", groups)
    db.set_metadata("duplicate_group_count", groups - db.count())
    db.set_metadata("scanned_scene_count", scanned_scenes)
    db.commit()
    return {
        "groups": groups,
        "unique_requests": db.count(),
        "duplicate_groups": groups - db.count(),
        "scenes": scanned_scenes,
    }


def check_disk_space(path: Path, minimum_free_bytes: int, required_bytes: int = 0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    threshold = minimum_free_bytes + max(0, required_bytes)
    if free < threshold:
        raise DiskSpaceReserveError(
            f"Insufficient disk space at {path}: free={format_bytes(free)}, "
            f"required reserve={format_bytes(threshold)}"
        )


def write_request_npz(path: Path, payload: RequestPayload) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        scene_id=np.asarray(payload.scene_id, dtype=np.int32),
        state_id=np.asarray(payload.state_id, dtype=np.int32),
        object_pose=np.asarray(payload.object_pose, dtype=np.float32),
        object_present=np.asarray(payload.object_present, dtype=bool),
        object_active=np.asarray(payload.object_active, dtype=bool),
        object_asset_ids=np.asarray(payload.object_asset_ids),
        object_model_ids=np.asarray(payload.object_model_ids),
        object_scales=np.asarray(payload.object_scales, dtype=np.float32),
        render_seed=np.asarray(payload.render_seed, dtype=np.int64),
        point_count=np.asarray(payload.point_count, dtype=np.int32),
        renderer_version=np.asarray(payload.renderer_version),
    )
    os.replace(temporary, path)


@contextlib.contextmanager
def exclusive_hash_lock(
    lock_path: Path,
    final_path: Path,
    *,
    expected_point_count: int,
    object_count: int,
    timeout_seconds: int,
    verify_values: bool,
):
    """Serialize writers for one content hash across independent tool processes."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max(60, timeout_seconds * 2)
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(
                descriptor,
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "thread": threading.get_ident(),
                        "created_utc": utc_now(),
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
        except FileExistsError:
            validation = validate_cache_file(
                final_path,
                expected_point_count=expected_point_count,
                object_count=object_count,
                check_values=verify_values,
            )
            if validation.valid:
                yield False
                return
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > max(600, timeout_seconds * 3):
                lock_path.unlink(missing_ok=True)
                continue
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for cache hash lock: {lock_path}")
            time.sleep(0.25)
    try:
        yield True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def render_one(
    payload: RequestPayload,
    *,
    cache_dir: Path,
    corrupt_dir: Path,
    request_dir: Path,
    pybullet_python: Path,
    worker_script: Path,
    scene_root: Path,
    runtime_mesh_root: Path,
    width: int,
    height: int,
    timeout_seconds: int,
    minimum_free_bytes: int,
    verify_values: bool,
    persistent_renderer: PersistentRendererManager | None = None,
) -> RenderResult:
    started = time.time()
    final_path = cache_path(cache_dir, payload.key)
    validation = validate_cache_file(
        final_path,
        expected_point_count=payload.point_count,
        object_count=payload.object_count,
        check_values=verify_values,
    )
    if validation.valid:
        return RenderResult(
            payload.key,
            STATUS_CACHED,
            validation.size_bytes,
            time.time() - started,
            payload.scene_id,
            payload.state_id,
            payload.task_index,
            payload.group_index,
        )

    check_disk_space(cache_dir, minimum_free_bytes)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = final_path.with_suffix(".lock")
    try:
        with exclusive_hash_lock(
            lock_path,
            final_path,
            expected_point_count=payload.point_count,
            object_count=payload.object_count,
            timeout_seconds=timeout_seconds,
            verify_values=verify_values,
        ) as owns_lock:
            if not owns_lock:
                validation = validate_cache_file(
                    final_path,
                    expected_point_count=payload.point_count,
                    object_count=payload.object_count,
                    check_values=verify_values,
                )
                return RenderResult(
                    payload.key,
                    STATUS_CACHED,
                    validation.size_bytes,
                    time.time() - started,
                    payload.scene_id,
                    payload.state_id,
                    payload.task_index,
                    payload.group_index,
                )

            request_dir.mkdir(parents=True, exist_ok=True)
            request_path = request_dir / (
                f"{payload.key}.{os.getpid()}.{threading.get_ident()}.request.npz"
            )
            output_path = final_path.with_name(
                f"{payload.key}.{os.getpid()}.{threading.get_ident()}.render.tmp.npz"
            )
            if final_path.exists():
                corrupt_dir.mkdir(parents=True, exist_ok=True)
                corrupt_path = corrupt_dir / f"{payload.key}.{int(time.time())}.npz"
                try:
                    os.replace(final_path, corrupt_path)
                except OSError:
                    final_path.unlink(missing_ok=True)
            try:
                write_request_npz(request_path, payload)
                if persistent_renderer is not None:
                    persistent_renderer.render(
                        request_path,
                        output_path,
                        payload.key,
                        timeout_seconds,
                    )
                else:
                    command = [
                        str(pybullet_python),
                        str(worker_script),
                        "--request",
                        str(request_path),
                        "--output",
                        str(output_path),
                        "--scene-root",
                        str(scene_root),
                        "--runtime-mesh-root",
                        str(runtime_mesh_root),
                        "--width",
                        str(width),
                        "--height",
                        str(height),
                    ]
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                        timeout=timeout_seconds,
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"renderer exit={completed.returncode}\n"
                            f"stdout:\n{completed.stdout[-4000:]}\n"
                            f"stderr:\n{completed.stderr[-4000:]}"
                        )
                validation = validate_cache_file(
                    output_path,
                    expected_point_count=payload.point_count,
                    object_count=payload.object_count,
                    check_values=verify_values,
                )
                if not validation.valid:
                    raise RuntimeError(f"rendered cache validation failed: {validation.reason}")
                check_disk_space(cache_dir, minimum_free_bytes, validation.size_bytes)
                os.replace(output_path, final_path)
                return RenderResult(
                    payload.key,
                    STATUS_GENERATED,
                    validation.size_bytes,
                    time.time() - started,
                    payload.scene_id,
                    payload.state_id,
                    payload.task_index,
                    payload.group_index,
                )
            finally:
                request_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                output_path.with_suffix(output_path.suffix + ".tmp.npz").unlink(missing_ok=True)
    except BaseException as error:
        return RenderResult(
            payload.key,
            STATUS_FAILED,
            0,
            time.time() - started,
            payload.scene_id,
            payload.state_id,
            payload.task_index,
            payload.group_index,
            error=f"{type(error).__name__}: {error}",
        )


def validate_existing_one(
    payload: RequestPayload,
    *,
    cache_dir: Path,
    verify_values: bool,
) -> RenderResult:
    started = time.time()
    validation = validate_cache_file(
        cache_path(cache_dir, payload.key),
        expected_point_count=payload.point_count,
        object_count=payload.object_count,
        check_values=verify_values,
    )
    if validation.valid:
        status = STATUS_CACHED
        error = ""
    elif validation.reason == "missing":
        status = STATUS_MISSING
        error = validation.reason
    else:
        status = STATUS_INVALID
        error = validation.reason
    return RenderResult(
        payload.key,
        status,
        validation.size_bytes,
        time.time() - started,
        payload.scene_id,
        payload.state_id,
        payload.task_index,
        payload.group_index,
        error=error,
    )


def print_progress(
    phase: str,
    counters: Counters,
    total: int,
    started: float,
    cache_dir: Path,
    *,
    scenes_completed: int,
    scenes_total: int,
    current: RenderResult | None,
) -> dict[str, Any]:
    elapsed = max(time.time() - started, 1e-6)
    rate_per_minute = counters.completed / elapsed * 60.0
    remaining = max(0, total - counters.completed)
    eta = remaining / max(counters.completed / elapsed, 1e-9)
    free = shutil.disk_usage(cache_dir).free
    record = {
        "phase": phase,
        "timestamp_utc": utc_now(),
        "completed": counters.completed,
        "total": total,
        "generated": counters.generated,
        "cached": counters.cached,
        "failed": counters.failed,
        "missing": counters.missing,
        "invalid": counters.invalid,
        "rate_per_minute": rate_per_minute,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "bytes_added": counters.bytes_added,
        "disk_free_bytes": free,
        "scenes_completed": scenes_completed,
        "scenes_total": scenes_total,
        "current_scene_id": None if current is None else current.scene_id,
        "current_state_id": None if current is None else current.state_id,
        "current_task_index": None if current is None else current.task_index,
        "current_hash": None if current is None else current.key,
    }
    current_text = (
        ""
        if current is None
        else f" current={current.scene_id}/{current.state_id}/task{current.task_index}"
    )
    print(
        f"[{phase}] requests={counters.completed:,}/{total:,} "
        f"scenes={scenes_completed:,}/{scenes_total:,}{current_text} "
        f"generated={counters.generated:,} cached={counters.cached:,} "
        f"failed={counters.failed:,} missing={counters.missing:,} invalid={counters.invalid:,} "
        f"speed={rate_per_minute:.2f}/min elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)} added={format_bytes(counters.bytes_added)} "
        f"free={format_bytes(free)}",
        flush=True,
    )
    return record


def apply_result(counters: Counters, result: RenderResult) -> None:
    counters.completed += 1
    if result.status == STATUS_GENERATED:
        counters.generated += 1
        counters.bytes_added += result.size_bytes
    elif result.status == STATUS_CACHED:
        counters.cached += 1
    elif result.status == STATUS_FAILED:
        counters.failed += 1
    elif result.status == STATUS_MISSING:
        counters.missing += 1
    elif result.status == STATUS_INVALID:
        counters.invalid += 1


def run_parallel(
    payloads: Iterable[RequestPayload],
    worker,
    *,
    total: int,
    workers: int,
    progress_interval: int,
    phase: str,
    db: ManifestDB,
    work_dir: Path,
    cache_dir: Path,
    increment_attempt: bool,
    scene_mode: str,
) -> Counters:
    started = time.time()
    counters = Counters()
    if scene_mode == "all":
        scene_rows = db.connection.execute(
            "SELECT scene_id,COUNT(*) FROM requests GROUP BY scene_id"
        ).fetchall()
    elif scene_mode == "nonfinal":
        scene_rows = db.connection.execute(
            """
            SELECT scene_id,COUNT(*) FROM requests
            WHERE status NOT IN (?,?) GROUP BY scene_id
            """,
            (STATUS_CACHED, STATUS_GENERATED),
        ).fetchall()
    else:
        raise ValueError(f"Unsupported scene_mode={scene_mode}")
    scenes_total = int(
        db.connection.execute("SELECT COUNT(DISTINCT scene_id) FROM requests").fetchone()[0]
    )
    scene_remaining = {int(scene): int(count) for scene, count in scene_rows}
    scenes_completed = scenes_total - len(scene_remaining)
    failures_path = work_dir / "failures.jsonl"
    pending: dict[Future[RenderResult], RequestPayload] = {}
    iterator = iter(payloads)
    exhausted = False
    max_pending = max(workers, workers * 2)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tcd-render") as executor:
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    payload = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[executor.submit(worker, payload)] = payload
            if not pending:
                continue
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                payload = pending.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    result = RenderResult(
                        payload.key,
                        STATUS_FAILED,
                        0,
                        0.0,
                        payload.scene_id,
                        payload.state_id,
                        payload.task_index,
                        payload.group_index,
                        error=f"worker crash {type(error).__name__}: {error}",
                    )
                db.update_result(result, increment_attempt=increment_attempt)
                apply_result(counters, result)
                if result.status in FINAL_STATUSES and result.scene_id in scene_remaining:
                    scene_remaining[result.scene_id] -= 1
                    if scene_remaining[result.scene_id] <= 0:
                        del scene_remaining[result.scene_id]
                        scenes_completed += 1
                if result.status in {STATUS_FAILED, STATUS_INVALID, STATUS_MISSING}:
                    append_jsonl(
                        failures_path,
                        {
                            "timestamp_utc": utc_now(),
                            "phase": phase,
                            "scene_id": result.scene_id,
                            "state_id": result.state_id,
                            "task_index": result.task_index,
                            "group_index": result.group_index,
                            "hash": result.key,
                            "status": result.status,
                            "error": result.error,
                        },
                    )
                if result.status == STATUS_FAILED and result.error.startswith(
                    "DiskSpaceReserveError:"
                ):
                    for outstanding in pending:
                        outstanding.cancel()
                    db.commit()
                    progress = print_progress(
                        phase,
                        counters,
                        total,
                        started,
                        cache_dir,
                        scenes_completed=scenes_completed,
                        scenes_total=scenes_total,
                        current=result,
                    )
                    progress["stopped_reason"] = result.error
                    atomic_json(work_dir / "progress.json", progress)
                    raise DiskSpaceReserveError(result.error)
                if (
                    counters.completed % max(1, progress_interval) == 0
                    or counters.completed == total
                ):
                    db.commit()
                    progress = print_progress(
                        phase,
                        counters,
                        total,
                        started,
                        cache_dir,
                        scenes_completed=scenes_completed,
                        scenes_total=scenes_total,
                        current=result,
                    )
                    atomic_json(work_dir / "progress.json", progress)
    db.commit()
    return counters


def classify_existing(
    db: ManifestDB,
    args: argparse.Namespace,
    cache_dir: Path,
    work_dir: Path,
) -> Counters:
    total = db.count()
    worker = lambda payload: validate_existing_one(  # noqa: E731
        payload,
        cache_dir=cache_dir,
        verify_values=not args.fast_verify,
    )
    return run_parallel(
        db.iter_requests(),
        worker,
        total=total,
        workers=max(1, args.workers),
        progress_interval=args.progress_interval,
        phase="verify",
        db=db,
        work_dir=work_dir,
        cache_dir=cache_dir,
        increment_attempt=False,
        scene_mode="all",
    )


def generate_missing(
    db: ManifestDB,
    args: argparse.Namespace,
    config: TCDPRGConfig,
    cache_dir: Path,
    work_dir: Path,
) -> Counters:
    pybullet_python = resolve_executable(config.observation.pybullet_python)
    worker_script = Path(config.observation.worker_script).resolve()
    scene_root = Path(config.dataset.root) / config.dataset.scene_subdir
    runtime_mesh_root = Path(config.observation.runtime_mesh_root).resolve()
    request_dir = work_dir / "requests"
    corrupt_dir = work_dir / "corrupt"
    minimum_free_bytes = int(args.min_free_gb * (1 << 30))
    batch_worker_script = Path(args.batch_worker_script).resolve()
    aggregate = Counters()
    max_attempts = args.max_retries + 1
    for attempt in range(max_attempts):
        eligible_statuses = (
            (STATUS_PENDING, STATUS_MISSING, STATUS_INVALID) if attempt == 0 else (STATUS_FAILED,)
        )
        total = int(
            db.connection.execute(
                f"SELECT COUNT(*) FROM requests WHERE status IN "
                f"({','.join('?' for _ in eligible_statuses)}) AND attempts < ?",
                (*eligible_statuses, max_attempts),
            ).fetchone()[0]
        )
        if total == 0:
            break
        print(
            f"[render] pass={attempt + 1}/{max_attempts} requests={total:,} "
            f"mode={args.renderer_mode}",
            flush=True,
        )
        manager = (
            PersistentRendererManager(
                python_executable=pybullet_python,
                batch_worker_script=batch_worker_script,
                scene_root=scene_root,
                runtime_mesh_root=runtime_mesh_root,
                width=config.observation.render_width,
                height=config.observation.render_height,
                startup_timeout=args.render_timeout,
                log_dir=work_dir / "renderer_logs" / f"pass_{attempt + 1}",
            )
            if args.renderer_mode == "persistent"
            else None
        )
        worker = lambda payload: render_one(  # noqa: E731
            payload,
            cache_dir=cache_dir,
            corrupt_dir=corrupt_dir,
            request_dir=request_dir,
            pybullet_python=pybullet_python,
            worker_script=worker_script,
            scene_root=scene_root,
            runtime_mesh_root=runtime_mesh_root,
            width=config.observation.render_width,
            height=config.observation.render_height,
            timeout_seconds=args.render_timeout,
            minimum_free_bytes=minimum_free_bytes,
            verify_values=not args.fast_verify,
            persistent_renderer=manager,
        )
        try:
            result = run_parallel(
                db.iter_requests(eligible_statuses, max_attempts=max_attempts),
                worker,
                total=total,
                workers=max(1, args.workers),
                progress_interval=args.progress_interval,
                phase=f"render-pass-{attempt + 1}",
                db=db,
                work_dir=work_dir,
                cache_dir=cache_dir,
                increment_attempt=True,
                scene_mode="nonfinal",
            )
        finally:
            if manager is not None:
                manager.close()
        aggregate.completed += result.completed
        aggregate.generated += result.generated
        aggregate.cached += result.cached
        aggregate.failed += result.failed
        aggregate.missing += result.missing
        aggregate.invalid += result.invalid
        aggregate.bytes_added += result.bytes_added
    return aggregate


def estimate_space(db: ManifestDB, fallback_bytes: int) -> dict[str, Any]:
    counts = db.status_counts()
    valid_count = counts.get(STATUS_CACHED, 0) + counts.get(STATUS_GENERATED, 0)
    total_size = db.total_file_size()
    average = total_size / valid_count if valid_count else float(fallback_bytes)
    missing = (
        counts.get(STATUS_PENDING, 0)
        + counts.get(STATUS_MISSING, 0)
        + counts.get(STATUS_INVALID, 0)
        + counts.get(STATUS_FAILED, 0)
    )
    return {
        "status_counts": counts,
        "valid_count": valid_count,
        "average_valid_file_bytes": average,
        "fallback_file_bytes": fallback_bytes,
        "remaining_request_count": missing,
        "estimated_additional_bytes": int(math.ceil(missing * average)),
        "estimated_total_bytes": int(math.ceil(db.count() * average)),
    }


def verify_active_resolver(
    adapter: TaskOrientedClutterAdapter,
    rows: np.ndarray,
    samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), min(samples, len(rows)), replace=False)
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for index in indices.tolist():
        grouped[int(rows[index, 1])].append(rows[index])
    for scene_id, selected in grouped.items():
        with h5py.File(adapter._h5_path(scene_id), "r", swmr=True) as handle:
            scene = adapter._scene_group(handle, scene_id)
            resolver = ActiveMaskResolver(scene)
            for row in selected:
                state_id, task_index = int(row[4]), int(row[3])
                expected = adapter._object_active(scene, state_id, task_index)
                actual = resolver.active(state_id, task_index)
                if not np.array_equal(actual, expected):
                    raise AssertionError(
                        f"active-mask mismatch scene={scene_id} state={state_id} task={task_index}"
                    )


def mesh_coverage_check(
    adapter: TaskOrientedClutterAdapter,
    scene_ids: Sequence[int],
    runtime_mesh_root: Path,
    sample_count: int,
) -> dict[str, Any]:
    selected = list(scene_ids[:sample_count])
    model_ids: set[str] = set()
    for scene_id in selected:
        with np.load(
            adapter.scene_root / f"scene_{scene_id:04d}" / "scene.npz",
            allow_pickle=False,
        ) as raw:
            count = int(raw["object_count"])
            model_ids.update(str(x) for x in raw["object_model_id"][:count])
    missing: list[dict[str, str]] = []
    for model_id in sorted(model_ids):
        visual = runtime_mesh_root / f"model_{model_id}.obj"
        collision = runtime_mesh_root / f"vhacd_v3_{model_id}.obj"
        if not visual.is_file() or not collision.is_file():
            missing.append(
                {
                    "model_id": model_id,
                    "visual": str(visual),
                    "collision": str(collision),
                }
            )
    return {
        "sampled_scenes": len(selected),
        "unique_models": len(model_ids),
        "complete_models": len(model_ids) - len(missing),
        "missing": missing,
    }


def run_check(
    args: argparse.Namespace,
    config: TCDPRGConfig,
    adapter: TaskOrientedClutterAdapter,
    scene_ids: Sequence[int],
    rows: np.ndarray,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    critical_errors: list[str] = []
    paths = {
        "project": PROJECT,
        "dataset_root": Path(config.dataset.root),
        "scene_root": adapter.scene_root,
        "step_labels": adapter.step_root,
        "action_labels": adapter.action_root,
        "training_index": adapter.training_index_path,
        "worker_script": Path(config.observation.worker_script),
        "batch_worker_script": Path(args.batch_worker_script),
        "runtime_mesh_root": Path(config.observation.runtime_mesh_root),
        "cache_dir": Path(config.cache.directory),
    }
    for name, path in paths.items():
        exists = path.exists()
        checks[f"path_{name}"] = {"path": str(path), "exists": exists}
        if not exists:
            critical_errors.append(f"missing {name}: {path}")
    try:
        executable = resolve_executable(config.observation.pybullet_python)
        completed = subprocess.run(
            [
                str(executable),
                "-c",
                "import pybullet,numpy; print('pybullet-ok')",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        checks["pybullet_python"] = {
            "path": str(executable),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        if completed.returncode != 0:
            critical_errors.append("PyBullet interpreter cannot import pybullet/numpy")
        if args.renderer_mode == "persistent" and completed.returncode == 0:
            batch_worker = Path(args.batch_worker_script).resolve()
            batch_check = subprocess.run(
                [str(executable), str(batch_worker), "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
                cwd=str(batch_worker.parent),
            )
            checks["persistent_batch_worker"] = {
                "path": str(batch_worker),
                "returncode": batch_check.returncode,
                "stdout": batch_check.stdout[-2000:].strip(),
                "stderr": batch_check.stderr[-2000:].strip(),
            }
            if batch_check.returncode != 0:
                critical_errors.append(
                    "Persistent batch worker cannot start in the PyBullet interpreter"
                )
    except BaseException as error:
        checks["pybullet_python"] = {"error": f"{type(error).__name__}: {error}"}
        critical_errors.append(str(error))

    metadata_path = adapter.scene_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cameras = [
        item
        for item in metadata.get("camera_parameters", [])
        if str(item.get("sensor_type", "")).lower() != "oracle"
    ]
    checks["formal_cameras"] = {
        "count": len(cameras),
        "types": [str(item.get("sensor_type")) for item in cameras],
        "profile": config.observation.camera_profile,
    }
    if len(cameras) != 3:
        critical_errors.append(f"expected exactly 3 non-oracle cameras, got {len(cameras)}")

    try:
        verify_active_resolver(
            adapter,
            rows,
            samples=args.check_active_samples,
            seed=config.training.seed,
        )
        checks["active_mask_equivalence"] = {
            "samples": min(args.check_active_samples, len(rows)),
            "passed": True,
        }
    except BaseException as error:
        checks["active_mask_equivalence"] = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
        critical_errors.append("batched object_active resolver differs from adapter")

    coverage = mesh_coverage_check(
        adapter,
        scene_ids,
        Path(config.observation.runtime_mesh_root),
        args.check_scenes,
    )
    checks["mesh_coverage"] = coverage
    if coverage["missing"]:
        critical_errors.append(
            f"runtime mesh cache incomplete for {len(coverage['missing'])} sampled model(s)"
        )

    cache_dir = Path(config.cache.directory)
    cache_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(cache_dir)
    checks["disk"] = {
        "path": str(cache_dir),
        "free_bytes": usage.free,
        "free": format_bytes(usage.free),
        "minimum_free_gb": args.min_free_gb,
    }
    if usage.free < args.min_free_gb * (1 << 30):
        critical_errors.append("free disk space is below configured reserve")
    checks["selection"] = {
        "scenes": len(scene_ids),
        "groups": len(rows),
        "scene_first": int(scene_ids[0]),
        "scene_last": int(scene_ids[-1]),
        "split": args.split,
    }
    checks["critical_errors"] = critical_errors
    checks["passed"] = not critical_errors
    return checks


def cache_only_smoke_test(
    config: TCDPRGConfig,
    db: ManifestDB,
    *,
    units: int,
    model_forward: bool,
) -> dict[str, Any]:
    selected = db.representative_units(units)
    if not selected:
        return {"skipped": True, "reason": "no valid cached requests"}
    provider = CachedObservationProvider(
        config.cache.directory,
        fallback=None,
        max_bytes=1 << 60,
        min_free_bytes=1,
    )
    adapter = build_adapter(config, provider)
    samples = [
        adapter.load_sample(
            scene_id,
            state_id,
            task_index,
            group_index,
            include_global_grasps=False,
        )
        for scene_id, state_id, task_index, group_index in selected
    ]
    result: dict[str, Any] = {
        "cache_only_load_sample": True,
        "units": selected,
        "point_counts": [len(sample.observation.xyz) for sample in samples],
    }
    if not model_forward:
        return result

    import torch

    from tcd_prg.datasets import collate_unified
    from tcd_prg.models import TCDPRGModel

    smoke_config = copy.deepcopy(config)
    smoke_config.ablation.use_gripper_scene_verifier = False
    batch = collate_unified(
        samples,
        grid_size_m=(
            smoke_config.backbone.grid_size_m
            if smoke_config.backbone.backend == "point_transformer_v3"
            else None
        ),
        training=False,
    )
    device = torch.device(
        smoke_config.training.device
        if torch.cuda.is_available() and smoke_config.training.device.startswith("cuda")
        else "cpu"
    )

    def move(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    model = TCDPRGModel(
        smoke_config.model,
        smoke_config.ablation,
        smoke_config.graph,
        smoke_config.router,
        smoke_config.backbone,
    ).to(device)
    model.eval()
    with torch.no_grad():
        output = model(move(batch))
    result.update(
        {
            "model_forward": True,
            "device": str(device),
            "output_keys": sorted(output.keys()),
        }
    )
    return result


def random_sample_verify(
    db: ManifestDB,
    cache_dir: Path,
    count: int,
    seed: int,
    fast_verify: bool,
) -> dict[str, Any]:
    total = db.count()
    if total == 0:
        return {"sampled": 0, "valid": 0, "invalid": 0}
    rows = db.connection.execute("SELECT hash FROM requests ORDER BY hash").fetchall()
    rng = random.Random(seed)
    chosen = rng.sample(rows, min(count, len(rows)))
    valid = 0
    invalid: list[dict[str, str]] = []
    for (key,) in chosen:
        payload_row = db.connection.execute(
            """
            SELECT hash,scene_id,state_id,task_index,group_index,object_count,
                   object_pose,object_active,object_present,object_asset_ids,
                   object_model_ids,object_scales,render_seed,camera_profile,
                   point_count,renderer_version
            FROM requests WHERE hash=?
            """,
            (key,),
        ).fetchone()
        payload = ManifestDB._decode_row(payload_row)
        result = validate_cache_file(
            cache_path(cache_dir, payload.key),
            expected_point_count=payload.point_count,
            object_count=payload.object_count,
            check_values=not fast_verify,
        )
        if result.valid:
            valid += 1
        else:
            invalid.append({"hash": payload.key, "reason": result.reason})
    return {
        "sampled": len(chosen),
        "valid": valid,
        "invalid": len(invalid),
        "invalid_examples": invalid[:20],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute exact TCD-PRG observation point clouds into the content-addressed cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "config.yaml")
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=PROJECT / "configs" / "local_paths.yaml",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--acronym-root", type=Path)
    parser.add_argument("--functional-region-root", type=Path)
    parser.add_argument("--pybullet-python")
    parser.add_argument("--runtime-mesh-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--point-count", type=int, default=None)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-count", type=int, default=None)
    parser.add_argument("--scene-ids-file", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--renderer-mode",
        choices=("persistent", "subprocess"),
        default="persistent",
        help="Persistent mode keeps one external PyBullet process per worker thread.",
    )
    parser.add_argument(
        "--batch-worker-script",
        type=Path,
        default=PROJECT / "scripts" / "render_observation_batch_worker_py38.py",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rescan", action="store_true")
    parser.add_argument(
        "--work-root", type=Path, default=PROJECT / "runtime" / "cache" / "observation_precompute"
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--render-timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--verify-sample", type=int, default=100)
    parser.add_argument("--smoke-units", type=int, default=1)
    parser.add_argument("--model-forward", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument(
        "--fast-verify",
        action="store_true",
        help="Skip expensive finite/range scans while classifying existing files.",
    )
    parser.add_argument(
        "--estimated-bytes-per-observation", type=int, default=DEFAULT_ESTIMATED_BYTES
    )
    parser.add_argument("--check-scenes", type=int, default=50)
    parser.add_argument("--check-active-samples", type=int, default=20)
    parser.add_argument(
        "--override", action="append", default=[], help="Additional OmegaConf dot-list override."
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.scene_count is not None and args.scene_count <= 0:
        parser.error("--scene-count must be positive")
    if args.point_count is not None and args.point_count < 0:
        parser.error("--point-count must be zero (variable) or positive")
    if args.min_free_gb <= 0:
        parser.error("--min-free-gb must be positive")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if sum(bool(x) for x in (args.dry_run, args.verify_only, args.check)) > 1:
        parser.error("--dry-run, --verify-only, and --check are mutually exclusive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(PROJECT)
    config = load_runtime_config(args)
    cache_dir = Path(config.cache.directory).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    adapter = build_adapter(config)
    scene_ids = select_scene_ids(adapter, args)
    rows = selected_index_rows(adapter, scene_ids, args.split)
    index_info = training_index_signature(adapter.training_index_path)
    signature = build_run_signature(config, scene_ids, args.split, index_info)
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir
        else (args.work_root / f"run_{signature[:16]}").resolve()
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "selected_scene_ids.txt").write_text(
        "\n".join(str(scene) for scene in scene_ids) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "tool_version": SCRIPT_VERSION,
        "created_utc": utc_now(),
        "signature": signature,
        "project": str(PROJECT),
        "config": str(args.config.resolve()),
        "paths_config": str(args.paths_config.resolve()),
        "dataset_root": str(Path(config.dataset.root).resolve()),
        "scene_root": str(adapter.scene_root.resolve()),
        "action_root": str(adapter.action_root.resolve()),
        "training_index": index_info,
        "cache_dir": str(cache_dir),
        "runtime_mesh_root": str(Path(config.observation.runtime_mesh_root).resolve()),
        "worker_script": str(Path(config.observation.worker_script).resolve()),
        "batch_worker_script": str(Path(args.batch_worker_script).resolve()),
        "renderer_mode": args.renderer_mode,
        "pybullet_python": str(config.observation.pybullet_python),
        "camera_profile": config.observation.camera_profile,
        "renderer_version": config.observation.renderer_version,
        "point_count": int(config.dataset.scene_points),
        "render_size": [config.observation.render_width, config.observation.render_height],
        "split": args.split,
        "scene_count": len(scene_ids),
        "scene_first": scene_ids[0],
        "scene_last": scene_ids[-1],
        "scene_ids_file": str((work_dir / "selected_scene_ids.txt").resolve()),
        "workers": args.workers,
        "minimum_free_gb": args.min_free_gb,
        "mode": (
            "check"
            if args.check
            else "dry-run"
            if args.dry_run
            else "verify-only"
            if args.verify_only
            else "generate"
        ),
        "lru_eviction_disabled": True,
    }
    atomic_json(work_dir / "manifest.json", manifest)
    print("TCD-PRG observation cache precompute", flush=True)
    print(f"  tool_version={SCRIPT_VERSION}", flush=True)
    print(f"  project={PROJECT}", flush=True)
    print(f"  dataset={config.dataset.root}", flush=True)
    print(
        f"  scenes={len(scene_ids):,} [{scene_ids[0]}..{scene_ids[-1]}] split={args.split}",
        flush=True,
    )
    print(f"  groups={len(rows):,}", flush=True)
    print(f"  point_count={config.dataset.scene_points} (0 means variable/full fusion)", flush=True)
    print(f"  cache={cache_dir}", flush=True)
    print(f"  work_dir={work_dir}", flush=True)
    print("  LRU eviction is NOT used by this tool.", flush=True)

    if args.check:
        report = run_check(args, config, adapter, scene_ids, rows)
        atomic_json(work_dir / "check_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report["passed"] else 2

    db = ManifestDB(work_dir / "manifest.sqlite3")
    try:
        existing_signature = db.get_metadata("signature")
        if existing_signature and existing_signature != signature:
            raise RuntimeError(
                "Existing work directory belongs to a different request signature. "
                "Use a different --work-dir or remove its manifest.sqlite3."
            )
        db.set_metadata("signature", signature)
        db.set_metadata("manifest", manifest)
        db.commit()
        scan_stats = scan_manifest(
            adapter,
            rows,
            db,
            work_dir,
            progress_interval=max(1, args.progress_interval),
            force_rebuild=args.force_rescan or not args.resume,
        )
        manifest.update(scan_stats)
        manifest["unique_request_count"] = db.count()
        atomic_json(work_dir / "manifest.json", manifest)
        print(
            f"[scan-complete] groups={scan_stats['groups']:,} "
            f"unique_hashes={db.count():,} "
            f"deduplicated={scan_stats['groups'] - db.count():,}",
            flush=True,
        )

        verification = classify_existing(db, args, cache_dir, work_dir)
        space = estimate_space(db, args.estimated_bytes_per_observation)
        print(
            f"[space] valid={space['valid_count']:,} "
            f"remaining={space['remaining_request_count']:,} "
            f"avg={format_bytes(space['average_valid_file_bytes'])} "
            f"additional≈{format_bytes(space['estimated_additional_bytes'])} "
            f"total≈{format_bytes(space['estimated_total_bytes'])}",
            flush=True,
        )
        if args.dry_run or args.verify_only:
            sample_report = random_sample_verify(
                db,
                cache_dir,
                count=max(100, args.verify_sample) if db.count() >= 100 else db.count(),
                seed=config.training.seed,
                fast_verify=args.fast_verify,
            )
            summary = {
                "completed_utc": utc_now(),
                "mode": "dry-run" if args.dry_run else "verify-only",
                "scan": scan_stats,
                "classification": asdict(verification),
                "space": space,
                "sample_validation": sample_report,
                "status_counts": db.status_counts(),
                "work_dir": str(work_dir),
            }
            atomic_json(work_dir / "summary.json", summary)
            return 0 if args.dry_run or not (verification.missing or verification.invalid) else 3

        check_disk_space(
            cache_dir,
            int(args.min_free_gb * (1 << 30)),
        )
        try:
            generation = generate_missing(db, args, config, cache_dir, work_dir)
        except DiskSpaceReserveError as error:
            stopped_summary = {
                "completed_utc": utc_now(),
                "mode": "stopped-low-disk",
                "scan": scan_stats,
                "status_counts": db.status_counts(),
                "space": estimate_space(db, args.estimated_bytes_per_observation),
                "error": str(error),
                "work_dir": str(work_dir),
            }
            atomic_json(work_dir / "summary.json", stopped_summary)
            print(json.dumps(stopped_summary, ensure_ascii=False, indent=2), flush=True)
            return 7
        sample_report = random_sample_verify(
            db,
            cache_dir,
            count=max(100, args.verify_sample) if db.count() >= 100 else db.count(),
            seed=config.training.seed,
            fast_verify=args.fast_verify,
        )
        smoke: dict[str, Any]
        if args.skip_smoke_test:
            smoke = {"skipped": True, "reason": "--skip-smoke-test"}
        else:
            try:
                smoke = cache_only_smoke_test(
                    config,
                    db,
                    units=max(1, args.smoke_units),
                    model_forward=args.model_forward,
                )
            except BaseException as error:
                smoke = {
                    "failed": True,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
        status_counts = db.status_counts()
        remaining_failures = status_counts.get(STATUS_FAILED, 0)
        summary = {
            "completed_utc": utc_now(),
            "mode": "generate",
            "scan": scan_stats,
            "initial_classification": asdict(verification),
            "generation": asdict(generation),
            "status_counts": status_counts,
            "space": estimate_space(db, args.estimated_bytes_per_observation),
            "sample_validation": sample_report,
            "cache_only_smoke_test": smoke,
            "cache_total_bytes": db.total_file_size(),
            "cache_average_bytes": (
                db.total_file_size()
                / max(
                    1, status_counts.get(STATUS_CACHED, 0) + status_counts.get(STATUS_GENERATED, 0)
                )
            ),
            "work_dir": str(work_dir),
            "failures_file": str(work_dir / "failures.jsonl"),
        }
        atomic_json(work_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if remaining_failures:
            print(
                f"[complete-with-failures] remaining failed requests={remaining_failures:,}; "
                "rerun the same command to resume/retry.",
                flush=True,
            )
            return 4
        if sample_report["invalid"]:
            return 5
        if smoke.get("failed"):
            return 6
        print("[complete] all selected unique requests are valid in cache.", flush=True)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
