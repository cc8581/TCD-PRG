"""Immutable, split-safe offline physics outcomes for exact PUSH actions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tcd_prg.constants import PUSH_DISTANCE_M, CandidateStatus

PROTOCOL_VERSION = "push-outcome-bank-v1"


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PushSceneState:
    split: str
    scene_id: int
    state_id: int
    state_hash: str
    object_geometry_ids: tuple[str, ...]
    object_poses_xyzw: tuple[tuple[float, ...], ...]
    target_object: int
    task_category_id: int
    task_region_id: int
    simulator_version: str
    physics_parameters: Mapping[str, Any]
    robot_configuration: str
    label_generation_version: str
    random_seed: int

    def validate(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError("Push Outcome Bank split must be train, val or test")
        if not self.state_hash:
            raise ValueError("Push Outcome Bank requires an immutable state_hash")
        if len(self.object_geometry_ids) != len(self.object_poses_xyzw):
            raise ValueError("Object geometry IDs and poses must align")
        if any(len(pose) != 7 for pose in self.object_poses_xyzw):
            raise ValueError("Object poses must use xyz+xyzw")

    @property
    def scene_state_hash(self) -> str:
        self.validate()
        return _canonical_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class PushAction:
    object_index: int
    contact_world: tuple[float, float, float]
    direction_world: tuple[float, float, float]
    push_distance_m: float = PUSH_DISTANCE_M

    @property
    def action_hash(self) -> str:
        if abs(float(self.push_distance_m) - PUSH_DISTANCE_M) > 1e-6:
            raise ValueError("Outcome Bank action does not use the formal 0.15 m PUSH")
        return _canonical_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class PushOutcomeRecord:
    scene: PushSceneState
    action: PushAction
    status: int
    improves_state: bool | None
    potential_delta: tuple[float, ...] | None
    object_displacement_m: float | None
    target_displacement_m: float | None
    out_of_workspace: bool | None
    unstable: bool | None
    collision_or_invalid: bool | None
    robust_success_count: int = 0
    robust_trial_count: int = 0
    protocol_version: str = PROTOCOL_VERSION

    def validate(self) -> None:
        self.scene.validate()
        _ = self.action.action_hash
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unsupported PUSH outcome protocol version")
        if self.status not in {
            int(CandidateStatus.UNKNOWN_UNTESTED),
            int(CandidateStatus.NEGATIVE),
            int(CandidateStatus.POSITIVE),
        }:
            raise ValueError("Outcome status must preserve UNKNOWN/NEGATIVE/POSITIVE")
        unknown = self.status == int(CandidateStatus.UNKNOWN_UNTESTED)
        if unknown and self.improves_state is not None:
            raise ValueError("UNKNOWN outcome cannot carry an effectiveness label")
        if not unknown and self.improves_state is None:
            raise ValueError("Executed outcome requires an effectiveness label")
        if not 0 <= self.robust_success_count <= self.robust_trial_count:
            raise ValueError("Invalid local robustness counts")

    @property
    def key(self) -> str:
        self.validate()
        return f"{self.scene.scene_state_hash}:{self.action.action_hash}:{self.protocol_version}"

    @property
    def local_robust_success_rate(self) -> float | None:
        return (
            self.robust_success_count / self.robust_trial_count
            if self.robust_trial_count
            else None
        )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"] = self.key
        payload["local_robust_success_rate"] = self.local_robust_success_rate
        return payload


class PushOutcomeBank:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append(self, records: Iterable[PushOutcomeRecord], *, mining: bool = False) -> int:
        incoming = list(records)
        existing = {item["key"] for item in self.records()}
        serialized: list[dict[str, Any]] = []
        for record in incoming:
            record.validate()
            if mining and record.scene.split != "train":
                raise ValueError("Hard-UNKNOWN mining may append train scenes only")
            if record.key in existing:
                raise ValueError(f"Duplicate PUSH outcome key: {record.key}")
            existing.add(record.key)
            serialized.append(record.to_json())
        if not serialized:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for payload in serialized:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return len(serialized)
