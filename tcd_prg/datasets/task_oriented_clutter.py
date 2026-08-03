"""Adapter for TaskOrientedClutterSceneDataset action-label format v2."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from tcd_prg.constants import (
    PUSH_DISTANCE_M,
    ActionType,
    CandidateStatus,
    OutcomeCode,
)
from tcd_prg.geometry.numpy_se3 import compose_pose_with_transform, quaternion_xyzw_to_matrix_numpy
from tcd_prg.observation.base import ObservationProvider, ObservationRequest
from tcd_prg.observation.saved import SavedObservationProvider

from .base import DatasetAdapter
from .capabilities import DatasetCapabilities
from .registries import FunctionalRegionRegistry, GraspLibraryRegistry
from .types import (
    ActionCandidateGroup,
    CameraParameters,
    GlobalGraspLabels,
    SceneObservation,
    SequenceLabels,
    StateLabels,
)

ACTION_GROUP_INDEX_CACHE_VERSION = "tcd_prg_action_group_strata_v1"
ACTION_GROUP_STRATA = (
    "direct_grasp",
    "pick_remove",
    "push",
    "push_failure",
    "unresolved_or_unknown",
)


def _show_index_progress() -> bool:
    """Only rank zero owns terminal progress in distributed training."""

    try:
        import torch
    except ImportError:
        return True
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values)


def _ragged(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    return values[int(offsets[index]) : int(offsets[index + 1])]


def _se3_diverse_rows(library, rows: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest-first object-frame pose/opening sampling."""

    # 以平移、接近轴、夹持轴和开口宽度联合距离做最远点采样，避免只保留单一抓取模式。
    rows = np.asarray(rows, np.int64)
    if count <= 0 or not len(rows):
        return np.empty(0, np.int64)
    if len(rows) <= count:
        return rows
    transform = library.transform_object[rows]
    translation = transform[:, :3, 3]
    approach = transform[:, :3, 2]
    closing_axis = transform[:, :3, 0]
    width = library.ag_width_m[rows]
    first = int(np.argmax(library.quality[rows]))
    selected = [first]
    minimum = np.full(len(rows), np.inf, np.float64)
    while len(selected) < count:
        prior = selected[-1]
        translation_distance = np.sqrt(np.sum((translation - translation[prior]) ** 2, axis=-1)) / 0.01
        approach_distance = 1.0 - np.clip(np.sum(approach * approach[prior], axis=-1), -1.0, 1.0)
        jaw_distance = 1.0 - np.abs(np.clip(np.sum(closing_axis * closing_axis[prior], axis=-1), -1.0, 1.0))
        width_distance = np.abs(width - width[prior]) / 0.005
        minimum = np.minimum(
            minimum, translation_distance + approach_distance + jaw_distance + width_distance
        )
        minimum[selected] = -1.0
        selected.append(int(np.argmax(minimum)))
    return rows[np.asarray(selected, np.int64)]


def _object_frame_nms_rows(
    library, rows: np.ndarray, translation_m: float = 0.01,
    rotation_cosine: float = 0.965925826, width_m: float = 0.005,
) -> np.ndarray:
    """NMS full evaluation truth under parallel-jaw symmetry."""

    rows = np.asarray(rows, np.int64)
    order = rows[np.argsort(library.quality[rows], kind="stable")[::-1]]
    selected: list[int] = []
    transforms = library.transform_object
    for row in order.tolist():
        if not selected:
            selected.append(row)
            continue
        prior = np.asarray(selected, np.int64)
        candidate = transforms[row]
        translation_close = np.sum(
            (transforms[prior, :3, 3] - candidate[:3, 3]) ** 2, axis=-1
        ) <= translation_m**2
        if not translation_close.any():
            selected.append(row)
            continue
        nearby = prior[translation_close]
        approach_close = np.sum(
            transforms[nearby, :3, 2] * candidate[:3, 2], axis=-1
        ) >= rotation_cosine
        jaw_close = np.abs(np.sum(
            transforms[nearby, :3, 0] * candidate[:3, 0], axis=-1
        )) >= rotation_cosine
        opening_close = np.abs(library.ag_width_m[nearby] - library.ag_width_m[row]) <= width_m
        if not np.any(approach_close & jaw_close & opening_close):
            selected.append(row)
    return np.asarray(selected, np.int64)


class TaskOrientedClutterAdapter(DatasetAdapter):
    """Map the current dataset to the model-independent contract.

    The adapter never opens files in ``.work`` and snapshots only atomically
    published ``scene_*.h5`` files present at construction time.
    """

    capabilities = DatasetCapabilities(
        has_instance_masks=True,
        has_task_regions=True,
        has_task_grasps=True,
        has_global_grasps=True,
        has_push_actions=True,
        has_pick_remove_actions=True,
        has_sequences=True,
        has_relation_graph=True,
        # Exact FR5/AG certification is provided at execution time; it is not a
        # ground-truth field in the published HDF5 action labels.
        has_exact_ik=False,
        has_intermediate_observations=False,
        supports_closed_loop=True,
    )

    def __init__(
        self,
        root: str | Path,
        observation_provider: ObservationProvider | None = None,
        point_count: int = 16_384,
        renderer_version: str = "tcd_prg_pybullet_v1",
        camera_profile: str = "mecheye_pro_s_three_view",
        functional_region_root: str | Path | None = None,
        verifier_wrong_region_negatives: int = 8,
        verifier_collision_negatives: int = 8,
        verifier_approach_negatives: int = 8,
        sampling_seed: int = 2026,
        scene_subdir: str = "task_clutter_scenes_20_categories",
        step_labels_subdir: str = "task_training_labels_steps1_6_v1",
        action_labels_subdir: str = "task_positive_multistep_sequences",
        global_positive_grasps_per_object: int = 64,
        global_negative_grasps_per_object: int = 32,
        index_cache_dir: str | Path = "runtime/cache/dataset_indexes",
    ) -> None:
        self.root = Path(root)
        self.scene_root = self.root / scene_subdir
        self.step_root = self.root / step_labels_subdir
        self.action_root = self.root / action_labels_subdir
        self.training_index_path = self.action_root / "training_index.h5"
        self.index_cache_dir = Path(index_cache_dir)
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        self._action_group_index: np.ndarray | None = None
        self._reported_strata_cache = False
        self.global_sampling = (
            int(global_positive_grasps_per_object),
            int(global_negative_grasps_per_object),
        )
        for path in (self.scene_root, self.step_root, self.action_root):
            if not path.is_dir():
                raise FileNotFoundError(path)
        self.metadata = json.loads((self.scene_root / "metadata.json").read_text(encoding="utf-8"))
        self.action_metadata = json.loads((self.action_root / "metadata.json").read_text(encoding="utf-8"))
        self.raw_relation_names = tuple(self.metadata["relations"])
        required_raw = {"near", "contact", "support", "occlude", "block_path"}
        if not required_raw.issubset(self.raw_relation_names):
            raise ValueError(
                f"Dataset relation channels do not satisfy the adapter contract: {self.raw_relation_names}"
            )
        self.relation_names = ("near", "contact", "support", "press", "occlude")
        self.point_count = point_count
        self.renderer_version = renderer_version
        self.camera_profile = camera_profile
        self.verifier_sampling = (
            int(verifier_wrong_region_negatives), int(verifier_collision_negatives),
            int(verifier_approach_negatives), int(sampling_seed),
        )
        # 初始化时快照已原子发布的 HDF5，训练期间不会扫描或读取仍在写入的 .work 文件。
        self._h5_paths = tuple(sorted((self.action_root / "scene_labels").glob("scene_*.h5")))
        self._scene_ids = tuple(int(p.stem.split("_")[-1]) for p in self._h5_paths)
        self._path_by_scene = dict(zip(self._scene_ids, self._h5_paths, strict=True))
        self.observation_provider = observation_provider or SavedObservationProvider(
            self.scene_root, self.scene_root / "metadata.json", point_count
        )
        self.camera_parameters = self._formal_camera_parameters()
        self.grasp_registry = GraspLibraryRegistry(self.step_root / "grasp_library")
        inferred_region_root = (
            self.root.parent
            / "Grasp_20_class_object_3D_model"
            / "data"
            / "manual_function_regions_v1"
        )
        region_root = Path(functional_region_root) if functional_region_root else inferred_region_root
        self.functional_region_registry = FunctionalRegionRegistry(region_root) if region_root.is_dir() else None

    @property
    def snapshot_scene_ids(self) -> tuple[int, ...]:
        return self._scene_ids

    def _formal_camera_parameters(self) -> tuple[CameraParameters, ...]:
        width, height = self.metadata["image_size"]
        cameras = []
        for item in self.metadata["camera_parameters"]:
            if item["sensor_type"].lower() == "oracle":
                continue
            cameras.append(
                CameraParameters(
                    sensor_type=item["sensor_type"],
                    width=width,
                    height=height,
                    eye_world=np.asarray(item["eye"], np.float32),
                    target_world=np.asarray(item["target"], np.float32),
                    up_world=np.asarray(item["up"], np.float32),
                    fx=float(item["fx"]),
                    fy=float(item["fy"]),
                    cx=float(item["cx"]),
                    cy=float(item["cy"]),
                    near_m=float(item["z_near"]),
                    far_m=float(item["z_far"]),
                )
            )
        if len(cameras) != 3:
            raise ValueError("Expected exactly three formal PRO S cameras")
        return tuple(cameras)

    def _scene_group(self, handle: h5py.File, scene_id: int) -> h5py.Group:
        key = f"scene_{scene_id:04d}"
        if key not in handle:
            raise KeyError(f"{key} absent from {handle.filename}")
        return handle[key]

    def _h5_path(self, scene_id: int) -> Path:
        try:
            return self._path_by_scene[scene_id]
        except KeyError as error:
            raise FileNotFoundError(f"No completed action HDF5 for scene {scene_id}") from error

    def iter_action_groups(self, split: str | None = None) -> Iterable[tuple[int, int, int, int]]:
        # 生成器发布的全局 training_index 是权威索引，避免每次训练重开 10,000 个 HDF5。
        if self.training_index_path.is_file():
            rows = self._load_action_group_index()
            if split is not None:
                split_code = {"train": 0, "val": 1, "test": 2}.get(split)
                if split_code is None:
                    raise ValueError(f"Unsupported split: {split}")
                rows = rows[rows[:, 5] == split_code]
            yield from (
                (int(row[1]), int(row[4]), int(row[3]), int(row[2]))
                for row in rows
            )
            return

        # 兼容没有全局索引的旧数据集；首次扫描会明确显示当前场景进度。
        for scene_id in tqdm(
            self._scene_ids,
            desc="scan action-group index",
            unit="scene",
            disable=not _show_index_progress(),
        ):
            if split is not None:
                scene_file = self.scene_root / f"scene_{scene_id:04d}" / "scene.npz"
                with np.load(scene_file, allow_pickle=False) as data:
                    if str(data["split"].item()) != split:
                        continue
            with h5py.File(self._h5_path(scene_id), "r", swmr=True) as handle:
                group = self._scene_group(handle, scene_id)["action_state_groups"]
                states = group["from_state"][:]
                tasks = group["task_index"][:]
            yield from (
                (scene_id, int(state), int(task), index)
                for index, (state, task) in enumerate(zip(states, tasks, strict=True))
            )

    def _load_action_group_index(self) -> np.ndarray:
        if self._action_group_index is None:
            with h5py.File(self.training_index_path, "r", swmr=True) as handle:
                if handle.attrs.get("format") != "task_conditioned_action_training_index_v2":
                    raise ValueError(
                        f"Unsupported training index format: {self.training_index_path}"
                    )
                rows = handle["action_state_group"][:]
            if rows.ndim != 2 or rows.shape[1] < 7:
                raise ValueError("action_state_group index must be [N,>=7]")
            self._action_group_index = rows.astype(np.int32, copy=False)
            if _show_index_progress():
                print(
                    f"[dataset-index] source=training_index.h5 groups={len(rows)}",
                    flush=True,
                )
        return self._action_group_index

    def _strata_cache_path(self) -> Path:
        stat = self.training_index_path.stat()
        with h5py.File(self.training_index_path, "r", swmr=True) as handle:
            signature = str(handle.attrs.get("generation_signature", ""))
        payload = "|".join((
            ACTION_GROUP_INDEX_CACHE_VERSION,
            str(self.action_root.resolve()),
            signature,
            str(stat.st_size),
            str(stat.st_mtime_ns),
        ))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.index_cache_dir / f"action_group_strata_{digest}.npz"

    @staticmethod
    def _load_strata_cache(path: Path, expected_rows: int) -> np.ndarray | None:
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as cached:
                if str(cached["version"].item()) != ACTION_GROUP_INDEX_CACHE_VERSION:
                    return None
                codes = cached["strata"].astype(np.int8, copy=False)
            return codes if codes.shape == (expected_rows,) else None
        except (OSError, KeyError, ValueError):
            return None

    def _build_strata_cache(self, path: Path, rows: np.ndarray) -> np.ndarray:
        codes = np.full(len(rows), ACTION_GROUP_STRATA.index("unresolved_or_unknown"), np.int8)
        scene_column = rows[:, 1]
        scene_ids, starts = np.unique(scene_column, return_index=True)
        stops = np.r_[starts[1:], len(rows)]
        for scene_id, start, stop in tqdm(
            zip(scene_ids.tolist(), starts.tolist(), stops.tolist(), strict=True),
            total=len(scene_ids),
            desc="cache action-group strata",
            unit="scene",
            disable=not _show_index_progress(),
        ):
            scene_rows = rows[start:stop]
            if not np.all(scene_rows[:, 1] == scene_id):
                raise ValueError("training_index.h5 must group action-state rows by scene")
            with h5py.File(self._h5_path(int(scene_id)), "r", swmr=True) as handle:
                scene = self._scene_group(handle, int(scene_id))
                groups = scene["action_state_groups"]
                offsets = groups["action_offsets"][:]
                group_action_ids = groups["action_ids"][:]
                actions = scene["actions"]
                action_type = actions["action_type"][:]
                executed = actions["executed"][:]
                success = actions["success"][:] | actions["potential_improved"][:]
                for row_index, group_index in enumerate(scene_rows[:, 2], start=start):
                    group_index = int(group_index)
                    ids = group_action_ids[
                        int(offsets[group_index]) : int(offsets[group_index + 1])
                    ]
                    evaluated_positive = executed[ids] & success[ids]
                    positive_types = set(int(x) for x in action_type[ids][evaluated_positive])
                    if int(ActionType.TASK_GRASP) in positive_types:
                        stratum = "direct_grasp"
                    elif int(ActionType.PICK_REMOVE) in positive_types:
                        stratum = "pick_remove"
                    elif int(ActionType.PUSH) in positive_types:
                        stratum = "push"
                    elif np.any(executed[ids] & (action_type[ids] == int(ActionType.PUSH))):
                        stratum = "push_failure"
                    else:
                        stratum = "unresolved_or_unknown"
                    codes[row_index] = ACTION_GROUP_STRATA.index(stratum)
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary,
            version=np.asarray(ACTION_GROUP_INDEX_CACHE_VERSION),
            strata=codes,
        )
        os.replace(temporary, path)
        return codes

    def _load_or_build_strata_cache(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self._load_action_group_index()
        path = self._strata_cache_path()
        cached = self._load_strata_cache(path, len(rows))
        if cached is not None:
            if _show_index_progress() and not getattr(self, "_reported_strata_cache", False):
                print(f"[dataset-index] strata_cache=hit path={path}", flush=True)
                self._reported_strata_cache = True
            return rows, cached
        lock = path.with_suffix(".lock")
        # 给 rank 0 留出优先创建锁的时间，避免非主进程抢到构建任务却不显示进度。
        if not _show_index_progress():
            for _ in range(240):
                cached = self._load_strata_cache(path, len(rows))
                if cached is not None:
                    return rows, cached
                time.sleep(0.25)
        while True:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(descriptor)
                break
            except FileExistsError:
                cached = self._load_strata_cache(path, len(rows))
                if cached is not None:
                    return rows, cached
                try:
                    lock_age = time.time() - lock.stat().st_mtime
                except FileNotFoundError:
                    continue
                if lock_age > 6 * 60 * 60:
                    lock.unlink(missing_ok=True)
                    continue
                time.sleep(0.25)
        try:
            cached = self._load_strata_cache(path, len(rows))
            if cached is None:
                cached = self._build_strata_cache(path, rows)
                if _show_index_progress():
                    print(f"[dataset-index] strata_cache=created path={path}", flush=True)
                    self._reported_strata_cache = True
            return rows, cached
        finally:
            lock.unlink(missing_ok=True)

    def action_group_strata(
        self, units: Iterable[tuple[int, int, int, int]] | None = None
    ) -> dict[tuple[int, int, int, int], str]:
        """Classify groups without loading observations or grasp libraries."""

        if self.training_index_path.is_file():
            rows, codes = self._load_or_build_strata_cache()
            requested = None if units is None else set(units)
            result: dict[tuple[int, int, int, int], str] = {}
            for row, code in zip(rows, codes, strict=True):
                unit = (int(row[1]), int(row[4]), int(row[3]), int(row[2]))
                if requested is None or unit in requested:
                    result[unit] = ACTION_GROUP_STRATA[int(code)]
            return result

        result: dict[tuple[int, int, int, int], str] = {}
        requested = None if units is None else set(units)
        scene_ids = self._scene_ids if requested is None else sorted({unit[0] for unit in requested})
        for scene_id in scene_ids:
            with h5py.File(self._h5_path(scene_id), "r", swmr=True) as handle:
                scene = self._scene_group(handle, scene_id)
                groups = scene["action_state_groups"]
                offsets = groups["action_offsets"][:]
                group_action_ids = groups["action_ids"][:]
                states = groups["from_state"][:]
                tasks = groups["task_index"][:]
                actions = scene["actions"]
                action_type = actions["action_type"][:]
                executed = actions["executed"][:]
                success = actions["success"][:] | actions["potential_improved"][:]
                for group_index, (state_id, task_index) in enumerate(zip(states, tasks, strict=True)):
                    unit = (scene_id, int(state_id), int(task_index), group_index)
                    if requested is not None and unit not in requested:
                        continue
                    ids = group_action_ids[int(offsets[group_index]) : int(offsets[group_index + 1])]
                    evaluated_positive = executed[ids] & success[ids]
                    positive_types = set(int(x) for x in action_type[ids][evaluated_positive])
                    if int(ActionType.TASK_GRASP) in positive_types:
                        stratum = "direct_grasp"
                    elif int(ActionType.PICK_REMOVE) in positive_types:
                        stratum = "pick_remove"
                    elif int(ActionType.PUSH) in positive_types:
                        stratum = "push"
                    elif np.any(executed[ids] & (action_type[ids] == int(ActionType.PUSH))):
                        stratum = "push_failure"
                    else:
                        stratum = "unresolved_or_unknown"
                    result[unit] = stratum
        return result

    def _object_active(
        self, scene: h5py.Group, state_id: int, task_index: int | None = None
    ) -> np.ndarray:
        object_count = len(scene["catalog/object_index"])
        actions = scene["actions"]
        from_state = actions["from_state"][:]
        to_state = actions["to_state"][:]
        action_type = actions["action_type"][:]
        payload_index = actions["payload_index"][:]
        after_state_valid = actions["after_state_valid"][:]
        action_task = actions["task_index"][:]
        acted_object = np.full(len(action_type), -1, dtype=np.int64)
        pick_payload = actions["pick_remove/acted_object"][:]
        for action_index in np.flatnonzero(action_type == int(ActionType.PICK_REMOVE)):
            acted_object[action_index] = int(pick_payload[int(payload_index[action_index])])
        sequence_depth = scene["states/sequence_depth"][:]
        if task_index is None:
            task_index = int(scene["states/task_index"][state_id])
        predecessors: dict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(to_state):
            source, target = int(from_state[index]), int(target)
            if (
                after_state_valid[index]
                and target >= 0
                and int(action_task[index]) == task_index
                and sequence_depth[source] < sequence_depth[target]
            ):
                predecessors[int(target)].append(index)
        memo: dict[int, frozenset[int]] = {
            int(state): frozenset() for state in np.flatnonzero(sequence_depth == 0)
        }
        visiting: set[int] = set()

        def removed(state: int) -> frozenset[int]:
            if state in memo:
                return memo[state]
            if state in visiting:
                # Strictly increasing sequence depth above should make this
                # unreachable; retain a diagnostic for corrupted labels.
                raise ValueError("Cycle in depth-monotone transition graph")
            visiting.add(state)
            histories: list[frozenset[int]] = []
            for transition_index in predecessors.get(state, []):
                history = set(removed(int(from_state[transition_index])))
                if int(action_type[transition_index]) == int(ActionType.PICK_REMOVE):
                    history.add(int(acted_object[transition_index]))
                histories.append(frozenset(history))
            visiting.remove(state)
            if not histories:
                # A state can be published without a successful-sequence parent.
                # It then has no justified PICK_REMOVE history; do not invent one.
                memo[state] = frozenset()
                return memo[state]
            # Alternative positive sequences may merge into the same state. A
            # removed object must remain inactive on every subsequent branch;
            # union is the conservative macro-action semantics.
            memo[state] = frozenset().union(*histories)
            return memo[state]

        inactive = removed(state_id)
        active = np.ones(object_count, dtype=bool)
        if inactive:
            active[np.fromiter(inactive, dtype=np.int64)] = False
        return active

    def load_observation(self, scene_id: int, state_id: int, task_index: int) -> SceneObservation:
        with h5py.File(self._h5_path(scene_id), "r", swmr=True) as handle:
            scene = self._scene_group(handle, scene_id)
            states = scene["states"]
            catalog = scene["catalog"]
            object_pose = states["object_pose"][state_id].astype(np.float32)
            required_grasp_count = int(states["required_grasp_count"][state_id])
            target_object = int(catalog["task_object_index"][task_index])
            task_region = int(catalog["task_label"][task_index])
            active = self._object_active(scene, state_id, task_index)
            with np.load(
                self.scene_root / f"scene_{scene_id:04d}" / "scene.npz", allow_pickle=False
            ) as raw_scene:
                h5_names = tuple(str(x) for x in raw_scene["object_h5_name"][: len(object_pose)])
                category_keys = tuple(str(x) for x in raw_scene["object_category_key"][: len(object_pose)])
                model_ids = tuple(str(x) for x in raw_scene["object_model_id"][: len(object_pose)])
                object_scales = raw_scene["object_scale"][: len(object_pose)].astype(np.float32)
            request = self._make_observation_request(
                scene_id=scene_id,
                state_id=state_id,
                object_pose=object_pose,
                active=active,
                h5_names=h5_names,
                model_ids=model_ids,
                object_scales=object_scales,
                render_seed=int(states["observation_render_seed"][state_id]),
            )
            object_count = len(object_pose)
            object_category_id = catalog["object_category_id"][:].astype(np.int64)
        points = self.observation_provider.get(request)
        target_point_mask = points.instance_id == target_object
        region_target = np.zeros(len(points.xyz), dtype=bool)
        region_valid = np.zeros(len(points.xyz), dtype=bool)
        task_region_visibility: float | None = None
        if self.functional_region_registry is not None and np.any(target_point_mask):
            target_labels, target_valid = self.functional_region_registry.visible_labels(
                points.xyz[target_point_mask],
                object_pose[target_object],
                category_keys[target_object],
                model_ids[target_object],
                float(object_scales[target_object]),
            )
            region_target[target_point_mask] = target_labels == task_region
            region_valid[target_point_mask] = target_valid
            task_region_visibility = float(np.any(region_target & region_valid))
        observation = SceneObservation(
            scene_id=scene_id,
            state_id=state_id,
            task_index=task_index,
            xyz=points.xyz,
            rgb=points.rgb,
            instance_id=points.instance_id,
            target_mask=target_point_mask,
            target_object=target_object,
            task_region_id=task_region,
            object_uuid=tuple(f"scene_{scene_id:04d}/object_{i:02d}" for i in range(object_count)),
            object_pose=object_pose,
            object_category_id=object_category_id,
            object_present=np.ones(object_count, dtype=bool),
            object_active=active,
            camera_parameters=self.camera_parameters,
            point_valid=np.ones(len(points.xyz), dtype=bool),
            source_view=points.source_view,
            region_target=region_target if self.functional_region_registry is not None else None,
            region_valid=region_valid if self.functional_region_registry is not None else None,
            task_region_visibility=task_region_visibility,
            metadata={
                "quaternion_order": "xyzw",
                "length_unit": "m",
                "oracle_excluded": True,
                # This is the task acceptance criterion, not the state's
                # verified-positive truth count.  Deployment supplies the same
                # criterion from the task/object registry.
                "required_grasp_count": required_grasp_count,
                "object_category_key": category_keys,
                "object_model_id": model_ids,
                "object_scale": object_scales,
            },
        )
        observation.validate()
        return observation

    def _make_observation_request(
        self,
        *,
        scene_id: int,
        state_id: int,
        object_pose: np.ndarray,
        active: np.ndarray,
        h5_names: tuple[str, ...],
        model_ids: tuple[str, ...],
        object_scales: np.ndarray,
        render_seed: int,
    ) -> ObservationRequest:
        return ObservationRequest(
            scene_id=scene_id,
            state_id=state_id,
            object_pose=object_pose,
            object_active=active,
            object_present=np.ones(len(object_pose), dtype=bool),
            object_asset_ids=h5_names,
            object_model_ids=model_ids,
            object_scales=object_scales,
            render_seed=render_seed,
            camera_profile=self.camera_profile,
            point_count=self.point_count,
            renderer_version=self.renderer_version,
        )

    def observation_available(self, scene_id: int, state_id: int, task_index: int) -> bool:
        """Check the exact content-addressed request without loading point arrays."""

        with h5py.File(self._h5_path(scene_id), "r", swmr=True) as handle:
            scene = self._scene_group(handle, scene_id)
            states = scene["states"]
            object_pose = states["object_pose"][state_id].astype(np.float32)
            active = self._object_active(scene, state_id, task_index)
            render_seed = int(states["observation_render_seed"][state_id])
        with np.load(
            self.scene_root / f"scene_{scene_id:04d}" / "scene.npz", allow_pickle=False
        ) as raw_scene:
            count = len(object_pose)
            h5_names = tuple(str(x) for x in raw_scene["object_h5_name"][:count])
            model_ids = tuple(str(x) for x in raw_scene["object_model_id"][:count])
            object_scales = raw_scene["object_scale"][:count].astype(np.float32)
        request = self._make_observation_request(
            scene_id=scene_id,
            state_id=state_id,
            object_pose=object_pose,
            active=active,
            h5_names=h5_names,
            model_ids=model_ids,
            object_scales=object_scales,
            render_seed=render_seed,
        )
        return self.observation_provider.is_available(request)

    @lru_cache(maxsize=64)
    def _object_match_files(self, scene_id: int) -> tuple[str, ...]:
        """Cache the existing per-object grasp-library links for one scene."""

        path = self.step_root / "scene_labels" / f"scene_{scene_id:04d}_labels.npz"
        with np.load(path, allow_pickle=False) as labels:
            return tuple(str(value) for value in labels["object_match_file"])

    @lru_cache(maxsize=128)
    def _pick_remove_grasp_records(
        self, scene_id: int, state_id: int
    ) -> dict[tuple[int, int], tuple[np.ndarray, int]]:
        """Aggregate native PICK_REMOVE attempts into conflict-safe three-state labels."""

        with h5py.File(self._h5_path(scene_id), "r", swmr=True) as handle:
            actions = self._scene_group(handle, scene_id)["actions"]
            action_type = actions["action_type"][:].astype(np.int8)
            from_state = actions["from_state"][:].astype(np.int64)
            action_ids = np.flatnonzero(
                (action_type == int(ActionType.PICK_REMOVE)) & (from_state == state_id)
            )
            if not len(action_ids):
                return {}
            payload = actions["payload_index"][:][action_ids].astype(np.int64)
            pick_remove = actions["pick_remove"]
            objects = pick_remove["acted_object"][:][payload].astype(np.int64)
            sources = pick_remove["removal_grasp_source_index"][:][payload].astype(np.int64)
            poses = pick_remove["removal_grasp_pose_world"][:][payload].astype(np.float32)
            executed = actions["executed"][:][action_ids].astype(bool)
            # Global grasp quality means that the removal grasp itself
            # succeeded. A merely improved post-state remains a failed grasp,
            # even though it can still be useful policy evidence.
            positive = executed & actions["success"][:][action_ids].astype(bool)

        grouped: dict[tuple[int, int], dict[str, object]] = {}
        for obj, source, pose, was_executed, was_positive in zip(
            objects, sources, poses, executed, positive, strict=True
        ):
            key = (int(obj), int(source))
            record = grouped.setdefault(key, {"pose": pose, "known": set()})
            if was_executed:
                known = record["known"]
                assert isinstance(known, set)
                known.add(int(CandidateStatus.POSITIVE if was_positive else CandidateStatus.NEGATIVE))

        result: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
        for key, record in grouped.items():
            known = record["known"]
            assert isinstance(known, set)
            status = next(iter(known)) if len(known) == 1 else int(CandidateStatus.UNKNOWN_UNTESTED)
            result[key] = (np.asarray(record["pose"], np.float32), status)
        return result

    def load_global_grasps(
        self, scene_id: int, state_id: int, observation: SceneObservation,
        training: bool = True,
    ) -> GlobalGraspLabels:
        """Build task-free removal-grasp supervision from the published dataset only.

        Geometry comes from the existing per-object ``grasp_library`` while the
        authoritative world pose and three-state outcome come from PICK_REMOVE
        rows in the action HDF5. The attempted set is not exhaustive, so
        unmatched queries are never treated as negatives.
        """

        records = self._pick_remove_grasp_records(scene_id, state_id)
        match_files = self._object_match_files(scene_id)
        object_indices: list[int] = []
        sources: list[int] = []
        contacts: list[np.ndarray] = []
        anchor_distances: list[float] = []
        poses: list[np.ndarray] = []
        approaches: list[np.ndarray] = []
        widths: list[float] = []
        scene_labels: list[int] = []

        for object_index, active in enumerate(observation.physical_active):
            if not active:
                continue
            object_records = {
                source: value for (obj, source), value in records.items() if obj == object_index
            }
            if not object_records:
                continue
            library = self.grasp_registry.load(match_files[object_index])
            source_to_row: dict[int, int] = {}
            for row, source in enumerate(library.source_index):
                source_to_row.setdefault(int(source), row)
            missing = sorted(set(object_records) - set(source_to_row))
            if missing:
                raise KeyError(
                    f"scene {scene_id} object {object_index} PICK_REMOVE sources missing "
                    f"from grasp_library: {missing[:8]}"
                )
            candidate_rows = np.asarray(
                [source_to_row[source] for source in object_records], dtype=np.int64
            )
            status_by_row = {
                source_to_row[source]: int(value[1]) for source, value in object_records.items()
            }
            pose_by_row = {
                source_to_row[source]: value[0] for source, value in object_records.items()
            }
            if training:
                positive_rows = np.asarray(
                    [row for row in candidate_rows if status_by_row[int(row)] == CandidateStatus.POSITIVE],
                    np.int64,
                )
                negative_rows = np.asarray(
                    [row for row in candidate_rows if status_by_row[int(row)] == CandidateStatus.NEGATIVE],
                    np.int64,
                )
                eligible = np.concatenate((
                    _se3_diverse_rows(library, positive_rows, self.global_sampling[0]),
                    _se3_diverse_rows(library, negative_rows, self.global_sampling[1]),
                ))
            else:
                eligible = _object_frame_nms_rows(library, candidate_rows)

            rotation_world_object = quaternion_xyzw_to_matrix_numpy(
                observation.object_pose[object_index, 3:]
            ).astype(np.float32)
            visible_points = observation.xyz[observation.instance_id == object_index]
            for row in eligible:
                row = int(row)
                local_contacts = library.contact_points_object[row]
                # Windows workers can load NumPy with a different OpenMP runtime
                # than PyTorch; explicit 2x3 arithmetic avoids a tiny BLAS call.
                world_contacts = np.empty((2, 3), np.float32)
                for side in range(2):
                    for axis in range(3):
                        world_contacts[side, axis] = (
                            sum(
                                float(rotation_world_object[axis, inner])
                                * float(local_contacts[side, inner])
                                for inner in range(3)
                            )
                            + float(observation.object_pose[object_index, axis])
                        )
                if len(visible_points):
                    distance_sq = np.sum(
                        (world_contacts[:, None] - visible_points[None]) ** 2, axis=-1
                    )
                    side_distance = np.min(distance_sq, axis=-1)
                    side = int(np.argmin(side_distance))
                    anchor = world_contacts[side]
                    anchor_distance = float(np.sqrt(side_distance[side]))
                else:
                    anchor = world_contacts[0]
                    anchor_distance = float("inf")
                pose = np.asarray(pose_by_row[row], np.float32)
                pose_rotation = quaternion_xyzw_to_matrix_numpy(pose[3:]).astype(np.float32)
                object_indices.append(object_index)
                sources.append(int(library.source_index[row]))
                contacts.append(anchor)
                anchor_distances.append(anchor_distance)
                poses.append(pose)
                approaches.append(pose_rotation[:, 2])
                widths.append(float(library.contact_span_m[row]))
                scene_labels.append(status_by_row[row])

        count = len(object_indices)
        return GlobalGraspLabels(
            object_index=np.asarray(object_indices, np.int64),
            source_grasp_index=np.asarray(sources, np.int64),
            contact_point_world=np.asarray(contacts, np.float32).reshape(count, 3),
            grasp_pose_world=np.asarray(poses, np.float32).reshape(count, 7),
            approach_direction_world=np.asarray(approaches, np.float32).reshape(count, 3),
            width_m=np.asarray(widths, np.float32),
            intrinsic_stable=np.ones(count, bool),
            scene_executable=np.asarray(scene_labels, np.int8),
            anchor_visible_distance_m=np.asarray(anchor_distances, np.float32),
            valid_mask=np.ones(count, bool),
            conversion_version="task_grasp_library+action_hdf5_v1",
            label_set_complete=False,
        )

    def load_state_labels(self, scene_id: int, state_id: int) -> StateLabels:
        with h5py.File(self._h5_path(scene_id), "r", swmr=True) as handle:
            scene = self._scene_group(handle, scene_id)
            states = scene["states"]
            occlusion = _ragged(states["occlusion_blockers"][:], states["occlusion_blocker_offsets"][:], state_id)
            task_occ = _ragged(
                states["task_occlusion_blockers"][:], states["task_occlusion_blocker_offsets"][:], state_id
            )
            task_index = int(states["task_index"][state_id])
            target_object = int(scene["catalog/task_object_index"][task_index])
            raw_relation = states["relation_graph"][state_id].astype(np.float32)
            object_count = raw_relation.shape[0]
            raw_index = {name: self.raw_relation_names.index(name) for name in self.raw_relation_names}
            approach = np.flatnonzero(
                raw_relation[:, target_object, raw_index["block_path"]] > 0.5
            ).astype(np.int64)
            task_block_graph = np.zeros((object_count, 3), dtype=np.float32)
            task_block_graph[np.asarray(task_occ, dtype=np.int64), 0] = 1.0
            task_block_graph[np.asarray(occlusion, dtype=np.int64), 1] = 1.0
            task_block_graph[approach, 2] = 1.0
            direct_mask = task_block_graph.any(axis=1)
            dependency_mask = direct_mask.copy()
            support_channel = raw_index["support"]
            contact_channel = raw_index["contact"]
            object_pose = states["object_pose"][state_id]
            frontier = list(np.flatnonzero(direct_mask))
            while frontier:
                supported_object = int(frontier.pop())
                above = (raw_relation[supported_object, :, support_channel] > 0.5) | (
                    (raw_relation[supported_object, :, contact_channel] > 0.5)
                    & (object_pose[:, 2] > object_pose[supported_object, 2] + 0.005)
                )
                for prerequisite in np.flatnonzero(above & ~dependency_mask):
                    dependency_mask[int(prerequisite)] = True
                    frontier.append(int(prerequisite))
            indirect_mask = dependency_mask & ~direct_mask
            actionable_mask = dependency_mask.copy()
            for dependent in np.flatnonzero(dependency_mask):
                has_active_prerequisite = np.any(
                    dependency_mask
                    & (
                        (raw_relation[int(dependent), :, support_channel] > 0.5)
                        | (
                            (raw_relation[int(dependent), :, contact_channel] > 0.5)
                            & (object_pose[:, 2] > object_pose[int(dependent), 2] + 0.005)
                        )
                    )
                )
                if has_active_prerequisite:
                    actionable_mask[int(dependent)] = False
            blockers = np.flatnonzero(dependency_mask).astype(np.int64)
            relation = np.stack(
                (
                    raw_relation[..., raw_index["near"]],
                    raw_relation[..., raw_index["contact"]],
                    raw_relation[..., raw_index["support"]],
                    raw_relation[..., raw_index["support"]].T,
                    raw_relation[..., raw_index["occlude"]],
                ),
                axis=-1,
            ).astype(np.float32)
            dependencies = scene["dependencies"]
            prerequisite_order = _ragged(
                dependencies["task_prerequisite_object_order"][:],
                dependencies["task_prerequisite_object_order_offsets"][:],
                task_index,
            ).astype(np.int64)
            sequence_task = scene["sequences/task_index"][:]
            task_sequences = np.flatnonzero(sequence_task == task_index)
            topology = dependencies["sequence_topology_valid"][:]
            sequence_topology_valid = bool(
                len(task_sequences) and np.all(topology[task_sequences])
            )
            return StateLabels(
                relation_graph=relation,
                task_block_graph=task_block_graph,
                blockers=blockers,
                task_pressed=bool(states["task_pressed"][state_id]),
                task_region_pressed=bool(states["task_region_pressed"][state_id]),
                verified_positive_grasp_count=int(states["verified_positive_grasp_count"][state_id]),
                required_grasp_count=int(states["required_grasp_count"][state_id]),
                direct_goal_valid=bool(states["direct_goal_valid"][state_id]),
                terminal_goal_valid=bool(states["terminal_goal_valid"][state_id]),
                potential_components=states["potential_components"][state_id].astype(np.float32),
                object_visible_pixels=states["object_visible_pixels"][state_id].astype(np.int64),
                sequence_depth=int(states["sequence_depth"][state_id]),
                target_visible_ratio=float(states["target_visible_ratio"][state_id]),
                relation_names=self.relation_names,
                direct_blocker_mask=direct_mask,
                indirect_blocker_mask=indirect_mask,
                actionable_blocker_mask=actionable_mask,
                prerequisite_object_order=prerequisite_order,
                sequence_topology_valid=sequence_topology_valid,
            )

    def load_action_group(self, scene_id: int, group_index: int) -> ActionCandidateGroup:
        with h5py.File(self._h5_path(scene_id), "r", swmr=True) as handle:
            scene = self._scene_group(handle, scene_id)
            groups = scene["action_state_groups"]
            action_ids = _ragged(groups["action_ids"][:], groups["action_offsets"][:], group_index).astype(np.int64)
            actions = scene["actions"]
            action_type = actions["action_type"][action_ids].astype(np.int64)
            payload_index = actions["payload_index"][action_ids].astype(np.int64)
            n = len(action_ids)
            acted_object = np.full(n, -1, np.int64)
            parameters = {
                "push_contact_world": np.full((n, 3), np.nan, np.float32),
                "push_direction_world": np.full((n, 3), np.nan, np.float32),
                "push_distance_m": np.full(n, np.nan, np.float32),
                "removal_grasp_pose_world": np.full((n, 7), np.nan, np.float32),
                "removal_grasp_source_index": np.full(n, -1, np.int64),
                "removal_destination_world": np.full((n, 3), np.nan, np.float32),
                "task_grasp_source_index": np.full(n, -1, np.int64),
                "task_grasp_pose_world": np.full((n, 7), np.nan, np.float32),
                "grasp_width_m": np.full(n, np.nan, np.float32),
                "grasp_approach_world": np.full((n, 3), np.nan, np.float32),
                "grasp_contact_points_world": np.full((n, 2, 3), np.nan, np.float32),
                "grasp_rotation_matrix_world": np.full((n, 3, 3), np.nan, np.float32),
                "grasp_depth_m": np.full(n, np.nan, np.float32),
                "grasp_confidence": np.full(n, np.nan, np.float32),
                "risk_unstable": np.zeros(n, np.float32),
                "risk_out_of_workspace": np.zeros(n, np.float32),
                "risk_other_invalid": np.zeros(n, np.float32),
                "stable": np.full(n, np.nan, np.float32),
            }
            for head in ("stability", "task_compatibility", "collision", "clearance", "approach", "overall"):
                parameters[f"verifier_{head}_target"] = np.full(n, np.nan, np.float32)
                parameters[f"verifier_{head}_valid"] = np.zeros(n, bool)
            parameters["proposal_geometry_valid"] = np.ones(n, bool)
            object_match_files = self._object_match_files(scene_id)
            state_pose_cache: dict[int, np.ndarray] = {}

            def world_grasp(
                object_index: int, state_id: int, source_index: int
            ) -> tuple[np.ndarray, float, np.ndarray, float, float, np.ndarray]:
                library = self.grasp_registry.load(object_match_files[object_index])
                library_row = int(library.rows_for_source(np.asarray([source_index]))[0])
                if state_id not in state_pose_cache:
                    state_pose_cache[state_id] = scene["states/object_pose"][state_id]
                object_pose = state_pose_cache[state_id][object_index]
                pose = compose_pose_with_transform(object_pose, library.transform_object[library_row])
                rotation = quaternion_xyzw_to_matrix_numpy(pose[3:]).astype(np.float32)
                object_rotation = quaternion_xyzw_to_matrix_numpy(object_pose[3:])
                local_contacts = library.contact_points_object[library_row]
                world_contacts = np.empty((2, 3), np.float32)
                for side in range(2):
                    for axis in range(3):
                        world_contacts[side, axis] = (
                            sum(float(object_rotation[axis, inner] * local_contacts[side, inner]) for inner in range(3))
                            + float(object_pose[axis])
                        )
                return (
                    pose,
                    float(library.contact_span_m[library_row]),
                    rotation,
                    float(library.depth_m[library_row]),
                    float(library.confidence[library_row]),
                    world_contacts,
                )
            for row, (kind, payload) in enumerate(zip(action_type, payload_index, strict=True)):
                if kind == ActionType.PUSH:
                    p = actions["push"]
                    acted_object[row] = p["acted_object"][payload]
                    parameters["push_contact_world"][row] = p["contact_point_world"][payload]
                    parameters["push_direction_world"][row] = p["direction_world"][payload]
                    parameters["push_distance_m"][row] = p["push_distance"][payload]
                elif kind == ActionType.PICK_REMOVE:
                    p = actions["pick_remove"]
                    acted_object[row] = p["acted_object"][payload]
                    parameters["removal_grasp_pose_world"][row] = p["removal_grasp_pose_world"][payload]
                    parameters["removal_grasp_source_index"][row] = p["removal_grasp_source_index"][payload]
                    parameters["removal_destination_world"][row] = p["removal_destination_world"][payload]
                    _, width, rotation, depth_m, confidence, contacts_world = world_grasp(
                        int(acted_object[row]), int(actions["from_state"][action_ids[row]]),
                        int(parameters["removal_grasp_source_index"][row])
                    )
                    parameters["grasp_width_m"][row] = width
                    parameters["grasp_approach_world"][row] = rotation[:, 2]
                    parameters["grasp_rotation_matrix_world"][row] = rotation
                    parameters["grasp_depth_m"][row] = depth_m
                    parameters["grasp_confidence"][row] = confidence
                    parameters["grasp_contact_points_world"][row] = contacts_world
                elif kind == ActionType.TASK_GRASP:
                    p = actions["task_grasp"]
                    acted_object[row] = p["target_object"][payload]
                    parameters["task_grasp_source_index"][row] = p["grasp_source_index"][payload]
                    pose, width, rotation, depth_m, confidence, contacts_world = world_grasp(
                        int(acted_object[row]), int(actions["from_state"][action_ids[row]]),
                        int(parameters["task_grasp_source_index"][row])
                    )
                    parameters["task_grasp_pose_world"][row] = pose
                    parameters["grasp_width_m"][row] = width
                    parameters["grasp_approach_world"][row] = rotation[:, 2]
                    parameters["grasp_rotation_matrix_world"][row] = rotation
                    parameters["grasp_depth_m"][row] = depth_m
                    parameters["grasp_confidence"][row] = confidence
                    parameters["grasp_contact_points_world"][row] = contacts_world
                else:
                    raise ValueError(f"Unknown action type {kind}")
            push_rows = action_type == ActionType.PUSH
            if np.any(np.abs(parameters["push_distance_m"][push_rows] - PUSH_DISTANCE_M) > 1e-6):
                raise ValueError("Dataset violates fixed 0.15 m PUSH primitive")
            executed = actions["executed"][action_ids].astype(bool)
            parameters["stable"][executed] = actions["stable"][action_ids][executed].astype(np.float32)
            outcome = actions["outcome_code"][action_ids].astype(np.int64)
            parameters["risk_unstable"] = (executed & ~actions["stable"][action_ids].astype(bool)).astype(np.float32)
            parameters["risk_out_of_workspace"] = (
                executed & actions["non_target_out_of_workspace"][action_ids].astype(bool)
            ).astype(np.float32)
            known_failure = executed & ~actions["success"][action_ids].astype(bool)
            parameters["risk_other_invalid"] = (
                known_failure
                & ~(parameters["risk_unstable"].astype(bool) | parameters["risk_out_of_workspace"].astype(bool))
            ).astype(np.float32)
            status = np.where(executed, CandidateStatus.NEGATIVE, CandidateStatus.UNKNOWN_UNTESTED).astype(np.int8)
            positive = (
                actions["success"][action_ids].astype(bool)
                | actions["potential_improved"][action_ids].astype(bool)
                | (outcome == OutcomeCode.TERMINAL_POSITIVE)
            )
            status[positive & executed] = CandidateStatus.POSITIVE
            grasp_rows = (action_type == int(ActionType.TASK_GRASP)) | (
                action_type == int(ActionType.PICK_REMOVE)
            )
            parameters["verifier_stability_target"][grasp_rows & executed] = parameters["stable"][grasp_rows & executed]
            parameters["verifier_stability_valid"][grasp_rows & executed] = True
            task_rows = action_type == int(ActionType.TASK_GRASP)
            parameters["verifier_task_compatibility_target"][task_rows] = 1.0
            parameters["verifier_task_compatibility_valid"][task_rows] = True
            parameters["verifier_overall_target"][grasp_rows & executed] = positive[grasp_rows & executed]
            parameters["verifier_overall_valid"][grasp_rows & executed] = True
            verified_grasp = grasp_rows & executed & positive
            parameters["verifier_collision_target"][verified_grasp] = 0.0
            parameters["verifier_collision_valid"][verified_grasp] = True
            parameters["verifier_clearance_target"][verified_grasp] = 1.0
            parameters["verifier_clearance_valid"][verified_grasp] = True
            candidate_group = ActionCandidateGroup(
                candidate_action_ids=action_ids,
                action_type=action_type,
                acted_object=acted_object,
                valid_mask=np.ones(n, dtype=bool),
                evaluation_status=status,
                outcome_code=outcome,
                from_state=actions["from_state"][action_ids].astype(np.int64),
                to_state=actions["to_state"][action_ids].astype(np.int64),
                after_state_valid=actions["after_state_valid"][action_ids].astype(bool),
                after_pose_valid=actions["after_pose_valid"][action_ids].astype(bool),
                potential_after_valid=actions["potential_after_valid"][action_ids].astype(bool),
                acted_object_motion_valid=actions["acted_object_motion_valid"][action_ids].astype(bool),
                target_motion_valid=actions["target_motion_valid"][action_ids].astype(bool),
                potential_delta=actions["potential_delta"][action_ids].astype(np.float32),
                success_mask=positive & executed,
                action_parameters=parameters,
                label_set_complete=(
                    bool(groups["label_set_complete"][group_index])
                    if "label_set_complete" in groups else False
                ),
            )
            from_state_id = int(groups["from_state"][group_index])
            task_index = int(groups["task_index"][group_index])
            if int(scene["states/sequence_depth"][from_state_id]) == 0:
                candidate_group = self._append_initial_verifier_candidates(
                    scene_id, scene, candidate_group, task_index, from_state_id,
                    object_match_files, world_grasp,
                )
        candidate_group.validate()
        return candidate_group

    def _append_initial_verifier_candidates(
        self, scene_id: int, scene: h5py.Group, base: ActionCandidateGroup,
        task_index: int, state_id: int, object_match_files: tuple[str, ...], world_grasp,
    ) -> ActionCandidateGroup:
        """Append evaluated region/collision/approach negatives from Steps 1--6."""

        target_object = int(scene["catalog/task_object_index"][task_index])
        task_label = int(scene["catalog/task_label"][task_index])
        library = self.grasp_registry.load(object_match_files[target_object])
        step_path = self.step_root / "scene_labels" / f"scene_{scene_id:04d}_labels.npz"
        with np.load(step_path, allow_pickle=False) as labels:
            counts = labels["object_candidate_count"].astype(np.int64)
            offsets = labels["candidate_byte_offsets"].astype(np.int64)
            count = int(counts[target_object])
            start, stop = int(offsets[target_object]), int(offsets[target_object + 1])

            def unpack(name: str) -> np.ndarray:
                return np.unpackbits(labels[name][start:stop], bitorder="little")[:count].astype(bool)

            collision_free = unpack("candidate_coarse_collision_free_packed")
            reachable = unpack("candidate_coarse_fr5_reachable_packed")
        if count != len(library.source_index):
            raise ValueError(
                f"Scene {scene_id} object {target_object}: packed candidate count {count} "
                f"!= grasp library count {len(library.source_index)}"
            )
        correct_region = library.task_label == task_label
        pools = (
            np.flatnonzero(~correct_region & collision_free & reachable),
            np.flatnonzero(correct_region & ~collision_free),
            np.flatnonzero(correct_region & collision_free & ~reachable),
        )
        requested = self.verifier_sampling[:3]
        rng = np.random.default_rng(
            self.verifier_sampling[3] + scene_id * 1009 + task_index * 9176
        )
        selected, kinds = [], []
        for kind, (pool, amount) in enumerate(zip(pools, requested, strict=True)):
            if len(pool):
                choice = rng.choice(pool, min(int(amount), len(pool)), replace=False)
                selected.extend(choice.tolist())
                kinds.extend([kind] * len(choice))
        if not selected:
            return base
        selected_array = np.asarray(selected, np.int64)
        kinds_array = np.asarray(kinds, np.int8)  # 0 wrong-region, 1 collision, 2 approach
        count_new = len(selected_array)
        extension: dict[str, np.ndarray] = {}
        for key, original in base.action_parameters.items():
            shape = (count_new,) + original.shape[1:]
            if original.dtype == bool:
                extension[key] = np.zeros(shape, bool)
            elif np.issubdtype(original.dtype, np.integer):
                extension[key] = np.full(shape, -1, original.dtype)
            else:
                extension[key] = np.full(shape, np.nan, original.dtype)
        for row, library_row in enumerate(selected_array):
            source_index = int(library.source_index[library_row])
            pose, width, rotation, depth_m, confidence, contacts_world = world_grasp(
                target_object, state_id, source_index
            )
            extension["task_grasp_source_index"][row] = source_index
            extension["task_grasp_pose_world"][row] = pose
            extension["grasp_width_m"][row] = width
            extension["grasp_approach_world"][row] = rotation[:, 2]
            extension["grasp_rotation_matrix_world"][row] = rotation
            extension["grasp_depth_m"][row] = depth_m
            extension["grasp_confidence"][row] = confidence
            extension["grasp_contact_points_world"][row] = contacts_world
            is_wrong, is_collision, is_approach = kinds_array[row] == np.arange(3)
            extension["proposal_geometry_valid"][row] = not is_wrong
            targets = {
                "task_compatibility": float(not is_wrong),
                "collision": float(is_collision),
                "clearance": float(not is_collision),
                "approach": float(not is_approach),
                "overall": 0.0,
            }
            for head, target in targets.items():
                extension[f"verifier_{head}_target"][row] = target
                extension[f"verifier_{head}_valid"][row] = True
        parameters = {
            key: np.concatenate((value, extension[key]), axis=0)
            for key, value in base.action_parameters.items()
        }
        def append(name: str, values: np.ndarray) -> np.ndarray:
            return np.concatenate((getattr(base, name), values), axis=0)

        return ActionCandidateGroup(
            candidate_action_ids=append(
                "candidate_action_ids", -np.arange(1, count_new + 1, dtype=np.int64)
            ),
            action_type=append(
                "action_type", np.full(count_new, int(ActionType.TASK_GRASP), np.int64)
            ),
            acted_object=append(
                "acted_object", np.full(count_new, target_object, np.int64)
            ),
            valid_mask=append("valid_mask", np.ones(count_new, bool)),
            evaluation_status=append(
                "evaluation_status", np.full(count_new, int(CandidateStatus.NEGATIVE), np.int8)
            ),
            outcome_code=append(
                "outcome_code", np.full(count_new, int(OutcomeCode.OTHER_INVALID), np.int64)
            ),
            from_state=append("from_state", np.full(count_new, state_id, np.int64)),
            to_state=append("to_state", np.full(count_new, -1, np.int64)),
            after_state_valid=append("after_state_valid", np.zeros(count_new, bool)),
            after_pose_valid=append("after_pose_valid", np.zeros(count_new, bool)),
            potential_after_valid=append("potential_after_valid", np.zeros(count_new, bool)),
            acted_object_motion_valid=append("acted_object_motion_valid", np.zeros(count_new, bool)),
            target_motion_valid=append("target_motion_valid", np.zeros(count_new, bool)),
            potential_delta=append(
                "potential_delta",
                np.full((count_new,) + base.potential_delta.shape[1:], np.nan, np.float32),
            ),
            success_mask=append("success_mask", np.zeros(count_new, bool)),
            action_parameters=parameters,
            label_set_complete=base.label_set_complete,
        )

    def load_sequences(self, scene_id: int, task_index: int | None = None) -> tuple[SequenceLabels, ...]:
        result = []
        with h5py.File(self._h5_path(scene_id), "r", swmr=True) as handle:
            scene = self._scene_group(handle, scene_id)
            sequences = scene["sequences"]
            topology = scene["dependencies/sequence_topology_valid"][:]
            tasks = sequences["task_index"][:]
            for index, task in enumerate(tasks):
                if task_index is not None and int(task) != task_index:
                    continue
                result.append(
                    SequenceLabels(
                        state_ids=_ragged(sequences["state_ids"][:], sequences["state_offsets"][:], index).astype(np.int64),
                        transition_ids=_ragged(
                            sequences["transition_ids"][:], sequences["transition_offsets"][:], index
                        ).astype(np.int64),
                        policy_action_ids=_ragged(
                            sequences["policy_action_ids"][:], sequences["policy_action_offsets"][:], index
                        ).astype(np.int64),
                        terminal_action_ids=_ragged(
                            sequences["terminal_action_ids"][:], sequences["terminal_action_offsets"][:], index
                        ).astype(np.int64),
                        final_grasp_source_indices=_ragged(
                            sequences["final_grasp_source_indices"][:], sequences["final_grasp_offsets"][:], index
                        ).astype(np.int64),
                        sequence_topology_valid=bool(topology[index]),
                        task_index=int(task),
                    )
                )
        return tuple(result)
