"""Adapter for TaskOrientedClutterSceneDataset action-label format v2."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np

from tcd_prg.constants import ActionType, CandidateStatus, OutcomeCode, PUSH_DISTANCE_M
from tcd_prg.geometry.numpy_se3 import compose_pose_with_transform, quaternion_xyzw_to_matrix_numpy
from tcd_prg.observation.base import ObservationProvider, ObservationRequest
from tcd_prg.observation.saved import SavedObservationProvider

from .base import DatasetAdapter
from .capabilities import DatasetCapabilities
from .types import (
    ActionCandidateGroup,
    CameraParameters,
    SceneObservation,
    SequenceLabels,
    StateLabels,
)
from .registries import FunctionalRegionRegistry, GraspLibraryRegistry


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values)


def _ragged(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
    return values[int(offsets[index]) : int(offsets[index + 1])]


class TaskOrientedClutterAdapter(DatasetAdapter):
    """Map the current dataset to the model-independent contract.

    The adapter never opens files in ``.work`` and snapshots only atomically
    published ``scene_*.h5`` files present at construction time.
    """

    capabilities = DatasetCapabilities(
        has_instance_masks=True,
        has_task_regions=True,
        has_task_grasps=True,
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
        functional_region_root: str | Path | None = None,
        verifier_wrong_region_negatives: int = 8,
        verifier_collision_negatives: int = 8,
        verifier_approach_negatives: int = 8,
        sampling_seed: int = 2026,
        scene_subdir: str = "task_clutter_scenes_20_categories",
        step_labels_subdir: str = "task_training_labels_steps1_6_v1",
        action_labels_subdir: str = "task_positive_multistep_sequences",
    ) -> None:
        self.root = Path(root)
        self.scene_root = self.root / scene_subdir
        self.step_root = self.root / step_labels_subdir
        self.action_root = self.root / action_labels_subdir
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
        self.verifier_sampling = (
            int(verifier_wrong_region_negatives), int(verifier_collision_negatives),
            int(verifier_approach_negatives), int(sampling_seed),
        )
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
        for scene_id in self._scene_ids:
            if split is not None:
                with np.load(self.scene_root / f"scene_{scene_id:04d}" / "scene.npz", allow_pickle=False) as data:
                    if str(data["split"].item()) != split:
                        continue
            with h5py.File(self._h5_path(scene_id), "r", swmr=True) as handle:
                group = self._scene_group(handle, scene_id)["action_state_groups"]
                states = group["from_state"][:]
                tasks = group["task_index"][:]
            yield from ((scene_id, int(s), int(t), i) for i, (s, t) in enumerate(zip(states, tasks, strict=True)))

    def action_group_strata(
        self, units: Iterable[tuple[int, int, int, int]] | None = None
    ) -> dict[tuple[int, int, int, int], str]:
        """Classify groups without loading observations or grasp libraries."""

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
            request = ObservationRequest(
                scene_id=scene_id,
                state_id=state_id,
                object_pose=object_pose,
                object_active=active,
                object_present=np.ones(len(object_pose), dtype=bool),
                object_asset_ids=h5_names,
                object_model_ids=model_ids,
                object_scales=object_scales,
                render_seed=int(states["observation_render_seed"][state_id]),
                camera_profile="mecheye_pro_s_three_view",
                point_count=self.point_count,
                renderer_version=self.renderer_version,
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
                "push_approach_mode": np.full(n, -1, np.int64),
                "push_distance_m": np.full(n, np.nan, np.float32),
                "removal_grasp_pose_world": np.full((n, 7), np.nan, np.float32),
                "removal_grasp_source_index": np.full(n, -1, np.int64),
                "removal_destination_world": np.full((n, 3), np.nan, np.float32),
                "task_grasp_source_index": np.full(n, -1, np.int64),
                "task_grasp_pose_world": np.full((n, 7), np.nan, np.float32),
                "grasp_width_m": np.full(n, np.nan, np.float32),
                "grasp_approach_world": np.full((n, 3), np.nan, np.float32),
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
            step_path = self.step_root / "scene_labels" / f"scene_{scene_id:04d}_labels.npz"
            with np.load(step_path, allow_pickle=False) as step_labels:
                object_match_files = tuple(str(x) for x in step_labels["object_match_file"])
            state_pose_cache: dict[int, np.ndarray] = {}

            def world_grasp(
                object_index: int, state_id: int, source_index: int
            ) -> tuple[np.ndarray, float, np.ndarray, float, float]:
                library = self.grasp_registry.load(object_match_files[object_index])
                library_row = int(library.rows_for_source(np.asarray([source_index]))[0])
                if state_id not in state_pose_cache:
                    state_pose_cache[state_id] = scene["states/object_pose"][state_id]
                object_pose = state_pose_cache[state_id][object_index]
                pose = compose_pose_with_transform(object_pose, library.transform_object[library_row])
                rotation = quaternion_xyzw_to_matrix_numpy(pose[3:]).astype(np.float32)
                return (
                    pose,
                    float(library.contact_span_m[library_row]),
                    rotation,
                    float(library.depth_m[library_row]),
                    float(library.confidence[library_row]),
                )
            for row, (kind, payload) in enumerate(zip(action_type, payload_index, strict=True)):
                if kind == ActionType.PUSH:
                    p = actions["push"]
                    acted_object[row] = p["acted_object"][payload]
                    parameters["push_contact_world"][row] = p["contact_point_world"][payload]
                    parameters["push_direction_world"][row] = p["direction_world"][payload]
                    parameters["push_approach_mode"][row] = p["approach_mode"][payload]
                    parameters["push_distance_m"][row] = p["push_distance"][payload]
                elif kind == ActionType.PICK_REMOVE:
                    p = actions["pick_remove"]
                    acted_object[row] = p["acted_object"][payload]
                    parameters["removal_grasp_pose_world"][row] = p["removal_grasp_pose_world"][payload]
                    parameters["removal_grasp_source_index"][row] = p["removal_grasp_source_index"][payload]
                    parameters["removal_destination_world"][row] = p["removal_destination_world"][payload]
                    _, width, rotation, depth_m, confidence = world_grasp(
                        int(acted_object[row]), int(actions["from_state"][action_ids[row]]),
                        int(parameters["removal_grasp_source_index"][row])
                    )
                    parameters["grasp_width_m"][row] = width
                    parameters["grasp_approach_world"][row] = rotation[:, 2]
                    parameters["grasp_rotation_matrix_world"][row] = rotation
                    parameters["grasp_depth_m"][row] = depth_m
                    parameters["grasp_confidence"][row] = confidence
                elif kind == ActionType.TASK_GRASP:
                    p = actions["task_grasp"]
                    acted_object[row] = p["target_object"][payload]
                    parameters["task_grasp_source_index"][row] = p["grasp_source_index"][payload]
                    pose, width, rotation, depth_m, confidence = world_grasp(
                        int(acted_object[row]), int(actions["from_state"][action_ids[row]]),
                        int(parameters["task_grasp_source_index"][row])
                    )
                    parameters["task_grasp_pose_world"][row] = pose
                    parameters["grasp_width_m"][row] = width
                    parameters["grasp_approach_world"][row] = rotation[:, 2]
                    parameters["grasp_rotation_matrix_world"][row] = rotation
                    parameters["grasp_depth_m"][row] = depth_m
                    parameters["grasp_confidence"][row] = confidence
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
            pose, width, rotation, depth_m, confidence = world_grasp(
                target_object, state_id, source_index
            )
            extension["task_grasp_source_index"][row] = source_index
            extension["task_grasp_pose_world"][row] = pose
            extension["grasp_width_m"][row] = width
            extension["grasp_approach_world"][row] = rotation[:, 2]
            extension["grasp_rotation_matrix_world"][row] = rotation
            extension["grasp_depth_m"][row] = depth_m
            extension["grasp_confidence"][row] = confidence
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
        n = len(base.action_type)

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
