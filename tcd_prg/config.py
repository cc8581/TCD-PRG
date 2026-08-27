"""Typed configuration and invariant validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import MAX_PREPARATION_ACTIONS, PUSH_DISTANCE_M
from .paths import project_path


@dataclass(slots=True)
class DatasetConfig:
    # 数据路径在公共配置中使用仓库相对值，正式入口再用本机 local_paths.yaml 覆盖。
    root: str = ""
    acronym_root: str = ""
    functional_region_root: str = ""
    fr5_ag_urdf: str = "assets/robots/FR5_AG-160-95/urdf/fr5_ag160_95.urdf"
    adapter: str = "task_oriented_clutter"
    scene_subdir: str = "task_clutter_scenes_20_categories"
    step_labels_subdir: str = "task_training_labels"
    action_labels_subdir: str = "task_positive_multistep_sequences"
    scene_points: int = 0
    # target_points 仅供独立资源分析器使用，不限制正式 PTv3 输入。
    target_points: int = 4096
    stageb_binary_root: str = "runtime/stageb_binary"


@dataclass(slots=True)
class ObservationConfig:
    provider: str = "cached"
    allow_render_on_miss: bool = True
    render_width: int = 320
    render_height: int = 200
    camera_profile: str = "mecheye_pro_s_three_view"
    renderer_version: str = "tcd_prg_pybullet_v3_sensor_only_instance_query"
    pybullet_python: str = "python"
    worker_script: str = "scripts/render_observation_worker_py38.py"
    runtime_mesh_root: str = "runtime/cache/meshes"
    certification_worker_script: str = "scripts/certify_actions_worker_py38.py"
    render_temporary_root: str = "runtime/tmp/render_requests"
    certification_temporary_root: str = "runtime/tmp/certification"


LEGACY_READ_ONLY_RENDERER = "tcd_prg_pybullet_v2_variable_grid"


@dataclass(slots=True)
class CacheConfig:
    directory: str = "runtime/cache/observations"
    index_directory: str = "runtime/cache/dataset_indexes"
    max_gb: float = 5.0
    min_free_gb: float = 20.0
    eviction: str = "lru"
    prefetch_workers: int = 4


@dataclass(slots=True)
class BackboneConfig:
    # grid_size_m 控制 PTv3 体素分辨率；patch_size 控制序列化注意力的最大 patch 长度。
    backend: str = "point_transformer_v3"
    source_root: str = "third_party/PointTransformerV3"
    pretrained_checkpoint: str | None = None
    pretrained_format: str = "auto"
    pretrained_auto_download: bool = False
    pretrained_url: str = ""
    pretrained_sha256: str = ""
    pretrained_cache_dir: str = "runtime/cache/pretrained"
    pretrained_required: bool = False
    pretrained_min_parameter_fraction: float = 0.35
    pretrained_freeze_steps: int = 0
    freeze: bool = False
    grid_size_m: float = 0.005
    enable_flash_attention: bool = False
    patch_size: int = 128
    attention_points: int = 1_024


@dataclass(slots=True)
class GraspNetConfig:
    # Official graspnet-baseline is installed from the pinned third_party.lock.yaml entry.
    source_root: str = ".deps/graspnet-baseline"
    checkpoint: str = ".deps/checkpoints/graspnet-rs.tar"
    freeze: bool = True
    scene_input_points: int = 20_000
    target_input_points: int = 20_000
    global_proposals: int = 128
    target_proposals: int = 128
    target_selection_mode: str = "quality_diverse"
    global_selection_mode: str = "quality_topk"
    diversity_quality_fraction: float = 0.5
    diversity_translation_m: float = 0.010
    diversity_rotation_deg: float = 12.0
    diversity_pool_factor: int = 4
    camera_view_index: int = 2
    target_crop_probability: float = 0.5
    target_min_crop_points: int = 16
    camera_transfer_max_distance_m: float = 0.010
    num_view: int = 300
    num_angle: int = 12
    num_depth: int = 4
    cylinder_radius: float = 0.05
    hmin: float = -0.02
    hmax_list: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04)


@dataclass(slots=True)
class RegionHeadConfig:
    enabled: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_weight: float = 1.0
    visibility_threshold: float = 0.5


@dataclass(slots=True)
class PlannerConfig:
    max_preparation_actions: int = MAX_PREPARATION_ACTIONS


@dataclass(slots=True)
class BaselineConfig:
    type: str = "tcd_prg"
    gapg_root: str = "third_party/GAPG"
    grasp_checkpoint: str = ".deps/checkpoints/gapg/grasp_model.pt"
    push_checkpoint: str = ".deps/checkpoints/gapg/push_model.pt"
    graspnet_checkpoint: str = ".deps/checkpoints/graspnet-rs.tar"


@dataclass(slots=True)
class ModelConfig:
    # feature_dim 是所有共享点/物体 token 的主维度，修改时需同步检查各 Transformer 头数。
    feature_dim: int = 256
    task_dim: int = 128
    num_categories: int = 64
    num_task_regions: int = 64
    # Sensor-only object decomposition. Q is a capacity, not a GT object index.
    instance_queries: int = 32
    instance_decoder_layers: int = 6
    instance_decoder_heads: int = 8
    instance_objectness_threshold: float = 0.5
    instance_matching_points: int = 2048
    target_query_temperature: float = 0.25
    # V2: observable 3D target identity prompt. Region semantics never identify
    # which same-category physical instance is requested.
    target_prompt_radius_m: float = 0.030
    target_prompt_sigma_m: float = 0.012
    target_prompt_jitter_std_m: float = 0.003
    target_prompt_weight: float = 4.0
    target_category_weight: float = 1.0
    target_objectness_weight: float = 0.5
    target_center_weight: float = 0.5
    target_learned_weight: float = 0.20
    target_reid_weight: float = 0.75
    target_reid_center_weight: float = 0.35
    target_reid_max_center_distance_m: float = 0.15
    target_prompt_min_support: float = 0.20
    target_prompt_min_margin: float = 0.25
    target_same_category_loss_boost: float = 2.0
    num_direction_bins: int = 16
    # 每个抓取 query 直接预测平移、连续 SO(3)、夹爪宽度和条件化质量。
    contact_heatmap_sigma_m: float = 0.008
    # Shared train/validation association gate for contact -> visible point.
    push_contact_match_max_distance_m: float = 0.024
    max_grasp_width_m: float = 0.095
    min_grasp_width_m: float = 0.0
    candidate_topk: int = 64
    task_grasp_candidates: int = 64
    task_grasp_probability_threshold: float = 0.5
    pick_remove_probability_threshold: float = 0.5
    global_grasp_candidates: int = 64
    task_grasp_scene_points: int = 256
    task_grasp_gripper_points: int = 128
    task_grasp_gripper_geometry: str = "assets/robots/FR5_AG-160-95/ag16095_open_tcp_128.npz"
    stageb_label_gripper_geometry: str = "assets/robots/FR5_AG-160-95/ag16095_open_tcp_4096.npz"
    pick_remove_candidates: int = 16
    push_candidates: int = 16
    # push_candidates 是接触点预算；每个点再展开多个方向，最终总量受 max_push_candidates 限制。
    push_directions_per_contact: int = 2
    max_push_candidates: int = 32
    # 稀疏方向计算；训练时仅把已评价 GT contact 强制并入预测 top-k。
    push_direction_contact_topk: int = 32
    push_object_topk: int = 4
    push_utility_threshold: float = 0.0
    push_candidate_probability_threshold: float = 0.0
    push_utility_temperature: float = 1.0
    pick_remove_target_margin_m: float = 0.05
    push_direction_feature_dim: int = 64
    push_direction_transformer_layers: int = 1
    push_direction_transformer_heads: int = 4
    # 默认关闭以避免反向传播时重复执行 PTv3；显存不足时再显式开启。
    activation_checkpointing: bool = False
    grasp_nms_translation_m: float = 0.010
    grasp_nms_rotation_deg: float = 12.0
    grasp_nms_width_m: float = 0.005
    grasp_nms_approach_deg: float = 12.0
    global_grasp_input_mode: str = "scene_only"
    global_grasp_nms_translation_m: float = 0.01
    global_grasp_nms_rotation_deg: float = 15.0
    global_grasp_nms_width_m: float = 0.005
    global_grasp_nms_approach_deg: float = 15.0
    push_nms_contact_m: float = 0.015
    push_nms_direction_deg: float = 15.0
    push_utility_component_weights: tuple[float, ...] = (0.05, 1.0, -1.0, -0.25, 1.0)
    push_failure_penalties: tuple[float, ...] = (1.0, 2.0, 1.0)


@dataclass(slots=True)
class AblationConfig:
    use_task_region_condition: bool = True
    use_push_potential: bool = True


@dataclass(slots=True)
class TrainingConfig:
    stage: str = "joint"
    # max_optimizer_steps 统计真实参数更新次数，不包含 AMP 溢出后被跳过的 step。
    seed: int = 2026
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "float16"
    amp_initial_scale: float = 4096.0
    batch_size: int = 1
    # Task/state selection is primary.  Strata only guide which group to use
    # inside an already-selected state; they never replace that task/state.
    action_batch_coverage_strata: tuple[str, ...] = (
        "direct_grasp",
        "pick_remove",
        "push",
        "push_failure",
        "unresolved_or_unknown",
    )
    # Optional stage-level filter over the immutable action-group snapshot.
    allowed_action_strata: tuple[str, ...] = ()
    # Unique scene-state Global Grasp stream is added once per action batch.
    gradient_accumulation_steps: int = 1
    max_optimizer_steps: int = 100_000
    # validation_interval=0 仅用于没有验证集的启动阶段，正式实验不应关闭验证。
    validation_interval: int = 1_000
    # 联合目标的全局梯度范数远大于单任务 Transformer；20 仅截断实测尖峰。
    gradient_clip_norm: float = 20.0
    ema_decay: float | None = 0.999
    early_stopping_patience: int = 20
    push_coverage_penalty_weight: float = 1.0
    deterministic: bool = True
    num_workers: int = 4
    # Validation runs while the train/global persistent workers are still
    # alive.  Keep it synchronous by default to avoid a third worker pool and
    # multiprocessing queue copies exhausting Windows commit memory.
    validation_num_workers: int = 0
    pin_memory: bool = True
    max_train_groups: int | None = None
    # Randomly select this many complete scenes from the already-created
    # validation split.  The deterministic selection is persisted and reused.
    validation_scene_count: int | None = None
    validation_scene_seed: int = 2026
    max_validation_groups: int | None = None
    # Used only when max_validation_groups is finite.  The resulting exact
    # subset is persisted to validation_subset.json and reused on resume.
    validation_stratum_quota: dict[str, int] = field(
        default_factory=lambda: {
            "direct_grasp": 64,
            "pick_remove": 64,
            "push": 48,
            "push_failure": 48,
            "unresolved_or_unknown": 32,
        }
    )
    # Restrict the published scene snapshot before deterministic splitting.
    scene_start: int = 0
    scene_count: int | None = None
    # Deterministically select this fraction of published scenes before splitting.
    data_fraction: float = 1.0
    # Scene-level train/val or train/val/test weights; values are normalized.
    split_ratios: tuple[float, ...] = (8.0, 1.0, 1.0)
    frozen_modules: tuple[str, ...] = ()
    unfreeze_at_optimizer_step: int | None = None
    ddp_backend: str = "auto"
    ddp_find_unused_parameters: bool = True
    validation_family_weights: dict[str, float] = field(
        default_factory=lambda: {
            "instance": 0.1,
            "region": 0.1,
            "task_grasp": 0.25,
            "push_object": 0.1,
            "push_contact": 0.05,
            "push_direction": 0.05,
            "push_potential": 0.1,
        }
    )


@dataclass(slots=True)
class EvaluationConfig:
    max_preparation_actions: int = MAX_PREPARATION_ACTIONS
    horizons: tuple[int, ...] = (0, 1, 3, 5)
    bootstrap_samples: int = 1_000
    confidence: float = 0.95
    max_groups: int | None = None
    # 这些阈值属于评测协议；修改后产生的结果不得与旧实验直接混合。
    region_probability_threshold: float = 0.5
    ranking_topk: tuple[int, ...] = (1, 5, 10)
    calibration_bins: int = 15
    global_grasp_tracks: tuple[str, ...] = ("scene_only", "instance_assisted")
    # Internal convergence diagnostics only. Public Global Grasp comparison
    # continues to use the official graspnetAPI evaluator.
    task_translation_threshold_m: float = 0.01
    task_rotation_threshold_deg: float = 12.0
    task_width_threshold_m: float = 0.005
    global_translation_threshold_m: float = 0.01
    global_rotation_threshold_deg: float = 15.0
    global_width_threshold_m: float = 0.005
    # NMS thresholds are intentionally independent from GT matching thresholds.
    task_nms_translation_m: float = 0.01
    task_nms_rotation_deg: float = 12.0
    task_nms_width_m: float = 0.005
    global_nms_translation_m: float = 0.01
    global_nms_rotation_deg: float = 15.0
    global_nms_width_m: float = 0.005
    # Deprecated compatibility field; internal diagnostics always apply the
    # explicit NMS configuration above and standard GraspNet uses graspnetAPI.
    global_metrics_after_nms: bool = True


@dataclass(slots=True)
class PushConfig:
    distance_m: float = PUSH_DISTANCE_M
    direction_bins: int = 16


@dataclass(slots=True)
class LossConfig:
    # 实例真值仅用于该 loss family，不进入模型 forward。
    instance: float = 1.0
    region: float = 1.0
    task_grasp: float = 1.0
    push_object: float = 0.25
    push_contact: float = 0.25
    push_direction: float = 0.25
    push_potential: float = 0.25
    internal: dict[str, float] = field(
        default_factory=lambda: {
            "instance_objectness": 1.0,
            "instance_mask": 2.0,
            "instance_dice": 2.0,
            "instance_category": 1.0,
            "target_query": 1.0,
            "instance_auxiliary": 0.5,
            "region_focal": 1.0,
            "region_dice": 1.0,
            "region_visibility": 0.2,
            "task_grasp_bce": 1.0,
        }
    )

    def family_weights(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in (
                "instance",
                "region",
                "task_grasp",
                "push_object",
                "push_contact",
                "push_direction",
                "push_potential",
            )
        }


@dataclass(slots=True)
class SamplingConfig:
    # 配额只描述每组抽样数量，不能据此推断抓取标签集合已经完备。
    positive_grasps: int = 8
    wrong_region_grasps: int = 8
    collision_or_approach_negative_grasps: int = 8
    perturbed_negative_grasps: int = 4
    global_positive_grasps_per_object: int = 64
    global_negative_grasps_per_object: int = 32
    unit: str = "action_state_group"


@dataclass(slots=True)
class OptimizerConfig:
    # 官方点云骨干使用更小学习率，其余预测头使用 learning_rate。
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 2e-5
    weight_decay: float = 0.01


@dataclass(slots=True)
class SchedulerConfig:
    warmup_steps: int = 2_000


@dataclass(slots=True)
class LoggingConfig:
    backend: str = "tensorboard"
    # 终端只按间隔打印核心摘要；JSONL 和 TensorBoard 保留每个成功优化器步的完整指标。
    log_interval: int = 20
    # Validation progress is measured in validation DataLoader batches.
    validation_log_interval: int = 20
    # 模块级梯度范数只低频统计，不增加额外 backward。
    gradient_diagnostics_interval: int = 200


@dataclass(slots=True)
class TCDPRGConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    graspnet: GraspNetConfig = field(default_factory=GraspNetConfig)
    region_head: RegionHeadConfig = field(default_factory=RegionHeadConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    push: PushConfig = field(default_factory=PushConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    push_distance_m: float = PUSH_DISTANCE_M
    output_dir: str = "outputs/default"
    extra: dict[str, Any] = field(default_factory=dict)
    name: str = "tcd-prg"

    def validate(self) -> None:
        if self.training.stage not in {"perception", "grasp", "push", "joint"}:
            raise ValueError("training.stage must be one of perception, grasp, push, or joint")
        stage_families = {
            "perception": {"instance", "region"},
            "grasp": {"task_grasp"},
            "push": {"push_object", "push_contact", "push_direction", "push_potential"},
        }
        expected_families = stage_families.get(self.training.stage)
        if expected_families is not None:
            configured_families = {
                name
                for name in (
                    "instance",
                    "region",
                    "task_grasp",
                    "push_object",
                    "push_contact",
                    "push_direction",
                    "push_potential",
                )
                if float(getattr(self.losses, name)) > 0
            }
            if configured_families != expected_families:
                raise ValueError(
                    f"Stage {self.training.stage!r} requires exactly loss families "
                    f"{sorted(expected_families)}, got {sorted(configured_families)}"
                )
        if (
            self.observation.renderer_version == LEGACY_READ_ONLY_RENDERER
            and self.observation.allow_render_on_miss
        ):
            raise ValueError(
                "The legacy v2 observation cache is read-only; "
                "observation.allow_render_on_miss must be false"
            )
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if self.training.num_workers < 0:
            raise ValueError("training.num_workers must be non-negative")
        if self.training.validation_num_workers < 0:
            raise ValueError("training.validation_num_workers must be non-negative")
        if self.training.validation_interval < 0:
            raise ValueError("training.validation_interval must be non-negative")
        expected_strata = {
            "direct_grasp",
            "pick_remove",
            "push",
            "push_failure",
            "unresolved_or_unknown",
        }
        coverage_strata = tuple(self.training.action_batch_coverage_strata)
        if len(set(coverage_strata)) != len(coverage_strata):
            raise ValueError("training.action_batch_coverage_strata must not contain duplicates")
        if not set(coverage_strata).issubset(expected_strata):
            raise ValueError(
                "training.action_batch_coverage_strata contains an unknown action stratum"
            )
        allowed_strata = tuple(self.training.allowed_action_strata)
        if len(set(allowed_strata)) != len(allowed_strata):
            raise ValueError("training.allowed_action_strata must not contain duplicates")
        if not set(allowed_strata).issubset(expected_strata):
            raise ValueError("training.allowed_action_strata contains an unknown action stratum")
        if allowed_strata and not set(coverage_strata).issubset(set(allowed_strata)):
            raise ValueError(
                "action_batch_coverage_strata must be a subset of allowed_action_strata "
                "when stage filtering is enabled"
            )
        if not 0.0 < self.training.data_fraction <= 1.0:
            raise ValueError("training.data_fraction must be in (0,1]")
        if self.training.scene_start < 0:
            raise ValueError("training.scene_start must be non-negative")
        if self.training.scene_count is not None and self.training.scene_count <= 0:
            raise ValueError("training.scene_count must be positive when configured")
        if (
            self.training.validation_scene_count is not None
            and self.training.validation_scene_count <= 0
        ):
            raise ValueError("training.validation_scene_count must be positive")
        if (
            self.training.max_validation_groups is not None
            and self.training.max_validation_groups <= 0
        ):
            raise ValueError("training.max_validation_groups must be positive")
        if self.training.max_validation_groups is not None:
            if set(self.training.validation_stratum_quota) != expected_strata:
                raise ValueError(
                    "training.validation_stratum_quota must define exactly the five action strata"
                )
            if any(int(value) < 0 for value in self.training.validation_stratum_quota.values()):
                raise ValueError("training.validation_stratum_quota values must be non-negative")
            if (
                sum(int(value) for value in self.training.validation_stratum_quota.values())
                != self.training.max_validation_groups
            ):
                raise ValueError(
                    "training.validation_stratum_quota must sum to max_validation_groups"
                )
            if allowed_strata and any(
                int(value) != 0 and name not in set(allowed_strata)
                for name, value in self.training.validation_stratum_quota.items()
            ):
                raise ValueError(
                    "validation_stratum_quota must be zero outside allowed_action_strata"
                )
        if self.training.amp_initial_scale <= 0:
            raise ValueError("training.amp_initial_scale must be positive")
        if not 0 < self.evaluation.region_probability_threshold < 1:
            raise ValueError("evaluation.region_probability_threshold must be in (0,1)")
        if not self.evaluation.ranking_topk or any(
            value <= 0 for value in self.evaluation.ranking_topk
        ):
            raise ValueError("evaluation.ranking_topk must contain positive integers")
        diagnostic_thresholds = (
            self.evaluation.task_translation_threshold_m,
            self.evaluation.task_rotation_threshold_deg,
            self.evaluation.task_width_threshold_m,
            self.evaluation.global_translation_threshold_m,
            self.evaluation.global_rotation_threshold_deg,
            self.evaluation.global_width_threshold_m,
            self.evaluation.task_nms_translation_m,
            self.evaluation.task_nms_rotation_deg,
            self.evaluation.task_nms_width_m,
            self.evaluation.global_nms_translation_m,
            self.evaluation.global_nms_rotation_deg,
            self.evaluation.global_nms_width_m,
        )
        if any(float(value) <= 0 for value in diagnostic_thresholds):
            raise ValueError("All grasp diagnostic matching/NMS thresholds must be positive")
        if self.evaluation.calibration_bins <= 1:
            raise ValueError("evaluation.calibration_bins must be greater than one")
        if self.logging.log_interval <= 0:
            raise ValueError("logging.log_interval must be positive")
        if self.logging.validation_log_interval <= 0:
            raise ValueError("logging.validation_log_interval must be positive")
        if self.logging.gradient_diagnostics_interval < 0:
            raise ValueError("gradient_diagnostics_interval must be non-negative")
        ratios = self.training.split_ratios
        if len(ratios) not in {2, 3}:
            raise ValueError("training.split_ratios must contain train/val or train/val/test")
        if any(value < 0 for value in ratios) or sum(ratios) <= 0:
            raise ValueError("training.split_ratios must be non-negative with a positive sum")
        if ratios[0] <= 0:
            raise ValueError("training.split_ratios must allocate scenes to train")
        if self.training.validation_interval > 0 and ratios[1] <= 0:
            raise ValueError(
                "training.split_ratios must allocate scenes to val when validation is enabled"
            )
        if self.backbone.backend not in {"point_transformer_v3", "legacy"}:
            raise ValueError("backbone.backend must be point_transformer_v3 or legacy")
        if self.backbone.pretrained_format not in {
            "auto",
            "tcd_prg",
            "sonata",
            "pointcept",
            "ptv3",
        }:
            raise ValueError("Unsupported backbone.pretrained_format")
        if not 0.0 <= self.backbone.pretrained_min_parameter_fraction <= 1.0:
            raise ValueError("pretrained_min_parameter_fraction must be in [0,1]")
        if self.backbone.pretrained_freeze_steps < 0:
            raise ValueError("pretrained_freeze_steps must be non-negative")
        if self.backbone.pretrained_auto_download and not self.backbone.pretrained_url:
            raise ValueError("pretrained_auto_download requires pretrained_url")
        if self.backbone.grid_size_m <= 0:
            raise ValueError("backbone.grid_size_m must be positive")
        if self.dataset.scene_points < 0:
            raise ValueError("dataset.scene_points must be zero (unlimited) or positive")
        if self.backbone.patch_size <= 0:
            raise ValueError("backbone.patch_size must be positive")
        if abs(self.push_distance_m - PUSH_DISTANCE_M) > 1e-7:
            raise ValueError("The main TCD-PRG primitive requires push_distance_m == 0.15")
        if abs(self.push.distance_m - PUSH_DISTANCE_M) > 1e-7:
            raise ValueError("push.distance_m must be 0.15 for the main experiment")
        if self.evaluation.max_preparation_actions != MAX_PREPARATION_ACTIONS:
            raise ValueError("The main experiment requires H=5")
        if self.planner.max_preparation_actions != MAX_PREPARATION_ACTIONS:
            raise ValueError("planner.max_preparation_actions must be H=5")
        if not 0 <= self.model.min_grasp_width_m < self.model.max_grasp_width_m:
            raise ValueError("Invalid AG gripper opening range")
        if self.model.contact_heatmap_sigma_m <= 0:
            raise ValueError("contact_heatmap_sigma_m must be positive")
        if (
            min(
                self.model.grasp_nms_translation_m,
                self.model.grasp_nms_rotation_deg,
                self.model.grasp_nms_width_m,
                self.model.grasp_nms_approach_deg,
            )
            <= 0
        ):
            raise ValueError("All grasp NMS thresholds must be positive")
        if (
            min(
                self.model.global_grasp_nms_translation_m,
                self.model.global_grasp_nms_rotation_deg,
                self.model.global_grasp_nms_width_m,
                self.model.global_grasp_nms_approach_deg,
            )
            <= 0
        ):
            raise ValueError("Global grasp NMS thresholds must be positive")
        if not 0.0 < self.model.task_grasp_probability_threshold < 1.0:
            raise ValueError("task_grasp_probability_threshold must be in (0,1)")
        if not 0.0 < self.model.pick_remove_probability_threshold < 1.0:
            raise ValueError("pick_remove_probability_threshold must be in (0,1)")
        if self.model.push_object_topk <= 0:
            raise ValueError("push_object_topk must be positive")
        if self.model.push_object_topk > self.model.instance_queries:
            raise ValueError("push_object_topk cannot exceed instance_queries")
        if self.model.pick_remove_target_margin_m <= 0:
            raise ValueError("pick_remove_target_margin_m must be positive")
        if min(self.model.push_nms_contact_m, self.model.push_nms_direction_deg) <= 0:
            raise ValueError("PUSH NMS thresholds must be positive")
        if not 1 <= self.model.push_directions_per_contact <= self.model.num_direction_bins:
            raise ValueError("push_directions_per_contact must be in [1,num_direction_bins]")
        if self.model.push_candidates <= 0 or self.model.max_push_candidates <= 0:
            raise ValueError("PUSH contact and final candidate budgets must be positive")
        if self.model.push_direction_contact_topk <= 0:
            raise ValueError("push_direction_contact_topk must be positive")
        if self.model.push_utility_temperature <= 0:
            raise ValueError("push_utility_temperature must be positive")
        if self.training.push_coverage_penalty_weight < 0:
            raise ValueError("push_coverage_penalty_weight must be non-negative")
        if not 0.0 <= self.model.push_candidate_probability_threshold < 1.0:
            raise ValueError("push_candidate_probability_threshold must be in [0,1)")
        if self.model.global_grasp_input_mode not in {"scene_only", "instance_assisted"}:
            raise ValueError("global_grasp_input_mode must be scene_only or instance_assisted")
        if (
            min(
                self.sampling.global_positive_grasps_per_object,
                self.sampling.global_negative_grasps_per_object,
            )
            < 0
        ):
            raise ValueError("Global grasp stratum sizes cannot be negative")
        if self.sampling.global_positive_grasps_per_object == 0:
            raise ValueError("Global grasp training requires positive samples")
        if self.model.global_grasp_candidates <= 0:
            raise ValueError("global_grasp_candidates must be positive")
        if self.model.instance_queries <= 0:
            raise ValueError("instance_queries must be positive")
        if self.model.instance_decoder_layers <= 0 or self.model.instance_decoder_heads <= 0:
            raise ValueError("instance decoder layers/heads must be positive")
        if self.model.feature_dim % self.model.instance_decoder_heads:
            raise ValueError("feature_dim must be divisible by instance_decoder_heads")
        if not 0.0 < self.model.instance_objectness_threshold < 1.0:
            raise ValueError("instance_objectness_threshold must be in (0,1)")
        if self.model.instance_matching_points <= 0:
            raise ValueError("instance_matching_points must be positive")
        if self.model.target_query_temperature <= 0:
            raise ValueError("target_query_temperature must be positive")
        if self.model.target_prompt_radius_m <= 0 or self.model.target_prompt_sigma_m <= 0:
            raise ValueError("target prompt radius/sigma must be positive")
        if self.model.target_prompt_jitter_std_m < 0:
            raise ValueError("target_prompt_jitter_std_m must be non-negative")
        if (
            min(
                self.model.target_prompt_weight,
                self.model.target_category_weight,
                self.model.target_objectness_weight,
                self.model.target_center_weight,
                self.model.target_learned_weight,
                self.model.target_reid_weight,
                self.model.target_reid_center_weight,
            )
            < 0
        ):
            raise ValueError("target selector weights must be non-negative")
        if self.model.target_reid_max_center_distance_m <= 0:
            raise ValueError("target_reid_max_center_distance_m must be positive")
        if not 0.0 <= self.model.target_prompt_min_support <= 1.0:
            raise ValueError("target_prompt_min_support must be in [0,1]")
        if self.model.target_prompt_min_margin < 0:
            raise ValueError("target_prompt_min_margin must be non-negative")
        if self.model.target_same_category_loss_boost < 1.0:
            raise ValueError("target_same_category_loss_boost must be >= 1")
        if (
            min(
                self.model.task_grasp_scene_points,
                self.model.task_grasp_gripper_points,
            )
            <= 0
        ):
            raise ValueError("Task-grasp point budgets must be positive")
        if (
            min(
                self.graspnet.scene_input_points,
                self.graspnet.target_input_points,
                self.graspnet.global_proposals,
                self.graspnet.target_proposals,
                self.graspnet.num_view,
                self.graspnet.num_angle,
                self.graspnet.num_depth,
                self.graspnet.target_min_crop_points,
            )
            <= 0
        ):
            raise ValueError("GraspNet point/proposal/discretization counts must be positive")
        if self.graspnet.camera_view_index < 0:
            raise ValueError("GraspNet camera_view_index must be non-negative")
        if not 0.0 < self.graspnet.target_crop_probability < 1.0:
            raise ValueError("GraspNet target_crop_probability must lie in (0,1)")
        if self.graspnet.camera_transfer_max_distance_m <= 0:
            raise ValueError("GraspNet camera_transfer_max_distance_m must be positive")
        selection_modes = {"quality_topk", "quality_diverse"}
        if self.graspnet.target_selection_mode not in selection_modes:
            raise ValueError(
                "graspnet.target_selection_mode must be quality_topk or quality_diverse"
            )
        if self.graspnet.global_selection_mode not in selection_modes:
            raise ValueError(
                "graspnet.global_selection_mode must be quality_topk or quality_diverse"
            )
        if not 0.0 < self.graspnet.diversity_quality_fraction <= 1.0:
            raise ValueError("graspnet.diversity_quality_fraction must be in (0,1]")
        if self.graspnet.diversity_translation_m <= 0:
            raise ValueError("graspnet.diversity_translation_m must be positive")
        if self.graspnet.diversity_rotation_deg <= 0:
            raise ValueError("graspnet.diversity_rotation_deg must be positive")
        if self.graspnet.diversity_pool_factor < 1:
            raise ValueError("graspnet.diversity_pool_factor must be >= 1")
        if self.losses.internal.get("task_grasp_bce", 1.0) <= 0:
            raise ValueError("task_grasp_bce must be positive")
        if self.model.push_direction_transformer_layers <= 0:
            raise ValueError("push_direction_transformer_layers must be positive")
        if self.model.push_direction_feature_dim % self.model.push_direction_transformer_heads:
            raise ValueError(
                "push_direction_feature_dim must be divisible by push_direction_transformer_heads"
            )
        if len(self.model.push_utility_component_weights) != 5:
            raise ValueError(
                "push_utility_component_weights must match the five dataset components"
            )
        if len(self.model.push_failure_penalties) != 3:
            raise ValueError("push_failure_penalties must cover unstable/workspace/other failures")
        if self.training.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("training.amp_dtype must be float16 or bfloat16")
        if self.cache.eviction != "lru":
            raise ValueError("Only deterministic LRU cache eviction is supported")
        if self.cache.max_gb <= 0 or self.cache.min_free_gb <= 0:
            raise ValueError("cache max_gb and min_free_gb must be positive")


def load_config(path: str | Path, overrides: list[str] | None = None) -> TCDPRGConfig:
    """Load a strict structured YAML configuration with optional dot-list overrides."""

    from omegaconf import OmegaConf

    def load_source(source: Path, stack: tuple[Path, ...] = ()) -> Any:
        source = source.resolve()
        if source in stack:
            raise ValueError(f"Recursive config defaults: {source}")
        raw_source = OmegaConf.load(source)
        defaults = list(raw_source.get("defaults", []))
        if "defaults" in raw_source:
            del raw_source["defaults"]
        parents = []
        for default in defaults:
            if str(default) == "_self_":
                continue
            parent = Path(str(default))
            if not parent.is_absolute():
                parent = source.parent / parent
            parents.append(load_source(parent, (*stack, source)))
        return OmegaConf.merge(*parents, raw_source)

    raw = load_source(Path(path))
    merged = OmegaConf.merge(
        OmegaConf.structured(TCDPRGConfig), raw, OmegaConf.from_dotlist(overrides or [])
    )
    config = OmegaConf.to_object(merged)
    if not isinstance(config, TCDPRGConfig):
        raise TypeError("Structured configuration did not produce TCDPRGConfig")
    # Source configuration remains portable. At runtime, path-bearing fields
    # are made absolute relative to the repository, not the launch directory.
    path_fields = (
        (config.dataset, "root"),
        (config.dataset, "acronym_root"),
        (config.dataset, "functional_region_root"),
        (config.dataset, "stageb_binary_root"),
        (config.dataset, "fr5_ag_urdf"),
        (config.observation, "worker_script"),
        (config.observation, "runtime_mesh_root"),
        (config.observation, "certification_worker_script"),
        (config.observation, "render_temporary_root"),
        (config.observation, "certification_temporary_root"),
        (config.cache, "directory"),
        (config.cache, "index_directory"),
        (config.backbone, "source_root"),
        (config.graspnet, "source_root"),
        (config.graspnet, "checkpoint"),
        (config.baseline, "gapg_root"),
        (config.baseline, "grasp_checkpoint"),
        (config.baseline, "push_checkpoint"),
        (config.baseline, "graspnet_checkpoint"),
        (config.model, "task_grasp_gripper_geometry"),
        (config.model, "stageb_label_gripper_geometry"),
        (config, "output_dir"),
    )
    for owner, name in path_fields:
        value = getattr(owner, name)
        if value:
            setattr(owner, name, str(project_path(value)))
    if config.backbone.pretrained_checkpoint:
        config.backbone.pretrained_checkpoint = str(
            project_path(config.backbone.pretrained_checkpoint)
        )
    config.validate()
    return config
