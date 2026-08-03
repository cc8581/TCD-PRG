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
    step_labels_subdir: str = "task_training_labels_steps1_6_v1"
    action_labels_subdir: str = "task_positive_multistep_sequences"
    global_grasp_library_subdir: str = "generic_grasp_library_v1"
    global_grasp_certification_subdir: str = "global_grasp_scene_certification_v2"
    pick_remove_global_association_subdir: str = "pick_remove_global_grasp_association_v1"
    scene_points: int = 16_384
    # target_points 仅供独立资源分析器使用；正式模型输入点数由 scene_points 决定。
    target_points: int = 4_096


@dataclass(slots=True)
class ObservationConfig:
    provider: str = "cached"
    render_width: int = 320
    render_height: int = 200
    camera_profile: str = "mecheye_pro_s_three_view"
    renderer_version: str = "tcd_prg_pybullet_v1"
    pybullet_python: str = "python"
    worker_script: str = "scripts/render_observation_worker_py38.py"
    runtime_mesh_root: str = "runtime/cache/meshes"
    gripper_worker_script: str = "scripts/sample_gripper_worker_py38.py"
    gripper_cache_dir: str = "runtime/cache/grippers"
    certification_worker_script: str = "scripts/certify_actions_worker_py38.py"
    render_temporary_root: str = "runtime/tmp/render_requests"
    certification_temporary_root: str = "runtime/tmp/certification"


@dataclass(slots=True)
class CacheConfig:
    directory: str = "runtime/cache/observations"
    max_gb: float = 15.0
    min_free_gb: float = 20.0
    eviction: str = "lru"
    prefetch_workers: int = 4


@dataclass(slots=True)
class BackboneConfig:
    # grid_size_m 控制 PTv3 体素分辨率；patch_size 控制序列化注意力的最大 patch 长度。
    backend: str = "point_transformer_v3"
    source_root: str = "third_party/PointTransformerV3"
    pretrained_checkpoint: str | None = None
    freeze: bool = False
    grid_size_m: float = 0.005
    enable_flash_attention: bool = False
    patch_size: int = 256
    attention_points: int = 1_024


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
    num_relation_types: int = 8
    num_direction_bins: int = 16
    # 每个抓取 query 直接预测平移、连续 SO(3)、夹爪宽度和条件化质量。
    contact_heatmap_sigma_m: float = 0.008
    max_grasp_width_m: float = 0.095
    min_grasp_width_m: float = 0.0
    candidate_topk: int = 64
    task_grasp_candidates: int = 64
    global_grasp_candidates: int = 64
    grasp_decoder_layers: int = 3
    grasp_decoder_heads: int = 8
    pick_remove_candidates: int = 16
    push_candidates: int = 16
    # push_candidates 是接触点预算；每个点再展开多个方向，最终总量受 max_push_candidates 限制。
    push_directions_per_contact: int = 2
    max_push_candidates: int = 32
    push_direction_feature_dim: int = 64
    push_direction_transformer_layers: int = 1
    push_direction_transformer_heads: int = 4
    activation_checkpointing: bool = True
    verifier_local_radius_m: float = 0.25
    verifier_candidate_micro_batch: int = 16
    verifier_transformer_layers: int = 2
    verifier_transformer_heads: int = 8
    verifier_validity_threshold: float = 0.5
    # Verifier 是学习到的策略证据，不等价于确定性几何认证；hard gate 只用于消融实验。
    verifier_hard_gate: bool = False
    graph_edge_threshold: float = 0.5
    graph_candidate_mode: str = "soft"
    graph_candidate_topk_objects: int = 4
    default_required_grasp_count: int = 20
    max_required_grasp_count: int = 20
    grasp_nms_translation_m: float = 0.010
    grasp_nms_rotation_deg: float = 12.0
    grasp_nms_width_m: float = 0.005
    grasp_nms_approach_deg: float = 12.0
    graph_candidate_fallback_objects: int = 1
    # 数据中允许推动目标物体自恢复；该开关显式保留此动作原语，避免图门控静默覆盖。
    allow_target_push_recovery: bool = True
    global_grasp_input_mode: str = "scene_only"
    global_grasp_nms_translation_m: float = 0.01
    global_grasp_nms_rotation_deg: float = 15.0
    global_grasp_nms_width_m: float = 0.005
    global_grasp_nms_approach_deg: float = 15.0
    policy_push_match_contact_m: float = 0.020
    policy_push_match_direction_deg: float = 22.5
    policy_grasp_match_translation_m: float = 0.020
    policy_grasp_match_rotation_deg: float = 20.0
    policy_grasp_match_width_m: float = 0.010
    push_nms_contact_m: float = 0.015
    push_nms_direction_deg: float = 15.0
    push_utility_component_weights: tuple[float, ...] = (0.05, 1.0, -1.0, -0.25, 1.0)
    push_failure_penalties: tuple[float, ...] = (1.0, 2.0, 1.0)


@dataclass(slots=True)
class AblationConfig:
    use_task_region_condition: bool = True
    use_dependency_graph: bool = True
    use_indirect_dependency_reasoning: bool = True
    use_gripper_scene_verifier: bool = True
    use_push_potential: bool = True
    router_type: str = "hierarchical"


@dataclass(slots=True)
class TrainingConfig:
    # max_optimizer_steps 统计真实参数更新次数，不包含 AMP 溢出后被跳过的 step。
    seed: int = 2026
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "float16"
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_optimizer_steps: int = 100_000
    # validation_interval=0 仅用于没有验证集的启动阶段，正式实验不应关闭验证。
    validation_interval: int = 1_000
    checkpoint_interval: int = 1_000
    gradient_clip_norm: float = 1.0
    ema_decay: float | None = 0.999
    early_stopping_patience: int = 20
    deterministic: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    max_validation_groups: int = 256
    max_train_groups: int | None = None
    frozen_modules: tuple[str, ...] = ()
    unfreeze_at_optimizer_step: int | None = None
    ddp_backend: str = "auto"
    ddp_find_unused_parameters: bool = True
    # Policy Stage B/C 可读取冻结前级网络生成的候选；ratio=0 保持纯 teacher 预训练。
    generated_policy_candidate_cache: str = ""
    generated_policy_candidate_ratio: float = 0.0
    generated_policy_checkpoint_sha256: str = ""
    # generated-only 启动前必须达到正样本覆盖率和有效排序行覆盖率，避免静默训练极少状态。
    generated_policy_min_positive_coverage: float = 0.05
    generated_policy_min_effective_coverage: float = 0.01
    validation_family_weights: dict[str, float] = field(default_factory=lambda: {
        "policy_candidate": 1.0,
        "verify_overall": 0.5,
        "physical_edge": 0.25,
        "task_edge": 0.25,
        "task_grasp": 0.25,
        "global_grasp": 0.25,
        "region": 0.1,
        "push_object": 0.1,
        "push_contact": 0.05,
        "push_direction": 0.05,
        "push_potential": 0.1,
    })


@dataclass(slots=True)
class EvaluationConfig:
    max_preparation_actions: int = MAX_PREPARATION_ACTIONS
    horizons: tuple[int, ...] = (0, 1, 3, 5)
    bootstrap_samples: int = 1_000
    confidence: float = 0.95
    max_groups: int | None = None
    global_grasp_tracks: tuple[str, ...] = ("scene_only", "instance_assisted")
    global_translation_threshold_m: float = 0.01
    global_rotation_threshold_deg: float = 15.0
    global_width_threshold_m: float = 0.005
    global_metrics_after_nms: bool = True


@dataclass(slots=True)
class GraspVerifierConfig:
    local_scene_points: int = 512
    gripper_points: int = 512


@dataclass(slots=True)
class GraphConfig:
    physical_relations: tuple[str, ...] = ("near", "contact", "support", "press", "occlude")
    task_relations: tuple[str, ...] = (
        "block_task_region", "block_task_grasp", "block_grasp_approach"
    )
    layers: int = 3
    heads: int = 4


@dataclass(slots=True)
class PushConfig:
    distance_m: float = PUSH_DISTANCE_M
    direction_bins: int = 16


@dataclass(slots=True)
class RouterConfig:
    type: str = "hierarchical"
    layers: int = 2
    heads: int = 4
    max_preparation_actions: int = MAX_PREPARATION_ACTIONS


@dataclass(slots=True)
class LossConfig:
    # 这些权重控制任务族之间的比例；族内子损失权重单独放在 internal 中。
    region: float = 1.0
    task_grasp: float = 1.0
    global_grasp: float = 1.0
    physical_edge: float = 1.0
    task_edge: float = 1.0
    verify_overall: float = 1.0
    push_object: float = 1.0
    push_contact: float = 1.0
    push_direction: float = 1.0
    push_potential: float = 1.0
    policy_candidate: float = 1.0
    internal: dict[str, float] = field(default_factory=lambda: {
        "region_focal": 1.0,
        "region_dice": 1.0,
        "region_visibility": 0.2,
    })

    def family_weights(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in (
                "region", "task_grasp", "global_grasp", "physical_edge", "task_edge",
                "verify_overall", "push_object", "push_contact", "push_direction",
                "push_potential", "policy_candidate",
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
    global_intrinsic_negative_grasps_per_object: int = 32
    global_scene_negative_grasps_per_object: int = 32
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


@dataclass(slots=True)
class TCDPRGConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    region_head: RegionHeadConfig = field(default_factory=RegionHeadConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    grasp_verifier: GraspVerifierConfig = field(default_factory=GraspVerifierConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    push: PushConfig = field(default_factory=PushConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
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
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if self.logging.log_interval <= 0:
            raise ValueError("logging.log_interval must be positive")
        if self.backbone.backend not in {"point_transformer_v3", "legacy"}:
            raise ValueError("backbone.backend must be point_transformer_v3 or legacy")
        if self.backbone.grid_size_m <= 0:
            raise ValueError("backbone.grid_size_m must be positive")
        if self.backbone.patch_size <= 0:
            raise ValueError("backbone.patch_size must be positive")
        if abs(self.push_distance_m - PUSH_DISTANCE_M) > 1e-7:
            raise ValueError("The main TCD-PRG primitive requires push_distance_m == 0.15")
        if abs(self.push.distance_m - PUSH_DISTANCE_M) > 1e-7:
            raise ValueError("push.distance_m must be 0.15 for the main experiment")
        if self.evaluation.max_preparation_actions != MAX_PREPARATION_ACTIONS:
            raise ValueError("The main experiment requires H=5")
        if self.router.max_preparation_actions != MAX_PREPARATION_ACTIONS:
            raise ValueError("router.max_preparation_actions must be H=5")
        if self.planner.max_preparation_actions != MAX_PREPARATION_ACTIONS:
            raise ValueError("planner.max_preparation_actions must be H=5")
        if not 0 <= self.model.min_grasp_width_m < self.model.max_grasp_width_m:
            raise ValueError("Invalid AG gripper opening range")
        if self.model.contact_heatmap_sigma_m <= 0:
            raise ValueError("contact_heatmap_sigma_m must be positive")
        if not 0 < self.model.graph_edge_threshold < 1:
            raise ValueError("graph_edge_threshold must be in (0,1)")
        if self.model.default_required_grasp_count <= 0:
            raise ValueError("default_required_grasp_count must be positive")
        if self.model.max_required_grasp_count < self.model.default_required_grasp_count:
            raise ValueError("max_required_grasp_count cannot be smaller than the default")
        if self.model.task_grasp_candidates < self.model.max_required_grasp_count:
            raise ValueError(
                "task_grasp_candidates must cover max_required_grasp_count so the "
                "state gate can be satisfied"
            )
        if min(
            self.model.grasp_nms_translation_m,
            self.model.grasp_nms_rotation_deg,
            self.model.grasp_nms_width_m,
            self.model.grasp_nms_approach_deg,
        ) <= 0:
            raise ValueError("All grasp NMS thresholds must be positive")
        if min(
            self.model.global_grasp_nms_translation_m,
            self.model.global_grasp_nms_rotation_deg,
            self.model.global_grasp_nms_width_m,
            self.model.global_grasp_nms_approach_deg,
        ) <= 0:
            raise ValueError("Global grasp NMS thresholds must be positive")
        if min(
            self.model.policy_push_match_contact_m,
            self.model.policy_push_match_direction_deg,
            self.model.policy_grasp_match_translation_m,
            self.model.policy_grasp_match_rotation_deg,
            self.model.policy_grasp_match_width_m,
        ) <= 0:
            raise ValueError("Generated policy matching thresholds must be positive")
        if self.model.graph_candidate_fallback_objects < 0:
            raise ValueError("graph_candidate_fallback_objects cannot be negative")
        if self.model.graph_candidate_mode not in {"hard", "soft", "none"}:
            raise ValueError("graph_candidate_mode must be hard, soft, or none")
        if self.model.graph_candidate_topk_objects <= 0:
            raise ValueError("graph_candidate_topk_objects must be positive")
        if min(self.model.push_nms_contact_m, self.model.push_nms_direction_deg) <= 0:
            raise ValueError("PUSH NMS thresholds must be positive")
        if not 1 <= self.model.push_directions_per_contact <= self.model.num_direction_bins:
            raise ValueError(
                "push_directions_per_contact must be in [1,num_direction_bins]"
            )
        if self.model.push_candidates <= 0 or self.model.max_push_candidates <= 0:
            raise ValueError("PUSH contact and final candidate budgets must be positive")
        if not 0.0 <= self.training.generated_policy_candidate_ratio <= 1.0:
            raise ValueError("generated_policy_candidate_ratio must be in [0,1]")
        if self.training.validation_interval < 0:
            raise ValueError("validation_interval must be non-negative")
        if not 0.0 <= self.training.generated_policy_min_positive_coverage <= 1.0:
            raise ValueError("generated_policy_min_positive_coverage must be in [0,1]")
        if not 0.0 <= self.training.generated_policy_min_effective_coverage <= 1.0:
            raise ValueError("generated_policy_min_effective_coverage must be in [0,1]")
        if self.training.generated_policy_candidate_ratio == 1.0 and (
            self.training.generated_policy_min_positive_coverage <= 0
            or self.training.generated_policy_min_effective_coverage <= 0
        ):
            raise ValueError(
                "Pure generated policy training requires positive non-zero coverage thresholds"
            )
        if (
            self.training.generated_policy_candidate_ratio > 0
            and not self.training.generated_policy_candidate_cache
        ):
            raise ValueError(
                "generated_policy_candidate_cache is required when generated policy "
                "candidate training is enabled"
            )
        if self.model.global_grasp_input_mode not in {"scene_only", "instance_assisted"}:
            raise ValueError("global_grasp_input_mode must be scene_only or instance_assisted")
        if min(
            self.sampling.global_positive_grasps_per_object,
            self.sampling.global_intrinsic_negative_grasps_per_object,
            self.sampling.global_scene_negative_grasps_per_object,
        ) < 0:
            raise ValueError("Global grasp stratum sizes cannot be negative")
        if self.sampling.global_positive_grasps_per_object == 0:
            raise ValueError("Global grasp training requires positive samples")
        if self.model.global_grasp_candidates <= 0:
            raise ValueError("global_grasp_candidates must be positive")
        if self.model.grasp_decoder_layers <= 0 or self.model.grasp_decoder_heads <= 0:
            raise ValueError("Grasp decoder layers and heads must be positive")
        if self.model.feature_dim % self.model.grasp_decoder_heads:
            raise ValueError("feature_dim must be divisible by grasp_decoder_heads")
        if self.model.verifier_transformer_layers <= 0:
            raise ValueError("verifier_transformer_layers must be positive")
        if self.model.feature_dim % self.model.verifier_transformer_heads:
            raise ValueError("feature_dim must be divisible by verifier_transformer_heads")
        if self.model.push_direction_transformer_layers <= 0:
            raise ValueError("push_direction_transformer_layers must be positive")
        if (
            self.model.push_direction_feature_dim
            % self.model.push_direction_transformer_heads
        ):
            raise ValueError(
                "push_direction_feature_dim must be divisible by "
                "push_direction_transformer_heads"
            )
        if len(self.model.push_utility_component_weights) != 5:
            raise ValueError(
                "push_utility_component_weights must match the five dataset components"
            )
        if len(self.model.push_failure_penalties) != 3:
            raise ValueError("push_failure_penalties must cover unstable/workspace/other failures")
        if self.ablation.router_type not in {
            "hierarchical",
            "fixed_priority",
            "flat_candidate_classifier",
        }:
            raise ValueError(f"Unsupported router_type={self.ablation.router_type}")
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
        (config.dataset, "fr5_ag_urdf"),
        (config.observation, "worker_script"),
        (config.observation, "runtime_mesh_root"),
        (config.observation, "gripper_worker_script"),
        (config.observation, "gripper_cache_dir"),
        (config.observation, "certification_worker_script"),
        (config.observation, "render_temporary_root"),
        (config.observation, "certification_temporary_root"),
        (config.cache, "directory"),
        (config.backbone, "source_root"),
        (config.training, "generated_policy_candidate_cache"),
        (config.baseline, "gapg_root"),
        (config.baseline, "grasp_checkpoint"),
        (config.baseline, "push_checkpoint"),
        (config.baseline, "graspnet_checkpoint"),
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
