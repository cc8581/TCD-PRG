# ruff: noqa: E501
from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock

from .camera import build_cameras
from .decision import DecisionSession
from .perception import fuse_frames
from .physics_client import PhysicsClient
from .predictor_client import PredictorClient
from .robot import build_robot
from .transforms import xyz_rpy_to_matrix
from .types import Prediction
from .workflow import scene_change_m, similar_push


class ExperimentController:
    def __init__(self, config):
        self.config = config
        self.cameras = build_cameras(config)
        self.robot = build_robot(config)
        self.predictor = None
        self.physics = None
        self._worker_lock = Lock()
        self._worker_epoch = 0
        self._closed = False
        self.scene = None
        self.prediction = None
        self.connected = False
        self.cameras_connected = False
        self.robot_connected = False
        self.robot_enabled = False
        self.robot_paused = False
        self.active_task = None
        self.task_finished = False
        self.last_scene_xyz = None
        self.no_change_count = 0
        self.last_executed_action = None
        self.stop_reason = None
        self.cycle_started = None
        self.cycle_timings = {}
        self.manual_session = None
        self.pending_task = None
        self.pending_push_rules = ()
        self.manual_stage = "capture"
        self.manual_config_signature = None
        self.pending_task_grasp = None

    def connect_cameras(self) -> str:
        connected = []
        if not self.cameras_connected:
            try:
                for camera in self.cameras:
                    if not camera.connect():
                        raise RuntimeError(f"Camera {camera.camera_id} failed")
                    connected.append(camera)
            except Exception:
                for camera in connected:
                    camera.disconnect()
                raise
            self.cameras_connected = True
            self.connected = True
        return f"已连接 {len(self.cameras)} 台相机"

    def connect_robot(self) -> str:
        if self.robot_connected:
            return "机械臂已连接；FR5 当前未使能" if not self.robot_enabled else "机械臂已连接并使能"
        self.robot_connected = bool(self.robot.connect())
        self.robot_enabled = False
        if not self.robot_connected:
            raise RuntimeError("FR5 连接失败；请检查网络、控制器及设备设置")
        return "机械臂已连接；FR5 当前未使能"

    def connect(self) -> str:
        """Compatibility path for callers that still request both devices."""
        self.connect_cameras()
        robot_error = None
        try:
            self.connect_robot()
        except Exception as error:
            self.robot_connected = False
            robot_error = error
        if self.robot_connected:
            return f"已连接 {len(self.cameras)} 台相机和机械臂；FR5 当前未使能"
        detail = "" if robot_error is None else f"（{type(robot_error).__name__}: {robot_error}）"
        return f"已连接 {len(self.cameras)} 台相机；机械臂未连接{detail}，仍可完成推理与仿真"

    def enable_robot(self) -> str:
        if not self.robot_connected:
            raise RuntimeError("FR5 尚未连接")
        self.robot.enable()
        self.robot_enabled = True
        self.robot_paused = False
        return "FR5 已切换自动模式并使能"

    def disable_robot(self) -> str:
        if not self.robot_connected:
            raise RuntimeError("FR5 尚未连接")
        self.robot.disable()
        self.robot_enabled = False
        self.robot_paused = False
        self.prediction = None
        return "FR5 已下使能；待执行动作已清除"

    def clear_errors(self) -> str:
        if not self.robot_connected:
            raise RuntimeError("FR5 尚未连接")
        self.robot.clear_errors()
        return "FR5 控制器故障已清除；确认现场安全后再使能"

    def pause_robot(self) -> str:
        if not self.robot_enabled:
            raise RuntimeError("FR5 未使能")
        self.robot.pause()
        self.robot_paused = True
        return "FR5 运动已暂停"

    def resume_robot(self) -> str:
        if not self.robot_enabled:
            raise RuntimeError("FR5 未使能")
        if not self.robot_paused:
            raise RuntimeError("FR5 当前没有暂停的运动")
        self.robot.resume()
        self.robot_paused = False
        return "FR5 已继续运动"

    def home_robot(self) -> str:
        if not self.robot_enabled:
            raise RuntimeError("FR5 未使能")
        self.prediction = None
        self.robot.home()
        self.scene = None
        return "FR5 已回安全原点；场景已失效，请重新采集"

    def robot_status(self) -> str:
        state = self.robot.status()
        self.robot_enabled = bool(state.get("enabled", 0))
        joints = state.get("joints_deg")
        return (
            f"使能={state.get('enabled')}  急停={state.get('emergency_stop')}  "
            f"运动完成={state.get('motion_done')}  队列={state.get('queue_len')}  "
            f"故障={state.get('main_code')}/{state.get('sub_code')}\n"
            f"关节角={None if joints is None else [round(float(x), 2) for x in joints]}"
        )

    def load_model(self, progress: Callable[[str], None] | None = None) -> str:
        if self.predictor is None:
            if progress:
                progress("正在加载TCD-PRG模型……")
            self._ensure_worker("predictor", lambda: PredictorClient(self.config.path))
        return "TCD-PRG模型已加载"

    def _ensure_worker(self, name: str, factory):
        """Publish a started subprocess only if no cancellation raced its startup."""
        with self._worker_lock:
            if self._closed:
                raise RuntimeError("窗口已关闭，不能启动后台进程")
            existing = getattr(self, name)
            if existing is not None:
                return existing
            epoch = self._worker_epoch
        created = factory()
        with self._worker_lock:
            cancelled = self._closed or epoch != self._worker_epoch
            existing = getattr(self, name)
            if not cancelled and existing is None:
                setattr(self, name, created)
                return created
        created.close()
        if cancelled:
            raise RuntimeError("后台任务已取消")
        return existing

    def acquire(self):
        """Capture -> raw fuse -> integrated TCD-PRG instance perception."""
        if not self.cameras_connected:
            raise RuntimeError("请先连接相机")
        self.scene = None
        self.prediction = None
        self.cycle_started = time.perf_counter()
        self.cycle_timings = {}
        frames = [camera.capture() for camera in self.cameras]
        stage = time.perf_counter()
        updated = fuse_frames(frames, None, self.config.raw["fusion"])
        self.cycle_timings["point_cloud_preprocess_s"] = time.perf_counter() - stage
        self.load_model()
        stage = time.perf_counter()
        updated = self.predictor.perceive(updated)
        self.cycle_timings["perception_s"] = time.perf_counter() - stage
        if self.last_scene_xyz is not None and self.active_task is not None:
            change = scene_change_m(self.last_scene_xyz, updated.xyz_m)
            threshold = float(
                self.config.raw.get("workflow", {}).get("scene_change_threshold_m", 0.005)
            )
            self.no_change_count = self.no_change_count + 1 if change < threshold else 0
            if self.no_change_count >= 2:
                self.stop_reason = (
                    f"连续两次重新感知场景变化低于阈值（{change:.4f} m < {threshold:.4f} m）"
                )
        self.scene = updated
        self.prediction = None
        return self.scene

    def acquire_with_progress(self, progress):
        """Automatic stages 1-2 with UI-visible boundaries."""
        if not self.cameras_connected:
            raise RuntimeError("请先连接相机")
        self.cycle_started = time.perf_counter()
        self.cycle_timings = {}
        frames = [camera.capture() for camera in self.cameras]
        stage = time.perf_counter()
        updated = fuse_frames(frames, None, self.config.raw["fusion"])
        self.cycle_timings["point_cloud_preprocess_s"] = time.perf_counter() - stage
        self.scene = updated
        progress({"stage": "LOAD_OR_CAPTURE", "message": "点云融合与正式预处理完成"})
        self.load_model()
        stage = time.perf_counter()
        updated = self.predictor.perceive(updated)
        self.cycle_timings["perception_s"] = time.perf_counter() - stage
        if self.last_scene_xyz is not None and self.active_task is not None:
            change = scene_change_m(self.last_scene_xyz, updated.xyz_m)
            threshold = float(
                self.config.raw.get("workflow", {}).get("scene_change_threshold_m", 0.005)
            )
            self.no_change_count = self.no_change_count + 1 if change < threshold else 0
            if self.no_change_count >= 2:
                self.stop_reason = (
                    f"连续两次重新感知场景变化低于阈值（{change:.4f} m < {threshold:.4f} m）"
                )
        self.scene = updated
        self.prediction = None
        progress({"stage": "PERCEPTION", "message": "实例感知完成"})
        return updated

    def capture_scene(self):
        """Manual stage 1: capture/fuse/preprocess without running perception."""
        if not self.cameras_connected:
            raise RuntimeError("请先连接相机")
        self.cycle_started = time.perf_counter()
        self.cycle_timings = {}
        frames = [camera.capture() for camera in self.cameras]
        stage = time.perf_counter()
        self.scene = fuse_frames(frames, None, self.config.raw["fusion"])
        self.cycle_timings["point_cloud_preprocess_s"] = time.perf_counter() - stage
        self.prediction = self.manual_session = self.pending_task = None
        self.manual_stage = "perceive"
        return self.scene

    def perceive_scene(self):
        """Manual stage 2: instance perception over the captured fused cloud."""
        if self.manual_stage != "perceive":
            raise RuntimeError(f"当前手动阶段为 {self.manual_stage}，不能执行感知")
        if self.scene is None:
            raise RuntimeError("请先加载或采集点云")
        self.load_model()
        stage = time.perf_counter()
        self.scene = self.predictor.perceive(self.scene)
        self.cycle_timings["perception_s"] = time.perf_counter() - stage
        self.prediction = self.manual_session = self.pending_task = None
        self.manual_stage = "select_target"
        return self.scene

    def load_offline_scene(self, scene):
        """Enter the explicit perception stage after a cache-only scene load."""
        self.reset_task()
        self.scene = scene
        self.cycle_started = time.perf_counter()
        self.manual_stage = "perceive"
        return scene

    def select_manual_target(self, target: int, category: int, region: int):
        """Manual stage 3; changing it invalidates every downstream stage."""
        if self.manual_stage != "select_target":
            raise RuntimeError(f"当前手动阶段为 {self.manual_stage}，不能选择目标")
        if self.scene is None or target not in self.scene.instance_ids:
            raise RuntimeError("目标实例不在当前感知场景")
        if self.active_task is not None and self.active_task != (int(category), int(region)):
            raise RuntimeError("继续原目标时不能改变类别或功能区；请先重置任务")
        self.pending_task = (int(target), int(category), int(region))
        self.active_task = (int(category), int(region))
        self.manual_session = self.prediction = None
        self.pending_push_rules = ()
        self.manual_stage = "target_grasp"
        return {
            "stage": "SELECT_TARGET",
            "target": int(target),
            "category": int(category),
            "region": int(region),
        }

    def manual_target_grasp(self):
        self._require_manual_stage("target_grasp")
        if self.pending_task is None:
            raise RuntimeError("请先选择目标")
        target, category, region = self.pending_task
        analysis = self.predictor.analyze(self.scene, target, category, region)
        analysis.timings = {**self.cycle_timings, **(analysis.timings or {})}
        physics = self._ensure_worker("physics", lambda: PhysicsClient(self.config))
        self.manual_session = DecisionSession(
            self.config.raw["workflow"], physics, self.scene, analysis
        )
        self.manual_config_signature = self._decision_config_signature()
        update = self.manual_session.check_target_grasp()
        if self.manual_session.result is not None:
            self.prediction = self._finish_manual_decision()
            self.manual_stage = "execute"
        else:
            self.manual_stage = "obstruction"
        return update

    def manual_obstruction(self):
        self._require_manual_stage("obstruction")
        if self.manual_session is None:
            raise RuntimeError("请先执行目标抓取与碰撞检查")
        if self.manual_session.result is not None:
            return {"stage": "OBSTRUCTION_INFERENCE", "skipped": True}
        update = self.manual_session.infer_obstruction()
        self.manual_stage = "push_rules" if self.manual_session.status == "running" else "stopped"
        if self.manual_stage == "stopped":
            self.prediction = self._finish_manual_decision()
        return update

    def manual_push_rules(self):
        self._require_manual_stage("push_rules")
        if self.manual_session is None:
            raise RuntimeError("请先执行压覆物体推断")
        self.pending_push_rules = (
            self.predictor.generate_push_rules(
                self.manual_session.pushable_objects,
                adjacent_objects=self.manual_session.adjacent_blockers,
            ) if self.manual_session.adjacent_blockers else
            self.predictor.generate_push_rules(self.manual_session.pushable_objects)
        )
        update = self.manual_session.record_push_rules(self.pending_push_rules)
        self.manual_stage = "push_scoring" if self.manual_session.status == "running" else "stopped"
        if self.manual_stage == "stopped":
            self.prediction = self._finish_manual_decision()
        return update

    def manual_push_scoring(self):
        self._require_manual_stage("push_scoring")
        if self.manual_session is None:
            raise RuntimeError("请先生成PUSH规则候选")
        scored = self.predictor.score_push_rules()
        update = self.manual_session.rank_pushes(scored)
        self.prediction = self._finish_manual_decision()
        self.manual_stage = "execute" if self.prediction.status == "ready" else "stopped"
        return update

    def _finish_manual_decision(self) -> Prediction:
        if self.manual_session is None:
            raise RuntimeError("手动决策会话不存在")
        prediction = self.manual_session.finish()
        if prediction.timings is not None and self.cycle_started is not None:
            prediction.timings["end_to_end_s"] = time.perf_counter() - self.cycle_started
        return prediction

    def _decision_config_signature(self):
        import json

        return json.dumps(
            {
                "workflow": self.config.raw.get("workflow", {}),
                "physics": self.config.raw.get("physics", {}),
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    def _require_manual_stage(self, expected: str) -> None:
        if self.manual_stage != expected:
            raise RuntimeError(f"当前手动阶段为 {self.manual_stage}，不能执行 {expected}")
        if (
            self.manual_session is not None
            and self.manual_config_signature != self._decision_config_signature()
        ):
            self.manual_session = self.prediction = None
            self.manual_stage = "target_grasp"
            raise RuntimeError("决策配置已变化，下游结果已失效；请从目标抓取阶段重新执行")

    def _analyze_and_decide(self, target, category: int, region: int, progress=None) -> Prediction:
        """Single decision module used by initial, continuation, manual and auto callers."""
        self.load_model()
        predictor = self.predictor
        if predictor is None:
            raise RuntimeError("模型工作进程不可用")
        try:
            if hasattr(predictor, "analyze"):
                analysis = predictor.analyze(
                    self.scene,
                    target,
                    category,
                    region,
                    progress=(
                        None
                        if progress is None
                        else lambda update: progress({**update, "target_query": target})
                    ),
                )
                analysis.timings = {**self.cycle_timings, **(analysis.timings or {})}
                physics = self._ensure_worker("physics", lambda: PhysicsClient(self.config))
                session = DecisionSession(
                    self.config.raw.get("workflow", {}), physics, self.scene, analysis
                )
                for method in (session.check_target_grasp, session.infer_obstruction):
                    if session.result is not None or session.status != "running":
                        break
                    update = method()
                    update["target_query"] = session.target
                    if progress is not None:
                        progress(update)
                if session.result is None and session.status == "running":
                    if hasattr(predictor, "generate_push_rules") and session.adjacent_blockers:
                        rules = predictor.generate_push_rules(
                            session.pushable_objects,
                            adjacent_objects=session.adjacent_blockers,
                        )
                    elif hasattr(predictor, "generate_push_rules"):
                        rules = predictor.generate_push_rules(session.pushable_objects)
                    else:
                        rules = tuple(
                            item for item in analysis.candidates
                            if int(item.get("action_type", -1)) == 0
                        )
                    update = session.record_push_rules(rules)
                    update["target_query"] = session.target
                    if progress is not None:
                        progress(update)
                if session.result is None and session.status == "running":
                    scored = (
                        predictor.score_push_rules()
                        if hasattr(predictor, "score_push_rules") else analysis.candidates
                    )
                    update = session.rank_pushes(scored)
                    update["target_query"] = session.target
                    if progress is not None:
                        progress(update)
                prediction = session.finish()
            else:
                prediction = predictor.predict(self.scene, target, category, region)
        except RuntimeError as error:
            if target is not None or "target query" not in str(error):
                raise
            self.stop_reason = f"目标丢失或身份不确定：{error}"
            prediction = Prediction({"reason": self.stop_reason}, 0.0, status="operator_attention")
        self.prediction = prediction
        self.active_task = (int(category), int(region))
        if prediction.timings is not None and self.cycle_started is not None:
            prediction.timings["end_to_end_s"] = time.perf_counter() - self.cycle_started
        workflow = self.config.raw.get("workflow", {})
        if prediction.status == "ready" and similar_push(
            self.last_executed_action,
            prediction.action,
            float(workflow.get("repeated_action_contact_threshold_m", 0.02)),
            float(workflow.get("repeated_action_direction_deg", 15)),
        ):
            prediction = Prediction(
                {"reason": "连续选择了同一物体的相近PUSH动作"},
                prediction.inference_seconds,
                prediction.candidates,
                prediction.timings,
                prediction.decision_path,
                prediction.target_query,
                status="operator_attention",
            )
            self.prediction = prediction
        return prediction

    def predict_with_progress(
        self, target: int, category: int, region: int, progress
    ) -> Prediction:
        """Automatic decision with observable stage boundaries."""
        if self.scene is None:
            raise RuntimeError("请先采集并感知场景")
        if self.stop_reason:
            return Prediction({"reason": self.stop_reason}, 0.0, status="operator_attention")
        continued = self.active_task is not None
        model_target = None if continued else target
        progress(
            {"stage": "SELECT_TARGET", "target_query": target, "message": "已确认目标与任务功能区"}
        )
        progress({"stage": "TARGET_GRASP_MODEL", "message": "正在生成目标抓取候选"})
        started = time.perf_counter()
        result = self._analyze_and_decide(model_target, category, region, progress)
        progress(
            {
                "stage": "TARGET_GRASP_MODEL",
                "target_query": result.target_query,
                "message": f"模型分析完成，用时 {time.perf_counter() - started:.3f} s，共生成 {len(result.candidates)} 个候选",
                "total_candidate_count": len(result.candidates),
            }
        )
        return result

    def predict(self, target: int, category: int, region: int):
        if self.task_finished:
            raise RuntimeError("请确认上次执行结果并重置任务，再选择目标")
        if self.active_task is not None:
            return self.predict_continue(category, region)
        if self.scene is None:
            raise RuntimeError("请先采集并融合点云")
        if self.stop_reason:
            return Prediction({"reason": self.stop_reason}, 0.0, status="operator_attention")
        if target not in self.scene.instance_ids:
            raise RuntimeError("目标实例不在当前预测场景")
        return self._analyze_and_decide(target, category, region)

    def predict_continue(self, category: int, region: int):
        """Continue the previously prompted physical target after re-observation."""
        if self.scene is None:
            raise RuntimeError("请先重新采集并融合点云")
        if self.task_finished or self.active_task is None:
            raise RuntimeError("没有可继续的目标任务，请重置并选择目标")
        if self.active_task != (int(category), int(region)):
            raise RuntimeError("继续原目标时不能改变类别或功能区；请先重置任务")
        return self._analyze_and_decide(None, category, region)

    def execute(self, target: int, category: int, region: int) -> str:
        if self.prediction is None:
            raise RuntimeError("没有待执行的预测动作")
        if not self.robot_connected:
            self.prediction.status = "prediction_only"
            return "机械臂未连接：动作已标记为仅预测完成，可载入下一场景继续测试"
        if self.prediction.status == "operator_attention":
            raise RuntimeError(self.prediction.action.get("reason", "需要人工处理"))
        if not self.robot_enabled:
            raise RuntimeError("FR5 未使能，禁止执行动作")
        prediction = self.prediction
        self.last_scene_xyz = None if self.scene is None else self.scene.xyz_m.copy()
        self.prediction = None
        self.scene = None
        try:
            self.robot.execute(prediction.action)
            self.last_executed_action = dict(prediction.action)
        except Exception:
            self.task_finished = True
            raise
        from tcd_prg.constants import ActionType

        is_task_grasp = int(prediction.action["action_type"]) == int(ActionType.TASK_GRASP)
        if is_task_grasp:
            self.pending_task_grasp = dict(prediction.action)
            self.task_finished = False
            return "目标抓取动作已执行，必须确认抓取结果后才能结束或继续任务"
        self.predictor.action_executed(prediction.action)
        self.task_finished = False
        return "动作执行完成；请重新采集场景"

    def confirm_task_grasp(self, succeeded: bool) -> str:
        """Commit task completion only after the physical result is confirmed."""
        if self.pending_task_grasp is None:
            raise RuntimeError("没有等待确认的目标抓取动作")
        action = self.pending_task_grasp
        self.pending_task_grasp = None
        self.pending_push_rules = ()
        if succeeded:
            self.predictor.action_executed(action)
            self.task_finished = True
            return "目标抓取已确认成功；任务结束，开始新任务前请重置"
        self.task_finished = False
        return "目标抓取未成功；请重新采集场景并继续原任务"

    def reset_task(self) -> None:
        if self.predictor is not None:
            self.predictor.reset()
        self.prediction = None
        self.active_task = None
        self.task_finished = False
        self.last_scene_xyz = None
        self.no_change_count = 0
        self.last_executed_action = None
        self.stop_reason = None
        self.cycle_started = None
        self.cycle_timings = {}
        self.manual_session = None
        self.pending_task = None
        self.pending_push_rules = ()
        self.manual_stage = "capture"
        self.manual_config_signature = None
        self.pending_task_grasp = None

    def stop(self) -> None:
        self.cancel_background("操作者点击停止")
        if self.robot_connected:
            self.robot.stop()
        self.robot_enabled = False
        self.robot_paused = False
        self.prediction = None

    def cancel_background(self, reason: str) -> None:
        """Cancel isolated workers so an in-flight automatic cycle cannot advance."""
        self.stop_reason = reason
        with self._worker_lock:
            self._worker_epoch += 1
            workers = (self.physics, self.predictor)
            self.physics = self.predictor = None
        for worker in workers:
            if worker is not None:
                try:
                    worker.close()
                except Exception:
                    pass

    def initialize_gripper(self) -> str:
        controller = getattr(self.robot, "controller", None)
        if controller is None:
            raise RuntimeError("当前机器人后端不支持夹爪初始化")
        error = controller.initialize_gripper()
        if error:
            raise RuntimeError(f"AG-160-95 初始化失败，错误码: {error}")
        return "AG-160-95 夹爪初始化完成"

    def open_gripper(self) -> str:
        controller = getattr(self.robot, "controller", None)
        if controller is None:
            raise RuntimeError("当前机器人后端不支持夹爪控制")
        error = controller.gripper_open()
        if error:
            raise RuntimeError(f"AG-160-95 打开失败，错误码: {error}")
        return "AG-160-95 夹爪已打开"

    def close_gripper(self) -> str:
        controller = getattr(self.robot, "controller", None)
        if controller is None:
            raise RuntimeError("当前机器人后端不支持夹爪控制")
        error = controller.gripper_close()
        if error:
            raise RuntimeError(f"AG-160-95 闭合失败，错误码: {error}")
        return "AG-160-95 夹爪已闭合"

    def set_tcp_compensation(self, xyz_mm_rpy_deg) -> str:
        transform = xyz_rpy_to_matrix(xyz_mm_rpy_deg, 0.001)
        if hasattr(self.robot, "tcp_transform"):
            self.robot.tcp_transform = transform
        self.config.raw["robot"]["model_tcp_to_robot_tcp"] = {
            "xyz_mm_rpy_deg": [float(x) for x in xyz_mm_rpy_deg]
        }
        self.config.save()
        return "TCP补偿已应用并保存到配置文件"

    def close(self) -> None:
        with self._worker_lock:
            self._closed = True
        self.cancel_background("窗口关闭")
        for camera in self.cameras:
            try:
                camera.disconnect()
            except Exception:
                pass
        try:
            self.robot.disconnect()
        except Exception:
            pass
        self.connected = False
        self.cameras_connected = False
        self.robot_connected = False
        self.robot_enabled = False
        self.robot_paused = False
