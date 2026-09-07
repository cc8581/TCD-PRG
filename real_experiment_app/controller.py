from __future__ import annotations

from typing import Callable

from .camera import build_cameras
from .perception import fuse_frames
from .predictor_client import PredictorClient
from .robot import build_robot
from .transforms import xyz_rpy_to_matrix


class ExperimentController:
    def __init__(self, config):
        self.config = config
        self.cameras = build_cameras(config)
        self.robot = build_robot(config)
        self.predictor = None
        self.scene = None
        self.prediction = None
        self.connected = False
        self.robot_enabled = False
        self.robot_paused = False
        self.active_task = None
        self.task_finished = False

    def connect(self) -> str:
        connected = []
        try:
            for camera in self.cameras:
                if not camera.connect():
                    raise RuntimeError(f"Camera {camera.camera_id} failed")
                connected.append(camera)
            if not self.robot.connect():
                raise RuntimeError("FR5 connection failed")
        except Exception:
            for camera in connected:
                camera.disconnect()
            raise
        self.connected = True
        self.robot_enabled = False
        return f"已连接 {len(self.cameras)} 台相机和机械臂；FR5 当前未使能"

    def enable_robot(self) -> str:
        if not self.connected: raise RuntimeError("请先连接设备")
        self.robot.enable(); self.robot_enabled = True
        self.robot_paused = False
        return "FR5 已切换自动模式并使能"

    def disable_robot(self) -> str:
        if not self.connected: raise RuntimeError("请先连接设备")
        self.robot.disable(); self.robot_enabled = False
        self.robot_paused = False
        self.prediction = None
        return "FR5 已下使能；待执行动作已清除"

    def clear_errors(self) -> str:
        if not self.connected: raise RuntimeError("请先连接设备")
        self.robot.clear_errors()
        return "FR5 控制器故障已清除；确认现场安全后再使能"

    def pause_robot(self) -> str:
        if not self.robot_enabled: raise RuntimeError("FR5 未使能")
        self.robot.pause(); self.robot_paused = True
        return "FR5 运动已暂停"

    def resume_robot(self) -> str:
        if not self.robot_enabled: raise RuntimeError("FR5 未使能")
        if not self.robot_paused: raise RuntimeError("FR5 当前没有暂停的运动")
        self.robot.resume(); self.robot_paused = False
        return "FR5 已继续运动"

    def home_robot(self) -> str:
        if not self.robot_enabled: raise RuntimeError("FR5 未使能")
        self.prediction = None
        self.robot.home()
        self.scene = None
        return "FR5 已回安全原点；场景已失效，请重新采集"

    def robot_status(self) -> str:
        state = self.robot.status()
        self.robot_enabled = bool(state.get("enabled", 0))
        joints = state.get("joints_deg")
        return (f"使能={state.get('enabled')}  急停={state.get('emergency_stop')}  "
                f"运动完成={state.get('motion_done')}  队列={state.get('queue_len')}  "
                f"故障={state.get('main_code')}/{state.get('sub_code')}\n"
                f"关节角={None if joints is None else [round(float(x), 2) for x in joints]}")

    def load_model(self, progress: Callable[[str], None] | None = None) -> str:
        if self.predictor is None:
            if progress:
                progress("正在加载TCD-PRG模型……")
            self.predictor = PredictorClient(self.config.path)
        return "TCD-PRG模型已加载"

    def acquire(self):
        """Capture -> raw fuse -> integrated TCD-PRG instance perception."""
        if not self.connected:
            raise RuntimeError("请先连接设备")
        self.scene = None
        self.prediction = None
        frames = [camera.capture() for camera in self.cameras]
        updated = fuse_frames(frames, None, self.config.raw["fusion"])
        self.load_model()
        updated = self.predictor.perceive(updated)
        self.scene = updated
        self.prediction = None
        return self.scene

    def predict(self, target: int, category: int, region: int):
        if self.task_finished:
            raise RuntimeError('请确认上次执行结果并重置任务，再选择目标')
        if self.active_task is not None:
            return self.predict_continue(category, region)
        if self.scene is None:
            raise RuntimeError("请先采集并融合点云")
        self.load_model()
        if target not in self.scene.instance_ids:
            raise RuntimeError("目标实例不在当前预测场景")
        self.prediction = None
        self.prediction = self.predictor.predict(
            self.scene, target, category, region
        )
        self.active_task = (int(category), int(region))
        return self.prediction

    def predict_continue(self, category: int, region: int):
        """Continue the previously prompted physical target after re-observation."""
        if self.scene is None:
            raise RuntimeError("请先重新采集并融合点云")
        if self.task_finished or self.active_task is None:
            raise RuntimeError('没有可继续的目标任务，请重置并选择目标')
        if self.active_task != (int(category), int(region)):
            raise RuntimeError('继续原目标时不能改变类别或功能区；请先重置任务')
        self.load_model()
        self.prediction = None
        self.prediction = self.predictor.predict(
            self.scene, None, category, region
        )
        return self.prediction

    def execute(self, target: int, category: int, region: int) -> str:
        if self.prediction is None:
            raise RuntimeError("没有待执行的预测动作")
        if not self.connected:
            raise RuntimeError('请先连接设备')
        if not self.robot_enabled:
            raise RuntimeError('FR5 未使能，禁止执行动作')
        prediction = self.prediction
        self.prediction = None
        self.scene = None
        try:
            self.robot.execute(prediction.action)
            self.predictor.action_executed(prediction.action)
        except Exception:
            self.task_finished = True
            raise
        from tcd_prg.constants import ActionType
        self.task_finished = int(prediction.action['action_type']) == int(ActionType.TASK_GRASP)
        if self.task_finished:
            return '目标抓取动作已执行，请检查实物结果；开始新任务前请重置'
        return "动作执行完成；请重新采集场景"

    def reset_task(self) -> None:
        if self.predictor is not None:
            self.predictor.reset()
        self.prediction = None
        self.active_task = None
        self.task_finished = False

    def stop(self) -> None:
        self.robot.stop()
        self.robot_enabled = False
        self.robot_paused = False
        self.prediction = None

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
        return "TCP补偿已应用到后续动作"

    def close(self) -> None:
        for camera in self.cameras:
            try:
                camera.disconnect()
            except Exception:
                pass
        try:
            self.robot.disconnect()
        except Exception:
            pass
        if self.predictor is not None:
            try:
                self.predictor.close()
            except Exception:
                pass
        self.connected = False
        self.robot_enabled = False
        self.robot_paused = False
