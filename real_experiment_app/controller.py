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
        return f"已连接 {len(self.cameras)} 台相机和机械臂"

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
        frames = [camera.capture() for camera in self.cameras]
        updated = fuse_frames(frames, None, self.config.raw["fusion"])
        self.load_model()
        updated = self.predictor.perceive(updated)
        self.scene = updated
        self.prediction = None
        return self.scene

    def predict(self, target: int, category: int, region: int):
        if self.scene is None:
            raise RuntimeError("请先采集并融合点云")
        self.load_model()
        if target not in self.scene.instance_ids:
            raise RuntimeError("目标实例不在当前预测场景")
        self.prediction = self.predictor.predict(
            self.scene, target, category, region
        )
        return self.prediction

    def predict_continue(self, category: int, region: int):
        """Continue the previously prompted physical target after re-observation."""
        if self.scene is None:
            raise RuntimeError("请先重新采集并融合点云")
        self.load_model()
        self.prediction = self.predictor.predict(
            self.scene, None, category, region
        )
        return self.prediction

    def execute(self, target: int, category: int, region: int) -> str:
        if self.prediction is None:
            raise RuntimeError("没有待执行的预测动作")
        self.robot.execute(self.prediction.action)
        self.predictor.action_executed(
            self.prediction.action
        )
        self.prediction = None
        return "动作执行完成；请重新采集场景"

    def reset_task(self) -> None:
        if self.predictor is not None:
            self.predictor.reset()
        self.prediction = None

    def stop(self) -> None:
        self.robot.stop()

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
