from __future__ import annotations

from typing import Callable

from .camera import build_cameras
from .perception import build_segmenter, fuse_frames
from .predictor_client import PredictorClient
from .robot import build_robot
from .transforms import xyz_rpy_to_matrix
import numpy as np


def preserve_instance_ids(previous, current, maximum_distance_m: float):
    """Greedy one-to-one centroid tracking between consecutive observations."""
    if previous is None: return current
    prior_centres={i:previous.xyz_m[previous.instance_id==i].mean(0)
                   for i in previous.instance_ids}
    current_centres={i:current.xyz_m[current.instance_id==i].mean(0)
                     for i in current.instance_ids}
    pairs=[]
    for old,old_centre in prior_centres.items():
        for new,new_centre in current_centres.items():
            pairs.append((float(np.linalg.norm(old_centre-new_centre)),old,new))
    assigned_old=set();assigned_new=set();mapping={}
    for distance,old,new in sorted(pairs):
        if distance<=maximum_distance_m and old not in assigned_old and new not in assigned_new:
            mapping[new]=old;assigned_old.add(old);assigned_new.add(new)
    next_id=max(previous.instance_ids,default=-1)+1
    for new in current.instance_ids:
        if new not in mapping: mapping[new]=next_id;next_id+=1
    original=current.instance_id.copy()
    current.instance_id=np.asarray([mapping[int(value)] for value in original],np.int64)
    current.category_by_instance={mapping[key]:value for key,value in current.category_by_instance.items()}
    return current


class ExperimentController:
    def __init__(self, config):
        self.config = config
        self.cameras = build_cameras(config)
        self.segmenter = build_segmenter(config)
        self.robot = build_robot(config)
        self.predictor = None
        self.scene = None
        self.prediction = None
        self.connected = False

    def connect(self) -> str:
        connected = []
        try:
            for camera in self.cameras:
                if not camera.connect(): raise RuntimeError(f"Camera {camera.camera_id} failed")
                connected.append(camera)
            if not self.robot.connect(): raise RuntimeError("FR5 connection failed")
        except Exception:
            for camera in connected: camera.disconnect()
            raise
        self.connected = True
        return f"已连接 {len(self.cameras)} 台相机和机械臂"

    def load_model(self, progress: Callable[[str],None] | None = None) -> str:
        if self.predictor is None:
            if progress: progress("正在加载TCD-PRG模型……")
            self.predictor = PredictorClient(self.config.path)
        return "TCD-PRG模型已加载"

    def acquire(self):
        if not self.connected: raise RuntimeError("请先连接设备")
        frames = [camera.capture() for camera in self.cameras]
        segments = [self.segmenter.segment(frame) for frame in frames]
        updated = fuse_frames(frames, segments, self.config.raw["fusion"])
        self.scene = preserve_instance_ids(
            self.scene, updated,
            float(self.config.raw["fusion"].get("temporal_instance_distance_m",.10)))
        self.prediction = None
        return self.scene

    def predict(self, target: int, category: int, region: int):
        if self.scene is None: raise RuntimeError("请先采集并融合点云")
        self.load_model()
        if target not in self.scene.instance_ids: raise RuntimeError("目标实例不在当前场景")
        required = int(self.config.raw["task"].get("required_grasp_count",1))
        self.prediction = self.predictor.predict(self.scene,target,category,region,required)
        return self.prediction

    def execute(self, target: int, category: int, region: int) -> str:
        if self.prediction is None: raise RuntimeError("没有待执行的预测动作")
        self.robot.execute(self.prediction.action)
        self.predictor.action_executed(self.prediction.action)
        self.prediction = None
        return "动作执行完成；请重新采集场景"

    def reset_task(self) -> None:
        if self.predictor is not None: self.predictor.reset()
        self.prediction = None

    def stop(self) -> None: self.robot.stop()

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
        transform = xyz_rpy_to_matrix(xyz_mm_rpy_deg, .001)
        if hasattr(self.robot, "tcp_transform"):
            self.robot.tcp_transform = transform
        self.config.raw["robot"]["model_tcp_to_robot_tcp"] = {
            "xyz_mm_rpy_deg": [float(x) for x in xyz_mm_rpy_deg]}
        return "TCP补偿已应用到后续动作"

    def close(self) -> None:
        for camera in self.cameras:
            try: camera.disconnect()
            except Exception: pass
        try: self.robot.disconnect()
        except Exception: pass
        if self.predictor is not None:
            try: self.predictor.close()
            except Exception: pass
        self.connected = False
