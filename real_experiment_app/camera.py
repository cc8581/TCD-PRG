from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Protocol

import numpy as np

from .types import RGBDFrame


class CameraSource(Protocol):
    camera_id: str
    def connect(self) -> bool: ...
    def capture(self) -> RGBDFrame: ...
    def disconnect(self) -> None: ...


class LegacyMechEyeCamera:
    """Adapter over the previously validated Mech-Eye camera implementation."""

    def __init__(self, camera_id: str, ip: str, camera_to_base, model_view_index: int,
                 sdk_windows_root: Path):
        import sys
        root = str(sdk_windows_root.resolve())
        if root not in sys.path: sys.path.insert(0, root)
        self.camera_id = camera_id
        self._ip = ip
        self._camera = None
        self.model_view_index = int(model_view_index)
        self._transform = (None if camera_to_base is None
                           else np.asarray(camera_to_base, np.float64))

    def connect(self) -> bool:
        if self._transform is None:
            raise RuntimeError(f"{self.camera_id}: 尚未配置相机到 FR5 基座的外参")
        try:
            from grasp_system.vision.camera import MechEyeCamera
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "未找到 Mech-Eye Python SDK，请安装与相机固件匹配的 mecheye 包"
            ) from error
        self._camera = MechEyeCamera(self._ip)
        return bool(self._camera.connect())

    def capture(self) -> RGBDFrame:
        if self._camera is None:
            raise RuntimeError(f"{self.camera_id}: 相机尚未连接")
        color, depth = self._camera.get_frame()
        if color is None or depth is None:
            raise RuntimeError(f"{self.camera_id}: capture failed")
        return RGBDFrame(self.camera_id, np.asarray(color, np.uint8),
                         np.asarray(depth, np.float32), self._camera.get_intrinsics(),
                         self._transform.copy(), self.model_view_index)

    def disconnect(self) -> None:
        if self._camera is not None:
            self._camera.disconnect()
            self._camera = None


def build_cameras(config) -> list[CameraSource]:
    entries = config.raw.get("cameras", [])
    enabled = [x for x in entries if x.get("enabled", True)]
    if not enabled:
        raise ValueError("至少需要启用一台 Mech-Eye 相机")
    view_indices = [int(item.get("model_view_index", index)) for index, item in enumerate(enabled)]
    if len(view_indices) != len(set(view_indices)) or any(index < 0 for index in view_indices):
        raise ValueError("启用相机的模型视角编号必须是互不重复的非负整数")
    root = config.resolve(config.raw["robot"]["sdk_windows_root"])
    return [LegacyMechEyeCamera(str(item["id"]), str(item["ip"]),
                               item.get("camera_to_robot_base"),
                               int(item.get("model_view_index", index)), root)
            for index, item in enumerate(enabled)]
