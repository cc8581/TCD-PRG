# ruff: noqa: E501
from __future__ import annotations

import argparse
import ipaddress
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .cloud_view import OpenGLCloudView
from .config import AppConfig
from .controller import ExperimentController
from .grasp_safety import grasp_local_z_is_downward_or_level
from .workflow import reachable_obstructors
from .workflow_view import COMMON_STAGES, WorkflowView

ACTION_NAMES = {0: "PUSH", 2: "TASK_GRASP"}

APP_STYLE = """
QWidget {
    background: #0b1220;
    color: #e7edf6;
    font-size: 15px;
}
QLabel { background: transparent; }
QMainWindow { background: #0b1220; }
QFrame#TopBar {
    background: #101a2b;
    border-bottom: 1px solid #24324a;
}
QLabel#Brand { color: #f5f8fc; font-size: 21px; font-weight: 600; }
QLabel#SectionTitle { color: #f1f5fb; font-size: 15px; font-weight: 600; }
QLabel#SectionHint { color: #8b99ad; font-size: 13px; }
QLabel#ModeLabel { color: #8796ab; font-size: 14px; font-weight: 600; }
QLabel#ModeLabel[active="true"] { color: #77c8ff; }
QSlider#ModeSwitch { min-width: 76px; max-width: 76px; min-height: 30px; }
QSlider#ModeSwitch::groove:horizontal {
    height: 28px; border-radius: 14px; background: #24364f; border: 1px solid #45617f;
}
QSlider#ModeSwitch::handle:horizontal {
    width: 30px; margin: -3px 0; border-radius: 15px;
    background: #77c8ff; border: 1px solid #b3e3ff;
}
QSlider#ModeSwitch:disabled::handle:horizontal { background: #66788f; border-color: #66788f; }
QLabel#StatusBadge {
    background: #172235;
    border: 1px solid #324158;
    border-radius: 14px;
    padding: 4px 10px;
    color: #aab6c8;
}
QLabel#StatusBadge[state="ready"] { color: #73dfb2; border-color: #25674f; background: #102c27; }
QLabel#StatusBadge[state="busy"] { color: #77c8ff; border-color: #285b82; background: #10283c; }
QLabel#StatusBadge[state="error"] { color: #ff929f; border-color: #773444; background: #321923; }
QFrame#Card {
    background: #111b2b;
    border: 1px solid #223149;
    border-radius: 12px;
}
QFrame#ViewportCard {
    background: #0f1928;
    border: 1px solid #223149;
    border-radius: 14px;
}
QLabel#WorkflowNode {
    background: #142238; border: 1px solid #34445c;
    border-radius: 6px; padding: 5px 7px; color: #b9c7d9;
}
QLabel#WorkflowNode[state="success"] { background: #163e35; border-color: #47bd91; color: #d6fff0; }
QLabel#WorkflowNode[state="running"] { background: #50391b; border-color: #ffd36a; color: #fff0c5; }
QLabel#WorkflowNode[state="failed"] { background: #4b2330; border-color: #ff8290; color: #ffe1e5; }
QLabel#WorkflowNode[state="inactive"] { color: #64728a; border-color: #263449; }
QFrame#WorkflowBranch { background: #0e1929; border: 1px solid #2b3b53; border-radius: 8px; }
QFrame#WorkflowBranch[active="true"] { border-color: #52a9ed; background: #142942; }
QLabel#Metric {
    color: #92a0b5;
    background: #0d1624;
    border: 1px solid #1f2c41;
    border-radius: 8px;
    padding: 5px 9px;
    font-size: 13px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
    background: #0c1522;
    color: #e8eef7;
    border: 1px solid #2b3a52;
    border-radius: 7px;
    padding: 5px 8px;
    selection-background-color: #176ca0;
}
QLineEdit {
    background: #0c1522;
    color: #e8eef7;
    border: 1px solid #2b3a52;
    border-radius: 7px;
    padding: 5px 8px;
}
QCheckBox { background: transparent; spacing: 7px; color: #c8d2e1; }
QDialog { background: #0b1220; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QTextEdit:hover { border-color: #46617f; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus { border-color: #35a7e8; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #111b2b;
    color: #e7edf6;
    border: 1px solid #31425e;
    selection-background-color: #176ca0;
}
QTextEdit { padding: 8px; }
QPushButton {
    min-height: 29px;
    background: #18253a;
    color: #dbe5f2;
    border: 1px solid #31425e;
    border-radius: 8px;
    padding: 0 11px;
    font-weight: 500;
}
QPushButton:hover { background: #255079; border-color: #65bdf0; color: #ffffff; }
QPushButton:pressed { background: #0d6fa8; border: 2px solid #a8ddff; padding: 0 10px; }
QPushButton:focus { border: 2px solid #56bff5; color: #ffffff; }
QPushButton:disabled { color: #718097; background: #111a28; border-color: #263349; }
QPushButton[role="primary"], QPushButton[role="execute"], QPushButton[role="danger"], QPushButton[role="ghost"] {
    background: #18253a; border-color: #31425e; color: #dbe5f2;
}
QPushButton[role="primary"]:hover, QPushButton[role="execute"]:hover,
QPushButton[role="danger"]:hover, QPushButton[role="ghost"]:hover {
    background: #255079; border-color: #65bdf0; color: #ffffff;
}
QPushButton[role="primary"]:pressed, QPushButton[role="execute"]:pressed,
QPushButton[role="danger"]:pressed, QPushButton[role="ghost"]:pressed {
    background: #0d6fa8; border: 2px solid #a8ddff;
}
QPushButton[role="primary"]:focus, QPushButton[role="execute"]:focus,
QPushButton[role="danger"]:focus, QPushButton[role="ghost"]:focus {
    border: 2px solid #56bff5; color: #ffffff;
}
QPushButton#TcpToggle { min-height: 25px; text-align: left; background: transparent; border: none; padding: 0; color: #aab6c8; }
QPushButton#RunCompact { min-height: 28px; padding: 0 6px; font-size: 13px; }
QPushButton#SimFlowButton { min-height: 30px; max-width: 210px; padding: 0 9px; }
QPushButton#SimFlowButton[workflow_state="success"] {
    background: #1d6147; border-color: #64d6a3; color: #effff7;
}
QPushButton#SimFlowButton[workflow_state="running"] {
    background: #2d291d; border: 2px solid #ffd35a; color: #fff2c7;
}
QPushButton#SimFlowButton[workflow_state="running"][pulse="true"] {
    border-color: #fff4a5; background: #51431c;
}
QComboBox#CompactTargetInstance { min-height: 26px; max-height: 26px; font-size: 12px; padding: 0 5px; }
QSpinBox#CompactTargetValue { min-height: 26px; max-height: 26px; font-size: 12px; padding: 0 3px; }
QLabel#CompactTargetLabel { color: #b6c4d5; font-size: 12px; font-weight: 600; }
QLabel#FlowArrow { color: #7b9aba; font-size: 18px; font-weight: 700; min-width: 18px; }
QLabel#SimBranchLabel { color: #91a6c0; font-size: 13px; font-weight: 600; }
QLabel#SimBranchLabel[active="true"] { color: #78dbad; }
QDockWidget { color: #e7edf6; font-weight: 600; }
QDockWidget::title { background: #101a2b; border: 1px solid #263750; padding: 7px 10px; text-align: left; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 4px 0; }
QScrollBar::handle:vertical { background: #2d3d55; min-height: 36px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background: transparent; width: 8px; }
QToolTip { color: #eff5fc; background: #172337; border: 1px solid #40536f; padding: 6px; }
"""


def preferred_ui_font() -> str:
    # Some bundled Qt runtimes do not enumerate Windows' registered CJK fonts.
    # Loading the font file explicitly keeps Chinese text reliable in those runtimes.
    for font_path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
        if font_path.exists():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
            if families:
                return families[0]
    available = set(QFontDatabase.families())
    for family in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Segoe UI"):
        if family in available:
            return family
    return QApplication.font().family()


def refresh_style(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class WorkerSignals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(object)


class Job(QRunnable):
    def __init__(self, function: Callable, with_progress: bool = False):
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()
        self.with_progress = with_progress

    @Slot()
    def run(self):
        try:
            self.signals.done.emit(
                self.function(self.signals.progress.emit) if self.with_progress else self.function()
            )
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class SectionCard(QFrame):
    def __init__(self, title: str, hint: str = ""):
        super().__init__()
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(13, 10, 13, 12)
        self.body.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("SectionTitle")
        self.body.addWidget(heading)
        if hint:
            note = QLabel(hint)
            note.setObjectName("SectionHint")
            note.setWordWrap(True)
            self.body.addWidget(note)


class DeviceSettingsDialog(QDialog):
    """Editable, persisted production device settings."""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("真机实验参数设置")
        self.resize(860, 820)
        self.setMinimumWidth(760)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(9)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        cards = QVBoxLayout(content)
        cards.setContentsMargins(4, 4, 4, 4)
        cards.setSpacing(9)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        model = config.raw["tcd_prg"]
        self.model_fields = {}
        model_card = SectionCard("A / B / C 模型", "相对路径以本配置文件所在目录为基准")
        model_form = QFormLayout()
        for label, key in (
            ("模型配置", "config"), ("路径配置", "paths_config"),
            ("阶段 A 权重", "perception_checkpoint"), ("阶段 B 权重", "grasp_checkpoint"),
            ("阶段 C 权重", "push_evaluator_checkpoint"),
        ):
            field = QLineEdit(str(model.get(key, "")))
            self.model_fields[key] = field
            model_form.addRow(label, field)
        self.model_device = QComboBox()
        self.model_device.addItems(["cuda", "cpu"])
        self.model_device.setCurrentText(str(model.get("device", "cuda")))
        model_form.addRow("推理设备", self.model_device)
        model_card.body.addLayout(model_form)
        cards.addWidget(model_card)

        robot = config.raw["robot"]
        self.robot_ip = QLineEdit(str(robot.get("ip", "192.168.58.2")))
        self.robot_ip.setPlaceholderText("例如 192.168.58.2")
        self.tool_id = self._spin(0, 14, int(robot.get("tool_id", 1)))
        self.user_id = self._spin(0, 14, int(robot.get("user_id", 0)))
        self.speed = self._spin(1, 100, int(robot.get("speed_percent", 10)))
        robot_card = SectionCard("FR5 机械臂")
        robot_form = QFormLayout()
        robot_form.setVerticalSpacing(9)
        robot_form.addRow("控制器 IP", self.robot_ip)
        robot_form.addRow("工具坐标系", self.tool_id)
        robot_form.addRow("工件坐标系", self.user_id)
        robot_form.addRow("运动速度 / %", self.speed)
        robot_card.body.addLayout(robot_form)
        cards.addWidget(robot_card)

        self.camera_rows = []
        camera_card = SectionCard("Mech-Eye 相机")
        camera_grid = QGridLayout()
        camera_grid.setHorizontalSpacing(10)
        camera_grid.addWidget(QLabel("启用"), 0, 0)
        camera_grid.addWidget(QLabel("相机 ID"), 0, 1)
        camera_grid.addWidget(QLabel("IP 地址"), 0, 2)
        camera_grid.addWidget(QLabel("模型视角"), 0, 3)
        camera_grid.addWidget(QLabel("相机到基座外参（4×4，行优先）"), 0, 4)
        for row, item in enumerate(config.raw.get("cameras", []), start=1):
            enabled = QCheckBox()
            enabled.setChecked(bool(item.get("enabled", True)))
            camera_id = QLineEdit(str(item.get("id", f"camera_{row-1}")))
            ip = QLineEdit(str(item.get("ip", "")))
            ip.setPlaceholderText("例如 192.168.3.100")
            matrix = QLineEdit(self._format_numbers(item.get("camera_to_robot_base")))
            matrix.setPlaceholderText("16 个数，例如 1,0,0,0,...,0,0,0,1")
            model_view = self._spin(0, 31, int(item.get("model_view_index", row - 1)))
            camera_grid.addWidget(enabled, row, 0)
            camera_grid.addWidget(camera_id, row, 1)
            camera_grid.addWidget(ip, row, 2)
            camera_grid.addWidget(model_view, row, 3)
            camera_grid.addWidget(matrix, row, 4)
            self.camera_rows.append((enabled, camera_id, ip, model_view, matrix, item))
        camera_card.body.addLayout(camera_grid)
        cards.addWidget(camera_card)

        fusion = config.raw["fusion"]
        fusion_card = SectionCard("点云融合与场景标定", "坐标单位为米；桌面法向量必须朝向工作空间")
        fusion_form = QFormLayout()
        self.workspace_min = QLineEdit(self._format_numbers(fusion.get("workspace_min_m")))
        self.workspace_max = QLineEdit(self._format_numbers(fusion.get("workspace_max_m")))
        plane = fusion.get("table_plane_base") or {}
        self.table_normal = QLineEdit(self._format_numbers(plane.get("normal")))
        self.table_offset = self._double(-5, 5, float(plane.get("offset_m", 0)), 6)
        self.table_clearance = self._double(0, .03, float(fusion.get("table_clearance_m", .003)), 4)
        self.voxel_size = self._double(0, .05, float(fusion.get("voxel_size_m", .005)), 4)
        self.depth_min = self._spin(1, 10000, int(fusion.get("depth_min_mm", 150)))
        self.depth_max = self._spin(1, 10000, int(fusion.get("depth_max_mm", 1500)))
        self.scene_points = self._spin(0, 2000000, int(fusion.get("target_scene_points", 0)))
        self.association_distance = self._double(0, 1, float(
            fusion.get("instance_association_distance_m", .06)), 3)
        self.temporal_distance = self._double(0, 1, float(
            fusion.get("temporal_instance_distance_m", .10)), 3)
        for label, widget in (
            ("工作空间最小 XYZ", self.workspace_min), ("工作空间最大 XYZ", self.workspace_max),
            ("桌面法向量 XYZ", self.table_normal), ("桌面平面 offset", self.table_offset),
            ("桌面清除距离", self.table_clearance), ("体素尺寸", self.voxel_size),
            ("最小深度 / mm", self.depth_min), ("最大深度 / mm", self.depth_max),
            ("融合点数上限（0=不限）", self.scene_points),
            ("跨视角关联距离", self.association_distance), ("跨帧重识别距离", self.temporal_distance),
        ):
            fusion_form.addRow(label, widget)
        fusion_card.body.addLayout(fusion_form)
        cards.addWidget(fusion_card)

        motion_card = SectionCard("动作与工具参数")
        motion_form = QFormLayout()
        self.motion_fields = {}
        for label, key, low, high, default in (
            ("预抓取距离 / m", "pregrasp_distance_m", 0, .5, .10),
            ("抓取抬升距离 / m", "lift_distance_m", 0, .5, .10),
            ("推动回撤距离 / m", "push_retreat_m", 0, .5, .05),
            ("夹爪最大宽度 / m", "gripper_max_width_m", 0, .3, .095),
            ("夹爪闭合余量 / m", "gripper_close_margin_m", 0, .05, .003),
        ):
            field = self._double(low, high, float(robot.get(key, default)), 4)
            self.motion_fields[key] = field
            motion_form.addRow(label, field)
        self.removal_pose = QLineEdit(self._format_numbers(robot.get("removal_pose_mm_rpy_deg")))
        motion_form.addRow("移除放置位姿 XYZRPY", self.removal_pose)
        tcp = robot.get("model_tcp_to_robot_tcp", {}).get("xyz_mm_rpy_deg", [0] * 6)
        self.tcp_pose = QLineEdit(self._format_numbers(tcp))
        motion_form.addRow("TCP 补偿 XYZRPY", self.tcp_pose)
        self.home_joints = QLineEdit(self._format_numbers(robot.get("home_joints_deg")))
        self.home_speed = self._spin(1, 30, int(robot.get("home_speed_percent", 10)))
        self.motion_workspace_min = QLineEdit(
            self._format_numbers(robot.get("motion_workspace_min_m")))
        self.motion_workspace_max = QLineEdit(
            self._format_numbers(robot.get("motion_workspace_max_m")))
        motion_form.addRow("安全原点关节角 J1～J6", self.home_joints)
        motion_form.addRow("回原点速度 / %", self.home_speed)
        motion_form.addRow("运动安全区最小 XYZ", self.motion_workspace_min)
        motion_form.addRow("运动安全区最大 XYZ", self.motion_workspace_max)
        motion_card.body.addLayout(motion_form)
        cards.addWidget(motion_card)

        gripper_card = SectionCard("AG-160-95 夹爪")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(9)
        specs = (
            ("厂商编号", "gripper_company", 1, 16, 4),
            ("夹爪编号", "gripper_index", 1, 16, 1),
            ("设备编号", "gripper_device", 0, 16, 0),
            ("总线编号", "gripper_bus", 0, 16, 0),
            ("软件版本", "gripper_softversion", 0, 100, 0),
            ("打开位置 / %", "gripper_open_position", 0, 100, 100),
            ("闭合位置 / %", "gripper_closed_position", 0, 100, 5),
            ("运动速度 / %", "gripper_speed", 1, 100, 48),
            ("闭合力 / %", "gripper_force", 1, 100, 30),
            ("打开力 / %", "gripper_open_force", 1, 100, 50),
            ("超时 / ms", "gripper_max_time_ms", 1000, 60000, 30000),
        )
        self.gripper_fields = {}
        for index, (label, key, low, high, default) in enumerate(specs):
            field = self._spin(low, high, int(robot.get(key, default)))
            self.gripper_fields[key] = field
            row, column = divmod(index, 2)
            cell = QVBoxLayout()
            caption = QLabel(label)
            caption.setObjectName("SectionHint")
            cell.addWidget(caption)
            cell.addWidget(field)
            grid.addLayout(cell, row, column)
        gripper_card.body.addLayout(grid)
        cards.addWidget(gripper_card)

        task = config.raw.setdefault("task", {})
        task_card = SectionCard("默认任务")
        task_form = QFormLayout()
        self.default_category = self._spin(0, 63, int(task.get("default_category_id", 0)))
        self.default_region = self._spin(0, 63, int(task.get("default_region_id", 0)))
        task_form.addRow("默认类别 ID", self.default_category)
        task_form.addRow("默认功能区 ID", self.default_region)
        task_card.body.addLayout(task_form)
        cards.addWidget(task_card)

        workflow = config.raw.setdefault("workflow", {})
        physics = config.raw.setdefault("physics", {})
        runtime_card = SectionCard("闭环决策与 PyBullet", "保存后下次启动自动载入")
        runtime_form = QFormLayout()
        self.grasp_top_n = self._spin(1, 256, int(workflow.get("grasp_top_n", 36)))
        self.push_top_k = self._spin(1, 256, int(workflow.get("push_top_k", 5)))
        self.push_workers = self._spin(1, 64, int(physics.get("parallel_workers", 8)))
        self.grasp_threshold = self._double(0, 1, float(
            workflow.get("grasp_confidence_threshold", .5)), 3)
        self.object_mass = self._double(.001, 100, float(physics.get("object_mass_kg", .2)), 3)
        self.object_friction = self._double(0, 5, float(physics.get("object_friction", .5)), 3)
        self.table_friction = self._double(0, 5, float(physics.get("table_friction", .6)), 3)
        self.push_speed = self._double(.001, 1, float(physics.get("push_speed_m_s", .1)), 3)
        self.overlap_margin = self._double(0, 2, float(
            workflow.get("overlap_margin_scale", .15)), 3)
        self.minimum_overlap = self._double(0, 1, float(workflow.get("minimum_xy_overlap", .05)), 3)
        self.height_gap = self._double(0, 2, float(
            workflow.get("minimum_height_gap_scale", .08)), 3)
        self.display_points = self._spin(300_000, 2_000_000, int(
            workflow.get("display_max_points", 300000)))
        self.scene_change = self._double(0, .2, float(
            workflow.get("scene_change_threshold_m", .005)), 4)
        self.repeat_contact = self._double(0, .2, float(
            workflow.get("repeated_action_contact_threshold_m", .02)), 4)
        self.repeat_direction = self._double(0, 180, float(
            workflow.get("repeated_action_direction_deg", 15)), 1)
        self.restitution = self._double(0, 1, float(physics.get("restitution", .05)), 3)
        self.convex_parts = self._spin(1, 64, int(physics.get("convex_parts", 8)))
        self.effective_displacement = self._double(
            0, .2, float(physics.get("effective_displacement_m", .01)), 4)
        for label, widget in (
            ("抓取 Top-N", self.grasp_top_n), ("PUSH Top-K", self.push_top_k),
            ("并行进程数", self.push_workers), ("抓取阈值", self.grasp_threshold),
            ("物体质量 / kg", self.object_mass), ("物体摩擦系数", self.object_friction),
            ("桌面摩擦系数", self.table_friction), ("推动速度 / m/s", self.push_speed),
            ("XY邻近尺度", self.overlap_margin), ("最小XY重叠率", self.minimum_overlap),
            ("最小高度差尺度", self.height_gap), ("显示点数上限", self.display_points),
            ("场景变化阈值 / m", self.scene_change), ("重复接触点阈值 / m", self.repeat_contact),
            ("重复方向阈值 / deg", self.repeat_direction), ("恢复系数", self.restitution),
            ("凸体分块数", self.convex_parts), ("有效方向位移 / m", self.effective_displacement),
        ):
            runtime_form.addRow(label, widget)
        runtime_card.body.addLayout(runtime_form)
        cards.addWidget(runtime_card)
        cards.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存设置")
        buttons.button(QDialogButtonBox.StandardButton.Save).setProperty("role", "primary")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _spin(low: int, high: int, value: int) -> QSpinBox:
        field = QSpinBox()
        field.setRange(low, high)
        field.setValue(value)
        return field

    @staticmethod
    def _double(low: float, high: float, value: float, decimals: int = 3) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setDecimals(decimals)
        field.setRange(low, high)
        field.setValue(value)
        return field

    @staticmethod
    def _format_numbers(value) -> str:
        if value is None:
            return ""
        return ", ".join(f"{float(x):.8g}" for x in np.asarray(value).reshape(-1))

    @staticmethod
    def _numbers(field: QLineEdit, count: int, name: str) -> list[float]:
        text = field.text().replace(";", ",").replace(" ", ",")
        values = [float(item) for item in text.split(",") if item]
        if len(values) != count or not np.isfinite(values).all():
            raise ValueError(f"{name}必须包含 {count} 个有限数字")
        return values

    def save(self):
        try:
            ipaddress.ip_address(self.robot_ip.text().strip())
            enabled_cameras = [row for row in self.camera_rows if row[0].isChecked()]
            if not enabled_cameras:
                raise ValueError("至少需要启用一台相机")
            model_views = [row[3].value() for row in enabled_cameras]
            if len(model_views) != len(set(model_views)):
                raise ValueError("启用相机的模型视角编号不能重复")
            if 2 not in model_views:
                raise ValueError("阶段 B 需要一台相机映射到模型参考视角 2")
            camera_matrices = []
            for _, camera_id, ip, _, matrix, _ in enabled_cameras:
                if not camera_id.text().strip():
                    raise ValueError("相机 ID 不能为空")
                ipaddress.ip_address(ip.text().strip())
                value = np.asarray(self._numbers(
                    matrix, 16, f"{camera_id.text()} 外参")).reshape(4, 4)
                if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-5):
                    raise ValueError(f"{camera_id.text()} 外参最后一行必须为 0,0,0,1")
                if not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-3):
                    raise ValueError(f"{camera_id.text()} 外参旋转矩阵无效")
                camera_matrices.append(value.tolist())
            workspace_min = self._numbers(self.workspace_min, 3, "工作空间最小 XYZ")
            workspace_max = self._numbers(self.workspace_max, 3, "工作空间最大 XYZ")
            if np.any(np.asarray(workspace_min) >= np.asarray(workspace_max)):
                raise ValueError("工作空间最大值必须逐轴大于最小值")
            table_normal = np.asarray(self._numbers(self.table_normal, 3, "桌面法向量"))
            length = np.linalg.norm(table_normal)
            if length < 1e-8 or table_normal[2] <= 0:
                raise ValueError("桌面法向量必须非零并朝上")
            table_normal = (table_normal / length).tolist()
            removal_pose = self._numbers(self.removal_pose, 6, "移除放置位姿")
            tcp_pose = self._numbers(self.tcp_pose, 6, "TCP 补偿")
            home_joints = self._numbers(self.home_joints, 6, "安全原点关节角")
            motion_min = self._numbers(self.motion_workspace_min, 3, "运动安全区最小 XYZ")
            motion_max = self._numbers(self.motion_workspace_max, 3, "运动安全区最大 XYZ")
            if np.any(np.asarray(motion_min) >= np.asarray(motion_max)):
                raise ValueError("运动安全区最大值必须逐轴大于最小值")
            if self.depth_min.value() >= self.depth_max.value():
                raise ValueError("最大深度必须大于最小深度")
            for key, field in self.model_fields.items():
                if not field.text().strip():
                    raise ValueError(f"模型参数 {key} 不能为空")
        except ValueError as error:
            QMessageBox.warning(self, "设置无效", str(error))
            return
        robot = self.config.raw["robot"]
        robot.update({
            "ip": self.robot_ip.text().strip(),
            "tool_id": self.tool_id.value(),
            "user_id": self.user_id.value(),
            "speed_percent": self.speed.value(),
        })
        for key, field in self.motion_fields.items():
            robot[key] = field.value()
        robot["removal_pose_mm_rpy_deg"] = removal_pose
        robot["model_tcp_to_robot_tcp"] = {"xyz_mm_rpy_deg": tcp_pose}
        robot["home_joints_deg"] = home_joints
        robot["home_speed_percent"] = self.home_speed.value()
        robot["motion_workspace_min_m"] = motion_min
        robot["motion_workspace_max_m"] = motion_max
        for key, field in self.gripper_fields.items():
            robot[key] = field.value()
        matrix_index = 0
        for enabled, camera_id, ip, model_view, matrix, item in self.camera_rows:
            item["enabled"] = enabled.isChecked()
            item["id"] = camera_id.text().strip()
            item["ip"] = ip.text().strip()
            item["model_view_index"] = model_view.value()
            if enabled.isChecked():
                item["camera_to_robot_base"] = camera_matrices[matrix_index]
                matrix_index += 1
            elif matrix.text().strip():
                item["camera_to_robot_base"] = np.asarray(
                    self._numbers(matrix, 16, f"{camera_id.text()} 外参")
                ).reshape(4, 4).tolist()
        fusion = self.config.raw["fusion"]
        fusion.update({
            "workspace_min_m": workspace_min, "workspace_max_m": workspace_max,
            "table_plane_base": {"normal": table_normal, "offset_m": self.table_offset.value() / length},
            "table_clearance_m": self.table_clearance.value(), "voxel_size_m": self.voxel_size.value(),
            "depth_min_mm": self.depth_min.value(), "depth_max_mm": self.depth_max.value(),
            "target_scene_points": self.scene_points.value(),
            "instance_association_distance_m": self.association_distance.value(),
            "temporal_instance_distance_m": self.temporal_distance.value(),
        })
        model = self.config.raw["tcd_prg"]
        model.update({key: field.text().strip() for key, field in self.model_fields.items()})
        model["device"] = self.model_device.currentText()
        self.config.raw["task"].update({
            "default_category_id": self.default_category.value(),
            "default_region_id": self.default_region.value(),
        })
        self.config.raw["workflow"].update({
            "mode": "manual",
            "target_mode": "specified",
            "grasp_top_n": self.grasp_top_n.value(),
            "push_top_k": self.push_top_k.value(),
            "grasp_confidence_threshold": self.grasp_threshold.value(),
            "overlap_margin_scale": self.overlap_margin.value(),
            "minimum_xy_overlap": self.minimum_overlap.value(),
            "minimum_height_gap_scale": self.height_gap.value(),
            "display_max_points": self.display_points.value(),
            "scene_change_threshold_m": self.scene_change.value(),
            "repeated_action_contact_threshold_m": self.repeat_contact.value(),
            "repeated_action_direction_deg": self.repeat_direction.value(),
        })
        self.config.raw["physics"].update({
            "parallel_workers": self.push_workers.value(),
            "object_mass_kg": self.object_mass.value(),
            "object_friction": self.object_friction.value(),
            "table_friction": self.table_friction.value(),
            "push_speed_m_s": self.push_speed.value(),
            "restitution": self.restitution.value(),
            "convex_parts": self.convex_parts.value(),
            "effective_displacement_m": self.effective_displacement.value(),
        })
        self.config.save()
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("TCD-PRG · 真实抓取控制台")
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)
        self.controller = ExperimentController(config)
        self.pool = QThreadPool(self)
        self.busy = False
        self.job_generation = 0
        self.closing = False
        self.offline_mode = False
        self.offline_target_click = None
        self._obstruction_highlight = None
        self.offline_task_category = 0
        self.offline_task_region = 0
        self.run_mode = QComboBox()
        self.run_mode.addItem("仿真 · 本地场景", "simulation")
        self.run_mode.addItem("真机 · 相机与机械臂", "real")
        self._mode_value = "simulation"
        self.mode_switch = QSlider(Qt.Orientation.Horizontal)
        self.mode_switch.setObjectName("ModeSwitch")
        self.mode_switch.setRange(0, 1)
        self.mode_switch.setPageStep(1)
        self.mode_switch.setAccessibleName("运行模式：左侧仿真，右侧真机")
        self.simulation_mode_label = QLabel("仿真")
        self.real_mode_label = QLabel("真机")
        for label in (self.simulation_mode_label, self.real_mode_label):
            label.setObjectName("ModeLabel")

        self.canvas = OpenGLCloudView(
            config.raw.get("workflow", {}).get("display_max_points", 300000)
        )
        self.instance = QComboBox()
        self.instance.setPlaceholderText("选择实例")
        self.category = QSpinBox()
        self.category.setRange(0, 63)
        self.category.setValue(int(config.raw.get("task", {}).get("default_category_id", 0)))
        self.region = QSpinBox()
        self.region.setRange(0, 63)
        self.region.setValue(int(config.raw.get("task", {}).get("default_region_id", 0)))
        tcp_values = config.raw["robot"]["model_tcp_to_robot_tcp"].get("xyz_mm_rpy_deg", [0] * 6)
        self.tcp_fields = []
        for index, value in enumerate(tcp_values):
            field = QDoubleSpinBox()
            field.setDecimals(3)
            field.setRange(-1000 if index < 3 else -360, 1000 if index < 3 else 360)
            field.setValue(float(value))
            self.tcp_fields.append(field)

        self.tcp_apply = QPushButton("应用 TCP 补偿")
        self.tcp_apply.setProperty("role", "ghost")
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText("模型预测结果将在这里显示")
        self.status = QLabel("设备未连接")
        self.status.setObjectName("StatusBadge")
        self.status.setProperty("state", "idle")
        self.status.setWordWrap(True)
        self.camera_connect_button = QPushButton("连接相机")
        self.robot_connect_button = QPushButton("连接机械臂")
        self.camera_connect_button.setProperty("role", "primary")
        self.robot_connect_button.setProperty("role", "primary")
        self.device_hint = QLabel("相机与机械臂可分别连接；发送动作前仍须单独使能 FR5。")
        self.device_hint.setObjectName("SectionHint")
        self.device_hint.setWordWrap(True)
        self.settings_button = QPushButton("设备设置")
        self.settings_button.setProperty("role", "ghost")
        self.gripper_init_button = QPushButton("初始化夹爪")
        self.gripper_open_button = QPushButton("打开夹爪")
        self.gripper_close_button = QPushButton("闭合夹爪")
        self.enable_button = QPushButton("FR5 使能")
        self.enable_button.setProperty("role", "execute")
        self.disable_button = QPushButton("FR5 下使能")
        self.clear_error_button = QPushButton("清除故障")
        self.status_button = QPushButton("读取状态")
        self.home_button = QPushButton("回安全原点")
        self.pause_button = QPushButton("暂停运动")
        self.resume_button = QPushButton("继续运动")
        self.acquire_button = QPushButton("点云采集")
        self.local_scene = QComboBox()
        self.local_scene.setPlaceholderText("")
        self.local_scene.setFixedWidth(140)
        self.local_state = QSpinBox()
        self.local_state.setRange(0, 0)
        self.local_task = QSpinBox()
        self.local_task.setRange(0, 0)
        self.local_refresh_button = QPushButton("刷新场景")
        self.local_load_button = QPushButton("1  加载本地场景")
        self.segment_button = QPushButton("实例分割")
        self.fuse_button = QPushButton("点云融合")
        self.predict_button = QPushButton("抓取预测")
        self.obstruction_button = QPushButton("压覆物体")
        self.push_rules_button = QPushButton("规则生成推动参数")
        self.push_score_button = QPushButton("PUSH evaluator 评估")
        self.execute_button = QPushButton("动作执行")
        self.sim_segment_button = QPushButton("2  实例分割")
        self.sim_target_button = QPushButton("3  确认目标")
        self.sim_predict_button = QPushButton("4  目标抓取预测")
        self.sim_obstruction_button = QPushButton("5  压覆推断")
        self.sim_push_rules_button = QPushButton("6  规则生成")
        self.sim_push_score_button = QPushButton("7  PUSH评估")
        self.sim_finish_button = QPushButton("任务完成 / 重置")
        self.sim_grasp_result = QLabel("抓取成功分支：等待预测")
        self.sim_grasp_result.setObjectName("SimBranchLabel")
        self.sim_grasp_candidates = QComboBox()
        self.sim_grasp_candidates.setObjectName("CompactTargetInstance")
        self.sim_grasp_candidates.setMinimumWidth(145)
        self.sim_grasp_candidates.setVisible(False)
        self.sim_push_branch_label = QLabel("抓取不可行 → PUSH")
        self.sim_push_branch_label.setObjectName("SimBranchLabel")
        self._grasp_display_candidates = ()
        self._grasp_display_target = None
        for button in (
            self.acquire_button,
            self.segment_button,
            self.fuse_button,
            self.predict_button,
            self.obstruction_button, self.push_rules_button,
            self.push_score_button,
            self.execute_button,
            self.sim_segment_button, self.sim_target_button, self.sim_predict_button,
            self.sim_obstruction_button, self.sim_push_rules_button,
            self.sim_push_score_button,
        ):
            button.setProperty("role", "primary")
        self.next_button = QPushButton("采集下一轮")
        self.finish_button = QPushButton("任务完成 / 重置")
        self.finish_button.setProperty("role", "ghost")
        self.stop_button = QPushButton("停止机械臂")
        self.stop_button.setProperty("role", "danger")
        self.scene_metric = QLabel("点云  —    实例  —")
        self.scene_metric.setObjectName("Metric")
        self.flow = WorkflowView(self)
        self._sim_flow_buttons = {
            "source": self.local_load_button,
            "perception": self.sim_segment_button,
            "target": self.sim_target_button,
            "grasp": self.sim_predict_button,
            "obstruction": self.sim_obstruction_button,
            "push_rules": self.sim_push_rules_button,
            "push_score": self.sim_push_score_button,
        }
        self._sim_pulse = False
        self.sim_pulse_timer = QTimer(self)
        self.sim_pulse_timer.setInterval(420)
        self.sim_pulse_timer.timeout.connect(self._pulse_sim_flow_buttons)
        self.current_step = None
        self.stage_log = []
        self.model_wait_started = None
        self.model_wait_timer = QTimer(self)
        self.model_wait_timer.setInterval(250)
        self.model_wait_timer.timeout.connect(self._show_model_wait)

        self._layout()
        self._signals()
        self._sync_mode_widgets()
        self._enable()
        self._reset_steps()

    def _header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("TopBar")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 11, 20, 11)
        brand = QLabel("TCD-PRG抓取控制台")
        brand.setObjectName("Brand")
        layout.addWidget(brand)
        layout.addStretch(1)
        layout.addWidget(self.simulation_mode_label)
        layout.addWidget(self.mode_switch)
        layout.addWidget(self.real_mode_label)
        return header

    def _viewport(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ViewportCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 12, 15, 15)
        layout.setSpacing(7)
        top = QHBoxLayout()
        title = QLabel("融合场景点云")
        title.setObjectName("SectionTitle")
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.scene_metric)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)
        return panel

    def _step_rail(self) -> QWidget:
        return self.flow

    @staticmethod
    def _flow_arrow(text: str) -> QLabel:
        arrow = QLabel(text)
        arrow.setObjectName("FlowArrow")
        return arrow

    @staticmethod
    def _sim_flow_row(button: QPushButton, companion: QWidget | None = None) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addStretch(1)
        layout.addWidget(button)
        if companion is not None:
            layout.addWidget(companion)
        layout.addStretch(1)
        return row

    def _control_panel(self) -> QWidget:
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(5, 0, 5, 6)
        column.setSpacing(8)

        device = SectionCard("设备连接")
        self.device_card = device
        device.body.addWidget(self.status)
        device.body.addWidget(self.device_hint)
        connection_buttons = QHBoxLayout()
        connection_buttons.setSpacing(8)
        connection_buttons.addWidget(self.camera_connect_button)
        connection_buttons.addWidget(self.robot_connect_button)
        device.body.addLayout(connection_buttons)
        device_buttons = QHBoxLayout()
        device_buttons.setSpacing(6)
        device_buttons.addWidget(self.settings_button)
        device.body.addLayout(device_buttons)
        self.device_advanced_toggle = QPushButton("设备高级操作  ▾")
        self.device_advanced_toggle.setCheckable(True)
        self.device_advanced_toggle.setProperty("role", "ghost")
        device.body.addWidget(self.device_advanced_toggle)
        self.device_advanced_panel = QWidget()
        advanced = QVBoxLayout(self.device_advanced_panel)
        advanced.setContentsMargins(0, 4, 0, 0)
        advanced.setSpacing(6)
        advanced.addWidget(self.gripper_init_button)
        servo_buttons = QHBoxLayout()
        servo_buttons.addWidget(self.enable_button)
        servo_buttons.addWidget(self.disable_button)
        servo_buttons.addWidget(self.clear_error_button)
        advanced.addLayout(servo_buttons)
        recovery_buttons = QHBoxLayout()
        recovery_buttons.addWidget(self.status_button)
        recovery_buttons.addWidget(self.home_button)
        advanced.addLayout(recovery_buttons)
        motion_buttons = QHBoxLayout()
        motion_buttons.addWidget(self.pause_button)
        motion_buttons.addWidget(self.resume_button)
        advanced.addLayout(motion_buttons)
        gripper_buttons = QHBoxLayout()
        gripper_buttons.setSpacing(6)
        gripper_buttons.addWidget(self.gripper_open_button)
        gripper_buttons.addWidget(self.gripper_close_button)
        advanced.addLayout(gripper_buttons)
        device.body.addWidget(self.device_advanced_panel)
        device.body.addWidget(self.stop_button)
        self.device_advanced_panel.setVisible(False)
        self.device_advanced_toggle.toggled.connect(self.device_advanced_panel.setVisible)
        self.device_advanced_toggle.toggled.connect(lambda checked: self.device_advanced_toggle.setText(
            "收起设备高级操作  ▴" if checked else "设备高级操作  ▾"))

        real_source = SectionCard("真机 · 点云采集")
        real_source.body.addWidget(self.acquire_button)
        real_source.body.addWidget(self.next_button)

        self.target_controls = QWidget()
        self.target_controls.setFixedWidth(270)
        target_layout = QHBoxLayout(self.target_controls)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setSpacing(3)
        self.instance.setObjectName("CompactTargetInstance")
        self.instance.setFixedWidth(108)
        for field in (self.category, self.region):
            field.setObjectName("CompactTargetValue")
            field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            field.setAlignment(Qt.AlignmentFlag.AlignCenter)
            field.setFixedWidth(40)
        category_label = QLabel("类别")
        region_label = QLabel("功能区")
        for label in (category_label, region_label):
            label.setObjectName("CompactTargetLabel")
        category_label.setFixedWidth(25)
        region_label.setFixedWidth(42)
        target_layout.addWidget(self.instance)
        target_layout.addWidget(category_label)
        target_layout.addWidget(self.category)
        target_layout.addWidget(region_label)
        target_layout.addWidget(self.region)

        action = SectionCard("真机 · 流程操作")
        for button, text, tip in (
            (self.acquire_button, "1  点云采集", "采集单相机 RGB-D 点云"),
            (self.segment_button, "2  实例分割", "执行实例分割"),
            (self.fuse_button, "3  确认目标", "确认目标实例、类别和功能区"),
            (self.predict_button, "4  目标抓取预测", "预测目标抓取并执行碰撞检查"),
            (self.obstruction_button, "5  压覆推断", "推断最上层压覆物体"),
            (self.push_rules_button, "6  规则生成", "针对压覆物体生成推动参数"),
            (self.push_score_button, "7  PUSH评估", "使用push_evaluator评价推动参数"),
            (self.execute_button, "8  确认并发送动作", "仅真机模式可执行"),
        ):
            button.setText(text)
            button.setToolTip(tip)
        action.body.addWidget(self.segment_button)
        action.body.addWidget(self.fuse_button)
        action.body.addWidget(self.predict_button)
        action.body.addWidget(QLabel("抓取失败后，按顺序执行 PUSH 分支"))
        action.body.addWidget(self.obstruction_button)
        action.body.addWidget(self.push_rules_button)
        action.body.addWidget(self.push_score_button)
        action.body.addWidget(self.execute_button)
        action.body.addWidget(self.finish_button)

        sim_action = SectionCard("仿真 · 流程操作")
        self.sim_action_card = sim_action
        for button in self._sim_flow_buttons.values():
            button.setObjectName("SimFlowButton")
            button.setMaximumWidth(210)
        sim_action.body.addWidget(
            self._sim_flow_row(self.local_load_button, self.local_scene))
        sim_action.body.addWidget(self._flow_arrow("↓"), alignment=Qt.AlignmentFlag.AlignHCenter)
        sim_action.body.addWidget(self._sim_flow_row(self.sim_segment_button))
        sim_action.body.addWidget(self._flow_arrow("↓"), alignment=Qt.AlignmentFlag.AlignHCenter)
        self.sim_target_row = self._sim_flow_row(self.sim_target_button, self.target_controls)
        self.sim_target_row_layout = self.sim_target_row.layout()
        sim_action.body.addWidget(self.sim_target_row)
        sim_action.body.addWidget(self._flow_arrow("↓"), alignment=Qt.AlignmentFlag.AlignHCenter)
        sim_action.body.addWidget(self._sim_flow_row(self.sim_predict_button))
        split = QHBoxLayout()
        split.setContentsMargins(0, 2, 0, 0)
        split.setSpacing(10)
        grasp_branch = QWidget()
        grasp_layout = QVBoxLayout(grasp_branch)
        grasp_layout.setContentsMargins(0, 0, 0, 0)
        grasp_layout.setSpacing(3)
        grasp_layout.addWidget(self._flow_arrow("↙"), alignment=Qt.AlignmentFlag.AlignHCenter)
        grasp_layout.addWidget(self.sim_grasp_result, alignment=Qt.AlignmentFlag.AlignHCenter)
        grasp_layout.addWidget(self.sim_grasp_candidates, alignment=Qt.AlignmentFlag.AlignHCenter)
        grasp_layout.addStretch(1)
        push_branch = QWidget()
        push_layout = QVBoxLayout(push_branch)
        push_layout.setContentsMargins(0, 0, 0, 0)
        push_layout.setSpacing(3)
        push_layout.addWidget(self._flow_arrow("↘"), alignment=Qt.AlignmentFlag.AlignHCenter)
        push_layout.addWidget(self.sim_push_branch_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        for button in (
            self.sim_obstruction_button, self.sim_push_rules_button, self.sim_push_score_button,
        ):
            push_layout.addWidget(self._sim_flow_row(button))
            if button is not self.sim_push_score_button:
                push_layout.addWidget(self._flow_arrow("↓"), alignment=Qt.AlignmentFlag.AlignHCenter)
        split.addWidget(grasp_branch, 1)
        split.addWidget(push_branch, 1)
        sim_action.body.addLayout(split)
        sim_action.body.addWidget(self.sim_finish_button)
        self.sim_finish_button.setProperty("role", "ghost")

        tcp = SectionCard("工具坐标补偿")
        self.tcp_card = tcp
        toggle = QPushButton("展开补偿参数  ▾")
        toggle.setObjectName("TcpToggle")
        toggle.setCheckable(True)
        tcp.body.addWidget(toggle)
        tcp_panel = QWidget()
        tcp_panel.setVisible(False)
        tcp_grid = QGridLayout(tcp_panel)
        tcp_grid.setContentsMargins(0, 0, 0, 0)
        tcp_grid.setHorizontalSpacing(8)
        tcp_grid.setVerticalSpacing(8)
        labels = ("X / mm", "Y / mm", "Z / mm", "Roll / °", "Pitch / °", "Yaw / °")
        for index, (label, field) in enumerate(zip(labels, self.tcp_fields, strict=True)):
            row, col = divmod(index, 2)
            cell = QVBoxLayout()
            cell.setSpacing(4)
            caption = QLabel(label)
            caption.setObjectName("SectionHint")
            cell.addWidget(caption)
            cell.addWidget(field)
            tcp_grid.addLayout(cell, row, col)
        tcp_grid.addWidget(self.tcp_apply, 3, 0, 1, 2)
        tcp.body.addWidget(tcp_panel)
        toggle.toggled.connect(tcp_panel.setVisible)
        toggle.toggled.connect(lambda checked: toggle.setText(
            "收起补偿参数  ▴" if checked else "展开补偿参数  ▾"))

        self.simulation_controls = QWidget()
        self.simulation_layout = QVBoxLayout(self.simulation_controls)
        self.simulation_layout.setContentsMargins(0, 0, 0, 0)
        self.simulation_layout.setSpacing(8)
        self.simulation_layout.addWidget(self.local_refresh_button)
        self.simulation_layout.addWidget(sim_action)

        self.real_controls = QWidget()
        self.real_layout = QVBoxLayout(self.real_controls)
        self.real_layout.setContentsMargins(0, 0, 0, 0)
        self.real_layout.setSpacing(8)
        for widget in (device, real_source, action, tcp):
            self.real_layout.addWidget(widget)

        for widget in (self.simulation_controls, self.real_controls):
            column.addWidget(widget)
        column.addStretch(1)
        lower = QScrollArea()
        lower.setWidgetResizable(True)
        lower.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lower.setWidget(content)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(lower, 1)
        return controls

    def _layout(self):
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 14, 14, 14)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._viewport())
        self.setMenuWidget(self._header())
        self.setCentralWidget(body)

        controls = self._control_panel()
        controls.setMinimumWidth(370)
        controls.setMaximumWidth(440)
        self.controls_dock = QDockWidget("", self)
        self.controls_dock.setObjectName("ControlDock")
        self.controls_dock.setTitleBarWidget(QWidget())
        self.controls_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.controls_dock.setWidget(controls)

        self.result_dock = QDockWidget("预测结果", self)
        self.result_dock.setObjectName("PredictionDock")
        self.result_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.result_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.result_dock.setWidget(self.result)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.controls_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.result_dock)
        self.splitDockWidget(self.controls_dock, self.result_dock, Qt.Orientation.Vertical)
        self.resizeDocks([self.controls_dock, self.result_dock], [640, 260], Qt.Orientation.Vertical)

    def _signals(self):
        self.run_mode.currentIndexChanged.connect(self._mode_changed)
        self.mode_switch.valueChanged.connect(self.run_mode.setCurrentIndex)
        self.camera_connect_button.clicked.connect(lambda: self.run_job(
            self.controller.connect_cameras, self.on_connected))
        self.robot_connect_button.clicked.connect(lambda: self.run_job(
            self.controller.connect_robot, self.on_connected))
        self.settings_button.clicked.connect(self.open_device_settings)
        self.gripper_init_button.clicked.connect(
            lambda: self.run_job(self.controller.initialize_gripper, self.on_device_action))
        self.gripper_open_button.clicked.connect(
            lambda: self.run_job(self.controller.open_gripper, self.on_device_action))
        self.gripper_close_button.clicked.connect(
            lambda: self.run_job(self.controller.close_gripper, self.on_device_action))
        self.enable_button.clicked.connect(lambda: self.run_job(
            self.controller.enable_robot, self.on_device_action))
        self.disable_button.clicked.connect(self.disable_robot)
        self.clear_error_button.clicked.connect(lambda: self.run_job(
            self.controller.clear_errors, self.on_device_action))
        self.status_button.clicked.connect(lambda: self.run_job(
            self.controller.robot_status, self.on_status_read))
        self.home_button.clicked.connect(self.home_robot)
        self.pause_button.clicked.connect(lambda: self.run_job(
            self.controller.pause_robot, self.on_device_action))
        self.resume_button.clicked.connect(lambda: self.run_job(
            self.controller.resume_robot, self.on_device_action))
        self.acquire_button.clicked.connect(self.start_first_stage)
        self.local_refresh_button.clicked.connect(self.refresh_local_scenes)
        self.local_scene.currentIndexChanged.connect(self._local_scene_changed)
        self.local_load_button.clicked.connect(self.load_selected_local_scene)
        self.segment_button.clicked.connect(self.start_second_stage)
        self.sim_segment_button.clicked.connect(self.start_second_stage)
        self.fuse_button.clicked.connect(self.start_third_stage)
        self.sim_target_button.clicked.connect(self.start_third_stage)
        self.predict_button.clicked.connect(self.start_fourth_stage)
        self.sim_predict_button.clicked.connect(self.start_fourth_stage)
        self.obstruction_button.clicked.connect(
            lambda: self.start_manual_decision_stage("obstruction", self.controller.manual_obstruction))
        self.sim_obstruction_button.clicked.connect(
            lambda: self.start_manual_decision_stage("obstruction", self.controller.manual_obstruction))
        self.push_rules_button.clicked.connect(
            lambda: self.start_manual_decision_stage("push_rules", self.controller.manual_push_rules))
        self.sim_push_rules_button.clicked.connect(
            lambda: self.start_manual_decision_stage("push_rules", self.controller.manual_push_rules))
        self.push_score_button.clicked.connect(
            lambda: self.start_manual_decision_stage("push_score", self.controller.manual_push_scoring))
        self.sim_push_score_button.clicked.connect(
            lambda: self.start_manual_decision_stage("push_score", self.controller.manual_push_scoring))
        self.sim_grasp_candidates.currentIndexChanged.connect(self._select_grasp_candidate)
        self.execute_button.clicked.connect(self.start_execution)
        self.next_button.clicked.connect(self.start_first_stage)
        self.finish_button.clicked.connect(self.reset_task)
        self.sim_finish_button.clicked.connect(self.reset_task)
        self.stop_button.clicked.connect(self.emergency_stop)
        self.tcp_apply.clicked.connect(self.apply_tcp)
        self.instance.currentIndexChanged.connect(self.refresh_highlight)

    def _set_status(self, text: str, state: str = "idle") -> None:
        self.status.setText(text)
        self.status.setProperty("state", state)
        refresh_style(self.status)

    def _reset_steps(self, completed: int = 0) -> None:
        self.current_step = None
        self.flow.reset(completed)
        self._sync_sim_flow_states()

    def _set_step(self, stage: str) -> None:
        self.current_step = stage
        self.flow.start(stage)
        self._sync_sim_flow_states()

    def _complete_steps(self, completed: int) -> None:
        self.current_step = None
        for stage in COMMON_STAGES[:completed]:
            self.flow.complete(stage)
        self._sync_sim_flow_states()

    def _fail_current_step(self) -> None:
        if self.current_step is None:
            return
        self.flow.fail(self.current_step)
        self._sync_sim_flow_states()

    def _pulse_sim_flow_buttons(self) -> None:
        self._sim_pulse = not self._sim_pulse
        for stage, button in self._sim_flow_buttons.items():
            if self.flow.states[stage] == "running":
                button.setProperty("pulse", self._sim_pulse)
                refresh_style(button)

    @staticmethod
    def _same_grasp_candidate(first, second) -> bool:
        if first is second:
            return True
        if int(first.get("action_type", -1)) != 2 or int(second.get("action_type", -1)) != 2:
            return False
        if int(first.get("acted_object", -1)) != int(second.get("acted_object", -2)):
            return False
        try:
            return np.allclose(
                np.asarray(first["grasp_pose_world"]), np.asarray(second["grasp_pose_world"]),
                atol=1e-6,
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _clear_grasp_candidates(self) -> None:
        self._grasp_display_candidates = ()
        self._grasp_display_target = None
        self.sim_grasp_candidates.blockSignals(True)
        self.sim_grasp_candidates.clear()
        self.sim_grasp_candidates.blockSignals(False)
        self.sim_grasp_candidates.setVisible(False)

    def _set_grasp_candidates(self, selected, candidates, target) -> None:
        displayed = [selected]
        for candidate in candidates:
            if not any(self._same_grasp_candidate(candidate, current) for current in displayed):
                displayed.append(candidate)
        self._grasp_display_candidates = tuple(displayed)
        self._grasp_display_target = target
        self.sim_grasp_candidates.blockSignals(True)
        self.sim_grasp_candidates.clear()
        for index, candidate in enumerate(self._grasp_display_candidates, start=1):
            score = candidate.get("proposal_score", float("nan"))
            checked = {"collision_free": " · 通过碰撞检查", "collision_rejected": " · 碰撞未通过"}.get(
                candidate.get("_visual_state"), "")
            self.sim_grasp_candidates.addItem(
                f"候选 {index}/{len(displayed)} · {float(score):.3f}{checked}", index - 1
            )
        self.sim_grasp_candidates.setCurrentIndex(0)
        self.sim_grasp_candidates.blockSignals(False)
        self.sim_grasp_candidates.setVisible(
            len(self._grasp_display_candidates) > 0
        )

    def _preview_grasp_update(self, target, update):
        """Keep failed collision checks inspectable without selecting an action."""
        if update.get("stage") != "TARGET_GRASP_AND_COLLISION":
            return None
        candidates = tuple(update.get("candidates", ()))
        if not candidates:
            self._clear_grasp_candidates()
            return None
        free = tuple(update.get("collision_free", ()))
        preview = tuple({**candidate, "_visual_state": (
            "collision_free" if any(self._same_grasp_candidate(candidate, passed) for passed in free)
            else "collision_rejected")} for candidate in candidates)
        selected = update.get("selected") or preview[0]
        self._set_grasp_candidates(selected, preview, target)
        return selected

    def _select_grasp_candidate(self, index: int) -> None:
        if not 0 <= index < len(self._grasp_display_candidates):
            return
        if self.controller.scene is None:
            return
        selected = self._grasp_display_candidates[index]
        self.canvas.set_data(
            self.controller.scene,
            self._grasp_display_target,
            selected,
            self._grasp_display_candidates,
            obstructions=self._stage_obstructions(self.controller.scene, self._grasp_display_target, {}),
        )

    def _sync_sim_flow_states(self) -> None:
        running = False
        for stage, button in self._sim_flow_buttons.items():
            state = self.flow.states[stage]
            visual = state if state in {"running", "success"} else "pending"
            button.setProperty("workflow_state", visual)
            button.setProperty("pulse", False)
            refresh_style(button)
            running = running or visual == "running"
        branch = self.flow.active_branch
        self.sim_grasp_result.setProperty("active", branch == "grasp")
        self.sim_push_branch_label.setProperty("active", branch == "push")
        if branch == "grasp":
            self.sim_grasp_result.setText("抓取成功：结果已显示")
        elif self._grasp_display_candidates:
            self.sim_grasp_result.setText("抓取未通过：候选可切换查看")
        else:
            self.sim_grasp_result.setText("抓取成功分支：等待预测")
        refresh_style(self.sim_grasp_result)
        self.sim_grasp_candidates.setVisible(
            len(self._grasp_display_candidates) > 0
        )
        refresh_style(self.sim_push_branch_label)
        if running and not self.sim_pulse_timer.isActive():
            self.sim_pulse_timer.start()
        elif not running:
            self.sim_pulse_timer.stop()

    def _enable(self):
        simulation = self.run_mode.currentData() == "simulation"
        camera_ready = not simulation and self.controller.cameras_connected and not self.busy
        robot_ready = not simulation and self.controller.robot_connected and not self.busy
        self.camera_connect_button.setEnabled(
            not simulation and not self.busy and not self.controller.cameras_connected)
        self.robot_connect_button.setEnabled(
            not simulation and not self.busy and not self.controller.robot_connected)
        self.device_advanced_toggle.setEnabled(not simulation)
        self.settings_button.setEnabled(
            not simulation and not self.controller.cameras_connected and not self.busy)
        self.gripper_init_button.setEnabled(robot_ready)
        self.gripper_open_button.setEnabled(robot_ready)
        self.gripper_close_button.setEnabled(robot_ready)
        self.enable_button.setEnabled(robot_ready and not self.controller.robot_enabled)
        self.disable_button.setEnabled(robot_ready and self.controller.robot_enabled)
        self.clear_error_button.setEnabled(robot_ready and not self.controller.robot_enabled)
        self.status_button.setEnabled(robot_ready)
        self.home_button.setEnabled(robot_ready and self.controller.robot_enabled)
        self.pause_button.setEnabled(
            robot_ready and self.controller.robot_enabled and not self.controller.robot_paused)
        self.resume_button.setEnabled(
            robot_ready and self.controller.robot_enabled and self.controller.robot_paused)
        mode_manual = True  # Both operator modes expose the same explicit stages.
        stage = self.controller.manual_stage
        self.acquire_button.setEnabled(camera_ready)
        self.segment_button.setEnabled(not self.busy and (
            (simulation and stage == "perceive" and self.controller.scene is not None)
            or (camera_ready and (not mode_manual or stage == "perceive"))
        ))
        self.fuse_button.setEnabled(not self.busy and mode_manual and stage == "select_target")
        has_scene = self.controller.scene is not None
        self.predict_button.setEnabled(not self.busy and has_scene and not self.controller.task_finished and (
            not mode_manual or stage == "target_grasp"))
        manual = mode_manual and not self.busy
        self.obstruction_button.setEnabled(manual and stage == "obstruction")
        self.push_rules_button.setEnabled(manual and stage == "push_rules")
        self.push_score_button.setEnabled(manual and stage == "push_scoring")
        self.instance.setEnabled(not self.busy and (
            self.controller.active_task is None or stage == "select_target"
        ))
        for widget in (self.category, self.region):
            widget.setEnabled(not self.busy and self.controller.active_task is None)
        self.execute_button.setEnabled(
            not simulation and not self.busy and self.controller.prediction is not None
            and self.controller.prediction.status == "ready" and stage == "execute"
        )
        self.next_button.setEnabled(camera_ready)
        self.finish_button.setEnabled(not self.busy)
        self.stop_button.setEnabled(self.busy or self.controller.robot_connected)
        self.local_refresh_button.setEnabled(simulation and not self.busy)
        self.local_scene.setEnabled(simulation and not self.busy)
        self.local_state.setEnabled(simulation and not self.busy)
        self.local_task.setEnabled(simulation and not self.busy)
        self.local_load_button.setEnabled(
            simulation and not self.busy and self.local_scene.currentIndex() >= 0
            and self.local_scene.currentData() is not None
        )
        self.run_mode.setEnabled(not self.busy and not self.controller.robot_connected
                                 and not self.controller.cameras_connected)
        self.mode_switch.setEnabled(self.run_mode.isEnabled())
        for sim_button, real_button in (
            (self.sim_segment_button, self.segment_button),
            (self.sim_target_button, self.fuse_button),
            (self.sim_predict_button, self.predict_button),
            (self.sim_obstruction_button, self.obstruction_button),
            (self.sim_push_rules_button, self.push_rules_button),
            (self.sim_push_score_button, self.push_score_button),
            (self.sim_finish_button, self.finish_button),
        ):
            sim_button.setEnabled(real_button.isEnabled())
        if simulation:
            for button in (
                self.gripper_init_button, self.gripper_open_button,
                self.gripper_close_button, self.enable_button,
                self.disable_button, self.clear_error_button,
                self.status_button, self.home_button, self.pause_button,
                self.resume_button, self.acquire_button,
                self.next_button,
            ):
                button.setEnabled(False)

    def _sync_mode_widgets(self) -> None:
        simulation = self.run_mode.currentData() == "simulation"
        target_layout = self.target_controls.parentWidget().layout()
        if target_layout is not None:
            target_layout.removeWidget(self.target_controls)
        if simulation:
            self.sim_target_row_layout.insertWidget(2, self.target_controls)
        else:
            self.real_layout.insertWidget(2, self.target_controls)
        self.simulation_controls.setVisible(simulation)
        self.real_controls.setVisible(not simulation)
        self.mode_switch.blockSignals(True)
        self.mode_switch.setValue(0 if simulation else 1)
        self.mode_switch.blockSignals(False)
        for label, active in (
            (self.simulation_mode_label, simulation),
            (self.real_mode_label, not simulation),
        ):
            label.setProperty("active", active)
            refresh_style(label)

    def _mode_changed(self, _index=None) -> None:
        mode = self.run_mode.currentData()
        if mode == self._mode_value:
            return
        if self.busy or self.controller.robot_connected or self.controller.cameras_connected:
            self.run_mode.blockSignals(True)
            self.run_mode.setCurrentIndex(self.run_mode.findData(self._mode_value))
            self.run_mode.blockSignals(False)
            self._sync_mode_widgets()
            self._set_status("请先停止当前任务并断开设备后切换模式", "error")
            return
        self.controller.reset_task()
        self.controller.scene = None
        self.offline_mode = False
        self.offline_target_click = None
        self._clear_grasp_candidates()
        self.canvas.set_data(None)
        self.instance.clear()
        self.result.clear()
        self.scene_metric.setText("点云  —    实例  —")
        self.camera_connect_button.setText("连接相机")
        self.robot_connect_button.setText("连接机械臂")
        self._mode_value = mode
        self.flow.set_mode(mode)
        self.current_step = None
        self._sync_sim_flow_states()
        self._sync_mode_widgets()
        self._set_status("仿真模式：请加载本地场景" if mode == "simulation"
                         else "真机模式：请连接设备后采集点云", "idle")
        self._enable()

    def refresh_local_scenes(self) -> None:
        from .offline_ui import local_scene_ids
        self.run_job(lambda: local_scene_ids(self.config), self._on_local_scenes)

    def _on_local_scenes(self, ids: tuple[int, ...]) -> None:
        self.local_scene.blockSignals(True)
        try:
            self.local_scene.clear()
            for scene_id in ids:
                self.local_scene.addItem(f"scene_{scene_id:04d}", scene_id)
            # Refreshing is discovery only; it must never choose a scene for the operator.
            self.local_scene.setCurrentIndex(-1)
        finally:
            self.local_scene.blockSignals(False)
        self._local_scene_changed()
        self._set_status(f"已发现 {len(ids)} 个可选本地场景（仅使用 F 盘现有缓存）", "ready")
        self._enable()

    def _local_scene_changed(self, _index=None) -> None:
        self._enable()

    def load_selected_local_scene(self) -> None:
        scene_id = self.local_scene.currentData()
        if scene_id is None or self.local_scene.currentText() != f"scene_{int(scene_id):04d}":
            self._set_status("请先刷新并选择本地场景", "error")
            return
        # The compact simulation UI deliberately exposes scene selection only.
        # State/task selection therefore uses the dataset's canonical initial
        # observation rather than silently choosing another cached state.
        self.start_offline_replay(int(scene_id), 0, 0)

    def start_offline_replay(self, scene_id: int, state_id: int, task_index: int) -> None:
        """Run a cached observation inside the production UI without devices."""
        if self.busy:
            return
        if self.run_mode.currentData() != "simulation":
            self._set_status("请先切换到仿真模式", "error")
            return
        if self.controller.robot_connected or self.controller.cameras_connected:
            self._set_status("设备已连接；本地离线验证必须在未连接设备时启动", "error")
            return
        self.controller.reset_task()
        self._clear_grasp_candidates()
        self.controller.scene = None
        self.canvas.set_data(None)
        self.instance.clear()
        self.offline_target_click = None
        self.offline_mode = True
        self.controller.connected = False
        self.setWindowTitle("TCD-PRG · 真实抓取控制台 · 数据集离线回放")
        self._reset_steps()
        self._set_step("source")
        self._set_status("正在读取 F 盘已有场景点云……", "busy")
        self._enable()

        def acquire_offline():
            from tcd_prg.observation.cached import ObservationCacheMissError

            from .offline_ui import cached_states_for_task, load_offline_scene

            try:
                return load_offline_scene(self.controller, scene_id, state_id, task_index)
            except ObservationCacheMissError:
                available = cached_states_for_task(self.config, scene_id, task_index)
                return {"cache_missing": True, "available_states": available}

        def acquired(payload):
            if isinstance(payload, dict) and payload.get("cache_missing"):
                available = payload["available_states"]
                choices = "、".join(str(state) for state in available)
                hint = (
                    f"同一场景、同一任务已缓存的状态索引（共 {len(available)} 个）：\n{choices}"
                    if available else "同一场景、同一任务没有可用的状态缓存"
                )
                self._fail_current_step()
                self.result.setPlainText(
                    f"scene_{scene_id:04d} / state {state_id} / task {task_index} 的点云未缓存。\n"
                    f"{hint}\n当前紧凑界面固定加载初始 state 0 / task 0；请选择其他场景后重新加载；"
                    "系统不会自动切换，"
                    "也不会生成或修改 F 盘训练缓存。"
                )
                self._set_status("所选场景的初始状态未缓存，请选择其他场景", "error")
                return
            scene, target_click, category, region, metadata = payload
            self.controller.load_offline_scene(scene)
            self.offline_target_click = np.asarray(target_click, np.float32)
            self.offline_task_category = int(category)
            self.offline_task_region = int(region)
            self.on_scene(scene)
            self.result.setPlainText(
                f"离线回放场景  scene={scene_id} state={state_id} task={task_index}\n"
                f"缓存有效点数    {metadata['input_points']:,}\n"
                f"数据集类别/功能区 {category}/{region}\n"
                "下一步          点击‘2 实例分割’，随后确认目标\n"
                "设备状态        未连接相机或机械臂，动作执行已强制禁用"
            )
            self._set_status("本地场景已加载；请手动点击‘2 实例分割’", "ready")

        self.run_job(acquire_offline, acquired)

    def on_offline_perceived_scene(self, scene):
        from .offline_ui import target_query_near_point

        self.on_scene(scene)
        if self.offline_target_click is None:
            raise RuntimeError("离线任务目标点缺失，请重新加载场景")
        target_query = target_query_near_point(scene, self.offline_target_click)
        index = self.instance.findData(int(target_query))
        if index < 0:
            raise RuntimeError(f"离线目标实例 {target_query} 不在感知结果中")
        self.instance.setCurrentIndex(index)
        self.category.setValue(self.offline_task_category)
        self.region.setValue(self.offline_task_region)
        self.result.append(
            f"\n实例分割完成：数据集目标对应实例 {target_query}；"
            "请核对目标、类别和功能区，再点击‘3 确认目标’"
        )
        self._set_status("实例分割完成；请核对目标后点击‘3 确认目标’", "ready")

    def run_job(self, function, callback):
        if self.busy:
            return
        self.busy = True
        self._set_status("正在处理，请稍候…", "busy")
        self._enable()
        job = Job(function)
        generation = self.job_generation
        job.signals.done.connect(
            lambda value: self._job_done_if_current(generation, value, callback))
        job.signals.failed.connect(lambda text: self._job_failed_if_current(generation, text))
        self.pool.start(job)

    def run_progress_job(self, function, callback):
        if self.busy:
            return
        self.busy = True
        self._set_status("正在执行自动闭环…", "busy")
        self._enable()
        job = Job(function, with_progress=True)
        generation = self.job_generation
        job.signals.progress.connect(lambda update: self.on_auto_progress(
            update) if generation == self.job_generation and not self.closing else None)
        job.signals.done.connect(
            lambda value: self._job_done_if_current(generation, value, callback))
        job.signals.failed.connect(lambda text: self._job_failed_if_current(generation, text))
        self.pool.start(job)

    def _job_done_if_current(self, generation, value, callback):
        if generation != self.job_generation or self.closing:
            return
        self.job_done(value, callback)

    def _job_failed_if_current(self, generation, text):
        if generation != self.job_generation or self.closing:
            return
        self.job_failed(text)

    def _visual_candidates(self, update):
        """Keep the viewport readable without changing decision candidates."""
        candidates = [dict(item) for item in update.get("candidates", ())]
        stage = str(update.get("stage", ""))
        workflow = self.config.raw.get("workflow", {})
        results = tuple(update.get("results", ()))
        if stage == "PYBULLET":
            limit = max(int(workflow.get("push_top_k", 5)), len(results))
            for index, result in enumerate(results[:len(candidates)]):
                candidates[index]["_visual_state"] = (
                    "simulated_effective" if result.get("effective") else "simulated_invalid"
                )
        elif stage in ("PUSH_RULE_GENERATION", "PUSH_EVALUATOR_SCORING"):
            limit = int(workflow.get("push_top_k", 5))
        else:
            limit = int(workflow.get("grasp_top_n", 36))
            free_ids = {id(item) for item in update.get("collision_free", ())}
            for original, visible in zip(
                update.get("candidates", ()), candidates, strict=True
            ):
                if id(original) in free_ids:
                    visible["_visual_state"] = "collision_free"
        return tuple(candidates[:limit])

    def _stage_obstructions(self, scene, target, update):
        if scene is None or target is None:
            return ()
        key = (id(scene), int(target))
        if "relations" in update:
            ids = tuple(sorted(set(reachable_obstructors(int(target), update["relations"]))
                               | {int(value) for value in update.get("adjacent_blockers", ())}))
            self._obstruction_highlight = (key, ids)
            return ids
        if "adjacent_blockers" in update:
            previous = self._obstruction_highlight
            prior = previous[1] if previous is not None and previous[0] == key else ()
            ids = tuple(sorted(set(prior) | {int(value) for value in update["adjacent_blockers"]}))
            self._obstruction_highlight = (key, ids)
            return ids
        previous = self._obstruction_highlight
        return previous[1] if previous is not None and previous[0] == key else ()

    def _append_stage_log(self, stage, message, candidate_count):
        entry = f"{len(self.stage_log) + 1:02d}  {stage:<32} 候选 {candidate_count:>3}  {message}".rstrip()
        self.stage_log.append(entry)
        self.stage_log = self.stage_log[-40:]
        self.result.setPlainText("自动流程阶段记录\n" + "\n".join(self.stage_log))
        self.result.verticalScrollBar().setValue(self.result.verticalScrollBar().maximum())

    def _show_model_wait(self):
        if self.model_wait_started is not None:
            elapsed = time.perf_counter() - self.model_wait_started
            self._set_status(f"模型正在编码场景并生成候选… {elapsed:.1f} s", "busy")

    def on_auto_progress(self, update):
        stage = str(update.get("stage", ""))
        message = str(update.get("message", ""))
        if update.get("stage") == "PUSH_EVALUATOR_SCORING" and update.get("selected"):
            evidence = update["selected"].get("blocked_grasps", ())
            if evidence:
                remaining = sorted({int(value) for row in evidence for value in row["other_neighbor_ids"]})
                message += f"；同一抓取还受邻居 {remaining} 阻挡" if remaining else ""
        if stage == "TARGET_GRASP_MODEL" and message.startswith("正在"):
            self.model_wait_started = time.perf_counter()
            self.model_wait_timer.start()
        elif self.model_wait_timer.isActive():
            self.model_wait_timer.stop()
            self.model_wait_started = None
        if message:
            self._set_status(message, "busy")
        stage_index = {
            "LOAD_OR_CAPTURE": "source", "PERCEPTION": "perception",
            "SELECT_TARGET": "target",
            "TARGET_GRASP_MODEL": "grasp", "MODEL_SCENE_ENCODING": "grasp",
            "MODEL_CANDIDATE_GENERATION": "grasp", "TARGET_GRASP_AND_COLLISION": "grasp",
            "OBSTRUCTION_INFERENCE": "obstruction",
            "PUSH_RULE_GENERATION": "push_rules",
            "PUSH_EVALUATOR_SCORING": "push_score",
        }.get(stage)
        if stage_index is not None:
            self._set_step(stage_index)
        candidates = self._visual_candidates(update)
        selected = update.get("selected")
        obstruction = update.get("obstruction")
        if obstruction is not None and self.controller.scene is not None:
            action = {"acted_object": int(obstruction)}
        else:
            action = selected
        if self.controller.scene is not None:
            target = update.get("target_query", self.instance.currentData())
            preview = self._preview_grasp_update(target, update)
            if action is None:
                action = preview
            obstructions = self._stage_obstructions(self.controller.scene, target, update)
            self.canvas.set_data(self.controller.scene, target, action, candidates, obstructions=obstructions)
        total = int(update.get("total_candidate_count", len(update.get("candidates", ()))))
        count_text = f"{len(candidates)}/{total}" if total != len(candidates) else str(len(candidates))
        self._append_stage_log(stage, message, count_text)

    def start_first_stage(self):
        self._reset_steps()
        self._set_step("source")
        self.run_job(self.controller.capture_scene, self.on_scene)

    def start_second_stage(self):
        self._set_step("perception")
        self.run_job(
            self.controller.perceive_scene,
            self.on_offline_perceived_scene if self.run_mode.currentData() == "simulation" else self.on_scene,
        )

    def start_third_stage(self):
        target, category, region = self.target(), self.category.value(), self.region.value()
        self._set_step("target")
        self.run_job(lambda: self.controller.select_manual_target(
            target, category, region), self.on_manual_update)

    def start_fourth_stage(self):
        self.start_manual_decision_stage("grasp", self.controller.manual_target_grasp)

    def start_manual_decision_stage(self, stage, function):
        self._set_step(stage)
        self.run_job(function, self.on_manual_update)

    def on_manual_update(self, update):
        if hasattr(update, "action"):
            self.on_prediction(update)
            return
        completed = self.current_step
        stage = update.get("stage", "阶段")
        if update.get("skipped"):
            self.result.setPlainText(f"{stage}\n当前决策分支不需要此阶段，已跳过")
        else:
            candidates = tuple(update.get("candidates", ()))
            selected = update.get("selected")
            grasp_details = "\n".join(
                f"  #{row['candidate_index']}: {row['status']}"
                f" 邻居={row.get('neighbor_ids', ())}"
                f" 桌面={bool(row.get('table_collision', False))}"
                f" 目标非夹指={bool(row.get('target_non_finger_collision', False))}"
                for row in update.get("grasp_diagnostics", ())
            )
            push_details = "\n".join(
                f"  PUSH #{item.get('candidate_index', '?')} 物体={item.get('acted_object', '?')}"
                f" 依据={item.get('push_reason', '—')}"
                f" 同一抓取仍受={[(row['candidate_index'], row['other_neighbor_ids']) for row in item.get('blocked_grasps', ())]}"
                for item in candidates if int(item.get("action_type", -1)) == 0
            )
            self.result.setPlainText(
                f"阶段        {stage}\n候选数量    {len(candidates)}\n"
                f"通过碰撞    {len(update.get('collision_free', ())) if 'collision_free' in update else '—'}\n"
                f"压覆物体    {', '.join(map(str, update['obstructions'])) if update.get('obstructions') else update.get('obstruction', '—')}\n"
                f"抓取候选诊断    {update.get('grasp_generation_status', '—')}\n"
                f"邻接阻挡物    {', '.join(map(str, update['adjacent_blockers'])) if update.get('adjacent_blockers') else '—'}"
                + (f"\n逐候选结果\n{grasp_details}" if grasp_details else "")
                + (f"\nPUSH 候选\n{push_details}" if push_details else "")
            )
            target = self.controller.manual_session.target if self.controller.manual_session else self.instance.currentData()
            obstruction = update.get("obstruction")
            action = {"acted_object": int(obstruction)} if obstruction is not None else selected
            preview = self._preview_grasp_update(target, update)
            if action is None:
                action = preview
            obstructions = self._stage_obstructions(self.controller.scene, target, update)
            self.canvas.set_data(self.controller.scene, target, action, candidates, obstructions=obstructions)
        if self.controller.prediction is not None:
            self.on_prediction(self.controller.prediction)
        elif completed is not None:
            self.flow.complete(completed)
            self.current_step = None
            self._sync_sim_flow_states()

    def job_done(self, value, callback):
        self.model_wait_timer.stop()
        self.model_wait_started = None
        self.busy = False
        try:
            callback(value)
        except Exception:
            self.job_failed(traceback.format_exc())
            return
        self._enable()

    def job_failed(self, text):
        self.model_wait_timer.stop()
        self.model_wait_started = None
        self.busy = False
        self._fail_current_step()
        self._set_status("操作失败，请查看详细信息", "error")
        self.result.setPlainText(text)
        self._enable()
        QMessageBox.critical(self, "操作失败", text.splitlines()[-1] if text else "未知错误")

    def on_connected(self, message):
        self._set_status(message, "ready")
        self.camera_connect_button.setText(
            "相机已连接" if self.controller.cameras_connected else "连接相机")
        self.robot_connect_button.setText(
            "机械臂已连接" if self.controller.robot_connected else "连接机械臂")

    def on_device_action(self, message):
        self._set_status(message, "ready")

    def on_status_read(self, message):
        self.result.setPlainText(message)
        self._set_status("FR5 状态已刷新", "ready")

    def disable_robot(self):
        answer = QMessageBox.question(self, "确认下使能", "确认停止当前运动并下使能 FR5？")
        if answer == QMessageBox.StandardButton.Yes:
            self.run_job(self.controller.disable_robot, self.on_device_action)

    def home_robot(self):
        answer = QMessageBox.warning(
            self, "确认回安全原点",
            "回原点会执行关节运动。确认路径周围无人、无障碍物且夹持物不会碰撞后继续。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.run_job(self.controller.home_robot, self.on_device_action)

    def open_device_settings(self):
        if self.controller.connected:
            QMessageBox.warning(self, "设备已连接", "请先断开设备，再修改连接与夹爪参数。")
            return
        dialog = DeviceSettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Rebuild disconnected adapters so all persisted values take effect.
            self.controller.close()
            self.controller = ExperimentController(self.config)
            self._set_status("设置已保存，将在下次连接时生效", "ready")
            self._enable()

    def on_scene(self, scene):
        # A loaded offline scene is intentionally unsegmented at this point,
        # but its raw point cloud still must be visible before step 2.
        self._obstruction_highlight = None
        self._clear_grasp_candidates()
        self.canvas.set_data(scene, None, None)
        self.instance.blockSignals(True)
        self.instance.clear()
        for value in scene.instance_ids:
            self.instance.addItem(f"实例 {value}", value)
        self.instance.setCurrentIndex(-1)
        self.instance.blockSignals(False)
        if self.instance.currentData() is not None and self.controller.active_task is None:
            self.category.setValue(
                int(scene.category_by_instance.get(self.instance.currentData(), 0)))
        self.result.setPlainText(
            f"融合点数    {len(scene.xyz_m):,}\n实例数量    {len(scene.instance_ids)}\n实例列表    {scene.instance_ids}")
        self.scene_metric.setText(f"点云  {len(scene.xyz_m):,}    实例  {len(scene.instance_ids)}")
        status = (
            "点云已准备；请点击‘2 实例分割’"
            if self.controller.manual_stage == "perceive"
            else "实例分割完成；请手动选择目标并点击‘3 确认目标’"
        )
        self._set_status(status, "ready")
        self._complete_steps(2 if scene.instance_ids else 1)
        self.refresh_highlight()

    def target(self):
        value = self.instance.currentData()
        if value is None:
            raise RuntimeError("请先选择目标实例")
        return int(value)

    def on_prediction(self, prediction):
        action = prediction.action
        if prediction.status == "operator_attention":
            self._complete_steps(4)
            if "OBSTRUCTION_INFERENCE" in prediction.decision_path:
                self.flow.select_branch("push")
                self.flow.complete("obstruction")
                if "PUSH_RULE_GENERATION" in prediction.decision_path:
                    self.flow.complete("push_rules")
                if "PUSH_EVALUATOR_SCORING" in prediction.decision_path:
                    self.flow.fail("push_score")
            self._sync_sim_flow_states()
            self.result.setPlainText(
                "流程已停止，需要人工处理\n"
                f"原因        {action.get('reason', '没有有效动作')}\n"
                f"决策路径    {' → '.join(prediction.decision_path)}"
            )
            self._set_status("没有有效动作，请人工处理", "error")
            self.canvas.set_data(
                self.controller.scene, prediction.target_query, None,
                obstructions=self._stage_obstructions(
                    self.controller.scene, prediction.target_query, {}),
            )
            return
        kind = int(action["action_type"])
        lines = [
            f"动作类型    {ACTION_NAMES.get(kind, kind)}",
            f"作用实例    {action['acted_object']}",
            f"模型耗时    {prediction.inference_seconds:.3f} s",
            f"候选分数    {action.get('proposal_score', float('nan')):.4f}",
        ]
        if kind == 0:
            lines += [
                f"接触点      {np.round(action['push_contact_world'], 4).tolist()}",
                f"推动方向    {np.round(action['push_direction_world'], 4).tolist()}",
                f"推动距离    {action['push_distance_m']:.3f} m",
                f"推动评估分数 {action.get('improvement_probability', float('nan')):.4f}",
            ]
        else:
            lines += [
                f"6DoF 位姿   {np.round(action['grasp_pose_world'], 5).tolist()}",
                f"夹爪宽度    {action['grasp_width_m']:.4f} m",
            ]
        if prediction.decision_path:
            lines.append(f"决策路径    {' → '.join(prediction.decision_path)}")
        for name, seconds in (prediction.timings or {}).items():
            lines.append(f"{name:<18} {seconds:.4f} s")
        self.result.setPlainText("\n".join(lines))
        self._set_status(
            "预测完成；离线回放模式禁止执行机械臂"
            if self.offline_mode else "预测完成，确认后可执行",
            "ready",
        )
        self._complete_steps(4)
        if kind == 2:
            self.flow.select_branch("grasp")
            if self.run_mode.currentData() == "simulation":
                self.flow.complete("grasp_execute")
        elif kind == 0:
            self.flow.select_branch("push")
            for stage in ("obstruction", "push_rules", "push_score"):
                self.flow.complete(stage)
            if self.run_mode.currentData() == "simulation":
                self.flow.complete("push_execute")
        self._sync_sim_flow_states()
        same_branch = [
            item for item in prediction.candidates
            if int(item.get("action_type", -1)) == kind
            and int(item.get("acted_object", -1)) == int(action.get("acted_object", -2))
            and (kind != 2 or grasp_local_z_is_downward_or_level(item))
        ]
        visual_limit = (
            int(self.config.raw.get("workflow", {}).get("push_top_k", 5))
            if kind == 0 else int(self.config.raw.get("workflow", {}).get("grasp_top_n", 36))
        )
        if kind == 2:
            self._set_grasp_candidates(action, same_branch[:visual_limit], prediction.target_query)
        self.canvas.set_data(
            self.controller.scene, prediction.target_query, action, tuple(
                same_branch[:visual_limit]),
            obstructions=self._stage_obstructions(
                self.controller.scene, prediction.target_query, {}),
        )
        # Even an automatic decision never authorizes physical motion. The
        # operator must explicitly confirm the selected branch's action.

    def start_execution(self):
        if self.run_mode.currentData() != "real":
            self._set_status("仿真模式只展示候选结果，不发送机械臂动作", "error")
            return
        if self.controller.prediction is None or self.controller.prediction.status != "ready":
            self._set_status("没有可确认执行的有效候选动作", "error")
            return
        target, category, region = -1, self.category.value(), self.region.value()
        kind = int(self.controller.prediction.action["action_type"])
        question = (
            "PUSH 候选未经 PyBullet 仿真；请现场确认推动路径、周围物体和人员安全。"
            "确认将当前推动参数发送给机械臂？"
            if kind == 0 else "确认将当前目标抓取动作发送给机械臂？"
        )
        answer = QMessageBox.question(self, "确认执行", question)
        if answer == QMessageBox.StandardButton.Yes:
            self._set_step("grasp_execute" if kind == 2 else "push_execute")
            self.run_job(lambda: self.controller.execute(
                target, category, region), self.on_executed)

    def on_executed(self, message):
        self._set_status(message, "ready")
        self.result.append("\n" + message)
        if self.current_step is not None:
            self.flow.complete(self.current_step)
            self.current_step = None
        self._sync_sim_flow_states()
        if self.controller.pending_task_grasp is not None:
            answer = QMessageBox.question(
                self,
                "确认目标抓取结果",
                "请观察实物：目标物是否已被稳定抓取并抬起？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            confirmation = self.controller.confirm_task_grasp(
                answer == QMessageBox.StandardButton.Yes
            )
            self.result.append("\n" + confirmation)
            self._set_status(confirmation, "ready")
        if (
            not self.offline_mode
            and self.config.raw.get("workflow", {}).get("mode") in ("auto", "automatic")
            and not self.controller.task_finished
            and self.controller.robot_connected
        ):
            self._set_status("动作已执行；确认后手动采集下一轮场景", "ready")

    def refresh_highlight(self):
        if self.controller.active_task is None and self.controller.scene is not None and self.instance.currentData() is not None:
            value = int(self.instance.currentData())
            self.category.setValue(int(self.controller.scene.category_by_instance.get(value, 0)))
            self.canvas.set_data(self.controller.scene, value, None)

    def reset_task(self):
        self.controller.reset_task()
        self._obstruction_highlight = None
        self._clear_grasp_candidates()
        scene = self.controller.scene
        if scene is not None:
            self.controller.manual_stage = "select_target" if scene.instance_ids else "perceive"
        self.stage_log.clear()
        self.result.clear()
        self._set_status(
            "任务已重置，可重新确认离线目标" if self.offline_mode else "任务已重置，可重新选择目标",
            "idle",
        )
        self.canvas.set_data(self.controller.scene, None, None)
        self._reset_steps(0 if scene is None else 2 if scene.instance_ids else 1)
        self._enable()

    def emergency_stop(self):
        # Vendor RPCs can block when the controller or network is unhealthy;
        # never issue them on the GUI thread.
        self.job_generation += 1
        self.busy = True
        self.model_wait_timer.stop()
        self.model_wait_started = None
        self._set_status("正在通过独立通道停止并下使能 FR5…", "error")
        self._enable()
        job = Job(self.controller.stop)
        job.signals.done.connect(self._on_emergency_stopped)
        job.signals.failed.connect(self._on_emergency_stop_failed)
        self.pool.start(job)

    def _on_emergency_stopped(self, _value):
        self.busy = False
        self._set_status("自动流程已停止；已取消后台模型/仿真任务", "error")
        self._enable()

    def _on_emergency_stop_failed(self, text):
        self.busy = False
        self._set_status("停止命令失败，请使用控制器急停", "error")
        self._enable()
        QMessageBox.critical(self, "停止失败", text)

    def apply_tcp(self):
        values = [field.value() for field in self.tcp_fields]
        self._set_status(self.controller.set_tcp_compensation(values), "ready")

    def closeEvent(self, event):
        self.closing = True
        self.job_generation += 1
        self.controller.cancel_background("窗口关闭")
        if not self.pool.waitForDone(5000):
            event.ignore()
            self.closing = False
            QMessageBox.warning(self, "正在关闭", "后台任务尚未退出，请稍后再次关闭窗口。")
            return
        self.controller.close()
        event.accept()


def configure_application(app: QApplication) -> None:
    family = preferred_ui_font()
    app.setFont(QFont(family, 10))
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0b1220"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e7edf6"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0c1522"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e7edf6"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#18253a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#dbe5f2"))
    app.setPalette(palette)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / \
                        "configs/real_experiment.yaml")
    parser.add_argument("--offline-scene", type=int)
    parser.add_argument("--offline-state", type=int, default=0)
    parser.add_argument("--offline-task", type=int, default=0)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    configure_application(app)
    window = MainWindow(AppConfig.load(args.config))
    window.show()
    if args.offline_scene is not None:
        QTimer.singleShot(
            0,
            lambda: window.start_offline_replay(
                args.offline_scene, args.offline_state, args.offline_task
            ),
        )
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
