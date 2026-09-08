from __future__ import annotations

import argparse
import ipaddress
import sys
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDockWidget,
    QDialog,
    QDialogButtonBox,
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
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .config import AppConfig
from .controller import ExperimentController
from .transforms import pose7_to_matrix


ACTION_NAMES = {0: "PUSH", 1: "PICK_REMOVE", 2: "TASK_GRASP"}
WORKFLOW_STEPS = ("点云采集", "实例分割", "点云融合", "抓取预测", "动作执行")

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
QFrame#StepRail { background: transparent; }
QFrame#StepDot {
    min-width: 18px;
    max-width: 18px;
    min-height: 18px;
    max-height: 18px;
    background: #182335;
    border: 2px solid #34445c;
    border-radius: 9px;
}
QFrame#StepDot[state="success"] { background: #21966f; border-color: #57d8aa; }
QFrame#StepDot[state="running"] { background: #d79a24; border-color: #ffd36a; }
QFrame#StepDot[state="failed"] { background: #bd3f50; border-color: #ff8290; }
QFrame#StepDot[state="pending"] { background: #536075; border-color: #7c899d; }
QFrame#StepConnector {
    min-height: 2px;
    max-height: 2px;
    background: #2a3850;
    border: none;
}
QFrame#StepConnector[state="success"] { background: #2fae85; }
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


class Job(QRunnable):
    def __init__(self, function: Callable):
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.signals.done.emit(self.function())
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


class CloudCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(9, 7), tight_layout=True, facecolor="#0f1928")
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.scene = None
        self.target = None
        self.action = None
        self.draw_scene()

    def set_data(self, scene, target=None, action=None):
        self.scene, self.target, self.action = scene, target, action
        self.draw_scene()

    def _style_axis(self):
        ax = self.axis
        ax.set_facecolor("#0f1928")
        ax.tick_params(colors="#718099", labelsize=8, pad=1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color("#9aa8bb")
            axis.pane.set_facecolor((0.075, 0.118, 0.184, 1.0))
            axis.pane.set_edgecolor((0.18, 0.25, 0.36, 1.0))
            axis._axinfo["grid"]["color"] = (0.20, 0.28, 0.40, 0.45)
            axis._axinfo["grid"]["linewidth"] = 0.6

    def draw_scene(self):
        ax = self.axis
        ax.clear()
        self._style_axis()
        if self.scene is None:
            ax.set_axis_off()
            ax.text2D(
                0.5,
                0.54,
                "等待场景数据",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#dce6f3",
                fontsize=16,
                fontweight="medium",
            )
            self.draw_idle()
            return

        ax.set_axis_on()
        self._style_axis()
        ax.set_xlabel("X / m", labelpad=7)
        ax.set_ylabel("Y / m", labelpad=7)
        ax.set_zlabel("Z / m", labelpad=7)
        xyz = self.scene.xyz_m
        stride = max(1, len(xyz) // 22000)
        points = xyz[::stride]
        instance = self.scene.instance_id[::stride]
        colors = self.scene.rgb[::stride].copy()
        if self.target is not None:
            mask = instance == self.target
            colors[~mask] *= 0.22
            colors[mask] = [0.20, 0.78, 1.00]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2.2, depthshade=False)
        if self.action:
            kind = int(self.action["action_type"])
            if kind == 0:
                p = np.asarray(self.action["push_contact_world"])
                d = np.asarray(self.action["push_direction_world"])
                ax.quiver(*p, *d, length=float(self.action["push_distance_m"]), color="#ff6f7d", linewidth=3)
            else:
                pose = np.asarray(self.action["grasp_pose_world"])
                t = pose[:3]
                rotation = pose7_to_matrix(pose)[:3, :3]
                for axis, color in zip(range(3), ("#ff6575", "#61d7a4", "#55b9f3")):
                    ax.quiver(*t, *rotation[:, axis], length=.08, color=color, linewidth=3)
        self.draw_idle()


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
        self.association_distance = self._double(0, 1, float(fusion.get("instance_association_distance_m", .06)), 3)
        self.temporal_distance = self._double(0, 1, float(fusion.get("temporal_instance_distance_m", .10)), 3)
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
            ("推动距离 / m", "push_distance_m", 0, .5, .15),
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
        self.motion_workspace_min = QLineEdit(self._format_numbers(robot.get("motion_workspace_min_m")))
        self.motion_workspace_max = QLineEdit(self._format_numbers(robot.get("motion_workspace_max_m")))
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
        cards.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
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
                value = np.asarray(self._numbers(matrix, 16, f"{camera_id.text()} 外参")).reshape(4, 4)
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
        self.pool = QThreadPool.globalInstance()
        self.busy = False

        self.canvas = CloudCanvas()
        self.instance = QComboBox()
        self.instance.setPlaceholderText("采集后选择实例")
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
        self.result.setMinimumHeight(120)
        self.result_dock = QDockWidget("预测结果", self)
        self.result_dock.setObjectName("PredictionDock")
        self.result_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.result_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.result_dock.setWidget(self.result)
        self.result_dock.setMinimumSize(460, 140)
        self.result_dock.setMaximumHeight(230)
        self.status = QLabel("设备未连接")
        self.status.setObjectName("StatusBadge")
        self.status.setProperty("state", "idle")
        self.status.setWordWrap(True)
        self.connect_button = QPushButton("连接相机与机械臂")
        self.connect_button.setProperty("role", "primary")
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
        self.segment_button = QPushButton("实例分割")
        self.fuse_button = QPushButton("点云融合")
        self.predict_button = QPushButton("抓取预测")
        self.execute_button = QPushButton("动作执行")
        for button in (
            self.acquire_button,
            self.segment_button,
            self.fuse_button,
            self.predict_button,
            self.execute_button,
        ):
            button.setProperty("role", "primary")
        self.next_button = QPushButton("采集下一轮")
        self.finish_button = QPushButton("任务完成 / 重置")
        self.finish_button.setProperty("role", "ghost")
        self.stop_button = QPushButton("停止机械臂")
        self.stop_button.setProperty("role", "danger")
        self.scene_metric = QLabel("点云  —    实例  —")
        self.scene_metric.setObjectName("Metric")
        self.step_dots = []
        self.step_connectors = []
        self.current_step = None

        self._layout()
        self._signals()
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
        mode = QLabel("●  运行")
        mode.setObjectName("StatusBadge")
        mode.setProperty("state", "ready")
        layout.addWidget(mode)
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
        rail = QFrame()
        rail.setObjectName("StepRail")
        layout = QHBoxLayout(rail)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        for index, text in enumerate(WORKFLOW_STEPS):
            dot = QFrame()
            dot.setObjectName("StepDot")
            dot.setProperty("state", "pending")
            dot.setToolTip(text)
            self.step_dots.append(dot)
            layout.addWidget(dot)
            if index < len(WORKFLOW_STEPS) - 1:
                connector = QFrame()
                connector.setObjectName("StepConnector")
                connector.setProperty("state", "pending")
                self.step_connectors.append(connector)
                layout.addWidget(connector, 1)
        layout.addStretch(1)
        return rail

    def _control_panel(self) -> QWidget:
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(5, 0, 5, 6)
        column.setSpacing(8)
        column.addWidget(self._step_rail())

        device = SectionCard("设备")
        device.body.addWidget(self.status)
        device.body.addWidget(self.connect_button)
        device_buttons = QHBoxLayout()
        device_buttons.setSpacing(6)
        device_buttons.addWidget(self.settings_button)
        device_buttons.addWidget(self.gripper_init_button)
        device.body.addLayout(device_buttons)
        servo_buttons = QHBoxLayout()
        servo_buttons.addWidget(self.enable_button)
        servo_buttons.addWidget(self.disable_button)
        servo_buttons.addWidget(self.clear_error_button)
        device.body.addLayout(servo_buttons)
        recovery_buttons = QHBoxLayout()
        recovery_buttons.addWidget(self.status_button)
        recovery_buttons.addWidget(self.home_button)
        device.body.addLayout(recovery_buttons)
        motion_buttons = QHBoxLayout()
        motion_buttons.addWidget(self.pause_button)
        motion_buttons.addWidget(self.resume_button)
        device.body.addLayout(motion_buttons)
        gripper_buttons = QHBoxLayout()
        gripper_buttons.setSpacing(6)
        gripper_buttons.addWidget(self.gripper_open_button)
        gripper_buttons.addWidget(self.gripper_close_button)
        device.body.addLayout(gripper_buttons)

        task = SectionCard("任务输入")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.addRow("目标实例", self.instance)
        form.addRow("类别 ID", self.category)
        form.addRow("功能区 ID", self.region)
        task.body.addLayout(form)

        action = SectionCard("运行控制")
        run_buttons = QHBoxLayout()
        run_buttons.setSpacing(5)
        for button, short_text, tip in (
            (self.acquire_button, "采集", "采集单相机 RGB-D 点云"),
            (self.segment_button, "分割", "执行实例分割"),
            (self.fuse_button, "融合", "生成机器人基座坐标系点云"),
            (self.predict_button, "预测", "预测闭环下一步动作"),
            (self.execute_button, "执行", "确认并执行当前预测动作"),
        ):
            button.setText(short_text)
            button.setToolTip(tip)
            button.setObjectName("RunCompact")
            button.setMinimumWidth(0)
            run_buttons.addWidget(button, 1)
        action.body.addLayout(run_buttons)
        secondary = QHBoxLayout()
        secondary.setSpacing(6)
        secondary.addWidget(self.next_button)
        secondary.addWidget(self.finish_button)
        action.body.addLayout(secondary)
        action.body.addWidget(self.stop_button)

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
        for index, (label, field) in enumerate(zip(labels, self.tcp_fields)):
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
        toggle.toggled.connect(lambda checked: toggle.setText("收起补偿参数  ▴" if checked else "展开补偿参数  ▾"))

        for widget in (device, tcp, action, task):
            column.addWidget(widget)
        column.addStretch(1)
        return content

    def _layout(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._header())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(14, 14, 14, 14)
        body_layout.setSpacing(0)
        controls = QScrollArea()
        controls.setWidgetResizable(True)
        controls.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls.setMinimumWidth(370)
        controls.setMaximumWidth(410)
        controls.setWidget(self._control_panel())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._viewport())
        splitter.addWidget(controls)
        splitter.setSizes([1010, 400])
        body_layout.addWidget(splitter)
        root_layout.addWidget(body, 1)
        self.setCentralWidget(root)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.result_dock)
        self.result_dock.setFloating(True)
        self.result_dock.resize(560, 200)

    def _signals(self):
        self.connect_button.clicked.connect(lambda: self.run_job(self.controller.connect, self.on_connected))
        self.settings_button.clicked.connect(self.open_device_settings)
        self.gripper_init_button.clicked.connect(
            lambda: self.run_job(self.controller.initialize_gripper, self.on_device_action))
        self.gripper_open_button.clicked.connect(
            lambda: self.run_job(self.controller.open_gripper, self.on_device_action))
        self.gripper_close_button.clicked.connect(
            lambda: self.run_job(self.controller.close_gripper, self.on_device_action))
        self.enable_button.clicked.connect(lambda: self.run_job(self.controller.enable_robot, self.on_device_action))
        self.disable_button.clicked.connect(self.disable_robot)
        self.clear_error_button.clicked.connect(lambda: self.run_job(self.controller.clear_errors, self.on_device_action))
        self.status_button.clicked.connect(lambda: self.run_job(self.controller.robot_status, self.on_status_read))
        self.home_button.clicked.connect(self.home_robot)
        self.pause_button.clicked.connect(lambda: self.run_job(self.controller.pause_robot, self.on_device_action))
        self.resume_button.clicked.connect(lambda: self.run_job(self.controller.resume_robot, self.on_device_action))
        self.acquire_button.clicked.connect(lambda: self.start_acquire(0))
        self.segment_button.clicked.connect(lambda: self.start_acquire(1))
        self.fuse_button.clicked.connect(lambda: self.start_acquire(2))
        self.predict_button.clicked.connect(self.start_prediction)
        self.execute_button.clicked.connect(self.start_execution)
        self.next_button.clicked.connect(lambda: self.start_acquire(0))
        self.finish_button.clicked.connect(self.reset_task)
        self.stop_button.clicked.connect(self.emergency_stop)
        self.tcp_apply.clicked.connect(self.apply_tcp)
        self.instance.currentIndexChanged.connect(self.refresh_highlight)

    def _set_status(self, text: str, state: str = "idle") -> None:
        self.status.setText(text)
        self.status.setProperty("state", state)
        refresh_style(self.status)

    def _paint_steps(self, states) -> None:
        labels = {"pending": "未执行", "running": "执行中", "success": "成功", "failed": "执行失败"}
        for index, (dot, state) in enumerate(zip(self.step_dots, states)):
            dot.setProperty("state", state)
            dot.setToolTip(f"{WORKFLOW_STEPS[index]} · {labels[state]}")
            refresh_style(dot)
        for index, connector in enumerate(self.step_connectors):
            connector.setProperty("state", "success" if states[index] == "success" else "pending")
            refresh_style(connector)

    def _reset_steps(self, completed: int = 0) -> None:
        self.current_step = None
        self._paint_steps(["success" if index < completed else "pending" for index in range(len(self.step_dots))])

    def _set_step(self, current: int) -> None:
        current = max(0, min(current, len(self.step_dots) - 1))
        self.current_step = current
        self._paint_steps([
            "success" if index < current else "running" if index == current else "pending"
            for index in range(len(self.step_dots))
        ])

    def _complete_steps(self, completed: int) -> None:
        self.current_step = None
        self._paint_steps(["success" if index < completed else "pending" for index in range(len(self.step_dots))])

    def _fail_current_step(self) -> None:
        if self.current_step is None:
            return
        failed = self.current_step
        self._paint_steps([
            "success" if index < failed else "failed" if index == failed else "pending"
            for index in range(len(self.step_dots))
        ])

    def _enable(self):
        connected = self.controller.connected and not self.busy
        self.connect_button.setEnabled(not self.controller.connected and not self.busy)
        self.settings_button.setEnabled(not self.controller.connected and not self.busy)
        self.gripper_init_button.setEnabled(connected)
        self.gripper_open_button.setEnabled(connected)
        self.gripper_close_button.setEnabled(connected)
        self.enable_button.setEnabled(connected and not self.controller.robot_enabled)
        self.disable_button.setEnabled(connected and self.controller.robot_enabled)
        self.clear_error_button.setEnabled(connected and not self.controller.robot_enabled)
        self.status_button.setEnabled(connected)
        self.home_button.setEnabled(connected and self.controller.robot_enabled)
        self.pause_button.setEnabled(connected and self.controller.robot_enabled and not self.controller.robot_paused)
        self.resume_button.setEnabled(connected and self.controller.robot_enabled and self.controller.robot_paused)
        self.acquire_button.setEnabled(connected)
        self.segment_button.setEnabled(connected)
        self.fuse_button.setEnabled(connected)
        has_scene = self.controller.scene is not None
        self.predict_button.setEnabled(connected and has_scene and not self.controller.task_finished)
        for widget in (self.instance, self.category, self.region):
            widget.setEnabled(not self.busy and self.controller.active_task is None)
        self.execute_button.setEnabled(connected and self.controller.prediction is not None)
        self.next_button.setEnabled(connected)
        self.finish_button.setEnabled(not self.busy)
        self.stop_button.setEnabled(self.controller.connected)

    def run_job(self, function, callback):
        if self.busy:
            return
        self.busy = True
        self._set_status("正在处理，请稍候…", "busy")
        self._enable()
        job = Job(function)
        job.signals.done.connect(lambda value: self.job_done(value, callback))
        job.signals.failed.connect(self.job_failed)
        self.pool.start(job)

    def start_acquire(self, step: int = 0):
        self._set_step(step)
        self.run_job(self.controller.acquire, self.on_scene)

    def job_done(self, value, callback):
        self.busy = False
        callback(value)
        self._enable()

    def job_failed(self, text):
        self.busy = False
        self._fail_current_step()
        self._set_status("操作失败，请查看详细信息", "error")
        self.result.setPlainText(text)
        self._enable()
        QMessageBox.critical(self, "操作失败", text.splitlines()[-1] if text else "未知错误")

    def on_connected(self, message):
        self._set_status(message, "ready")
        self.connect_button.setText("设备已连接")
        self._reset_steps()

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
        previous = self.instance.currentData()
        self.instance.blockSignals(True)
        self.instance.clear()
        for value in scene.instance_ids:
            self.instance.addItem(f"实例 {value}", value)
        index = self.instance.findData(previous)
        self.instance.setCurrentIndex(max(0, index))
        self.instance.blockSignals(False)
        if self.instance.currentData() is not None and self.controller.active_task is None:
            self.category.setValue(int(scene.category_by_instance.get(self.instance.currentData(), 0)))
        self.result.setPlainText(f"融合点数    {len(scene.xyz_m):,}\n实例数量    {len(scene.instance_ids)}\n实例列表    {scene.instance_ids}")
        self.scene_metric.setText(f"点云  {len(scene.xyz_m):,}    实例  {len(scene.instance_ids)}")
        self._set_status("场景已更新，预测将重识别并继续原目标" if self.controller.active_task else "场景已更新，请选择目标并运行模型预测", "ready")
        self._complete_steps(3)
        self.refresh_highlight()

    def target(self):
        value = self.instance.currentData()
        if value is None:
            raise RuntimeError("请先选择目标实例")
        return int(value)

    def start_prediction(self):
        target = -1 if self.controller.active_task else self.target()
        category, region = self.category.value(), self.region.value()
        self._set_step(3)
        self.run_job(lambda: self.controller.predict(target, category, region), self.on_prediction)

    def on_prediction(self, prediction):
        action = prediction.action
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
                f"单步推动价值 {action['push_value']:.4f}",
            ]
        else:
            lines += [
                f"6DoF 位姿   {np.round(action['grasp_pose_world'], 5).tolist()}",
                f"夹爪宽度    {action['grasp_width_m']:.4f} m",
            ]
        self.result.setPlainText("\n".join(lines))
        self._set_status("预测完成，确认后可执行", "ready")
        self._complete_steps(4)
        self.canvas.set_data(self.controller.scene, action.get('target_query'), action)

    def start_execution(self):
        target, category, region = -1, self.category.value(), self.region.value()
        answer = QMessageBox.question(self, "确认执行", "是否将当前预测动作发送给机械臂？")
        if answer == QMessageBox.StandardButton.Yes:
            self._set_step(4)
            self.run_job(lambda: self.controller.execute(target, category, region), self.on_executed)

    def on_executed(self, message):
        self._set_status(message, "ready")
        self.result.append("\n" + message)
        self._complete_steps(5)

    def refresh_highlight(self):
        if self.controller.active_task is None and self.controller.scene is not None and self.instance.currentData() is not None:
            value = int(self.instance.currentData())
            self.category.setValue(int(self.controller.scene.category_by_instance.get(value, 0)))
            self.canvas.set_data(self.controller.scene, value, None)

    def reset_task(self):
        self.controller.reset_task()
        self.result.clear()
        self._set_status("任务已重置，可重新选择目标", "idle")
        self.canvas.set_data(self.controller.scene, None, None)
        self._reset_steps(0 if self.controller.scene is None else 3)
        self._enable()

    def emergency_stop(self):
        try:
            self.controller.stop()
            self._set_status("已发送机械臂停止命令", "error")
            self._enable()
        except Exception as error:
            QMessageBox.critical(self, "停止失败", str(error))

    def apply_tcp(self):
        values = [field.value() for field in self.tcp_fields]
        self._set_status(self.controller.set_tcp_compensation(values), "ready")

    def closeEvent(self, event):
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
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [family, "Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/real_experiment.yaml")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    configure_application(app)
    window = MainWindow(AppConfig.load(args.config))
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
