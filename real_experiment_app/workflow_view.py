"""Operator-visible, branched workflow state for simulation and real devices."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

COMMON_STAGES = ("source", "perception", "target", "grasp")
GRASP_STAGES = ("grasp_execute",)
PUSH_STAGES = ("obstruction", "push_rules", "push_score", "push_execute")


class WorkflowView(QWidget):
    """Show both outcome branches while highlighting only the chosen one."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "simulation"
        self.active_branch: str | None = None
        self.states = {stage: "pending" for stage in (*COMMON_STAGES, *GRASP_STAGES, *PUSH_STAGES)}
        self.nodes: dict[str, QLabel] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(5)
        self.title = QLabel("运行步骤 · 仿真")
        self.title.setObjectName("SectionTitle")
        root.addWidget(self.title)
        for stage in COMMON_STAGES:
            root.addWidget(self._node(stage))
        split = QHBoxLayout()
        split.setSpacing(6)
        self.grasp_branch = self._branch("抓取成功", GRASP_STAGES)
        self.push_branch = self._branch("抓取不可行 → PUSH", PUSH_STAGES)
        split.addWidget(self.grasp_branch, 1)
        split.addWidget(self.push_branch, 1)
        root.addLayout(split)
        self.set_mode("simulation")

    def _node(self, stage: str) -> QLabel:
        node = QLabel()
        node.setObjectName("WorkflowNode")
        node.setWordWrap(True)
        node.setMinimumHeight(31)
        self.nodes[stage] = node
        return node

    def _branch(self, title: str, stages: tuple[str, ...]) -> QFrame:
        panel = QFrame()
        panel.setObjectName("WorkflowBranch")
        panel.setProperty("active", False)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 6, 5, 6)
        layout.setSpacing(5)
        caption = QLabel(title)
        caption.setWordWrap(True)
        layout.addWidget(caption)
        for stage in stages:
            layout.addWidget(self._node(stage))
        layout.addStretch(1)
        return panel

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def set_mode(self, mode: str) -> None:
        if mode not in {"simulation", "real"}:
            raise ValueError(f"Unknown workflow mode: {mode}")
        self.mode = mode
        self.title.setText("运行步骤 · 仿真" if mode == "simulation" else "运行步骤 · 真机")
        labels = {
            "source": "1  加载场景" if mode == "simulation" else "1  点云采集",
            "perception": "2  实例分割",
            "target": "3  目标选择",
            "grasp": "4  目标抓取预测",
            "grasp_execute": "展示抓取结果" if mode == "simulation" else "确认并执行目标抓取",
            "obstruction": "压覆关系推断",
            "push_rules": "规则生成推动参数",
            "push_score": "PUSH evaluator 评估",
            "push_execute": "展示推动结果" if mode == "simulation" else "确认并执行推动",
        }
        for stage, label in labels.items():
            self.nodes[stage].setText(label)
        self.reset()

    def reset(self, completed_common: int = 0) -> None:
        self.states = {stage: "pending" for stage in self.states}
        for stage in COMMON_STAGES[:completed_common]:
            self.states[stage] = "success"
        self.active_branch = None
        self._render()

    def select_branch(self, branch: str) -> None:
        if branch not in {"grasp", "push"}:
            raise ValueError(f"Unknown branch: {branch}")
        self.active_branch = branch
        self._render()

    def start(self, stage: str) -> None:
        self._set(stage, "running")

    def complete(self, stage: str) -> None:
        self._set(stage, "success")

    def fail(self, stage: str) -> None:
        self._set(stage, "failed")

    def _set(self, stage: str, state: str) -> None:
        if stage not in self.states:
            raise ValueError(f"Unknown stage: {stage}")
        if stage in GRASP_STAGES:
            self.active_branch = "grasp"
        elif stage in PUSH_STAGES:
            self.active_branch = "push"
        self.states[stage] = state
        self._render()

    def _render(self) -> None:
        for stage, node in self.nodes.items():
            branch = "grasp" if stage in GRASP_STAGES else "push" if stage in PUSH_STAGES else None
            state = self.states[stage]
            if branch and branch != self.active_branch and self.active_branch is not None:
                state = "inactive"
            node.setProperty("state", state)
            self._repolish(node)
        for branch, panel in (("grasp", self.grasp_branch), ("push", self.push_branch)):
            panel.setProperty("active", branch == self.active_branch)
            self._repolish(panel)
