import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from real_experiment_app.workflow_view import WorkflowView


def test_both_branches_are_visible_and_only_entered_branch_is_highlighted():
    app = QApplication.instance() or QApplication([])
    view = WorkflowView()
    view._test_app = app
    assert view.grasp_branch.isVisibleTo(view)
    assert view.push_branch.isVisibleTo(view)
    assert "加载场景" in view.nodes["source"].text()
    view.complete("grasp")
    view.select_branch("push")
    view.start("obstruction")
    assert view.active_branch == "push"
    assert view.push_branch.property("active") is True
    assert view.grasp_branch.property("active") is False
    assert view.nodes["obstruction"].property("state") == "running"
    assert view.nodes["grasp_execute"].property("state") == "inactive"
    view.set_mode("real")
    assert "点云采集" in view.nodes["source"].text()
    assert "执行目标抓取" in view.nodes["grasp_execute"].text()
    assert view.active_branch is None
