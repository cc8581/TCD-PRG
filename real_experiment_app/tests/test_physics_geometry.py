import numpy as np

from real_experiment_app.physics_worker import _grasp_contact_report, transformed_cloud


def test_final_pose_contact_report_keeps_every_neighbor_id_and_other_collision():
    class Bullet:
        def getContactPoints(self, _robot, body, physicsClientId):
            assert physicsClientId == 0
            return [(None, None, None, link) for link in {
                10: [], 20: [3], 30: [4], 40: [5], 50: [6],
            }[body]]

    report = _grasp_contact_report(Bullet(), 1, 10, {2: 20, 7: 30, 8: 40, 9: 50},
                                   target=2, allowed={3}, client=0)
    assert report["neighbor_ids"] == [7, 8, 9]
    assert report["collision_free"] is False
    assert report["table_collision"] is False
    assert report["target_non_finger_collision"] is False


def test_final_gap_transforms_both_dynamic_objects():
    points = np.asarray([[0, 0, 0], [0.01, 0, 0]], float)
    moved = transformed_cloud(points, np.zeros(3), [0.10, 0, 0], [0, 0, 0, 1])
    target = transformed_cloud(
        points + [0.04, 0, 0], np.asarray([0.04, 0, 0]), [0.12, 0, 0], [0, 0, 0, 1]
    )
    # Final gap is .01, whereas comparing moved with the stale initial target
    # would incorrectly report a much larger separation.
    assert np.isclose(
        np.min(np.linalg.norm(moved[:, None, :2] - target[None, :, :2], axis=2)), 0.01
    )
