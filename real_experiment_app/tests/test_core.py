import numpy as np
from pathlib import Path
import yaml

from real_experiment_app.config import AppConfig
from real_experiment_app.perception import fuse_frames
from real_experiment_app.transforms import model_pose_to_robot_pose, xyz_rpy_to_matrix
from real_experiment_app.controller import preserve_instance_ids
from real_experiment_app.types import RGBDFrame, SegmentationResult


def synthetic_observation(camera_id: str, view_index: int = 0):
    height, width = 80, 120
    yy, xx = np.mgrid[:height, :width]
    color = np.zeros((height, width, 3), np.uint8)
    depth = np.zeros((height, width), np.float32)
    labels = np.full((height, width), -1, np.int32)
    objects = (
        ((xx - 45) ** 2 + (yy - 44) ** 2 < 13 ** 2, 690., (50, 160, 240)),
        ((xx - 78) ** 2 + (yy - 40) ** 2 < 11 ** 2, 745., (235, 120, 70)),
        ((xx - 62) ** 2 + (yy - 22) ** 2 < 8 ** 2, 650., (90, 220, 120)),
    )
    for instance_id, (mask, z, rgb) in enumerate(objects):
        depth[mask], color[mask], labels[mask] = z, rgb, instance_id
    transform = np.eye(4)
    transform[:3, 3] = (0.5, 0.01 * view_index, -0.55)
    frame = RGBDFrame(camera_id, color, depth,
                      {"fx": 115., "fy": 115., "cx": width / 2, "cy": height / 2}, transform)
    return frame, SegmentationResult(labels, {0: 0, 1: 1, 2: 2})


def test_tcp_identity_and_translation():
    pose = [.4,.1,.3,0,0,0,1]
    assert np.allclose(model_pose_to_robot_pose(pose,np.eye(4)),[400,100,300,0,0,0])
    compensation = xyz_rpy_to_matrix([10,-20,30,0,0,0],.001)
    assert np.allclose(model_pose_to_robot_pose(pose,compensation)[:3],[410,80,330])


def test_two_view_fusion_associates_instances():
    observations = [synthetic_observation("a", 0), synthetic_observation("b", 1)]
    scene=fuse_frames([x[0] for x in observations], [x[1] for x in observations],{
        "depth_min_mm":150,"depth_max_mm":1500,"voxel_size_m":.005,
        "workspace_min_m":[.1,-.5,-.05],"workspace_max_m":[.95,.5,.65],
        "instance_association_distance_m":.06,
        "instance_association_color_distance":.35})
    assert scene.instance_ids == [0,1,2]
    assert set(scene.source_view.tolist()) == {0,1}


def test_temporal_tracking_preserves_ids():
    frame, segmentation = synthetic_observation("a", 0)
    settings={"depth_min_mm":150,"depth_max_mm":1500,"voxel_size_m":.005,
        "workspace_min_m":[.1,-.5,-.05],"workspace_max_m":[.95,.5,.65]}
    first=fuse_frames([frame],[segmentation],settings)
    second=fuse_frames([frame],[segmentation],settings)
    second.instance_id=np.where(second.instance_id==0,9,second.instance_id)
    second.category_by_instance[9]=second.category_by_instance.pop(0)
    tracked=preserve_instance_ids(first,second,.10)
    assert 0 in tracked.instance_ids


def test_operator_settings_are_persisted(tmp_path: Path):
    path = tmp_path / "settings.yaml"
    path.write_text("robot:\n  ip: 192.168.58.2\n", encoding="utf-8")
    config = AppConfig.load(path)
    config.raw["robot"]["tool_id"] = 1
    config.raw["robot"]["gripper_company"] = 4
    config.save()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["robot"]["tool_id"] == 1
    assert saved["robot"]["gripper_company"] == 4
