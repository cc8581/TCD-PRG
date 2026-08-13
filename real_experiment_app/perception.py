from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import numpy as np

from .types import FusedScene, RGBDFrame, SegmentationResult


class ExternalCommandSegmenter:
    """External model contract: output NPZ needs instance_image and optional mapping."""

    def __init__(self, command: list[str]):
        self.command = [str(x) for x in command]

    def segment(self, frame: RGBDFrame) -> SegmentationResult:
        if not self.command:
            raise RuntimeError("尚未配置实例分割程序，请在配置文件中填写 segmentation.command")
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"input.npz", Path(directory)/"output.npz"
            np.savez_compressed(source, color_rgb=frame.color_rgb, depth_mm=frame.depth_mm)
            completed = subprocess.run(self.command+["--input",str(source),"--output",str(output)],
                                       capture_output=True, text=True, check=False)
            if completed.returncode or not output.is_file():
                raise RuntimeError(f"segmentation failed: {completed.stderr}")
            with np.load(output, allow_pickle=False) as data:
                labels = data["instance_image"].astype(np.int32)
                mapping = ({int(k): int(v) for k,v in zip(
                    data["category_keys"], data["category_values"], strict=True)}
                    if "category_keys" in data and "category_values" in data else {})
            return SegmentationResult(labels, mapping)


def build_segmenter(config):
    section = config.raw["segmentation"]
    return ExternalCommandSegmenter(section.get("command", []))


def _points(frame: RGBDFrame, segmentation: SegmentationResult,
            depth_min: float, depth_max: float):
    depth = frame.depth_mm
    valid = ((depth >= depth_min) & (depth <= depth_max)
             & (segmentation.instance_image >= 0))
    v, u = np.nonzero(valid)
    z = depth[v,u].astype(np.float64) * .001
    intr = frame.intrinsics
    x = (u-intr["cx"])*z/intr["fx"]; y = (v-intr["cy"])*z/intr["fy"]
    camera = np.column_stack((x,y,z,np.ones_like(z)))
    base = (frame.camera_to_base @ camera.T).T[:, :3]
    rgb = frame.color_rgb[v,u].astype(np.float32)/255.
    return base.astype(np.float32), rgb, segmentation.instance_image[v,u].astype(np.int64)


def fuse_frames(frames: list[RGBDFrame], segments: list[SegmentationResult],
                settings: dict) -> FusedScene:
    xyzs, rgbs, ids, views = [], [], [], []
    categories: dict[int,int] = {}
    global_descriptors: list[tuple[np.ndarray,np.ndarray,int]] = []
    association_distance = float(settings.get("instance_association_distance_m", .06))
    association_color = float(settings.get("instance_association_color_distance", .35))
    for view, (frame, result) in enumerate(zip(frames, segments, strict=True)):
        xyz, rgb, instance = _points(frame, result, settings["depth_min_mm"],
                                     settings["depth_max_mm"])
        associated = np.full_like(instance, -1)
        for local_id in sorted(int(x) for x in np.unique(instance) if int(x) >= 0):
            rows = instance == local_id
            centroid, mean_color = xyz[rows].mean(0), rgb[rows].mean(0)
            match = None; best = float("inf")
            for global_id, (prior_xyz, prior_rgb, observations) in enumerate(global_descriptors):
                spatial = float(np.linalg.norm(centroid-prior_xyz))
                color_distance = float(np.linalg.norm(mean_color-prior_rgb))
                if spatial <= association_distance and color_distance <= association_color and spatial < best:
                    match, best = global_id, spatial
            if match is None:
                match = len(global_descriptors)
                global_descriptors.append((centroid,mean_color,1))
            else:
                prior_xyz, prior_rgb, observations = global_descriptors[match]
                count = observations+1
                global_descriptors[match] = ((prior_xyz*observations+centroid)/count,
                                             (prior_rgb*observations+mean_color)/count,count)
            associated[rows] = match
            categories[match] = int(result.category_by_instance.get(local_id,0))
        xyzs.append(xyz); rgbs.append(rgb); ids.append(instance)
        ids[-1] = associated
        views.append(np.full(len(xyz), view, np.int16))
    if not xyzs or not sum(map(len, xyzs)): raise RuntimeError("No segmented 3D points")
    xyz, rgb, instance, source = map(np.concatenate, (xyzs,rgbs,ids,views))
    low, high = np.asarray(settings["workspace_min_m"]), np.asarray(settings["workspace_max_m"])
    keep = np.all((xyz >= low) & (xyz <= high), axis=1)
    xyz, rgb, instance, source = xyz[keep],rgb[keep],instance[keep],source[keep]
    voxel = float(settings["voxel_size_m"])
    keys = np.floor(xyz/voxel).astype(np.int64)
    _, selected = np.unique(keys, axis=0, return_index=True)
    selected.sort()
    xyz, rgb, instance, source = xyz[selected],rgb[selected],instance[selected],source[selected]
    return FusedScene(xyz, rgb, instance.astype(np.int64), source, categories)
