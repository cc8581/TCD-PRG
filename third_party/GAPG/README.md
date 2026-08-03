# Minimal GAPG runtime subset

This directory contains only the original GAPG modules imported by
`scripts/run_gapg_baseline_worker_py38.py`: the grasp and push evaluators,
PointNet++ utilities, geometric preprocessing, constants and the PyTorch3D
farthest-point fallback.

The files were moved without algorithmic changes from TCD-PRG provenance commit
`2523dec701c7b1c01c5e481c595d622b1e0a47d2`. GAPG data collection, training,
demo, UR5e/YCB simulation and visualization assets are intentionally excluded
because the unified TCD-PRG baseline interface does not import them.

GraspNet source, GraspNetAPI and all pretrained weights remain external,
Git-ignored dependencies under `.deps/`.
