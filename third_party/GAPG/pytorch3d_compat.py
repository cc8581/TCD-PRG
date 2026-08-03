"""Small Windows fallback for the PyTorch3D farthest-point sampling op.

The repository's documented PyTorch3D package is a Linux-only conda build.
This module implements the single PyTorch3D operation used by GAPG so the
project can run in a native Windows conda environment.
"""

import torch


def sample_farthest_points(points, lengths=None, K=50, random_start_point=False):
    """Return farthest-point samples and their indices.

    This mirrors the subset of ``pytorch3d.ops.sample_farthest_points`` used by
    GAPG: ``points`` has shape ``[B, N, D]`` and ``K`` is an integer.
    """
    if points.ndim != 3:
        raise ValueError("points must have shape [B, N, D]")
    batch_size, point_count, dims = points.shape
    k = int(K)
    if k < 0:
        raise ValueError("K must be non-negative")

    if lengths is None:
        lengths = torch.full(
            (batch_size,), point_count, dtype=torch.int64, device=points.device
        )
    else:
        lengths = lengths.to(device=points.device, dtype=torch.int64)

    indices = torch.full(
        (batch_size, k), -1, dtype=torch.int64, device=points.device
    )
    if k == 0:
        return points.new_empty((batch_size, 0, dims)), indices

    valid = torch.arange(point_count, device=points.device)[None, :] < lengths[:, None]
    if torch.any(lengths <= 0):
        raise ValueError("all point clouds must contain at least one valid point")

    if random_start_point:
        current = (torch.rand(batch_size, device=points.device) * lengths).long()
    else:
        current = torch.zeros(batch_size, dtype=torch.int64, device=points.device)

    min_distances = torch.full(
        (batch_size, point_count), torch.inf, dtype=points.dtype, device=points.device
    )
    batch = torch.arange(batch_size, device=points.device)
    steps = min(k, int(lengths.max().item()))
    for step in range(steps):
        active = step < lengths
        indices[active, step] = current[active]
        selected = points[batch, current]
        distances = ((points - selected[:, None, :]) ** 2).sum(dim=-1)
        min_distances = torch.minimum(min_distances, distances)
        min_distances = min_distances.masked_fill(~valid, -1)
        current = min_distances.argmax(dim=1)

    gather_idx = indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, dims)
    sampled = points.gather(1, gather_idx)
    sampled = sampled.masked_fill((indices < 0).unsqueeze(-1), 0)
    return sampled, indices
