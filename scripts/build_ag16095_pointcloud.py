"""Build the fixed 128-point open AG-160-95 cloud from repository CAD hulls."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def binary_stl_triangles(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    count = struct.unpack_from("<I", payload, 80)[0]
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("tag", "<u2")])
    return np.frombuffer(payload, dtype=dtype, offset=84, count=count)["vertices"]


def sample_part(paths: list[Path], count: int, rng: np.random.Generator) -> np.ndarray:
    triangle = np.concatenate([binary_stl_triangles(path) for path in paths])
    area = (
        np.linalg.norm(
            np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1
        )
        * 0.5
    )
    chosen = rng.choice(len(triangle), count, replace=True, p=area / area.sum())
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    reflected = u + v > 1
    u[reflected] = 1 - u[reflected]
    v[reflected] = 1 - v[reflected]
    selected = triangle[chosen]
    return (
        selected[:, 0]
        + u * (selected[:, 1] - selected[:, 0])
        + v * (selected[:, 2] - selected[:, 0])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="assets/robots/FR5_AG-160-95/ag16095_open_tcp_128.npz")
    args = parser.parse_args()
    root = Path("assets/robots/FR5_AG-160-95/meshes/ag16095/cad_open_reference_collision")
    rng = np.random.default_rng(16095)
    counts = {"base": 64, "left": 32, "right": 32}
    points = []
    parts = []
    for part_id, (name, count) in enumerate(counts.items(), start=1):
        sampled = sample_part(sorted(root.glob(f"{name}_*.stl")), count, rng)
        sampled[:, 2] -= 0.19  # published AG mount -> TCP offset
        points.append(sampled.astype(np.float32))
        parts.append(np.full(count, part_id, np.int64))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        points_tcp=np.concatenate(points),
        part_id=np.concatenate(parts),
        width_m=np.float32(0.095),
        source=np.asarray("cad_open_reference_collision"),
    )


if __name__ == "__main__":
    main()
