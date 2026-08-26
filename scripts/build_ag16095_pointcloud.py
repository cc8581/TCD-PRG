"""Build an open AG-160-95 surface cloud from repository CAD hulls."""

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
    parser.add_argument("--points", type=int, default=128)
    args = parser.parse_args()
    root = Path("assets/robots/FR5_AG-160-95/meshes/ag16095/cad_open_reference_collision")
    rng = np.random.default_rng(16095)
    if args.points < 16:
        raise ValueError("--points must be at least 16")
    base = args.points // 2
    left = (args.points - base) // 2
    counts = {"base": base, "left": left, "right": args.points - base - left}
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
