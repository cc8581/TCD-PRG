#!/usr/bin/env python3
"""Interactive launcher for read-only legacy-cache validation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TOOL = PROJECT / "scripts" / "precompute_observation_cache.py"
DEFAULT_CACHE = Path(
    os.environ.get("TCD_PRG_CACHE_DIR", PROJECT / "runtime" / "cache" / "observations")
)


def ask(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def main() -> int:
    print("TCD-PRG 旧版训练点云缓存只读验证工具")
    print("本工具不会生成、补齐、删除或修改任何缓存条目。")
    scene_start = ask("起始 scene_id", "0")
    scene_count = ask("场景数量", "100")
    split = ask("索引范围 train/val/test/all", "all").lower()
    cache_dir = ask("缓存目录", str(DEFAULT_CACHE))
    command = [
        sys.executable,
        str(TOOL),
        "--verify-only",
        "--scene-start", scene_start,
        "--scene-count", scene_count,
        "--split", split,
        "--cache-dir", cache_dir,
    ]
    completed = subprocess.run(command, cwd=PROJECT, check=False)
    input("按回车键关闭窗口...")
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
