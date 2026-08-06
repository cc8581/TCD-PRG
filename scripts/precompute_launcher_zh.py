#!/usr/bin/env python3
"""TCD-PRG observation-cache Chinese interactive launcher."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TOOL = PROJECT / "scripts" / "precompute_observation_cache.py"
DEFAULT_CACHE = Path(
    os.environ.get("TCD_PRG_CACHE_DIR", PROJECT / "runtime" / "cache" / "observations")
)


def ask(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]：").strip()
    return value or default


def main() -> int:
    print("=" * 68)
    print("TCD-PRG 训练观测点云缓存生成工具")
    print(f"项目目录：{PROJECT}")
    print(f"默认缓存：{DEFAULT_CACHE}")
    print("=" * 68)
    print("[1] 检查环境和数据")
    print("[2] 预估数量与空间（不生成）")
    print("[3] 正式生成缓存")
    print("[4] 校验已有缓存")
    print("[5] 退出")
    choice = input("请选择 [1-5]：").strip()
    if choice == "5":
        return 0
    modes = {"1": "--check", "2": "--dry-run", "3": "", "4": "--verify-only"}
    if choice not in modes:
        print("输入无效。")
        return 1

    default_count = {"1": "10", "2": "2500", "3": "1", "4": "100"}[choice]
    scene_start = ask("起始 scene_id", "0")
    scene_count = ask("场景数量", default_count)
    split = ask("索引范围 train/val/test/all", "all").lower()
    workers = ask("并行渲染 worker 数", "4")
    cache_dir = ask("缓存保存目录", str(DEFAULT_CACHE))
    min_free = ask("最小剩余空间（GiB）", "50")

    command = [
        sys.executable,
        str(TOOL),
    ]
    if modes[choice]:
        command.append(modes[choice])
    command.extend(
        [
            "--scene-start", scene_start,
            "--scene-count", scene_count,
            "--split", split,
            "--workers", workers,
            "--renderer-mode", "persistent",
            "--cache-dir", cache_dir,
            "--min-free-gb", min_free,
        ]
    )

    print()
    print(f"将处理 scene_{int(scene_start):04d} 开始的 {int(scene_count)} 个场景。")
    print(f"索引范围：{split}")
    print(f"缓存目录：{cache_dir}")
    if choice == "3":
        confirmation = input("确认开始正式生成？输入 YES 继续：").strip()
        if confirmation != "YES":
            print("已取消。")
            return 0
    print()
    completed = subprocess.run(command, cwd=PROJECT, check=False)
    print()
    if completed.returncode == 0:
        print("操作已成功完成。")
    else:
        print(f"程序结束，返回码：{completed.returncode}")
    input("按回车键关闭窗口...")
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
