"""Orderly shutdown for the application's own subprocess workers."""

from __future__ import annotations

import json
import os
import subprocess


def stop_worker_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return

    stream = getattr(process, "stdin", None)
    if stream is not None and not stream.closed:
        try:
            stream.write(json.dumps({"command": "close"}) + "\n")
            stream.flush()
            process.wait(timeout=2)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass

    # A busy PyBullet worker can own a ProcessPoolExecutor. Terminating only
    # its parent leaves Python children behind, so stop this worker's tree.
    pid = getattr(process, "pid", None)
    if os.name == "nt" and isinstance(pid, int) and process.poll() is None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
            )
            process.wait(timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
