"""Start the standalone Stage-B task-grasp training run."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if any(arg == "--stage" or arg.startswith("--stage=") for arg in sys.argv[1:]):
        raise SystemExit("train_grasp.py fixes --stage=grasp; do not pass --stage")
    project = Path(__file__).resolve().parent
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from train import main as launcher_main

    sys.argv = ["train_grasp.py", "--stage", "grasp", *sys.argv[1:]]
    launcher_main()


if __name__ == "__main__":
    main()
