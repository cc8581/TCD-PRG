"""Start the standalone Stage-A perception training run."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if any(arg == "--stage" or arg.startswith("--stage=") for arg in sys.argv[1:]):
        raise SystemExit("train_perception.py fixes --stage=perception; do not pass --stage")
    project = Path(__file__).resolve().parent
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from train import main as launcher_main

    sys.argv = ["train_perception.py", "--stage", "perception", *sys.argv[1:]]
    launcher_main()


if __name__ == "__main__":
    main()
