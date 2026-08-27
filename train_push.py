"""Start the standalone Stage-C push training run."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parent
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from train import main as launcher_main

    sys.argv = ["train_push.py", "--stage", "push", *sys.argv[1:]]
    launcher_main()


if __name__ == "__main__":
    main()
