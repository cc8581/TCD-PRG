"""Launch the DDP integration smoke with the Windows-safe file store."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch.multiprocessing as mp

from ddp_smoke_worker import run


def main() -> None:
    output_dir = Path(sys.argv[1]).resolve()
    rendezvous = output_dir.parent / "ddp_test.store"
    rendezvous.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=rendezvous.parent, suffix=".store")
    os.close(descriptor)
    Path(temporary).unlink(missing_ok=True)
    try:
        mp.spawn(
            run,
            args=(2, Path(temporary).resolve().as_uri(), str(output_dir)),
            nprocs=2,
            join=True,
        )
    finally:
        Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
