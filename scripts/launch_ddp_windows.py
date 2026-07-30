"""Native-Windows multi-GPU launcher using a DNS-independent file store."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch.multiprocessing as mp


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    train_arguments: list[str],
) -> None:
    os.environ.update(
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        TCD_DDP_INIT_METHOD=init_method,
    )
    sys.argv = ["tcd-prg-train", *train_arguments]
    from tcd_prg.scripts.train import main

    main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc-per-node", type=int, required=True)
    args, train_arguments = parser.parse_known_args()
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be positive")
    repository = Path(__file__).resolve().parents[1]
    rendezvous_root = repository / "runtime" / "tmp" / "ddp_rendezvous"
    rendezvous_root.mkdir(parents=True, exist_ok=True)
    file_descriptor, store_name = tempfile.mkstemp(dir=rendezvous_root, suffix=".store")
    os.close(file_descriptor)
    store = Path(store_name)
    store.unlink(missing_ok=True)
    try:
        mp.spawn(
            _worker,
            args=(args.nproc_per_node, store.resolve().as_uri(), train_arguments),
            nprocs=args.nproc_per_node,
            join=True,
        )
    finally:
        store.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
