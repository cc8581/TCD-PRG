"""Two-process CPU/Gloo integration worker invoked outside pytest collection."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from tcd_prg.config import LoggingConfig, TCDPRGConfig, TrainingConfig
from tcd_prg.trainers import Trainer


def run(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size
    )
    config = TCDPRGConfig(
        training=TrainingConfig(
            device="cpu",
            amp=False,
            max_optimizer_steps=1,
            gradient_accumulation_steps=1,
            validation_interval=100,
        ),
        logging=LoggingConfig(backend="none", log_interval=1),
        output_dir=str(Path(output_dir).resolve()),
    )
    model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(3, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def loss_step(module, batch):
        loss = torch.nn.functional.mse_loss(module(batch["x"]), batch["y"])
        return loss, {"loss_total": loss}

    generator = torch.Generator().manual_seed(100 + rank)
    loader = [
        {"x": torch.randn(2, 3, generator=generator),
         "xyz": torch.randn(2, 4, 3, generator=generator),
         "y": torch.randn(2, 1, generator=generator)}
        for _ in range(1)
    ]
    trainer = Trainer(model, optimizer, config, loss_step)
    state = trainer.train(loader, groups_per_effective_epoch=4)
    trainer.save_checkpoint(Path(config.output_dir) / "last.pt")
    torch.distributed.barrier()
    if rank == 0:
        checkpoint = torch.load(
            Path(config.output_dir) / "last.pt", map_location="cpu", weights_only=False
        )
        assert state.optimizer_steps == 1
        assert state.candidate_groups_seen == 4
        assert len(checkpoint["rng_by_rank"]) == world_size
        metrics = [
            json.loads(line)
            for line in (Path(config.output_dir) / "train_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert metrics[-1]["metric_scope"] == "global"
    torch.distributed.destroy_process_group()


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run(rank, world_size, "env://", sys.argv[1])


if __name__ == "__main__":
    main()
