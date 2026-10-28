"""
Generic segmentation training loop with AMP and exact checkpoint/resume (src/fl/checkpoint.py).

Determinism contract (what makes resume exact):
  - sample order of epoch e = permutation drawn from Generator(seed * 1_000_003 + e), independent of any running RNG,
    and the DataLoader gets its own generator so creating an iterator never consumes the global torch RNG;
  - dataset augmentations depend on (seed, epoch, index) via dataset.set_epoch(e) (XBDPairDataset does this);
  - model / optimizer / GradScaler / RNG states are restored from the checkpoint;
  so a run interrupted after step k and resumed from its last checkpoint follows the same trajectory as an
  uninterrupted run (bit-identical on CPU; on GPU up to cuDNN nondeterminism).

The loop is model/task agnostic: `batch_to_io(batch, device) -> (inputs, target)` adapts dataset items to the model.
It is used for centralized training now and as the client update in Silo 2 FL later.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from src.fl.checkpoint import CheckpointManager, rng_state, set_rng_state


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    num_workers: int = 0
    seed: int = 0
    max_steps: int | None = None       # stop early (feasibility runs, tests)
    log_every: int = 25


class EpochPermutationSampler(Sampler[int]):
    """Deterministic per-epoch shuffle; `start` skips the first `start` positions (mid-epoch resume)."""

    def __init__(self, n: int, seed: int, epoch: int, start: int = 0):
        self.n, self.seed, self.epoch, self.start = n, seed, epoch, start

    def order(self) -> list[int]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        return torch.randperm(self.n, generator=g).tolist()

    def __iter__(self) -> Iterator[int]:
        return iter(self.order()[self.start:])

    def __len__(self) -> int:
        return max(0, self.n - self.start)


def loc_batch_to_io(batch: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage-1 localisation: pre image -> building mask."""
    return batch["pre"].to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)


def train(model: nn.Module, dataset: Dataset, cfg: TrainConfig, loss_fn: Callable, device: str | torch.device,
          ckpt: CheckpointManager | None = None,
          batch_to_io: Callable = loc_batch_to_io,
          on_step: Callable[[int, float], None] | None = None) -> dict:
    """
    Train for cfg.epochs (or cfg.max_steps) and return timing/memory stats. Resumes automatically from the newest
    checkpoint in `ckpt` if there is one. `on_step(global_step, loss)` is called after every optimizer step.
    """
    device = torch.device(device)
    use_amp = cfg.amp and device.type == "cuda"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epoch, step_in_epoch, global_step, resumed_from = 0, 0, 0, None

    state = ckpt.load_latest(map_location=device) if ckpt else None
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        epoch, step_in_epoch, global_step = state["epoch"], state["step_in_epoch"], state["global_step"]
        resumed_from = global_step

    def save() -> None:
        if ckpt is not None:
            ckpt.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                       "rng": rng_state(), "epoch": epoch, "step_in_epoch": step_in_epoch,
                       "train_config": asdict(cfg)}, global_step)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses, data_wait, compute, t_start = [], 0.0, 0.0, time.perf_counter()
    done = False
    while epoch < cfg.epochs and not done:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler = EpochPermutationSampler(len(dataset), cfg.seed, epoch, start=step_in_epoch * cfg.batch_size)
        # A DataLoader draws its worker base seed from the GLOBAL torch RNG whenever an iterator is created; a
        # resumed run creates one mid-epoch, which would shift every later dropout mask. A dedicated per-epoch
        # generator keeps the global RNG untouched (and seeds workers deterministically).
        loader_gen = torch.Generator().manual_seed(cfg.seed * 1_000_003 + epoch + 7)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers,
                            pin_memory=device.type == "cuda", drop_last=False, persistent_workers=False,
                            generator=loader_gen)
        model.train()
        t_wait = time.perf_counter()
        for batch in loader:
            t0 = time.perf_counter()
            data_wait += t0 - t_wait
            x, y = batch_to_io(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss = loss_fn(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1
            step_in_epoch += 1
            loss_v = loss.item()                     # syncs the GPU, so `compute` measures real work
            compute += time.perf_counter() - t0
            if global_step % cfg.log_every == 0 or global_step == 1:
                losses.append({"step": global_step, "epoch": epoch, "loss": loss_v})
            if on_step is not None:
                on_step(global_step, loss_v)
            if ckpt is not None and ckpt.due(global_step):
                save()
            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                done = True
                break
            t_wait = time.perf_counter()
        if not done:
            epoch, step_in_epoch = epoch + 1, 0
            save()                                   # end-of-epoch checkpoint
    wall = time.perf_counter() - t_start
    stats = {"global_step": global_step, "epoch": epoch, "resumed_from_step": resumed_from, "wall_s": wall,
             "data_wait_s": data_wait, "compute_s": compute, "loss_log": losses}
    if device.type == "cuda":
        stats["peak_vram_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        stats["peak_vram_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    return stats
