"""
One persistent DataLoader for many clients, driven by a client-aware batch sampler.

Why: on Windows every new DataLoader with workers spawns fresh processes (~10 s measured for 4 workers). Creating one
loader per client per round made worker start-up ~1/3 of a projected FL run. Here the workers start ONCE
(persistent_workers=True) and a schedule decides which client's batches come next.

Sampling contract (identical to what a per-client DataLoader with EpochPermutationSampler would produce):
  - client c, local epoch e: order = torch.randperm(n_c, Generator(seed_c * 1_000_003 + e)), cut into batches;
  - each sampled index is the tuple (client, local_index, epoch) and the dataset returns
    datasets[client].get(local_index, epoch) — the epoch travels with the index because persistent workers keep
    their own dataset copies and would never see set_epoch() calls from the main process;
  - sampling happens in the main process, so a new schedule takes effect at the next iteration.
"""

from __future__ import annotations

from typing import Callable, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

Key = tuple[str, int, int]          # (client, local index, epoch)


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """The per-epoch permutation shared with seg_train.EpochPermutationSampler."""
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1_000_003 + epoch)).tolist()


class ScheduledBatchSampler(Sampler[list]):
    """Yields batches of (client, local_index, epoch) for a schedule of (client, epoch, start_batch) entries."""

    def __init__(self, sizes: dict[str, int], batch_size: int | dict[str, int], seed_of: Callable[[str], int]):
        self.sizes, self.seed_of = dict(sizes), seed_of
        self.batch_size = batch_size if isinstance(batch_size, dict) else {c: batch_size for c in sizes}
        self.schedule: list[tuple[str, int, int]] = []

    def set_schedule(self, schedule: Sequence[tuple[str, int, int]]) -> None:
        unknown = {c for c, _, _ in schedule} - set(self.sizes)
        if unknown:
            raise KeyError(f"unknown clients in schedule: {sorted(unknown)}")
        self.schedule = list(schedule)

    def batches(self, client: str, epoch: int, start_batch: int = 0) -> list[list[Key]]:
        order = epoch_order(self.sizes[client], self.seed_of(client), epoch)
        b = self.batch_size[client]
        return [[(client, i, epoch) for i in order[k:k + b]] for k in range(start_batch * b, len(order), b)]

    def n_batches(self, client: str, start_batch: int = 0) -> int:
        return max(0, -(-self.sizes[client] // self.batch_size[client]) - start_batch)

    def __iter__(self) -> Iterator[list[Key]]:
        for client, epoch, start in self.schedule:
            yield from self.batches(client, epoch, start)

    def __len__(self) -> int:
        return sum(self.n_batches(c, s) for c, _, s in self.schedule)


class MultiClientDataset(Dataset):
    """Routes (client, local_index, epoch) keys to the client's dataset. Datasets without get(i, epoch) fall back to
    set_epoch + __getitem__, executed inside the worker on the worker's own copy."""

    def __init__(self, datasets: dict[str, Dataset]):
        self.datasets = datasets

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets.values())

    def __getitem__(self, key: Key):
        client, i, epoch = key
        ds = self.datasets[client]
        if hasattr(ds, "get"):
            return ds.get(i, epoch)
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        return ds[i]


class PersistentClientLoader:
    """
    loader = PersistentClientLoader({"a": ds_a, "b": ds_b}, batch_size=4, num_workers=4, seed_of=...)
    for client, batch in loader.run([("a", epoch, 0), ("b", epoch, 0)]): ...
    Workers are created once, on the first run(), and reused for every later schedule. `batch_size` may be a dict per
    key, so one loader (one worker pool) can serve train batches and full-resolution eval batches of a whole run;
    call close() when the run ends.
    """

    def __init__(self, datasets: dict[str, Dataset], batch_size: int | dict[str, int], num_workers: int = 0,
                 seed_of: Callable[[str], int] | int = 0, pin_memory: bool = False, loader_seed: int = 0):
        seed_fn = seed_of if callable(seed_of) else (lambda _c, s=seed_of: s)
        self.sampler = ScheduledBatchSampler({c: len(d) for c, d in datasets.items()}, batch_size, seed_fn)
        self.loader = DataLoader(MultiClientDataset(datasets), batch_sampler=self.sampler, num_workers=num_workers,
                                 persistent_workers=num_workers > 0, pin_memory=pin_memory,
                                 generator=torch.Generator().manual_seed(loader_seed))

    def run(self, schedule: Sequence[tuple[str, int, int]]) -> Iterator[tuple[str, dict]]:
        """Yield (client, batch) for the whole schedule, in order."""
        self.sampler.set_schedule(schedule)
        owners = [c for c, _, s in schedule for _ in range(self.sampler.n_batches(c, s))]
        for client, batch in zip(owners, self.loader):
            yield client, batch

    def close(self) -> None:
        """Terminate the persistent workers now (each one is a full process holding torch's CUDA libraries; on
        Windows too many alive at once exhausts the commit limit)."""
        it = getattr(self.loader, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
        self.loader._iterator = None
