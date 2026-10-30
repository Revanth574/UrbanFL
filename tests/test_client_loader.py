"""The persistent multi-client loader must yield exactly the batches the old per-client DataLoaders produced."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.fl.client_loader import PersistentClientLoader, ScheduledBatchSampler
from src.fl.seg_train import EpochPermutationSampler
from src.fl.xbd_data import XBDPairDataset
from tests.test_xbd import make_synthetic_xbd

B = 3
SEEDS = {"ev-a": 11, "ev-b": 22}


@pytest.fixture(scope="module")
def client_datasets(tmp_path_factory):
    split = make_synthetic_xbd(tmp_path_factory.mktemp("xbd_loader"), {"ev-a": (8, 3), "ev-b": (6, 2)})
    ids = {ev: sorted(p.name.split("_pre_")[0] for p in (split / "images").glob(f"{ev}_*_pre_disaster.png"))
           for ev in SEEDS}
    return {ev: XBDPairDataset(split, ids[ev], "damage", train=True, crop=32, seed=5) for ev in SEEDS}


def old_batches(ds, seed, epoch, start=0):
    """The pre-refactor path: one DataLoader per client per local epoch."""
    ds.set_epoch(epoch)
    loader = DataLoader(ds, batch_size=B, sampler=EpochPermutationSampler(len(ds), seed, epoch, start=start * B),
                        num_workers=0)
    return list(loader)


def assert_same(a, b):
    assert a["pair_id"] == b["pair_id"]
    for k in ("pre", "post", "target", "building"):
        assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("num_workers", [0, 2])
def test_batches_identical_to_per_client_loaders(client_datasets, num_workers):
    loader = PersistentClientLoader(client_datasets, B, num_workers=num_workers, seed_of=SEEDS.__getitem__)
    schedule = [("ev-a", 0, 0), ("ev-b", 0, 0), ("ev-a", 1, 0), ("ev-b", 1, 0)]       # two FL rounds, E = 1
    got = list(loader.run(schedule))
    expected = [(c, b) for c, e, _ in schedule for b in old_batches(client_datasets[c], SEEDS[c], e)]
    assert [c for c, _ in got] == [c for c, _ in expected]
    for (_, a), (_, b) in zip(got, expected):
        assert_same(a, b)


def test_resume_offset_matches(client_datasets):
    loader = PersistentClientLoader(client_datasets, B, num_workers=0, seed_of=SEEDS.__getitem__)
    got = [b for _, b in loader.run([("ev-a", 3, 2)])]
    for a, b in zip(got, old_batches(client_datasets["ev-a"], SEEDS["ev-a"], 3, start=2), strict=True):
        assert_same(a, b)


def test_workers_persist_across_schedules(client_datasets):
    loader = PersistentClientLoader(client_datasets, B, num_workers=2, seed_of=SEEDS.__getitem__)
    list(loader.run([("ev-a", 0, 0)]))
    first = loader.loader._iterator                     # torch keeps the worker pool here when persistent
    list(loader.run([("ev-b", 0, 0), ("ev-a", 1, 0)]))
    assert first is not None and loader.loader._iterator is first


def test_sampler_lengths_and_unknown_client():
    s = ScheduledBatchSampler({"a": 10, "b": 4}, batch_size=3, seed_of=lambda c: 0)
    s.set_schedule([("a", 0, 0), ("b", 0, 1)])
    assert len(s) == 4 + 1 and sum(1 for _ in s) == 5
    with pytest.raises(KeyError):
        s.set_schedule([("zzz", 0, 0)])
