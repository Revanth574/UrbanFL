"""
Unseen-disaster protocol: no row of the held-out event may reach training, class weights or checkpoint selection.
Embeddings are replaced by a fake whose single feature is a unique row code, so we can see exactly which tweets each
training call received.
"""

import numpy as np
import pytest
import torch

import experiments.silo1_baselines as B
import experiments.silo1_unseen as U
from src.fl.partition import build_partition

HOLDOUT = "2014_Chile_Earthquake_cl"


@pytest.fixture(scope="module")
def part():
    return build_partition(seed=0)


@pytest.fixture
def data(monkeypatch, part):
    code = {tid: float(i) for i, tid in enumerate(part.df["tweet_id"])}
    monkeypatch.setattr(B, "load_embeddings", lambda ids: np.array([[code[t]] for t in ids], dtype=np.float32))
    return U.event_arrays(part)


def codes(data, events, splits):
    return {float(x) for e in events for s in splits for x in data[e][s][0][:, 0]}


def test_heldout_set_is_every_row_of_the_event(data, part):
    X, y, ids, split = U.heldout_set(data, HOLDOUT)
    assert sorted(ids) == sorted(t for s in U.SPLITS for t in part.clients[HOLDOUT][s])
    assert len(X) == len(y) == len(split) == len(ids)


@pytest.mark.parametrize("method", ["centralized", U.FL_NAME])
def test_heldout_rows_never_reach_training(monkeypatch, data, method):
    seen = []

    def fake_train_sgd(X_tr, y_tr, X_va, y_va, lr, seed, device):
        seen.append((set(X_tr[:, 0]), set(X_va[:, 0]), set()))
        return None, {}

    def fake_run_fl(algorithm, clients, *a, class_weights=None, **k):
        seen.append(tuple({float(x) for c in clients for x in getattr(c, f)[:, 0]}
                          for f in ("X_train", "X_val", "X_test")))
        raise StopIteration   # inspected the inputs; skip the model build

    monkeypatch.setattr(U, "train_sgd", fake_train_sgd)
    monkeypatch.setattr(U, "run_fl", fake_run_fl)
    train_events = [e for e in sorted(data) if e != HOLDOUT]
    train = U.train_fedavg if method == U.FL_NAME else U.train_centralized
    try:
        train({e: data[e] for e in train_events}, train_events, 0.1, 0, "cpu")
    except StopIteration:
        pass

    heldout = codes(data, [HOLDOUT], U.SPLITS)
    assert len(seen) == 1
    for used in seen[0]:
        assert not used & heldout
    tr, va, _ = seen[0]
    assert tr == codes(data, train_events, ["train"]) and va == codes(data, train_events, ["val"])


def test_fedavg_class_weights_exclude_heldout(monkeypatch, data):
    got = {}

    def fake_run_fl(algorithm, clients, *a, class_weights=None, **k):
        got["w"] = class_weights
        raise StopIteration

    monkeypatch.setattr(U, "run_fl", fake_run_fl)
    train_events = [e for e in sorted(data) if e != HOLDOUT]
    with pytest.raises(StopIteration):
        U.train_fedavg({e: data[e] for e in train_events}, train_events, 0.1, 0, "cpu")
    counts = sum(np.bincount(data[e]["train"][1], minlength=U.N_CLASSES) for e in train_events)
    assert torch.allclose(got["w"], U.class_weights_from_counts(counts))


def test_local_model_trains_on_own_event_only(monkeypatch, data):
    seen = []
    monkeypatch.setattr(U, "train_sgd", lambda X_tr, y_tr, X_va, y_va, lr, seed, device:
                        seen.append((set(X_tr[:, 0]), set(X_va[:, 0]))))
    U.train_local(data, HOLDOUT, 0.1, 0, "cpu")
    assert seen == [(codes(data, [HOLDOUT], ["train"]), codes(data, [HOLDOUT], ["val"]))]


def test_sweep_never_scores_heldout(monkeypatch, tmp_path, data, part):
    """The per-fold lr sweep only trains on the 9; it never predicts on (or loads) the held-out event."""
    seen = []

    def fake_train_centralized(d, train_events, lr, seed, device):
        seen.append(set(d))
        return None, {"best_val_loss": lr, "best_epoch": 1, "history": [{"val_loss": lr}]}

    monkeypatch.setattr(U, "train_centralized", fake_train_centralized)
    monkeypatch.setattr(U, "predict", lambda *a: pytest.fail("sweep must not predict"))
    monkeypatch.setattr(U, "SWEEP_DIR", tmp_path)
    m = U.sweep_run("centralized", part, data, HOLDOUT, 0.01, "cpu", {"commit": "x", "dirty": False})
    assert seen == [set(data) - {HOLDOUT}] and m["val_loss"] == 0.01


def test_resource_guard(tmp_path):
    log = tmp_path / "campaign_x.log"
    log.write_text("CRASH old\n", encoding="utf-8")
    guard = U.ResourceGuard(None, str(tmp_path / "campaign_*.log"))
    guard.check()                                   # pre-existing CRASH lines are ignored
    with log.open("a", encoding="utf-8") as f:
        f.write("CKPT ok\nCRASH step (attempt 1)\n")
    with pytest.raises(U.GuardStop):
        guard.check()
    with pytest.raises(U.GuardStop):
        U.ResourceGuard(1e9, None).check()           # RAM floor no machine meets


def test_fedavg_vs_centralized_verdict():
    events = [f"e{i}" for i in range(10)]
    seeds = list(range(5))

    def per_seed(diff_fn):
        return {s: {U.FL_NAME: {"per_event": {h: {"all_rows": {"d4_macro_f1": 0.5 + diff_fn(i, s)}}
                                              for i, h in enumerate(events)}},
                    "centralized": {"per_event": {h: {"all_rows": {"d4_macro_f1": 0.5}} for h in events}}}
                for s in seeds}

    noisy = U.fedavg_vs_centralized(per_seed(lambda i, s: 0.01 * (-1) ** s), seeds, events)
    assert noisy["verdict"] == "comparable" and noisy["clear"] == {"fedavg_better": 0, "centralized_better": 0}
    clear = U.fedavg_vs_centralized(per_seed(lambda i, s: 0.05 + 0.001 * s), seeds, events)
    assert clear["verdict"] == "fedavg_better" and clear["wins_by_seed_mean"] == 10


def test_holm():
    adj = U.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
