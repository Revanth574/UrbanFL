"""
The test split of every client must never reach training or early stopping. Embeddings are replaced by a fake
whose single feature is a unique row code, so we can see exactly which tweets each train_head call received.
"""

import numpy as np
import pytest

import experiments.silo1_baselines as B
from src.fl.partition import build_partition


@pytest.fixture(scope="module")
def part():
    return build_partition(seed=0)


@pytest.mark.parametrize("method", ["local_only", "centralized"])
def test_test_rows_never_used_for_training(monkeypatch, part, method):
    code = {tid: float(i) for i, tid in enumerate(part.df["tweet_id"])}
    monkeypatch.setattr(B, "load_embeddings", lambda ids: np.array([[code[t]] for t in ids], dtype=np.float32))

    seen = []

    def fake_train_sgd(X_tr, y_tr, X_va, y_va, lr, seed, device):
        seen.append((set(X_tr[:, 0]), set(X_va[:, 0])))
        return None, {"best_val_loss": 1.0, "n_val": len(y_va)}

    monkeypatch.setattr(B, "train_sgd", fake_train_sgd)
    monkeypatch.setattr(B, "predict", lambda model, X: np.zeros(len(X), dtype=int))

    B.run_method(method, part, seed=0, device="cpu", lr=0.1)

    test_codes = {code[t] for parts in part.clients.values() for t in parts["test"]}
    assert seen, "train_head was never called"
    assert len(seen) == (len(part.clients) if method == "local_only" else 1)
    for train_codes, val_codes in seen:
        assert not train_codes & test_codes
        assert not val_codes & test_codes
        assert not train_codes & val_codes
