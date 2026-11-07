"""Silo 1 embedding-cache provenance: only the canonical cache loads, every run records it, and summaries / figures
refuse to combine runs made with different (or unrecorded) caches."""

import hashlib

import numpy as np
import pytest
import torch

import experiments.silo1_embed_cache as E


def _run(sha):
    return {"config": {"environment": {"embedding_cache": None if sha is None else {"sha256": sha}}}, "metrics": {}}


def test_require_single_cache_accepts_one_hash_and_refuses_mixed_or_unrecorded():
    a, b = "a" * 64, "b" * 64
    assert E.require_single_cache({"r1": _run(a), "r2": _run(a)}, "t") == a
    with pytest.raises(ValueError, match="different embedding caches") as err:
        E.require_single_cache({"r1": _run(a), "r2": _run(b), "r3": _run(a)}, "t")
    assert "aaaaaaaaaaaa" in str(err.value) and "bbbbbbbbbbbb" in str(err.value)      # names both groups
    with pytest.raises(ValueError, match="record no embedding-cache hash"):
        E.require_single_cache({"r1": _run(a), "legacy": _run(None)}, "t")
    with pytest.raises(ValueError, match="record no embedding-cache hash"):           # legacy payload: no environment
        E.require_single_cache({"r1": _run(a), "legacy": {"config": {"lr": 0.1}}}, "t")
    with pytest.raises(ValueError, match="no runs"):
        E.require_single_cache({}, "t")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    emb, index = tmp_path / "silo1_emb.npy", tmp_path / "silo1_emb_index.json"
    np.save(emb, np.arange(6, dtype=np.float32).reshape(3, 2))
    index.write_text('{"model": "m", "device": "cuda", "dtype_during_encoding": "float16", "tweet_ids": ["x", "y", "z"]}')
    monkeypatch.setattr(E, "EMB_PATH", emb)
    monkeypatch.setattr(E, "INDEX_PATH", index)
    monkeypatch.setattr(E, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(E, "_LOADED", None)
    monkeypatch.setattr(E, "CANONICAL_EMB_SHA256", hashlib.sha256(emb.read_bytes()).hexdigest())
    return emb


def test_only_the_canonical_cache_loads_and_runs_record_it(cache, monkeypatch):
    assert E.silo1_environment("cpu")["embedding_cache"] is None          # nothing loaded yet
    out = E.load_embeddings(["z", "x"])
    assert out.tolist() == [[4.0, 5.0], [0.0, 1.0]]
    env = E.silo1_environment("cpu")
    assert env["embedding_cache"]["sha256"] == E.CANONICAL_EMB_SHA256
    assert env["embedding_cache"]["encode_device"] == "cuda" and env["torch"] == torch.__version__
    assert env["device"] == "cpu"
    assert E.require_single_cache({"new": {"config": {"environment": env}}}, "t") == E.CANONICAL_EMB_SHA256

    np.save(cache, np.zeros((3, 2), dtype=np.float32))                     # swapped for another encoding
    with pytest.raises(ValueError, match="not the canonical Silo 1 cache"):
        E.load_embeddings(["x"])
    monkeypatch.setattr(E, "CANONICAL_EMB_SHA256", hashlib.sha256(cache.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="changed while this process"):  # even if it were canonical: mid-run swap
        E.load_embeddings(["x"])


def test_figure_is_not_written_from_mixed_caches(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    import experiments.make_figures as F
    monkeypatch.setattr(F, "OUT", tmp_path)
    for runs, ok in (({"silo1_dp/a": _run("a" * 64), "silo1_fl/b": _run("b" * 64)}, False),
                     ({"silo1_dp/a": _run("a" * 64), "silo1_fl/b": _run("a" * 64)}, True)):
        F._USED.clear()
        F._USED.update(runs)
        fig, _ = plt.subplots()
        if ok:
            assert F.save(fig, "ok.png", "source").exists()
        else:
            with pytest.raises(ValueError, match="different embedding caches"):
                F.save(fig, "mixed.png", "source")
            assert not (tmp_path / "mixed.png").exists()
        assert not F._USED                                                   # reset for the next figure
