"""Pre-registered stage-2 follow-ups: D1 diagnostics agree with the official scorer; the grid-edge rule; step order."""

import numpy as np
import pytest
import torch

from src.fl.dmg_diag import collect_scored, confusion, damage_counts, fit_offsets, harmonic, mean_softmax, predict
from src.fl.dmg_eval import xview2_evaluate
from src.fl.xbd_metrics import XView2Scorer


def _images(rng, n=4, h=24, w=24):
    """Synthetic (loc mask, pre target, damage class 0..3, post target 0..4) per image."""
    out = []
    for _ in range(n):
        post = rng.integers(0, 5, (h, w))
        post[rng.random((h, w)) < 0.4] = 0
        pre = ((post > 0) | (rng.random((h, w)) < 0.1)).astype(np.int64)
        out.append((rng.random((h, w)) < 0.7, pre, rng.integers(0, 4, (h, w)), post))
    return out


def test_counts_and_harmonic_equal_the_official_scorer():
    rng = np.random.default_rng(0)
    s, dts, preds = XView2Scorer(), [], []
    for lp, pre, dcls, post in _images(rng):
        s.update(lp, pre, dcls, post)
        scored = post > 0
        dts.append(post[scored])
        preds.append(((dcls + 1) * lp)[scored])
    counts = damage_counts(np.concatenate(dts), np.concatenate(preds))
    assert (counts == s.dmg.astype(np.int64)).all()
    assert harmonic(counts) == pytest.approx(s.compute()["damage_f1_harmonic"], abs=1e-12)
    conf = confusion(np.concatenate(dts), np.concatenate(preds))
    assert conf.shape == (4, 5) and conf.sum() == len(np.concatenate(dts))
    assert (np.diag(conf[:, 1:]) == counts[:, 0]).all()                    # diagonal = TP


class _Loader:
    """Stands in for PersistentClientLoader.run: one client, fixed batches."""

    def __init__(self, batches):
        self.batches = batches

    def run(self, keys):
        for key, _, _ in keys:
            for b in self.batches:
                yield key, b


def test_collect_scored_reproduces_xview2_evaluate():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    batches = []
    for k in range(2):
        post = torch.as_tensor(rng.integers(0, 5, (2, 16, 16)))
        batches.append({"pre": torch.randn(2, 3, 16, 16), "post": torch.randn(2, 3, 16, 16),
                        "target": torch.zeros(2, 16, 16, dtype=torch.long), "pre_target": (post > 0).long(),
                        "post_target": post, "pair_id": [f"p{k}a", f"p{k}b"]})
    loc, dmg = torch.nn.Conv2d(3, 1, 3, padding=1), torch.nn.Conv2d(6, 4, 3, padding=1)
    ref = xview2_evaluate(loc, dmg, _Loader(batches), "test/", ["c"], ".", "cpu", with_buildings=False)["c"]
    got = collect_scored(loc, dmg, _Loader(batches), "test/", ["c"], "cpu")
    assert got["loc_tp_fp_fn"] == ref["counts"]["loc_tp_fp_fn"] and got["n_images"] == 4
    counts = damage_counts(got["dt"], predict(got["logits"], got["lp"]))
    assert counts.tolist() == [ref["counts"]["damage_tp_fp_fn"][c] for c in ref["damage_f1"]]


def test_fit_offsets_recovers_an_underpredicted_class_and_never_lowers_val_hm():
    rng = np.random.default_rng(2)
    dt = rng.choice([1, 2, 3, 4], size=20000, p=[0.7, 0.1, 0.1, 0.1]).astype(np.int8)
    z = rng.normal(0, 1, (len(dt), 4)).astype(np.float32)
    z[np.arange(len(dt)), dt - 1] += 2.0                                    # informative logits ...
    z[:, 1] -= 3.0                                                           # ... but minor pushed down
    lp = rng.random(len(dt)) < 0.9
    b, info = fit_offsets(z, lp, dt)
    assert b[0] == 0 and b[1] > 1.0                                          # no-damage fixed, minor raised
    assert info["val_harmonic_after"] > info["val_harmonic_before"] + 0.05
    assert info["val_harmonic_after"] == pytest.approx(harmonic(damage_counts(dt, predict(z, lp, b))))
    assert ((predict(z, lp, b) == 0) == ~lp).all()                          # offsets never change "not localised"
    p = mean_softmax(z, dt == 2)
    assert len(p) == 4 and sum(p) == pytest.approx(1.0) and mean_softmax(z, dt == 9) is None


def test_edge_rule_needs_more_than_the_margin():
    from experiments.silo2_dmg import EDGE_MARGIN, edge_decision
    ref = 0.9201758630347975
    assert EDGE_MARGIN == 0.03
    assert edge_decision(0.80, ref) and edge_decision(ref - 0.0301, ref)
    assert not edge_decision(ref - 0.03, ref)                                # exactly 0.03: keep 1e-4
    assert not edge_decision(0.91, ref) and not edge_decision(0.95, ref)


def test_post_steps_follow_the_recorded_order():
    from experiments.silo2_campaign import post_steps
    steps = post_steps(["--smoke", "2"])
    assert [n for n, _ in steps] == ["edge_check_lr3e-05", "S1_fedavg_R60_lr1e-4", "S2_fedavg_R30_lr3e-4",
                                     "D1_diagnostics"]
    assert all(a[-2:] == ["--smoke", "2"] for _, a in steps)
