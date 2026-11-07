"""
Pre-registered D1 (docs/STAGE2_PROTOCOL.md): FedAvg vs centralized damage diagnostics on the saved stage-2 best
models of split seed 0. Sensitivity / diagnostics only: the headline numbers are unchanged whatever this shows.

For each method, from one forward pass over val and one over test (src/fl/dmg_diag.collect_scored):
  - reproduction: test damage counts recomputed here vs the run's committed counts (reported, not assumed)
  - test confusion matrix on the scored building pixels (rows true class, columns predicted incl. "not localised")
  - mean softmax probability vector on true-minor and true-major test pixels
  - per-class logit offsets fitted on VALIDATION only (max val damage harmonic mean), applied unchanged to test:
    test damage F1s, harmonic mean and xView2 score without and with the offsets

Run:  python -m experiments.silo2_dmg --diagnostics        (after the campaign; writes results/silo2_dmg/diagnostics/)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from experiments.silo2_dmg import DmgData, DmgHub, _load, _load_dmg, load_unet
from src.fl.dmg_diag import (OFFSET_GRID, collect_scored, confusion, damage_counts, fit_offsets, harmonic,
                             mean_softmax, per_class_f1, predict)
from src.fl.utils import write_metrics
from src.fl.xbd_data import DAMAGE_CLASSES
from src.fl.xbd_metrics import _f1

D1_METHODS, D1_SEED = ("centralized", "fedavg"), 0
LABEL = ("diagnostics (pre-registered D1 in docs/STAGE2_PROTOCOL.md, split seed 0 only); the 30-round, lr 1e-4 "
         "headline is unchanged whatever this shows")


def committed_test_counts(run: dict) -> np.ndarray:
    """The run's own test damage TP / FP / FN, summed over events (as pooled_score does)."""
    tot = np.zeros((len(DAMAGE_CLASSES), 3), dtype=np.int64)
    for ev in run["metrics"]["test_per_event"].values():
        tot += np.asarray([ev["counts"]["damage_tp_fp_fn"][c] for c in DAMAGE_CLASSES], dtype=np.int64)
    return tot


def method_diagnostics(run: dict, hub: DmgHub, clients: list[str], device) -> dict:
    dmg = _load_dmg(str(Path(run["metrics"]["checkpoint_dir"]) / "best_model.pt")).to(device)
    loc = load_unet(run["metrics"]["stage1_source"]["best_model"]).to(device)
    val = collect_scored(loc, dmg, hub.loader, "val/", clients, device)
    offsets, fit = fit_offsets(val["logits"], val["lp"], val["dt"])
    del val
    test = collect_scored(loc, dmg, hub.loader, "test/", clients, device)
    dt, lp, z = test["dt"], test["lp"], test["logits"]
    loc_f1 = _f1(*test["loc_tp_fp_fn"])
    out = {"offsets": dict(zip(DAMAGE_CLASSES, offsets.tolist())), "offset_fit": fit,
           "n_scored_test_pixels": int(len(dt)), "n_test_images": test["n_images"], "test_loc_f1": loc_f1}
    for tag, b in (("without_offsets", None), ("with_offsets", offsets)):
        pred = predict(z, lp, b)
        counts = damage_counts(dt, pred)
        h = harmonic(counts)
        out[tag] = {"damage_f1": per_class_f1(counts), "damage_f1_harmonic": h, "xview2_score": 0.3 * loc_f1 + 0.7 * h,
                    "damage_tp_fp_fn": dict(zip(DAMAGE_CLASSES, counts.tolist())),
                    "confusion": {"rows_true": list(DAMAGE_CLASSES), "cols_pred": ["not-localised", *DAMAGE_CLASSES],
                                  "pixels": confusion(dt, pred).tolist()}}
    committed = committed_test_counts(run)
    mine = np.asarray([out["without_offsets"]["damage_tp_fp_fn"][c] for c in DAMAGE_CLASSES])
    out["reproduction"] = {"exact": bool((committed == mine).all()),
                           "max_abs_count_diff": int(np.abs(committed - mine).max()),
                           "committed_score": run["metrics"]["test_xview2_score"],
                           "recomputed_score": out["without_offsets"]["xview2_score"]}
    out["mean_softmax_test"] = {f"true_{c}": dict(zip(DAMAGE_CLASSES, mean_softmax(z, dt == k) or [None] * 4))
                                for k, c in ((2, "minor-damage"), (3, "major-damage"))}
    return out


def run_diagnostics(device, git, stage1_root, limit=None, out_root=None) -> dict:
    from experiments.silo2_dmg import EXPERIMENT_DIR
    out_root = Path(out_root or EXPERIMENT_DIR)
    runs = {m: _load(out_root / f"{m}_seed{D1_SEED}" / "metrics.json") for m in D1_METHODS}
    missing = [m for m, r in runs.items() if r is None]
    if missing:
        raise FileNotFoundError(f"D1 needs the finished seed-{D1_SEED} runs of {missing}")
    t0 = time.perf_counter()
    data = DmgData(D1_SEED, limit)
    hub = DmgHub(data, "local_only", device)
    try:
        res = {m: method_diagnostics(runs[m], hub, data.clients, device) for m in D1_METHODS}
    finally:
        hub.close()
    for m, r in res.items():
        print(f"D1 {m}: test score {r['without_offsets']['xview2_score']:.4f} -> {r['with_offsets']['xview2_score']:.4f} "
              f"with val-fitted offsets {r['offsets']}; reproduces committed counts: {r['reproduction']['exact']}",
              flush=True)
    metrics = {"label": LABEL, "methods": res,
               "sources": {m: {"metrics": str(out_root / f"{m}_seed{D1_SEED}" / "metrics.json"),
                               "training_commit": runs[m]["metrics"]["training_git"]["commit"]} for m in D1_METHODS},
               "definitions": {"scored_pixels": "post-target building pixels with a damage label (official score)",
                               "not_localised": "the method's stage-1 model predicts no building there (dp = 0)",
                               "mean_softmax": "damage-head softmax over the 4 classes, mean over all true-class test "
                                               "pixels (localised or not)",
                               "offsets": "added to the damage logits; no-damage fixed at 0; coordinate ascent over "
                                          f"{OFFSET_GRID[0]:g}..{OFFSET_GRID[-1]:g} step "
                                          f"{OFFSET_GRID[1] - OFFSET_GRID[0]:g}, val damage harmonic mean only"},
               "training_git": git}
    write_metrics(out_root / "diagnostics" / f"D1_seed{D1_SEED}" / "metrics.json", metrics, seed=D1_SEED,
                  config={"methods": list(D1_METHODS), "split_file": f"data/splits/silo2_seed{D1_SEED}.json",
                          "smoke_limit": limit},
                  split_sizes=data.split_sizes(), runtime_s=time.perf_counter() - t0)
    return metrics
