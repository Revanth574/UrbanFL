"""
Silo 1 unseen-disaster test (leave-one-event-out): each of the 10 events in turn is the held-out NEW disaster; the
other 9 train, the held-out event is test data only.

Methods (recipes, lrs and budgets are those of the in-distribution runs; nothing is re-tuned):
  - local_only:  each training event's own model (silo1_baselines local_only: single-client SGD, own class weights,
                 own-val-loss checkpoint). A local model depends only on (event, seed), so the 10 local models per seed
                 are trained once and each is scored on the 9 events it never saw. Reported as the mean over the 9
                 other-event models.
  - fedavg_E1:   FedAvg over the 9 (silo1_fl fedavg_E1: E=1, 100 rounds, global class weights from the federated sum
                 of the 9 label histograms); checkpoint = lowest federated val loss over the 9 clients' val sets.
  - centralized: single-client SGD on the pooled train sets of the 9; checkpoint = lowest pooled val loss of the 9.

The held-out event is never used for training, class weights or checkpoint selection. Lr choices come from the
saved seed-0 sweeps (results/silo1_baselines/lr_sweep/choice.json, results/silo1_fl/lr_sweep/choice.json), which
were made with all 10 events' val sets; that is recorded in every metrics.json.

Held-out test set: ALL rows of the held-out event (its train + val + test split of data/splits/silo1_seed{k}.json),
since none of them is seen. The subset in its own test split is also scored (secondary). Near-duplicates were
removed globally before splitting (src/fl/partition.py D3), so no held-out tweet has a copy in a training event.

Metrics:
  primary   - pooled held-out macro-F1 (9 classes): every fold's held-out predictions concatenated, so each tweet is
              scored exactly once per method and seed. For local_only each held-out tweet is scored by the 9
              other-event models, all predictions concatenated (= macro-F1 of the model-averaged confusion matrix).
  per event - D4 macro-F1 on the held-out event (classes with >= 3 true examples); local_only = mean over 9 models.
  tests     - paired t over seeds on pooled macro-F1 (n=5); paired t and exact Wilcoxon over events on the
              seed-mean per-event macro-F1 (n=10); Holm-adjusted within each family of three comparisons.

Per-fold lr (--lr-sweep, --per-fold-lr): the shared lrs above saw every fold's held-out val set. --lr-sweep re-chooses
the FedAvg E=1 and centralized lr PER FOLD from LR_SWEEP_GRID on seed 0 with the same rules as the in-distribution
sweeps (FedAvg: lowest best federated val loss; centralized: lowest best pooled val loss), using only the 9 training
events' val sets -> lr_sweep/choice.json. --per-fold-lr reruns seeds 0-4 only for the (method, fold) pairs whose lr
changed (training is deterministic, so an unchanged lr reproduces the shared-lr run) -> per_fold_lr/. The local_only
lr is not re-chosen: it is one lr shared by all 10 local models (size-weighted val loss of all 10).

Reference column: each event's own-label local-only score (its own model on its own test split) from the committed
Review 1 runs, results/silo1_baselines/local_only_seed<k>/metrics.json.

Outputs: results/silo1_unseen/{local_only_seed<k>, <method>_holdout-<event>_seed<k>}/{metrics.json, predictions.json},
lr_sweep/, per_fold_lr/, summary.json (shared lr) and summary_per_fold_lr.json. Best states go to
<checkpoint_dir>/silo1_unseen_<commit> (git-ignored).

--min-free-ram-gb / --stop-log: checked before every run; the process stops (exit 3, resumable) when available RAM
falls below the floor or a watched log gains a line containing --stop-pattern.

Usage:  python -m experiments.silo1_unseen [--seeds 0 1 2 3 4] [--threads 3]
        python -m experiments.silo1_unseen --lr-sweep --threads 3
        python -m experiments.silo1_unseen --per-fold-lr --threads 3
        python -m experiments.silo1_unseen --summarize
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from experiments.silo1_baselines import CONFIG as BASELINE_CONFIG
from experiments.silo1_baselines import EXPERIMENT_DIR as BASELINE_DIR
from experiments.silo1_baselines import SWEEP_DIR as BASELINE_SWEEP_DIR
from experiments.silo1_baselines import client_arrays, predict, train_sgd
from experiments.silo1_embed_cache import require_single_cache, silo1_environment
from experiments.silo1_fl import COMMON as FL_COMMON
from experiments.silo1_fl import CONFIGS as FL_CONFIGS
from experiments.silo1_fl import SWEEP_DIR as FL_SWEEP_DIR
from src.fl.checkpoint import resolve_checkpoint_dir, save_atomic
from src.fl.models import MLPHead, class_weights_from_counts
from src.fl.partition import LABEL_SHORT, LABELS, load_partition
from src.fl.runner import ClientData, run_fl
from src.fl.utils import (RESULTS_DIR, accuracy, d4_metrics, macro_f1, per_class_report, require_clean_tree,
                          write_metrics)

EXPERIMENT_DIR = RESULTS_DIR / "silo1_unseen"
SWEEP_DIR = EXPERIMENT_DIR / "lr_sweep"
PER_FOLD_DIR = EXPERIMENT_DIR / "per_fold_lr"
LR_SWEEP_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]   # in-distribution grid + 0.003 (centralized chose its edge 0.01)
SWEEP_SEED = 0
SEEDS = [0, 1, 2, 3, 4]
N_CLASSES = len(LABELS)
LABEL_IDS = list(range(N_CLASSES))
SPLITS = ("train", "val", "test")
FL_NAME = "fedavg_E1"
METHODS = ("local_only", FL_NAME, "centralized")
PAIRS = ((FL_NAME, "local_only"), (FL_NAME, "centralized"), ("centralized", "local_only"))
MIN_SUPPORT = 3   # D4 per-event macro-F1


def load_lrs() -> dict:
    """The lrs used in-distribution, read from the saved sweeps (validation loss, seed 0; never test data)."""
    base = json.loads((BASELINE_SWEEP_DIR / "choice.json").read_text(encoding="utf-8"))
    fl = json.loads((FL_SWEEP_DIR / "choice.json").read_text(encoding="utf-8"))
    if fl["E1"]["rounds"] != FL_CONFIGS[FL_NAME]["rounds"]:
        raise ValueError("FedAvg E=1 lr was swept at a different round budget")
    return {"local_only": base["local_only"]["lr"], "centralized": base["centralized"]["lr"], FL_NAME: fl["E1"]["lr"],
            "sources": {"local_only": "results/silo1_baselines/lr_sweep/choice.json",
                        "centralized": "results/silo1_baselines/lr_sweep/choice.json",
                        FL_NAME: "results/silo1_fl/lr_sweep/choice.json (E1)"},
            "sweep_commits": {"baselines": base["training_git"]["commit"], "fl": fl["training_git"]["commit"]},
            "note": "chosen on seed 0 with all 10 events' val sets, i.e. including each fold's held-out event; "
                    "kept unchanged so the recipes match the in-distribution runs"}


def event_arrays(part) -> dict:
    """event -> split -> (X, y, tweet_ids)."""
    return {e: {s: (*client_arrays(part, e, s), list(part.clients[e][s])) for s in SPLITS} for e in sorted(part.clients)}


def heldout_set(data: dict, event: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """All rows of `event` (train + val + test): X, y, tweet_ids, and which split each row came from."""
    X = np.concatenate([data[event][s][0] for s in SPLITS])
    y = np.concatenate([data[event][s][1] for s in SPLITS])
    ids = [t for s in SPLITS for t in data[event][s][2]]
    split = [s for s in SPLITS for _ in data[event][s][2]]
    return X, y, ids, split


def score(y_true, y_pred, split: list[str]) -> dict:
    """Held-out metrics for one model on one event: all rows (primary) and the event's own test split."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    test = np.array([s == "test" for s in split])
    out = {}
    for name, mask in (("all_rows", np.ones(len(y_true), bool)), ("test_split", test)):
        yt, yp = y_true[mask], y_pred[mask]
        d4 = d4_metrics({"e": yt}, {"e": yp}, LABEL_IDS, min_support=MIN_SUPPORT)["per_client"]["e"]
        out[name] = {"n": int(mask.sum()), "macro_f1_9class": macro_f1(yt, yp, LABEL_IDS), "accuracy": accuracy(yt, yp),
                     "d4_macro_f1": d4["macro_f1"], "d4_n_qualifying_classes": d4["n_qualifying_classes"]}
    return out


def seen_test_score(preds: dict) -> dict:
    """In-distribution reference: the training events' own test splits (pooled 9-class macro-F1 + D4 client mean)."""
    yt = {c: p["y_true"] for c, p in preds.items()}
    yp = {c: p["y_pred"] for c, p in preds.items()}
    d4 = d4_metrics(yt, yp, LABEL_IDS, min_support=MIN_SUPPORT)
    return {"pooled_macro_f1": d4["pooled_macro_f1"], "client_mean_macro_f1": d4["client_mean_macro_f1"]}


# ── Training (one call = one model; the held-out event's arrays are never passed in) ──────────────────────────────

def train_local(data: dict, event: str, lr: float, seed: int, device: str):
    return train_sgd(*data[event]["train"][:2], *data[event]["val"][:2], lr=lr, seed=seed, device=device)


def train_centralized(data: dict, train_events: list[str], lr: float, seed: int, device: str):
    X_tr = np.concatenate([data[c]["train"][0] for c in train_events])
    y_tr = np.concatenate([data[c]["train"][1] for c in train_events])
    X_va = np.concatenate([data[c]["val"][0] for c in train_events])
    y_va = np.concatenate([data[c]["val"][1] for c in train_events])
    return train_sgd(X_tr, y_tr, X_va, y_va, lr=lr, seed=seed, device=device)


def train_fedavg(data: dict, train_events: list[str], lr: float, seed: int, device: str):
    cfg = FL_CONFIGS[FL_NAME]
    clients, hist = [], np.zeros(N_CLASSES, dtype=np.int64)
    for c in train_events:
        t = [torch.as_tensor(a, device=device) for s in SPLITS for a in data[c][s][:2]]
        clients.append(ClientData(c, *t))
        hist += np.bincount(data[c]["train"][1], minlength=N_CLASSES)   # what each client would send
    res = run_fl(cfg["algorithm"], clients, cfg["rounds"], cfg["local_epochs"], lr, FL_COMMON["batch_size"],
                 FL_COMMON["client_fraction"], seed, mu=cfg["mu"], class_weights=class_weights_from_counts(hist),
                 n_classes=N_CLASSES, head=FL_COMMON["head"], device=device)
    model = MLPHead(clients[0].X_train.shape[1], N_CLASSES, **FL_COMMON["head"]).to(device)
    model.load_state_dict(res["best_state"])
    info = {"best_round": res["best_round"], "best_fed_val_loss": res["best_fed_val_loss"],
            "federated_label_counts": hist.tolist(), "seen_test_predictions": res["test_predictions"],
            "history": [{k: h[k] for k in ("round", "fed_val_loss", "train_ce", "pooled_val_macro_f1")}
                        for h in res["history"]]}
    return model, info


# ── Resource guard ─────────────────────────────────────────────────────────

class GuardStop(RuntimeError):
    pass


def available_ram_gb() -> float:
    """Available physical memory (Windows GlobalMemoryStatusEx; POSIX sysconf fallback)."""
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MemoryStatus()
        st.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 2**30
    import os
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30


class ResourceGuard:
    """Stop between runs when available RAM < `min_free_gb` or a log matching `log_glob` gains a `pattern` line
    (lines already present at start-up are ignored)."""

    def __init__(self, min_free_gb: float | None, log_glob: str | None, pattern: str = "CRASH"):
        self.min_free_gb, self.log_glob, self.pattern = min_free_gb, log_glob, pattern
        self.baseline = self._hits()

    def _hits(self) -> dict:
        if not self.log_glob:
            return {}
        out = {}
        for path in glob.glob(self.log_glob):
            with open(path, encoding="utf-8", errors="replace") as f:
                out[path] = sum(self.pattern in line for line in f)
        return out

    def check(self) -> None:
        if self.min_free_gb is not None and (free := available_ram_gb()) < self.min_free_gb:
            raise GuardStop(f"available RAM {free:.2f} GB < {self.min_free_gb} GB")
        new = {p: n - self.baseline.get(p, 0) for p, n in self._hits().items() if n > self.baseline.get(p, 0)}
        if new:
            raise GuardStop(f"new '{self.pattern}' lines in {new}")


# ── Runs ───────────────────────────────────────────────────────────────────

def _done(run_dir: Path, git: dict) -> bool:
    path = run_dir / "metrics.json"
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]["training_git"]["commit"] == git["commit"]


def shared_lr_meta(lrs: dict, method: str) -> dict:
    return {"lr_mode": "shared", "lr_source": lrs["sources"][method], "lr_note": lrs["note"],
            "lr_sweep_commits": lrs["sweep_commits"]}


def _config(method: str, lr: float, seed: int, device: str, lr_meta: dict, **extra) -> dict:
    recipe = ({**FL_COMMON, **FL_CONFIGS[FL_NAME]} if method == FL_NAME else
              {k: BASELINE_CONFIG[k] for k in ("head", "optimizer", "batch_size", "epochs", "checkpoint_selection")})
    return {"method": method, "protocol": "leave-one-event-out (unseen disaster)", **recipe, "lr": lr, **lr_meta,
            "heldout_rows": "all rows of the held-out event (train+val+test of its split); never used for training, "
                            "class weights or checkpoint selection",
            "device": device, "labels": list(LABELS), "split_file": f"data/splits/silo1_seed{seed}.json",
            "environment": silo1_environment(device), **extra}


def run_local(part, data: dict, seed: int, lrs: dict, device: str, git: dict, ckpt_dir: Path) -> None:
    """10 local models for one seed; each scored on every other event (held-out) and on its own test split."""
    run_dir = EXPERIMENT_DIR / f"local_only_seed{seed}"
    if _done(run_dir, git):
        print(f"  skip {run_dir.name} (done at this commit)", flush=True)
        return
    events, lr = sorted(data), lrs["local_only"]
    heldout = {e: heldout_set(data, e) for e in events}
    t0 = time.perf_counter()
    models, predictions = {}, {}
    for c in events:
        model, info = train_local(data, c, lr, seed, device)
        save_atomic(model.state_dict(), ckpt_dir / f"local_only_{c}_seed{seed}.pt")
        own = predict(model, data[c]["test"][0])
        predictions[c] = {"own_test": {"tweet_ids": data[c]["test"][2], "y_true": data[c]["test"][1].tolist(),
                                       "y_pred": own.tolist()},
                          "heldout": {h: predict(model, heldout[h][0]).tolist() for h in events if h != c}}
        models[c] = {"best_epoch": info["best_epoch"], "best_val_loss": info["best_val_loss"],
                     "own_test": seen_test_score({c: predictions[c]["own_test"]}),
                     "heldout": {h: score(heldout[h][1], predictions[c]["heldout"][h], heldout[h][3])
                                 for h in events if h != c}}
        print(f"    local {c:36s} best epoch {info['best_epoch']:3d}", flush=True)
    runtime = time.perf_counter() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.json").write_text(json.dumps({
        "heldout_rows": {h: {"tweet_ids": heldout[h][2], "split": heldout[h][3], "y_true": heldout[h][1].tolist()}
                         for h in events},
        "models": predictions}), encoding="utf-8")
    metrics = {"models": models, "own_test_pooled": seen_test_score({c: p["own_test"] for c, p in predictions.items()}),
               "training_git": git}
    write_metrics(run_dir / "metrics.json", metrics, seed=seed,
                  config=_config("local_only", lr, seed, device, shared_lr_meta(lrs, "local_only"),
                                 class_weights="each event's own train label counts",
                                 checkpoint_dir=str(ckpt_dir)),
                  split_sizes=part.split_sizes(), runtime_s=runtime)


def run_fold(method: str, part, data: dict, holdout: str, seed: int, lr: float, lr_meta: dict, device: str,
             git: dict, ckpt_dir: Path, out_dir: Path = EXPERIMENT_DIR) -> None:
    run_dir = out_dir / f"{method}_holdout-{holdout}_seed{seed}"
    if _done(run_dir, git):
        print(f"  skip {run_dir.name} (done at this commit)", flush=True)
        return
    train_events = [e for e in sorted(data) if e != holdout]
    t0 = time.perf_counter()
    train = train_fedavg if method == FL_NAME else train_centralized
    model, info = train({e: data[e] for e in train_events}, train_events, lr, seed, device)
    runtime = time.perf_counter() - t0
    save_atomic(model.state_dict(), ckpt_dir / f"{run_dir.parent.name}_{run_dir.name}.pt")

    X_h, y_h, ids_h, split_h = heldout_set(data, holdout)
    pred_h = predict(model, X_h)
    if method == FL_NAME:
        seen = info.pop("seen_test_predictions")
        training = info
    else:
        seen = {c: {"y_true": data[c]["test"][1].tolist(), "y_pred": predict(model, data[c]["test"][0]).tolist()}
                for c in train_events}
        training = {"best_epoch": info["best_epoch"], "best_val_loss": info["best_val_loss"], "n_val": info["n_val"],
                    "history": info["history"]}

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.json").write_text(json.dumps({
        "heldout": {"event": holdout, "tweet_ids": ids_h, "split": split_h, "y_true": y_h.tolist(),
                    "y_pred": pred_h.tolist()},
        "seen_test": seen}), encoding="utf-8")
    metrics = {"holdout_event": holdout, "train_events": train_events, "heldout": score(y_h, pred_h, split_h),
               "seen_test": seen_test_score(seen), "training": training, "training_git": git}
    write_metrics(run_dir / "metrics.json", metrics, seed=seed,
                  config=_config(method, lr, seed, device, lr_meta, holdout_event=holdout, train_events=train_events,
                                 class_weights=("global, federated sum of the 9 training events' label histograms"
                                                if method == FL_NAME else "pooled train label counts of the 9"),
                                 checkpoint_dir=str(ckpt_dir)),
                  split_sizes={e: part.split_sizes()[e] for e in sorted(data)}, runtime_s=runtime)
    print(f"  {run_dir.name:70s} held-out D4 F1 {metrics['heldout']['all_rows']['d4_macro_f1']:.4f}  "
          f"({runtime:.0f}s)", flush=True)


# ── Per-fold lr ────────────────────────────────────────────────────────────

def sweep_run(method: str, part, data: dict, holdout: str, lr: float, device: str, git: dict) -> dict:
    """One sweep point: train on the 9, record the selection criterion only (no held-out data is touched)."""
    run_dir = SWEEP_DIR / f"{method}_holdout-{holdout}_lr{lr:g}_seed{SWEEP_SEED}"
    if _done(run_dir, git):
        return _load(run_dir / "metrics.json")["metrics"]
    train_events = [e for e in sorted(data) if e != holdout]
    t0 = time.perf_counter()
    train = train_fedavg if method == FL_NAME else train_centralized
    _, info = train({e: data[e] for e in train_events}, train_events, lr, SWEEP_SEED, device)
    runtime = time.perf_counter() - t0
    if method == FL_NAME:
        info.pop("seen_test_predictions")
        crit = {"criterion": "best federated val loss over the 9", "val_loss": info["best_fed_val_loss"],
                "best_round_or_epoch": info["best_round"],
                "val_loss_curve": [h["fed_val_loss"] for h in info["history"]]}
    else:
        crit = {"criterion": "best pooled val loss of the 9", "val_loss": info["best_val_loss"],
                "best_round_or_epoch": info["best_epoch"], "val_loss_curve": [h["val_loss"] for h in info["history"]]}
    metrics = {"holdout_event": holdout, "train_events": train_events, "lr": lr, **crit, "training_git": git}
    write_metrics(run_dir / "metrics.json", metrics, seed=SWEEP_SEED,
                  config={"method": method, "purpose": "per-fold lr sweep (val loss of the 9 training events only)",
                          "lr": lr, "grid": LR_SWEEP_GRID, "device": device, "holdout_event": holdout,
                          "split_file": f"data/splits/silo1_seed{SWEEP_SEED}.json",
                          "environment": silo1_environment(device)},
                  split_sizes={e: part.split_sizes()[e] for e in train_events}, runtime_s=runtime)
    print(f"  sweep {method:12s} holdout {holdout:36s} lr {lr:<6g} val loss {crit['val_loss']:.5f}  "
          f"best {crit['best_round_or_epoch']:3d}  ({runtime:.0f}s)", flush=True)
    return metrics


def lr_sweep(device: str, git: dict, guard: ResourceGuard, holdouts: list[str] | None) -> dict:
    part = load_partition(SWEEP_SEED)
    data = event_arrays(part)
    events = holdouts or sorted(data)
    shared = load_lrs()
    results = {m: {} for m in (FL_NAME, "centralized")}
    for h in events:
        for method in (FL_NAME, "centralized"):
            for lr in LR_SWEEP_GRID:
                guard.check()
                results[method].setdefault(h, {})[lr] = sweep_run(method, part, data, h, lr, device, git)
    cache_sha = require_single_cache(
        {f"{m}/{h}/lr{lr:g}": _load(SWEEP_DIR / f"{m}_holdout-{h}_lr{lr:g}_seed{SWEEP_SEED}" / "metrics.json")
         for m, folds in results.items() for h, by_lr in folds.items() for lr in by_lr}, "silo1_unseen lr sweep")
    choice = {}
    for method, folds in results.items():
        choice[method] = {}
        for h, by_lr in folds.items():
            losses = {lr: m["val_loss"] for lr, m in by_lr.items()}
            best = min(losses, key=losses.get)
            choice[method][h] = {"lr": best, "val_loss_by_lr": {f"{k:g}": v for k, v in losses.items()},
                                 "best_round_or_epoch_by_lr": {f"{k:g}": m["best_round_or_epoch"]
                                                               for k, m in by_lr.items()},
                                 "chosen_lr_at_grid_edge": best in (LR_SWEEP_GRID[0], LR_SWEEP_GRID[-1]),
                                 "shared_lr": shared[method], "same_as_shared": best == shared[method]}
    complete = all(len(choice[m]) == 10 for m in choice)
    payload = {"selection": f"per fold: min best val loss of the 9 training events, seed {SWEEP_SEED}; FedAvg: "
                            "client-size-weighted federated val loss; centralized: pooled val loss; held-out event "
                            "never evaluated", "grid": LR_SWEEP_GRID, "complete": complete,
               "all_folds_same_as_shared": complete and all(c["same_as_shared"] for m in choice.values()
                                                            for c in m.values()),
               "n_changed": {m: sum(not c["same_as_shared"] for c in v.values()) for m, v in choice.items()},
               "shared_lrs": {m: shared[m] for m in choice}, "training_git": git,
               "embedding_cache_sha256": cache_sha, **choice}
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "choice.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def per_fold_runs(seeds: list[int], device: str, git: dict, ckpt_dir: Path, guard: ResourceGuard,
                  check_unchanged: str | None) -> None:
    """Rerun the (method, fold) pairs whose per-fold lr differs from the shared lr; optionally also one unchanged
    fold (seed 0) as a determinism check against the shared-lr run."""
    choice = _load(SWEEP_DIR / "choice.json")
    if not choice["complete"]:
        raise RuntimeError("lr sweep incomplete: run --lr-sweep first")
    meta = {"lr_mode": "per_fold", "lr_source": "results/silo1_unseen/lr_sweep/choice.json",
            "lr_note": "chosen per fold on seed 0 from the 9 training events' val losses only",
            "lr_sweep_commits": {"per_fold": choice["training_git"]["commit"]}}
    for seed in seeds:
        part = load_partition(seed)
        data = event_arrays(part)
        print(f"seed {seed}", flush=True)
        for h in sorted(data):
            for method in (FL_NAME, "centralized"):
                c = choice[method][h]
                if not c["same_as_shared"] or (seed == SWEEP_SEED and f"{method}:{h}" == check_unchanged):
                    guard.check()
                    run_fold(method, part, data, h, seed, c["lr"], meta, device, git, ckpt_dir, PER_FOLD_DIR)


# ── Summary ────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ms(vals) -> dict:
    vals = np.asarray(vals, dtype=float)
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals),
            "values": vals.tolist()}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down adjusted p-values within one family."""
    order = sorted(pvals, key=pvals.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        adj[k] = running
    return adj


def shared_fold_dir(method: str, holdout: str, seed: int) -> Path:
    return EXPERIMENT_DIR / f"{method}_holdout-{holdout}_seed{seed}"


def per_fold_dir_resolver() -> callable:
    """Per-fold-lr run of (method, fold): the rerun if its lr changed, else the identical shared-lr run."""
    choice = _load(SWEEP_DIR / "choice.json")

    def resolve(method: str, holdout: str, seed: int) -> Path:
        if choice[method][holdout]["same_as_shared"]:
            return shared_fold_dir(method, holdout, seed)
        return PER_FOLD_DIR / f"{method}_holdout-{holdout}_seed{seed}"
    return resolve


def combined_runs(seeds: list[int], events: list[str], fold_dir, per_fold: bool) -> dict:
    """Every run a summary combines, as {name: metrics.json payload}: the local models, both methods' fold runs, the
    Review 1 local-only reference runs and, for per-fold lr, the sweep points the lrs were chosen from."""
    paths = [EXPERIMENT_DIR / f"local_only_seed{s}" for s in seeds]
    paths += [fold_dir(m, h, s) for m in (FL_NAME, "centralized") for h in events for s in seeds]
    paths += [BASELINE_DIR / f"local_only_seed{s}" for s in seeds if (BASELINE_DIR / f"local_only_seed{s}").exists()]
    if per_fold:
        paths += sorted(p.parent for p in SWEEP_DIR.glob("*/metrics.json"))
    return {str(p.relative_to(RESULTS_DIR)).replace("\\", "/"): _load(p / "metrics.json") for p in paths}


def collect_seed(seed: int, events: list[str], fold_dir=shared_fold_dir) -> dict:
    """Per-event and pooled held-out results for one seed, read from the saved predictions only."""
    local = _load(EXPERIMENT_DIR / f"local_only_seed{seed}" / "predictions.json")
    local_m = _load(EXPERIMENT_DIR / f"local_only_seed{seed}" / "metrics.json")["metrics"]
    rows = local["heldout_rows"]
    out = {m: {"per_event": {}, "pooled": {}} for m in METHODS}
    for subset in ("all_rows", "test_split"):
        pooled = {m: ([], []) for m in METHODS}
        for h in events:
            mask = np.ones(len(rows[h]["y_true"]), bool) if subset == "all_rows" else \
                np.array([s == "test" for s in rows[h]["split"]])
            yt = np.asarray(rows[h]["y_true"])[mask]
            for c in events:
                if c != h:
                    pooled["local_only"][0].append(yt)
                    pooled["local_only"][1].append(np.asarray(local["models"][c]["heldout"][h])[mask])
            for m in (FL_NAME, "centralized"):
                p = _load(fold_dir(m, h, seed) / "predictions.json")["heldout"]
                if p["tweet_ids"] != rows[h]["tweet_ids"]:
                    raise AssertionError(f"{m} holdout {h} seed {seed}: held-out rows differ from the local run")
                pooled[m][0].append(yt)
                pooled[m][1].append(np.asarray(p["y_pred"])[mask])
        for m, (yts, yps) in pooled.items():
            yt, yp = np.concatenate(yts), np.concatenate(yps)
            out[m]["pooled"][subset] = {"macro_f1": macro_f1(yt, yp, LABEL_IDS), "accuracy": accuracy(yt, yp),
                                        "n_scored": int(len(yt))}
            if subset == "all_rows":
                out[m]["pooled"]["per_class"] = per_class_report(yt, yp, LABEL_IDS, LABEL_SHORT)

    for h in events:
        others = [local_m["models"][c]["heldout"][h] for c in events if c != h]
        out["local_only"]["per_event"][h] = {
            subset: {key: float(np.mean([o[subset][key] for o in others]))
                     for key in ("d4_macro_f1", "macro_f1_9class", "accuracy")} for subset in ("all_rows", "test_split")}
        out["local_only"]["per_event"][h]["all_rows"]["d4_macro_f1_by_source_model"] = {
            c: local_m["models"][c]["heldout"][h]["all_rows"]["d4_macro_f1"] for c in events if c != h}
        for m in (FL_NAME, "centralized"):
            fm = _load(fold_dir(m, h, seed) / "metrics.json")["metrics"]
            out[m]["per_event"][h] = {subset: {k: fm["heldout"][subset][k] for k in
                                               ("d4_macro_f1", "macro_f1_9class", "accuracy")}
                                      for subset in ("all_rows", "test_split")}
            out[m]["per_event"][h]["seen_test_pooled_macro_f1"] = fm["seen_test"]["pooled_macro_f1"]
            out[m]["per_event"][h]["best_round_or_epoch"] = fm["training"].get("best_round",
                                                                               fm["training"].get("best_epoch"))
    out["local_only"]["own_test_pooled_macro_f1"] = local_m["own_test_pooled"]["pooled_macro_f1"]
    return out


def reproduction_check(seeds: list[int]) -> dict:
    """The local models are the committed silo1_baselines local_only models when the embedding cache reproduces:
    compare their own-test predictions with the committed predictions.json."""
    out = {}
    for s in seeds:
        old_path = BASELINE_DIR / f"local_only_seed{s}" / "predictions.json"
        if not old_path.exists():
            continue
        old = _load(old_path)
        new = _load(EXPERIMENT_DIR / f"local_only_seed{s}" / "predictions.json")["models"]
        agree = [np.mean(np.asarray(old[c]["y_pred"]) == np.asarray(new[c]["own_test"]["y_pred"])) for c in new]
        old_f1 = _load(BASELINE_DIR / f"local_only_seed{s}" / "metrics.json")["metrics"]["pooled_test_macro_f1"]
        new_f1 = macro_f1(np.concatenate([new[c]["own_test"]["y_true"] for c in sorted(new)]),
                          np.concatenate([new[c]["own_test"]["y_pred"] for c in sorted(new)]), LABEL_IDS)
        out[f"seed{s}"] = {"prediction_agreement_mean_over_events": float(np.mean(agree)),
                           "committed_pooled_macro_f1": old_f1, "this_run_pooled_macro_f1": new_f1}
    return out


def review1_local_reference(seeds: list[int], events: list[str]) -> dict:
    """Each event's own-label local-only score from the committed Review 1 runs: its own model on its own test split
    (D4 macro-F1), mean over seeds; plus the pooled in-distribution local-only macro-F1."""
    runs = {s: _load(BASELINE_DIR / f"local_only_seed{s}" / "metrics.json") for s in seeds}
    return {"source": "results/silo1_baselines/local_only_seed<k>/metrics.json",
            "training_commits": sorted({r["metrics"]["training_git"]["commit"] for r in runs.values()}),
            "per_event_d4_macro_f1": {h: _ms([runs[s]["metrics"]["d4"]["per_client"][h]["macro_f1"] for s in seeds])
                                      for h in events},
            "pooled_macro_f1": _ms([runs[s]["metrics"]["pooled_test_macro_f1"] for s in seeds]),
            "note": "in-distribution (trained on the event's own labels, tested on its own test split); compare with "
                    "the held-out test_split columns"}


T_CRIT_DF4 = float(stats.t.ppf(0.975, 4))   # two-sided 95 %, 5 seeds


def fedavg_vs_centralized(per_seed: dict, seeds: list[int], events: list[str]) -> dict:
    """Wins/losses over events. Per event, the FedAvg - centralized difference is paired over seeds; it counts as a
    clear difference only when |mean| > t(0.975, 4) * SE (SE = std of the 5 paired differences / sqrt 5), i.e. it
    exceeds seed noise. Verdict 'comparable' unless clear differences favour one method in >= 6 of 10 events."""
    rows, clear = {}, {"fedavg_better": 0, "centralized_better": 0}
    for h in events:
        d = np.array([per_seed[s][FL_NAME]["per_event"][h]["all_rows"]["d4_macro_f1"] -
                      per_seed[s]["centralized"]["per_event"][h]["all_rows"]["d4_macro_f1"] for s in seeds])
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        threshold = T_CRIT_DF4 * se
        outcome = "comparable" if abs(d.mean()) <= threshold else ("fedavg_better" if d.mean() > 0 else
                                                                    "centralized_better")
        if outcome != "comparable":
            clear[outcome] += 1
        rows[h] = {"mean_diff": float(d.mean()), "se_over_seeds": se, "noise_threshold": threshold,
                   "outcome": outcome, "diffs_by_seed": d.tolist()}
    signs = np.array([r["mean_diff"] for r in rows.values()])
    verdict = "comparable"
    for side in clear:
        if clear[side] >= 6:
            verdict = side
    return {"rule": f"clear iff |mean paired diff| > t(0.975,4)={T_CRIT_DF4:.3f} x SE over seeds; verdict "
                    "'comparable' unless one side has >= 6/10 clear events",
            "wins_by_seed_mean": int((signs > 0).sum()), "losses_by_seed_mean": int((signs < 0).sum()),
            "clear": clear, "verdict": verdict, "per_event": rows}


def summarize(seeds: list[int], per_fold: bool = False) -> dict:
    events = sorted(_load(EXPERIMENT_DIR / f"local_only_seed{seeds[0]}" / "predictions.json")["heldout_rows"])
    fold_dir = per_fold_dir_resolver() if per_fold else shared_fold_dir
    cache_sha = require_single_cache(combined_runs(seeds, events, fold_dir, per_fold),
                                     f"silo1_unseen summary ({'per-fold' if per_fold else 'shared'} lr)")
    per_seed = {s: collect_seed(s, events, fold_dir) for s in seeds}

    def pooled(m, s, subset="all_rows"):
        return per_seed[s][m]["pooled"][subset]["macro_f1"]

    def event_mean(m, h, subset="all_rows"):   # seed-mean per-event D4 macro-F1
        return float(np.mean([per_seed[s][m]["per_event"][h][subset]["d4_macro_f1"] for s in seeds]))

    table = {m: {"pooled_heldout_macro_f1": _ms([pooled(m, s) for s in seeds]),
                 "pooled_heldout_accuracy": _ms([per_seed[s][m]["pooled"]["all_rows"]["accuracy"] for s in seeds]),
                 "pooled_heldout_test_split_macro_f1": _ms([pooled(m, s, "test_split") for s in seeds]),
                 "event_mean_d4_macro_f1": _ms([np.mean([per_seed[s][m]["per_event"][h]["all_rows"]["d4_macro_f1"]
                                                         for h in events]) for s in seeds])}
             for m in METHODS}

    per_event = {}
    for h in events:
        per_event[h] = {m: {"d4_macro_f1": _ms([per_seed[s][m]["per_event"][h]["all_rows"]["d4_macro_f1"] for s in seeds]),
                            "macro_f1_9class": _ms([per_seed[s][m]["per_event"][h]["all_rows"]["macro_f1_9class"]
                                                    for s in seeds]),
                            "test_split_d4_macro_f1": _ms([per_seed[s][m]["per_event"][h]["test_split"]["d4_macro_f1"]
                                                           for s in seeds])}
                        for m in METHODS}
        for m in (FL_NAME, "centralized"):
            per_event[h][m]["seen_events_test_pooled_macro_f1"] = _ms(
                [per_seed[s][m]["per_event"][h]["seen_test_pooled_macro_f1"] for s in seeds])
        src = {c: _ms([per_seed[s]["local_only"]["per_event"][h]["all_rows"]["d4_macro_f1_by_source_model"][c]
                       for s in seeds])["mean"] for c in events if c != h}
        per_event[h]["local_only"]["d4_macro_f1_by_source_model_seed_mean"] = src
        per_event[h]["local_only"]["best_single_source"] = max(src, key=src.get)
        per_event[h]["local_only"]["best_single_source_note"] = "chosen by held-out score: descriptive only"

    tests = {"over_seeds_pooled_macro_f1": {}, "over_events_seed_mean_d4_macro_f1": {}}
    for a, b in PAIRS:
        key = f"{a}_minus_{b}"
        x, y = np.array([pooled(a, s) for s in seeds]), np.array([pooled(b, s) for s in seeds])
        tests["over_seeds_pooled_macro_f1"][key] = {**_ms(x - y), "p_paired_t": float(stats.ttest_rel(x, y).pvalue)}
        xe, ye = np.array([event_mean(a, h) for h in events]), np.array([event_mean(b, h) for h in events])
        d = xe - ye
        tests["over_events_seed_mean_d4_macro_f1"][key] = {
            **_ms(d), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
            "p_paired_t": float(stats.ttest_rel(xe, ye).pvalue),
            "p_wilcoxon_exact": float(stats.wilcoxon(xe, ye, method="exact").pvalue)}
    for fam, col in (("over_seeds_pooled_macro_f1", "p_paired_t"), ("over_events_seed_mean_d4_macro_f1", "p_paired_t"),
                     ("over_events_seed_mean_d4_macro_f1", "p_wilcoxon_exact")):
        adj = holm({k: v[col] for k, v in tests[fam].items()})
        for k, p in adj.items():
            tests[fam][k][f"{col}_holm"] = p

    reference = review1_local_reference(seeds, events)
    for h in events:
        per_event[h]["reference_review1_local_own_event_d4_macro_f1"] = reference["per_event_d4_macro_f1"][h]

    local_cfg = _load(EXPERIMENT_DIR / f"local_only_seed{seeds[0]}" / "metrics.json")["config"]
    fold_cfg = {m: {h: _load(fold_dir(m, h, seeds[0]) / "metrics.json")["config"] for h in events}
                for m in (FL_NAME, "centralized")}
    lrs = {"local_only": local_cfg["lr"],
           **{m: {h: c["lr"] for h, c in by_h.items()} for m, by_h in fold_cfg.items()}}
    lr_notes = {"local_only": local_cfg["lr_note"] + " (not re-chosen per fold)",
                **{m: sorted({c["lr_note"] for c in by_h.values()}) for m, by_h in fold_cfg.items()}}
    if per_fold:
        lr_notes.update({m: "chosen per fold on seed 0 from the 9 training events' val losses only "
                            "(lr_sweep/choice.json); folds whose choice equals the shared lr reuse the identical "
                            "shared-lr run" for m in fold_cfg})
    summary = {
        "protocol": "leave-one-event-out: train on 9 events, test on all rows of the held-out 10th; checkpoint "
                    "selection on the 9 training events' val sets only",
        "lr_mode": "per_fold" if per_fold else "shared",
        "embedding_cache_sha256": cache_sha,
        "seeds": seeds, "events": events, "methods": list(METHODS),
        "lrs": lrs, "lr_notes": lr_notes,
        "run_sources": {m: {h: str(fold_dir(m, h, seeds[0]).relative_to(RESULTS_DIR)).replace(
                            f"_seed{seeds[0]}", "_seed<k>") for h in events} for m in (FL_NAME, "centralized")},
        "metric_definitions": {
            "pooled_heldout_macro_f1": "9-class macro-F1 over all held-out predictions of the 10 folds concatenated "
                                       "(each tweet once; local_only: once per other-event model, 9x)",
            "event_d4_macro_f1": f"macro-F1 on one held-out event over classes with >= {MIN_SUPPORT} true examples; "
                                 "local_only = mean over the 9 other-event models",
            "test_split": "same, restricted to the held-out event's own test split (secondary)"},
        "table": table, "per_event": per_event, "paired_tests": tests,
        "fedavg_vs_centralized_over_events": fedavg_vs_centralized(per_seed, seeds, events),
        "in_distribution_reference": {
            "local_only_own_test_pooled_macro_f1": _ms([per_seed[s]["local_only"]["own_test_pooled_macro_f1"]
                                                        for s in seeds]),
            "review1_local_only": reference,
            "note": "fedavg/centralized seen-event test F1 per fold is in per_event[*][method]"},
        "embedding_cache_reproduction_check": reproduction_check(seeds),
        "per_seed": {f"seed{s}": {m: per_seed[s][m]["pooled"] for m in METHODS} for s in seeds},
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    name = "summary_per_fold_lr.json" if per_fold else "summary.json"
    (EXPERIMENT_DIR / name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_summary(summary)
    return summary


def print_summary(sm: dict) -> None:
    f = lambda c, d=4: f"{c['mean']:.{d}f} ± {c['std']:.{d}f}"
    print(f"\nUnseen-disaster (leave-one-event-out), lr mode {sm['lr_mode']}, seeds {sm['seeds']}")
    for m in (FL_NAME, "centralized"):
        print(f"  {m} lr per fold: {sm['lrs'][m]}")
    print(f"{'method':12s} | {'pooled held-out F1':18s} | {'pooled acc':17s} | {'event-mean D4 F1':17s} | test-split F1")
    for m, r in sm["table"].items():
        print(f"{m:12s} | {f(r['pooled_heldout_macro_f1']):18s} | {f(r['pooled_heldout_accuracy']):17s} | "
              f"{f(r['event_mean_d4_macro_f1']):17s} | {f(r['pooled_heldout_test_split_macro_f1'])}")
    print(f"\nPer held-out event: D4 macro-F1, mean ± std over seeds (ref = Review 1 own-label local-only, own test)")
    print(f"{'held-out event':36s} | " + " | ".join(f"{m:15s}" for m in sm["methods"]) + " | ref (in-dist)")
    for h, r in sm["per_event"].items():
        print(f"{h:36s} | " + " | ".join(f"{f(r[m]['d4_macro_f1'], 3):15s}" for m in sm["methods"]) +
              f" | {f(r['reference_review1_local_own_event_d4_macro_f1'], 3)}")
    fc = sm["fedavg_vs_centralized_over_events"]
    print(f"\nFedAvg vs centralized over events: wins/losses {fc['wins_by_seed_mean']}/{fc['losses_by_seed_mean']}, "
          f"clear (beyond seed noise) {fc['clear']} -> {fc['verdict']}")
    print("\nPaired tests")
    for k, v in sm["paired_tests"]["over_seeds_pooled_macro_f1"].items():
        print(f"  seeds  {k:32s} Δ {v['mean']:+.4f} ± {v['std']:.4f}  t p={v['p_paired_t']:.4f} "
              f"(Holm {v['p_paired_t_holm']:.4f})")
    for k, v in sm["paired_tests"]["over_events_seed_mean_d4_macro_f1"].items():
        print(f"  events {k:32s} Δ {v['mean']:+.4f}  W/L {v['wins']}/{v['losses']}  t p={v['p_paired_t']:.4f} "
              f"Wilcoxon p={v['p_wilcoxon_exact']:.4f} (Holm {v['p_wilcoxon_exact_holm']:.4f})")
    print(f"\nEmbedding-cache reproduction check: {json.dumps(sm['embedding_cache_reproduction_check'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Silo 1 unseen-disaster (leave-one-event-out) test.")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--holdouts", nargs="+", default=None, help="subset of held-out events (default: all)")
    ap.add_argument("--summarize", action="store_true", help="shared-lr summary, plus per-fold if the sweep exists")
    ap.add_argument("--lr-sweep", action="store_true", help="per-fold lr for FedAvg E=1 and centralized (seed 0)")
    ap.add_argument("--per-fold-lr", action="store_true", help="rerun the folds whose per-fold lr changed")
    ap.add_argument("--check-unchanged", default=None, metavar="METHOD:EVENT",
                    help="with --per-fold-lr: also rerun this unchanged fold on seed 0 (determinism check)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--checkpoint-dir", default=None, help="default: URBANFL_CHECKPOINT_DIR / user config")
    ap.add_argument("--min-free-ram-gb", type=float, default=None)
    ap.add_argument("--stop-log", default=None, help="glob of log files to watch, e.g. campaign_*.log")
    ap.add_argument("--stop-pattern", default="CRASH")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.summarize:
        summarize(args.seeds)
        choice = SWEEP_DIR / "choice.json"
        if choice.exists() and _load(choice)["complete"]:
            summarize(args.seeds, per_fold=True)
        return

    git = require_clean_tree(args.allow_dirty)
    guard = ResourceGuard(args.min_free_ram_gb, args.stop_log, args.stop_pattern)
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, f"silo1_unseen_{(git['commit'] or 'nogit')[:7]}")
    try:
        if args.lr_sweep:
            choice = lr_sweep(args.device, git, guard, args.holdouts)
            print(json.dumps({k: choice[k] for k in ("complete", "all_folds_same_as_shared", "n_changed")}))
            for m in (FL_NAME, "centralized"):
                print(f"  {m}: " + ", ".join(f"{h}={choice[m][h]['lr']:g}" for h in sorted(choice[m])))
            return
        if args.per_fold_lr:
            per_fold_runs(args.seeds, args.device, git, ckpt_dir, guard, args.check_unchanged)
            summarize(args.seeds, per_fold=True)
            return

        lrs = load_lrs()
        print(f"lrs {({m: lrs[m] for m in METHODS})}; checkpoints -> {ckpt_dir}", flush=True)
        for seed in args.seeds:
            part = load_partition(seed)
            data = event_arrays(part)
            holdouts = args.holdouts or sorted(data)
            print(f"seed {seed}", flush=True)
            guard.check()
            run_local(part, data, seed, lrs, args.device, git, ckpt_dir)
            for h in holdouts:
                for method in (FL_NAME, "centralized"):
                    guard.check()
                    run_fold(method, part, data, h, seed, lrs[method], shared_lr_meta(lrs, method), args.device,
                             git, ckpt_dir)
        if args.holdouts is None:
            summarize(args.seeds)
    except GuardStop as e:
        print(f"GUARD STOP: {e} (completed runs are kept; rerun the same command to resume)", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
