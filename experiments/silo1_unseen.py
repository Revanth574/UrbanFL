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

Outputs: results/silo1_unseen/{local_only_seed<k>, <method>_holdout-<event>_seed<k>}/{metrics.json, predictions.json}
and results/silo1_unseen/summary.json. Best states go to <checkpoint_dir>/silo1_unseen_<commit> (git-ignored).

Usage:  python -m experiments.silo1_unseen [--seeds 0 1 2 3 4] [--threads 3]
        python -m experiments.silo1_unseen --summarize
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from experiments.silo1_baselines import CONFIG as BASELINE_CONFIG
from experiments.silo1_baselines import EXPERIMENT_DIR as BASELINE_DIR
from experiments.silo1_baselines import SWEEP_DIR as BASELINE_SWEEP_DIR
from experiments.silo1_baselines import client_arrays, predict, train_sgd
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


# ── Runs ───────────────────────────────────────────────────────────────────

def _done(run_dir: Path, git: dict) -> bool:
    path = run_dir / "metrics.json"
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]["training_git"]["commit"] == git["commit"]


def _config(method: str, lr: float, seed: int, device: str, lrs: dict, **extra) -> dict:
    recipe = ({**FL_COMMON, **FL_CONFIGS[FL_NAME]} if method == FL_NAME else
              {k: BASELINE_CONFIG[k] for k in ("head", "optimizer", "batch_size", "epochs", "checkpoint_selection")})
    return {"method": method, "protocol": "leave-one-event-out (unseen disaster)", **recipe, "lr": lr,
            "lr_source": lrs["sources"][method], "lr_note": lrs["note"], "lr_sweep_commits": lrs["sweep_commits"],
            "heldout_rows": "all rows of the held-out event (train+val+test of its split); never used for training, "
                            "class weights or checkpoint selection",
            "device": device, "labels": list(LABELS), "split_file": f"data/splits/silo1_seed{seed}.json", **extra}


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
                  config=_config("local_only", lr, seed, device, lrs,
                                 class_weights="each event's own train label counts",
                                 checkpoint_dir=str(ckpt_dir)),
                  split_sizes=part.split_sizes(), runtime_s=runtime)


def run_fold(method: str, part, data: dict, holdout: str, seed: int, lrs: dict, device: str, git: dict,
             ckpt_dir: Path) -> None:
    run_dir = EXPERIMENT_DIR / f"{method}_holdout-{holdout}_seed{seed}"
    if _done(run_dir, git):
        print(f"  skip {run_dir.name} (done at this commit)", flush=True)
        return
    train_events = [e for e in sorted(data) if e != holdout]
    lr = lrs[method]
    t0 = time.perf_counter()
    train = train_fedavg if method == FL_NAME else train_centralized
    model, info = train({e: data[e] for e in train_events}, train_events, lr, seed, device)
    runtime = time.perf_counter() - t0
    save_atomic(model.state_dict(), ckpt_dir / f"{run_dir.name}.pt")

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
                  config=_config(method, lr, seed, device, lrs, holdout_event=holdout, train_events=train_events,
                                 class_weights=("global, federated sum of the 9 training events' label histograms"
                                                if method == FL_NAME else "pooled train label counts of the 9"),
                                 checkpoint_dir=str(ckpt_dir)),
                  split_sizes={e: part.split_sizes()[e] for e in sorted(data)}, runtime_s=runtime)
    print(f"  {run_dir.name:70s} held-out D4 F1 {metrics['heldout']['all_rows']['d4_macro_f1']:.4f}  "
          f"({runtime:.0f}s)", flush=True)


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


def collect_seed(seed: int, events: list[str]) -> dict:
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
                p = _load(EXPERIMENT_DIR / f"{m}_holdout-{h}_seed{seed}" / "predictions.json")["heldout"]
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
            fm = _load(EXPERIMENT_DIR / f"{m}_holdout-{h}_seed{seed}" / "metrics.json")["metrics"]
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


def summarize(seeds: list[int]) -> dict:
    events = sorted(_load(EXPERIMENT_DIR / f"local_only_seed{seeds[0]}" / "predictions.json")["heldout_rows"])
    per_seed = {s: collect_seed(s, events) for s in seeds}

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

    configs = {m: _load(EXPERIMENT_DIR / (f"local_only_seed{seeds[0]}" if m == "local_only" else
                                          f"{m}_holdout-{events[0]}_seed{seeds[0]}") / "metrics.json")["config"]
               for m in METHODS}
    summary = {
        "protocol": "leave-one-event-out: train on 9 events, test on all rows of the held-out 10th; checkpoint "
                    "selection on the 9 training events' val sets only",
        "seeds": seeds, "events": events, "methods": list(METHODS),
        "lrs": {m: c["lr"] for m, c in configs.items()},
        "lr_note": configs[FL_NAME]["lr_note"],
        "metric_definitions": {
            "pooled_heldout_macro_f1": "9-class macro-F1 over all held-out predictions of the 10 folds concatenated "
                                       "(each tweet once; local_only: once per other-event model, 9x)",
            "event_d4_macro_f1": f"macro-F1 on one held-out event over classes with >= {MIN_SUPPORT} true examples; "
                                 "local_only = mean over the 9 other-event models",
            "test_split": "same, restricted to the held-out event's own test split (secondary)"},
        "table": table, "per_event": per_event, "paired_tests": tests,
        "in_distribution_reference": {
            "local_only_own_test_pooled_macro_f1": _ms([per_seed[s]["local_only"]["own_test_pooled_macro_f1"]
                                                        for s in seeds]),
            "note": "fedavg/centralized seen-event test F1 per fold is in per_event[*][method]"},
        "embedding_cache_reproduction_check": reproduction_check(seeds),
        "per_seed": {f"seed{s}": {m: per_seed[s][m]["pooled"] for m in METHODS} for s in seeds},
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_summary(summary)
    return summary


def print_summary(sm: dict) -> None:
    f = lambda c, d=4: f"{c['mean']:.{d}f} ± {c['std']:.{d}f}"
    print(f"\nUnseen-disaster (leave-one-event-out), seeds {sm['seeds']}, lrs {sm['lrs']}")
    print(f"{'method':12s} | {'pooled held-out F1':18s} | {'pooled acc':17s} | {'event-mean D4 F1':17s} | test-split F1")
    for m, r in sm["table"].items():
        print(f"{m:12s} | {f(r['pooled_heldout_macro_f1']):18s} | {f(r['pooled_heldout_accuracy']):17s} | "
              f"{f(r['event_mean_d4_macro_f1']):17s} | {f(r['pooled_heldout_test_split_macro_f1'])}")
    print(f"\nPer held-out event: D4 macro-F1, mean ± std over seeds")
    print(f"{'held-out event':36s} | " + " | ".join(f"{m:15s}" for m in sm["methods"]))
    for h, r in sm["per_event"].items():
        print(f"{h:36s} | " + " | ".join(f"{f(r[m]['d4_macro_f1'], 3):15s}" for m in sm["methods"]))
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
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--checkpoint-dir", default=None, help="default: URBANFL_CHECKPOINT_DIR / user config")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.summarize:
        summarize(args.seeds)
        return

    git = require_clean_tree(args.allow_dirty)
    lrs = load_lrs()
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, f"silo1_unseen_{(git['commit'] or 'nogit')[:7]}")
    print(f"lrs {({m: lrs[m] for m in METHODS})}; checkpoints -> {ckpt_dir}", flush=True)
    for seed in args.seeds:
        part = load_partition(seed)
        data = event_arrays(part)
        holdouts = args.holdouts or sorted(data)
        print(f"seed {seed}", flush=True)
        run_local(part, data, seed, lrs, args.device, git, ckpt_dir)
        for h in holdouts:
            for method in (FL_NAME, "centralized"):
                run_fold(method, part, data, h, seed, lrs, args.device, git, ckpt_dir)
    if args.holdouts is None:
        summarize(args.seeds)


if __name__ == "__main__":
    main()
