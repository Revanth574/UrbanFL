"""
Silo 1 non-federated reference points (CLAUDE.md rule 6): majority-class, local-only (lower bound) and
centralized (upper bound), all on the same per-client test sets as the FL runs.

Trained baselines use exactly the FL client recipe, so FL-vs-baseline differences come from federation alone:
  - training runs through src.fl.runner.run_fl with a SINGLE client (single-client FedAvg == plain SGD is a tested
    invariant): plain SGD, batch 64, the same minibatch-order RNG and the same checkpoint rule;
  - budget: 100 epochs = the same number of passes over each client's data as FL E=1 x 100 rounds; no early stop;
  - checkpoint: epoch with the lowest validation LOSS (class-weighted CE). Local-only: each client's own val set.
    Centralized: pooled val set. Val macro-F1 is logged only;
  - client lr chosen from the FL grid on seed 0 by validation loss (--lr-sweep): local-only by the
    client-val-size-weighted mean of the clients' best val losses (same rule as FL), centralized by its best
    pooled val loss.

Class weights: centralized - pooled train label counts (equal to the FL federated count sum);
               local_only  - the client's own train label counts (an isolated agency cannot see others).

Each (method, seed) writes results/silo1_baselines/<method>_seed<k>/{metrics.json, predictions.json}.

Usage:  python -m experiments.silo1_baselines --lr-sweep
        python -m experiments.silo1_baselines [--seeds 0 1 2 3 4]
        python -m experiments.silo1_baselines --metrics-only   # recompute metrics from saved predictions
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from experiments.silo1_embed_cache import load_embeddings
from src.fl.models import MLPHead, class_weights
from src.fl.partition import LABEL_SHORT, LABELS, load_partition
from src.fl.runner import ClientData, run_fl
from src.fl.utils import (RESULTS_DIR, accuracy, d4_metrics, macro_f1, per_class_report, require_clean_tree,
                          write_metrics)

EXPERIMENT_DIR = RESULTS_DIR / "silo1_baselines"
SWEEP_DIR = EXPERIMENT_DIR / "lr_sweep"
LABEL_IDS = list(range(len(LABELS)))
METHODS = ("majority_global", "majority_local", "local_only", "centralized")
TRAINED = ("local_only", "centralized")
LR_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]   # identical to the FL client lr grid

CONFIG = {
    "encoder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (frozen, cached)",
    "head": {"hidden": 256, "dropout": 0.2},
    "optimizer": "SGD (no momentum, no weight decay) via single-client run_fl",
    "batch_size": 64,
    "epochs": 100,
    "checkpoint_selection": "epoch with lowest val loss (class-weighted CE); local-only: own val, "
                            "centralized: pooled val",
    "lr_grid": LR_GRID,
}


def train_sgd(X_tr, y_tr, X_va, y_va, lr: float, seed: int, device: str) -> tuple[MLPHead, dict]:
    """Single-client run_fl == plain SGD for CONFIG['epochs'] epochs; returns the lowest-val-loss checkpoint."""
    t = {k: torch.as_tensor(v, device=device) for k, v in (("Xtr", X_tr), ("ytr", y_tr), ("Xva", X_va), ("yva", y_va))}
    client = ClientData("single", t["Xtr"], t["ytr"], t["Xva"], t["yva"], t["Xva"], t["yva"])
    res = run_fl("fedavg", [client], CONFIG["epochs"], 1, lr, CONFIG["batch_size"], 1.0, seed,
                 class_weights=class_weights(y_tr, len(LABELS)), n_classes=len(LABELS), head=CONFIG["head"],
                 device=device)
    model = MLPHead(t["Xtr"].shape[1], len(LABELS), **CONFIG["head"]).to(device)
    model.load_state_dict(res["best_state"])
    history = [{"epoch": h["round"], "train_ce": h["train_ce"], "val_loss": h["fed_val_loss"],
                "val_macro_f1": h["pooled_val_macro_f1"]} for h in res["history"]]
    return model, {"lr": lr, "best_epoch": res["best_round"], "best_val_loss": res["best_fed_val_loss"],
                   "n_val": len(y_va), "history": history}


@torch.no_grad()
def predict(model: MLPHead, X) -> np.ndarray:
    model.eval()
    X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, device=next(model.parameters()).device)
    return model(X).argmax(dim=1).cpu().numpy()


def client_arrays(part, client: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    ids = part.clients[client][split]
    y = part.df.set_index("tweet_id").loc[ids, "y"].to_numpy(copy=True)  # pandas 3 CoW views are read-only
    return load_embeddings(ids), y


def run_method(method: str, part, seed: int, device: str, lr: float | None = None) -> tuple[dict, dict]:
    """Returns (predictions {client: {y_true, y_pred}}, training info)."""
    clients = sorted(part.clients)
    data = {c: {s: client_arrays(part, c, s) for s in ("train", "val", "test")} for c in clients}
    preds, info = {}, {}

    if method == "majority_global":
        pooled_y = np.concatenate([data[c]["train"][1] for c in clients])
        cls = int(np.bincount(pooled_y, minlength=len(LABELS)).argmax())
        info["predicted_class"] = LABELS[cls]
        preds = {c: np.full(len(data[c]["test"][1]), cls) for c in clients}

    elif method == "majority_local":
        for c in clients:
            cls = int(np.bincount(data[c]["train"][1], minlength=len(LABELS)).argmax())
            info[c] = {"predicted_class": LABELS[cls]}
            preds[c] = np.full(len(data[c]["test"][1]), cls)

    elif method == "local_only":
        for c in clients:
            model, info[c] = train_sgd(*data[c]["train"], *data[c]["val"], lr=lr, seed=seed, device=device)
            preds[c] = predict(model, data[c]["test"][0])
        n_val = np.array([info[c]["n_val"] for c in clients], dtype=float)
        info["size_weighted_best_val_loss"] = float(
            (np.array([info[c]["best_val_loss"] for c in clients]) * n_val).sum() / n_val.sum())

    elif method == "centralized":
        X_tr = np.concatenate([data[c]["train"][0] for c in clients])
        y_tr = np.concatenate([data[c]["train"][1] for c in clients])
        X_va = np.concatenate([data[c]["val"][0] for c in clients])
        y_va = np.concatenate([data[c]["val"][1] for c in clients])
        model, info = train_sgd(X_tr, y_tr, X_va, y_va, lr=lr, seed=seed, device=device)
        preds = {c: predict(model, data[c]["test"][0]) for c in clients}

    else:
        raise ValueError(method)

    predictions = {c: {"y_true": data[c]["test"][1].tolist(), "y_pred": preds[c].tolist()} for c in clients}
    return predictions, info


def sweep_score(method: str, info: dict) -> float:
    """Validation-loss criterion for lr choice (never test data)."""
    return info["size_weighted_best_val_loss"] if method == "local_only" else info["best_val_loss"]


def compute_metrics(predictions: dict) -> dict:
    y_true = np.concatenate([p["y_true"] for p in predictions.values()])
    y_pred = np.concatenate([p["y_pred"] for p in predictions.values()])
    metrics = {
        "pooled_test_macro_f1": macro_f1(y_true, y_pred, LABEL_IDS),
        "pooled_test_accuracy": accuracy(y_true, y_pred),
        "pooled_per_class": per_class_report(y_true, y_pred, LABEL_IDS, LABEL_SHORT),
        "per_client_test_accuracy": {c: accuracy(p["y_true"], p["y_pred"]) for c, p in predictions.items()},
    }
    d4 = d4_metrics({c: p["y_true"] for c, p in predictions.items()},
                    {c: p["y_pred"] for c, p in predictions.items()}, LABEL_IDS, min_support=3)
    if abs(d4["pooled_macro_f1"] - metrics["pooled_test_macro_f1"]) > 1e-9:
        raise AssertionError("d4_metrics pooled macro-F1 disagrees with macro_f1 on the pooled test set")
    metrics["d4"] = d4
    return metrics


def execute(method: str, part, seed: int, device: str, lr: float | None, run_dir, training_git: dict) -> dict:
    t0 = time.perf_counter()
    predictions, info = run_method(method, part, seed, device, lr)
    runtime = time.perf_counter() - t0
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    metrics = {**compute_metrics(predictions), "training": info, "training_git": training_git}
    write_metrics(run_dir / "metrics.json", metrics, seed=seed,
                  config={**CONFIG, "method": method, "lr": lr, "device": device, "labels": list(LABELS),
                          "split_file": f"data/splits/silo1_seed{seed}.json", "dedup_stats": part.stats},
                  split_sizes=part.split_sizes(), runtime_s=runtime)
    return metrics


def lr_sweep(device: str, training_git: dict) -> dict:
    part = load_partition(0)
    choice = {}
    for method in TRAINED:
        scores = {}
        for lr in LR_GRID:
            m = execute(method, part, 0, device, lr, SWEEP_DIR / f"{method}_lr{lr:g}_seed0", training_git)
            scores[lr] = sweep_score(method, m["training"])
            print(f"  sweep {method:12s} lr={lr:<5g} val loss {scores[lr]:.4f}")
        choice[method] = {"lr": min(scores, key=scores.get), "val_loss_by_lr": {str(k): v for k, v in scores.items()}}
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "choice.json").write_text(json.dumps({
        "selection": "seed 0; local_only: client-val-size-weighted mean best val loss; centralized: best pooled "
                     "val loss; no test data", "grid": LR_GRID, "training_git": training_git, **choice},
        indent=1), encoding="utf-8")
    return choice


def summarize(seeds: list[int]) -> dict:
    """Mean / sample std over seeds, read back from the per-run metrics.json files."""
    summary = {}
    for method in METHODS:
        runs = [EXPERIMENT_DIR / f"{method}_seed{s}" / "metrics.json" for s in seeds]
        runs = [r for r in runs if r.exists()]
        if not runs:
            continue
        ms = [json.loads(r.read_text(encoding="utf-8"))["metrics"] for r in runs]
        row = {"n_seeds": len(ms), "sources": [str(r.relative_to(EXPERIMENT_DIR.parent.parent)) for r in runs]}
        keys = {"pooled_test_macro_f1": lambda m: m["pooled_test_macro_f1"],
                "pooled_test_accuracy": lambda m: m["pooled_test_accuracy"],
                "client_mean_macro_f1": lambda m: m["d4"]["client_mean_macro_f1"]}
        for key, get in keys.items():
            vals = np.array([get(m) for m in ms], dtype=float)
            row[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                        "values": vals.tolist()}
        summary[method] = row
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def print_table(summary: dict) -> None:
    def fmt(cell):
        if not cell:
            return "not computed"
        return f"{cell['mean']:.4f} ± {cell['std']:.4f}" if cell["std"] is not None else f"{cell['mean']:.4f}"
    print(f"\n{'method':18s} | {'pooled macro-F1':18s} | {'pooled accuracy':18s} | {'client-mean macro-F1 (D4)':26s} | seeds")
    print("-" * 100)
    for method, row in summary.items():
        print(f"{method:18s} | {fmt(row['pooled_test_macro_f1']):18s} | {fmt(row['pooled_test_accuracy']):18s} | "
              f"{fmt(row.get('client_mean_macro_f1')):26s} | {row['n_seeds']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Silo 1 majority / local-only / centralized baselines.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--lr-sweep", action="store_true", help="choose local-only / centralized lr on seed 0")
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute metrics.json from saved predictions.json without retraining")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    git = require_clean_tree(args.allow_dirty)

    if args.lr_sweep:
        print(json.dumps(lr_sweep(args.device, git), indent=1))
        return

    choice = None
    if not args.metrics_only:
        path = SWEEP_DIR / "choice.json"
        if any(m in TRAINED for m in args.methods) and not path.exists():
            raise FileNotFoundError("run `python -m experiments.silo1_baselines --lr-sweep` first")
        choice = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    for seed in args.seeds:
        part = load_partition(seed)
        for method in args.methods:
            run_dir = EXPERIMENT_DIR / f"{method}_seed{seed}"
            if args.metrics_only:
                old = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
                metrics = {**compute_metrics(predictions), "training": old["metrics"]["training"],
                           "training_git": old["metrics"].get("training_git", old["git"])}
                write_metrics(run_dir / "metrics.json", metrics, seed=seed, config=old["config"],
                              split_sizes=old["split_sizes"], runtime_s=old["runtime_s"])
            else:
                lr = choice[method]["lr"] if method in TRAINED else None
                metrics = execute(method, part, seed, args.device, lr, run_dir, git)
            print(f"seed {seed} {method:16s} pooled macro-F1 {metrics['pooled_test_macro_f1']:.4f}  "
                  f"acc {metrics['pooled_test_accuracy']:.4f}")

    print_table(summarize(args.seeds))


if __name__ == "__main__":
    main()
