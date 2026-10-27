"""
Silo 1 non-federated reference points (CLAUDE.md rule 6): majority-class, local-only (lower bound) and
centralized (upper bound), all on the same per-client test sets as the FL runs.

Each (method, seed) writes results/silo1_baselines/<method>_seed<k>/{metrics.json, predictions.json}.
predictions.json keeps every client's test y_true / y_pred, so metrics can be recomputed without retraining
(`--metrics-only`), e.g. after src.fl.utils.d4_metrics is implemented.

Class weights:
  centralized  - pooled train label counts.
  local_only   - the client's own train label counts (an isolated agency cannot see other agencies' histograms).

Usage:  python -m experiments.silo1_baselines [--seeds 0 1 2 3 4] [--metrics-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.silo1_embed_cache import load_embeddings
from src.fl.models import MLPHead, weighted_ce
from src.fl.partition import LABEL_SHORT, LABELS, load_partition
from src.fl.utils import (RESULTS_DIR, accuracy, d4_metrics, macro_f1, per_class_report, set_seed,
                          write_metrics)

EXPERIMENT_DIR = RESULTS_DIR / "silo1_baselines"
LABEL_IDS = list(range(len(LABELS)))
METHODS = ("majority_global", "majority_local", "local_only", "centralized")

CONFIG = {
    "encoder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (frozen, cached)",
    "head": {"hidden": 256, "dropout": 0.2},
    "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4},
    "batch_size": 64,
    "max_epochs": 200,
    "early_stopping": {"metric": "val macro-F1 over all 9 labels", "patience": 15},
}


def train_head(X_tr, y_tr, X_va, y_va, seed: int, device: str) -> tuple[MLPHead, dict]:
    """Train an MLPHead with class-weighted CE; keep the epoch with the best validation macro-F1."""
    set_seed(seed)
    X_tr, y_tr = torch.as_tensor(X_tr, device=device), torch.as_tensor(y_tr, device=device)
    X_va = torch.as_tensor(X_va, device=device)

    model = MLPHead(X_tr.shape[1], len(LABELS), **CONFIG["head"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), **{k: v for k, v in CONFIG["optimizer"].items() if k != "name"})
    criterion = weighted_ce(y_tr.cpu().numpy(), len(LABELS), device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bs, patience = CONFIG["batch_size"], CONFIG["early_stopping"]["patience"]

    best = {"val_macro_f1": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(y_tr), generator=gen).to(device)
        total = 0.0
        for start in range(0, len(perm), bs):
            idx = perm[start:start + bs]
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_f1 = macro_f1(y_va, predict(model, X_va), LABEL_IDS)
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_macro_f1": val_f1})
        if val_f1 > best["val_macro_f1"]:
            best = {"val_macro_f1": val_f1, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= patience:
            break
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_val_macro_f1": best["val_macro_f1"],
                   "epochs_run": len(history), "history": history}


@torch.no_grad()
def predict(model: MLPHead, X) -> np.ndarray:
    model.eval()
    X = X if isinstance(X, torch.Tensor) else torch.as_tensor(X, device=next(model.parameters()).device)
    return model(X).argmax(dim=1).cpu().numpy()


def client_arrays(part, client: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    ids = part.clients[client][split]
    y = part.df.set_index("tweet_id").loc[ids, "y"].to_numpy(copy=True)  # pandas 3 CoW views are read-only
    return load_embeddings(ids), y


def run_method(method: str, part, seed: int, device: str) -> tuple[dict, dict]:
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
            model, info[c] = train_head(*data[c]["train"], *data[c]["val"], seed=seed, device=device)
            preds[c] = predict(model, data[c]["test"][0])

    elif method == "centralized":
        X_tr = np.concatenate([data[c]["train"][0] for c in clients])
        y_tr = np.concatenate([data[c]["train"][1] for c in clients])
        X_va = np.concatenate([data[c]["val"][0] for c in clients])
        y_va = np.concatenate([data[c]["val"][1] for c in clients])
        model, info = train_head(X_tr, y_tr, X_va, y_va, seed=seed, device=device)
        preds = {c: predict(model, data[c]["test"][0]) for c in clients}

    else:
        raise ValueError(method)

    predictions = {c: {"y_true": data[c]["test"][1].tolist(), "y_pred": preds[c].tolist()} for c in clients}
    return predictions, info


def compute_metrics(predictions: dict) -> dict:
    y_true = np.concatenate([p["y_true"] for p in predictions.values()])
    y_pred = np.concatenate([p["y_pred"] for p in predictions.values()])
    metrics = {
        "pooled_test_macro_f1": macro_f1(y_true, y_pred, LABEL_IDS),
        "pooled_test_accuracy": accuracy(y_true, y_pred),
        "pooled_per_class": per_class_report(y_true, y_pred, LABEL_IDS, LABEL_SHORT),
        "per_client_test_accuracy": {c: accuracy(p["y_true"], p["y_pred"]) for c, p in predictions.items()},
    }
    try:
        d4 = d4_metrics({c: p["y_true"] for c, p in predictions.items()},
                        {c: p["y_pred"] for c, p in predictions.items()}, LABEL_IDS, min_support=3)
        if abs(d4["pooled_macro_f1"] - metrics["pooled_test_macro_f1"]) > 1e-9:
            raise AssertionError("d4_metrics pooled macro-F1 disagrees with macro_f1 on the pooled test set")
        metrics["d4"] = d4
    except NotImplementedError:
        metrics["d4"] = None
        metrics["d4_status"] = "not computed: src.fl.utils.d4_metrics is not implemented yet"
    return metrics


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
                "pooled_test_accuracy": lambda m: m["pooled_test_accuracy"]}
        if all(m.get("d4") for m in ms):
            keys["client_mean_macro_f1"] = lambda m: m["d4"]["client_mean_macro_f1"]
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
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute metrics.json from saved predictions.json without retraining")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in args.seeds:
        part = load_partition(seed)
        for method in args.methods:
            run_dir = EXPERIMENT_DIR / f"{method}_seed{seed}"
            pred_path = run_dir / "predictions.json"
            if args.metrics_only:
                old = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                predictions = json.loads(pred_path.read_text(encoding="utf-8"))
                info, runtime = old["metrics"].get("training", {}), old["runtime_s"]
                info = {**info, "predictions_git": old["metrics"].get("training", {}).get("predictions_git", old["git"])}
            else:
                t0 = time.perf_counter()
                predictions, info = run_method(method, part, seed, device)
                runtime = time.perf_counter() - t0
                run_dir.mkdir(parents=True, exist_ok=True)
                pred_path.write_text(json.dumps(predictions), encoding="utf-8")
            metrics = {**compute_metrics(predictions), "training": info}
            write_metrics(run_dir / "metrics.json", metrics, seed=seed,
                          config={**CONFIG, "method": method, "device": device, "labels": list(LABELS),
                                  "split_file": f"data/splits/silo1_seed{seed}.json",
                                  "dedup_stats": part.stats},
                          split_sizes=part.split_sizes(), runtime_s=runtime)
            print(f"seed {seed} {method:16s} pooled macro-F1 {metrics['pooled_test_macro_f1']:.4f}  "
                  f"acc {metrics['pooled_test_accuracy']:.4f}  ({runtime:.1f}s)")

    print_table(summarize(args.seeds))


if __name__ == "__main__":
    main()
