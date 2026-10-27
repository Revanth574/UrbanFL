"""
Silo 1 federated experiments: FedAvg and FedProx over the 10 event clients.

Reuses the saved splits (data/splits/silo1_seed{k}.json), the embedding cache, MLPHead and the class-weighting
scheme of the centralized baseline. Class weights are GLOBAL, obtained by a federated count sum: each client sends
its 9-integer train-label histogram once (logged as one-off communication) and the server sums them. FL then
optimises the same weighted objective as centralized, so the FL-vs-centralized gap reflects the training procedure,
not a different loss. The server does learn each client's label histogram (not covered by any DP accounting).

Modes:
  --lr-sweep   FedAvg, seed 0, lr grid x local_epochs; chooses client lr per local_epochs by the best FEDERATED
               val loss (no test data). Writes results/silo1_fl/lr_sweep/{runs..., choice.json}.
  (default)    all CONFIGS x seeds -> results/silo1_fl/<config>_seed<k>/{metrics.json, predictions.json}
  --summarize  tables + results/silo1_fl/summary.json from the saved JSON only (also runs after the default mode)

Usage:  python -m experiments.silo1_fl --lr-sweep
        python -m experiments.silo1_fl [--seeds 0 1 2 3 4] [--configs ...]
        python -m experiments.silo1_fl --summarize
        python -m experiments.silo1_fl --metrics-only    # after a metric change, no retraining
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from experiments.silo1_baselines import CONFIG as BASELINE_CONFIG
from experiments.silo1_baselines import EXPERIMENT_DIR as BASELINE_DIR
from experiments.silo1_baselines import client_arrays, compute_metrics
from src.fl.models import class_weights_from_counts
from src.fl.partition import LABELS, load_partition
from src.fl.runner import ClientData, run_fl
from src.fl.utils import RESULTS_DIR, accuracy, write_metrics

EXPERIMENT_DIR = RESULTS_DIR / "silo1_fl"
SWEEP_DIR = EXPERIMENT_DIR / "lr_sweep"
SEEDS = [0, 1, 2, 3, 4]
N_CLASSES = len(LABELS)

COMMON = {"rounds": 100, "batch_size": 64, "client_fraction": 1.0, "head": BASELINE_CONFIG["head"],
          "client_optimizer": "SGD (no momentum, no weight decay)",
          "class_weights": "global, federated sum of client train-label histograms",
          "checkpoint_selection": "round with lowest client-size-weighted mean val loss (class-weighted CE)"}
LR_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]   # 0.01 added after E=5 picked the old lower edge (0.03)


def config_name(algorithm: str, mu: float, local_epochs: int) -> str:
    return f"fedavg_E{local_epochs}" if algorithm == "fedavg" else f"fedprox_mu{mu:g}_E{local_epochs}"


CONFIGS = {config_name(a, mu, e): {"algorithm": a, "mu": mu, "local_epochs": e}
           for e in (1, 5) for a, mu in (("fedavg", 0.0), ("fedprox", 0.001), ("fedprox", 0.01), ("fedprox", 0.1))}


def build_clients(part, device: str) -> tuple[list[ClientData], np.ndarray]:
    clients, hist = [], np.zeros(N_CLASSES, dtype=np.int64)
    for name in sorted(part.clients):
        arrays = {s: client_arrays(part, name, s) for s in ("train", "val", "test")}
        t = {s: (torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)) for s, (X, y) in arrays.items()}
        clients.append(ClientData(name, *t["train"], *t["val"], *t["test"]))
        hist += np.bincount(arrays["train"][1], minlength=N_CLASSES)   # what each client would send
    return clients, hist


def run_one(run_dir, part, cfg: dict, lr: float, seed: int, device: str) -> dict:
    clients, label_counts = build_clients(part, device)
    t0 = time.perf_counter()
    res = run_fl(cfg["algorithm"], clients, COMMON["rounds"], cfg["local_epochs"], lr, COMMON["batch_size"],
                 COMMON["client_fraction"], seed, mu=cfg["mu"], class_weights=class_weights_from_counts(label_counts),
                 n_classes=N_CLASSES, head=COMMON["head"], device=device)
    runtime = time.perf_counter() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.json").write_text(json.dumps(res["test_predictions"]), encoding="utf-8")
    histogram_bytes = len(clients) * N_CLASSES * 8   # one-off, int64 histograms up
    metrics = {
        **compute_metrics(res["test_predictions"]),
        "fl": {"best_round": res["best_round"], "best_fed_val_loss": res["best_fed_val_loss"],
               "n_params": res["n_params"], "clients_per_round": res["clients_per_round"],
               "bytes_per_round": res["bytes_per_round"], "total_model_bytes": res["total_bytes"],
               "one_off_histogram_bytes": histogram_bytes, "federated_label_counts": label_counts.tolist()},
        "history": res["history"],
    }
    write_metrics(run_dir / "metrics.json", metrics, seed=seed,
                  config={**COMMON, **cfg, "lr": lr, "device": device, "labels": list(LABELS),
                          "split_file": f"data/splits/silo1_seed{seed}.json"},
                  split_sizes=part.split_sizes(), runtime_s=runtime)
    return metrics


def lr_sweep(device: str) -> dict:
    part = load_partition(0)
    choice = {}
    for e in (1, 5):
        losses = {}
        for lr in LR_GRID:
            run_dir = SWEEP_DIR / f"fedavg_E{e}_lr{lr:g}_seed0"
            m = _load(run_dir / "metrics.json")
            m = m["metrics"] if m else run_one(run_dir, part, {"algorithm": "fedavg", "mu": 0.0, "local_epochs": e},
                                               lr, 0, device)
            losses[lr] = m["fl"]["best_fed_val_loss"]
            print(f"  sweep E={e} lr={lr:<5g} best fed val loss {losses[lr]:.4f} @ round {m['fl']['best_round']}")
        best_lr = min(losses, key=losses.get)
        choice[f"E{e}"] = {"lr": best_lr, "fed_val_loss_by_lr": {str(k): v for k, v in losses.items()}}
    (SWEEP_DIR / "choice.json").write_text(json.dumps({
        "selection": "min best federated val loss, FedAvg, seed 0, no test data", "grid": LR_GRID, **choice},
        indent=1), encoding="utf-8")
    return choice


# ── Summary ────────────────────────────────────────────────────────────────

def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ms(vals):
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) == 0:
        return None
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)}


def _fmt(cell, digits=4):
    return "n/a" if cell is None else f"{cell['mean']:.{digits}f} ± {cell['std']:.{digits}f}"


def _d4_mean(m):
    return m["d4"]["client_mean_macro_f1"] if m.get("d4") else None


def summarize(seeds: list[int]) -> dict:
    base = {meth: {s: _load(BASELINE_DIR / f"{meth}_seed{s}" / "metrics.json") for s in seeds}
            for meth in ("majority_global", "majority_local", "local_only", "centralized")}
    fl = {name: {s: _load(EXPERIMENT_DIR / f"{name}_seed{s}" / "metrics.json") for s in seeds} for name in CONFIGS}
    fl = {n: runs for n, runs in fl.items() if all(runs.values())}
    missing_base = [f"{m}_seed{s}" for m, r in base.items() for s, v in r.items() if v is None]
    if missing_base:
        raise FileNotFoundError(f"baseline runs missing for {missing_base}; run experiments.silo1_baselines first")

    def pooled(runs, s):
        return runs[s]["metrics"]["pooled_test_macro_f1"]

    local, central = base["local_only"], base["centralized"]
    rows = {}
    for name, runs in {**{k: base[k] for k in base}, **fl}.items():
        per_seed = []
        for s in seeds:
            f, lo, ce = pooled(runs, s), pooled(local, s), pooled(central, s)
            per_seed.append({"seed": s, "pooled_macro_f1": f, "d4_client_mean": _d4_mean(runs[s]["metrics"]),
                             "delta_vs_local": f - lo,
                             "gap_recovery": (f - lo) / (ce - lo) if ce != lo else None,
                             "best_round": runs[s]["metrics"].get("fl", {}).get("best_round"),
                             "best_fed_val_loss": runs[s]["metrics"].get("fl", {}).get("best_fed_val_loss")})
        rows[name] = {"per_seed": per_seed, **{k: _ms([p[k] for p in per_seed]) for k in
                      ("pooled_macro_f1", "d4_client_mean", "delta_vs_local", "gap_recovery", "best_round",
                       "best_fed_val_loss")}}

    # Headline FL configs chosen by FEDERATED VAL LOSS (mean over seeds), never by test metrics.
    def pick(prefix):
        cands = {n: rows[n]["best_fed_val_loss"]["mean"] for n in fl if n.startswith(prefix)}
        return min(cands, key=cands.get) if cands else None
    headline = {"fedavg": pick("fedavg"), "fedprox": pick("fedprox")}

    per_client = {}
    for label, name in headline.items():
        if name is None:
            continue
        per_client[label] = {"config": name, "clients": {}}
        for c in sorted(local[seeds[0]]["metrics"]["per_client_test_accuracy"]):
            acc_d, f1_d = [], []
            for s in seeds:
                acc_d.append(fl[name][s]["metrics"]["per_client_test_accuracy"][c]
                             - local[s]["metrics"]["per_client_test_accuracy"][c])
                d_fl, d_lo = fl[name][s]["metrics"].get("d4"), local[s]["metrics"].get("d4")
                f1_d.append(d_fl["per_client"][c]["macro_f1"] - d_lo["per_client"][c]["macro_f1"]
                            if d_fl and d_lo and d_fl["per_client"][c]["macro_f1"] is not None
                            and d_lo["per_client"][c]["macro_f1"] is not None else None)
            per_client[label]["clients"][c] = {"delta_accuracy": _ms(acc_d), "delta_d4_macro_f1": _ms(f1_d)}

    comm = {n: runs[seeds[0]]["metrics"]["fl"] for n, runs in fl.items()}
    summary = {"seeds": seeds, "rows": rows, "headline_selection": "min mean best federated val loss over seeds",
               "headline": headline, "per_client_delta_vs_local": per_client,
               "communication": {n: {k: c[k] for k in ("n_params", "bytes_per_round", "total_model_bytes",
                                                         "one_off_histogram_bytes")} for n, c in comm.items()}}
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_summary(summary)
    return summary


def print_summary(sm: dict) -> None:
    rows, seeds, hl = sm["rows"], sm["seeds"], sm["headline"]
    print(f"\nAll configs — mean ± std over seeds {seeds}")
    print(f"{'method':22s} | {'pooled macro-F1':17s} | {'D4 client-mean':17s} | {'Δ vs local':17s} | "
          f"{'gap recovery':15s} | best rnd")
    print("-" * 118)
    for name, r in rows.items():
        best_round = "" if r["best_round"] is None else f"{r['best_round']['mean']:.0f}"
        print(f"{name:22s} | {_fmt(r['pooled_macro_f1']):17s} | {_fmt(r['d4_client_mean']):17s} | "
              f"{_fmt(r['delta_vs_local']):17s} | {_fmt(r['gap_recovery'], 2):15s} | {best_round}")

    order = ["majority_global", "local_only", hl.get("fedavg"), hl.get("fedprox"), "centralized"]
    order = [o for o in order if o]
    print(f"\nReview 1 table — pooled test macro-F1 per seed (FL configs picked by federated val loss)")
    print(f"{'seed':5s} | " + " | ".join(f"{o:18s}" for o in order))
    for i, s in enumerate(seeds):
        print(f"{s:<5d} | " + " | ".join(f"{rows[o]['per_seed'][i]['pooled_macro_f1']:<18.4f}" for o in order))
    print(f"{'mean':5s} | " + " | ".join(f"{_fmt(rows[o]['pooled_macro_f1']):18s}" for o in order))

    for label, pc in sm["per_client_delta_vs_local"].items():
        print(f"\nPer-client Δ vs local-only — {label} = {pc['config']} (mean ± std over seeds)")
        for c, d in pc["clients"].items():
            print(f"  {c:38s} Δacc {_fmt(d['delta_accuracy'], 3):15s}  ΔD4-F1 {_fmt(d['delta_d4_macro_f1'], 3)}")

    name = hl.get("fedavg")
    if name:
        c = sm["communication"][name]
        print(f"\nCommunication ({name}): {c['n_params']:,} params, {c['bytes_per_round'] / 2**20:.2f} MiB/round, "
              f"{c['total_model_bytes'] / 2**20:.1f} MiB total + {c['one_off_histogram_bytes']} B label histograms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Silo 1 FedAvg / FedProx experiments.")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--lr-sweep", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute test metrics from saved predictions.json (keeps FL logs) without retraining")
    # The 100k-parameter head is kernel-launch bound on GPU: CPU measured ~3.4x faster per round (RTX 3050).
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (1 when running seeds in parallel)")
    args = ap.parse_args()
    device = args.device
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.lr_sweep:
        print(json.dumps(lr_sweep(device), indent=1))
        return
    if args.metrics_only:
        for seed in args.seeds:
            for name in args.configs:
                run_dir = EXPERIMENT_DIR / f"{name}_seed{seed}"
                old = _load(run_dir / "metrics.json")
                if old is None:
                    continue
                kept = {k: old["metrics"][k] for k in ("fl", "history")}
                kept["fl"] = {**kept["fl"], "predictions_git": kept["fl"].get("predictions_git", old["git"])}
                preds = _load(run_dir / "predictions.json")
                write_metrics(run_dir / "metrics.json", {**compute_metrics(preds), **kept}, seed=seed,
                              config=old["config"], split_sizes=old["split_sizes"], runtime_s=old["runtime_s"])
    elif not args.summarize:
        choice = _load(SWEEP_DIR / "choice.json")
        if choice is None:
            raise FileNotFoundError("run `python -m experiments.silo1_fl --lr-sweep` first")
        for seed in args.seeds:
            part = load_partition(seed)
            for name in args.configs:
                cfg = CONFIGS[name]
                lr = choice[f"E{cfg['local_epochs']}"]["lr"]
                m = run_one(EXPERIMENT_DIR / f"{name}_seed{seed}", part, cfg, lr, seed, device)
                print(f"seed {seed} {name:22s} lr {lr:<4g} pooled macro-F1 {m['pooled_test_macro_f1']:.4f}  "
                      f"best round {m['fl']['best_round']:3d}")
    summarize(args.seeds)


if __name__ == "__main__":
    main()
