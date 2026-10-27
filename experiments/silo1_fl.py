"""
Silo 1 federated experiments: FedAvg, FedProx and FedAvg + local fine-tuning over the 10 event clients.

Reuses the saved splits (data/splits/silo1_seed{k}.json), the embedding cache, MLPHead and the class-weighting
scheme of the centralized baseline. Class weights are GLOBAL, obtained by a federated count sum: each client sends
its 9-integer train-label histogram once (logged as one-off communication) and the server sums them. FL then
optimises the same weighted objective as centralized, so the FL-vs-centralized gap reflects the training procedure,
not a different loss. The server does learn each client's label histogram (not covered by any DP accounting).

Checkpoints are selected by federated validation loss only (see src/fl/runner.py). Test data never selects anything.

Modes:
  --lr-sweep   FedAvg, seed 0, LR_GRID x local_epochs, each at its experiment budget (E=1: 100 rounds, E=5: 200);
               client lr per local_epochs = lowest best federated val loss -> results/silo1_fl/lr_sweep/choice.json
  (default)    CONFIGS x seeds -> results/silo1_fl/<config>_seed<k>/{metrics.json, predictions.json, checkpoint.pt}
  --finetune   FedAvg+FT: every client fine-tunes the selected FedAvg E=1 global model on its own train set
               (plain SGD, FL client lr) for up to 1 / 2 epochs; each client keeps the epoch in {0..k} with the
               lowest loss on ITS OWN val set (0 = keep the global model) -> ft<k>_fedavg_E1_seed<s>/
  --summarize  tables + results/silo1_fl/summary.json from saved JSON only (also runs after the other modes)

Usage:  python -m experiments.silo1_fl --lr-sweep --workers 5 --threads 2
        python -m experiments.silo1_fl --workers 5 --threads 2
        python -m experiments.silo1_fl --finetune
        python -m experiments.silo1_fl --summarize
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from experiments.silo1_baselines import CONFIG as BASELINE_CONFIG
from experiments.silo1_baselines import EXPERIMENT_DIR as BASELINE_DIR
from experiments.silo1_baselines import client_arrays, compute_metrics
from src.fl.algorithms import local_train
from src.fl.models import MLPHead, class_weights_from_counts
from src.fl.partition import LABELS, load_partition
from src.fl.runner import ClientData, evaluate, run_fl
from src.fl.utils import RESULTS_DIR, require_clean_tree, set_seed, write_metrics

EXPERIMENT_DIR = RESULTS_DIR / "silo1_fl"
SWEEP_DIR = EXPERIMENT_DIR / "lr_sweep"
SEEDS = [0, 1, 2, 3, 4]
N_CLASSES = len(LABELS)

COMMON = {"batch_size": 64, "client_fraction": 1.0, "head": BASELINE_CONFIG["head"],
          "client_optimizer": "SGD (no momentum, no weight decay)",
          "class_weights": "global, federated sum of client train-label histograms",
          "checkpoint_selection": "round with lowest client-size-weighted mean val loss (class-weighted CE)"}
LR_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
ROUNDS_FOR_EPOCHS = {1: 100, 5: 200}     # E=5 gets 200 rounds: at 100 it was still improving (budget-limited)

CONFIGS = {
    "fedavg_E1": {"algorithm": "fedavg", "mu": 0.0, "local_epochs": 1, "rounds": 100},
    "fedprox_mu0.001_E1": {"algorithm": "fedprox", "mu": 0.001, "local_epochs": 1, "rounds": 100},
    "fedprox_mu0.01_E1": {"algorithm": "fedprox", "mu": 0.01, "local_epochs": 1, "rounds": 100},
    "fedprox_mu0.1_E1": {"algorithm": "fedprox", "mu": 0.1, "local_epochs": 1, "rounds": 100},
    "fedavg_E5_R200": {"algorithm": "fedavg", "mu": 0.0, "local_epochs": 5, "rounds": 200},
}
FT_BASE = "fedavg_E1"
FT_MAX_EPOCHS = 2
FT_NAMES = [f"ft{k}_{FT_BASE}" for k in range(1, FT_MAX_EPOCHS + 1)]
PLATEAU_TOL = 0.001   # mean fed val loss drop over the last 10% of rounds below this -> plateaued
STRONG_REFERENCES = ("local_only_adamw", "centralized_adamw")
PAIRED_FL = "fedavg_E1"


def build_clients(part, device: str) -> tuple[list[ClientData], np.ndarray]:
    clients, hist = [], np.zeros(N_CLASSES, dtype=np.int64)
    for name in sorted(part.clients):
        arrays = {s: client_arrays(part, name, s) for s in ("train", "val", "test")}
        t = {s: (torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)) for s, (X, y) in arrays.items()}
        clients.append(ClientData(name, *t["train"], *t["val"], *t["test"]))
        hist += np.bincount(arrays["train"][1], minlength=N_CLASSES)   # what each client would send
    return clients, hist


def run_one(run_dir: Path, part, cfg: dict, lr: float, seed: int, device: str, training_git: dict) -> dict:
    clients, label_counts = build_clients(part, device)
    t0 = time.perf_counter()
    res = run_fl(cfg["algorithm"], clients, cfg["rounds"], cfg["local_epochs"], lr, COMMON["batch_size"],
                 COMMON["client_fraction"], seed, mu=cfg["mu"], class_weights=class_weights_from_counts(label_counts),
                 n_classes=N_CLASSES, head=COMMON["head"], device=device)
    runtime = time.perf_counter() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.json").write_text(json.dumps(res["test_predictions"]), encoding="utf-8")
    torch.save(res["best_state"], run_dir / "checkpoint.pt")      # git-ignored; consumed by --finetune
    metrics = {
        **compute_metrics(res["test_predictions"]),
        "fl": {"best_round": res["best_round"], "best_fed_val_loss": res["best_fed_val_loss"],
               "n_params": res["n_params"], "clients_per_round": res["clients_per_round"],
               "bytes_per_round": res["bytes_per_round"], "total_model_bytes": res["total_bytes"],
               "one_off_histogram_bytes": len(clients) * N_CLASSES * 8,
               "federated_label_counts": label_counts.tolist()},
        "history": res["history"],
        "training_git": training_git,
    }
    write_metrics(run_dir / "metrics.json", metrics, seed=seed,
                  config={**COMMON, **cfg, "lr": lr, "device": device, "labels": list(LABELS),
                          "split_file": f"data/splits/silo1_seed{seed}.json"},
                  split_sizes=part.split_sizes(), runtime_s=runtime)
    return metrics


def _job(job: dict) -> dict:
    """Process-pool worker (top level so it pickles on Windows spawn)."""
    if job.get("threads"):
        torch.set_num_threads(job["threads"])
    m = run_one(Path(job["run_dir"]), load_partition(job["seed"]), job["cfg"], job["lr"], job["seed"],
                job["device"], job["git"])
    return {"label": job["label"], "pooled": m["pooled_test_macro_f1"], "best_round": m["fl"]["best_round"],
            "best_fed_val_loss": m["fl"]["best_fed_val_loss"]}


def run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    out = []
    if workers <= 1:
        for j in jobs:
            out.append(_job(j))
            print(f"  done {out[-1]['label']}", flush=True)
        return out
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, j) for j in jobs]
        for f in as_completed(futures):
            r = f.result()
            out.append(r)
            print(f"  done {r['label']:34s} pooled macro-F1 {r['pooled']:.4f}  best round {r['best_round']}",
                  flush=True)
    return out


def lr_sweep(device: str, git: dict, workers: int, threads: int | None) -> dict:
    jobs = []
    for e, rounds in ROUNDS_FOR_EPOCHS.items():
        for lr in LR_GRID:
            jobs.append({"label": f"sweep_E{e}_lr{lr:g}", "run_dir": str(SWEEP_DIR / f"fedavg_E{e}_R{rounds}_lr{lr:g}_seed0"),
                         "cfg": {"algorithm": "fedavg", "mu": 0.0, "local_epochs": e, "rounds": rounds},
                         "lr": lr, "seed": 0, "device": device, "threads": threads, "git": git, "E": e})
    results = {r["label"]: r for r in run_jobs(jobs, workers)}
    choice = {}
    for e, rounds in ROUNDS_FOR_EPOCHS.items():
        losses = {lr: results[f"sweep_E{e}_lr{lr:g}"]["best_fed_val_loss"] for lr in LR_GRID}
        rounds_at = {lr: results[f"sweep_E{e}_lr{lr:g}"]["best_round"] for lr in LR_GRID}
        best = min(losses, key=losses.get)
        choice[f"E{e}"] = {"lr": best, "rounds": rounds, "fed_val_loss_by_lr": {str(k): v for k, v in losses.items()},
                           "best_round_by_lr": {str(k): v for k, v in rounds_at.items()},
                           "chosen_lr_at_grid_edge": best in (LR_GRID[0], LR_GRID[-1])}
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "choice.json").write_text(json.dumps({
        "selection": "min best federated val loss, FedAvg, seed 0, no test data", "grid": LR_GRID,
        "training_git": git, **choice}, indent=1), encoding="utf-8")
    return choice


def finetune(seed: int, device: str, git: dict) -> None:
    """FedAvg+FT for one seed: per-client fine-tuning from the selected global checkpoint, own-val-loss selection."""
    base_dir = EXPERIMENT_DIR / f"{FT_BASE}_seed{seed}"
    base = json.loads((base_dir / "metrics.json").read_text(encoding="utf-8"))
    ckpt = torch.load(base_dir / "checkpoint.pt", map_location=device)
    lr = base["config"]["lr"]
    part = load_partition(seed)
    clients, counts = build_clients(part, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_counts(counts).to(device))

    t0 = time.perf_counter()
    set_seed(seed)
    traj = {}
    for i, c in enumerate(clients):
        model = MLPHead(c.X_train.shape[1], N_CLASSES, **COMMON["head"]).to(device)
        model.load_state_dict(ckpt)
        gen = torch.Generator(device="cpu").manual_seed(seed * 10_000 + 5_000 + i)   # distinct from FL streams
        states, val_losses = [], []
        for epoch in range(FT_MAX_EPOCHS + 1):
            if epoch > 0:
                local_train(model, c.X_train, c.y_train, criterion, epochs=1, lr=lr,
                            batch_size=COMMON["batch_size"], generator=gen)
            val_losses.append(evaluate(model, c.X_val, c.y_val, criterion)[0])
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        traj[c.name] = (c, states, val_losses)
    runtime = time.perf_counter() - t0

    for k in range(1, FT_MAX_EPOCHS + 1):
        predictions, chosen = {}, {}
        for name, (c, states, val_losses) in traj.items():
            e = int(np.argmin(val_losses[:k + 1]))          # validation loss only
            model = MLPHead(c.X_train.shape[1], N_CLASSES, **COMMON["head"]).to(device)
            model.load_state_dict(states[e])
            _, pred = evaluate(model, c.X_test, c.y_test, criterion)
            predictions[name] = {"y_true": c.y_test.cpu().numpy().tolist(), "y_pred": pred.tolist()}
            chosen[name] = {"epoch": e, "val_loss_by_epoch": val_losses[:k + 1]}
        n_val = np.array([len(traj[n][0].y_val) for n in chosen], dtype=float)
        sel = np.array([chosen[n]["val_loss_by_epoch"][chosen[n]["epoch"]] for n in chosen])
        run_dir = EXPERIMENT_DIR / f"ft{k}_{FT_BASE}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
        metrics = {**compute_metrics(predictions),
                   "ft": {"max_epochs": k, "chosen": chosen,
                          "n_clients_fine_tuned": int(sum(v["epoch"] > 0 for v in chosen.values())),
                          "fed_val_loss_after_selection": float((sel * n_val).sum() / n_val.sum()),
                          "base_fed_val_loss": base["metrics"]["fl"]["best_fed_val_loss"]},
                   "training_git": git}
        write_metrics(run_dir / "metrics.json", metrics, seed=seed,
                      config={"method": "FedAvg + local fine-tuning", "base_run": f"silo1_fl/{base_dir.name}",
                              "base_training_git": base["metrics"]["training_git"], "lr": lr,
                              "max_ft_epochs": k, "candidate_epochs": list(range(k + 1)),
                              "selection": "per client: epoch with lowest loss on its own val set (0 = global model)",
                              "criterion": COMMON["class_weights"], "batch_size": COMMON["batch_size"],
                              "generator": "seed*10000 + 5000 + client_index", "device": device},
                      split_sizes=part.split_sizes(), runtime_s=runtime)
        print(f"  FT seed {seed} k={k}: pooled macro-F1 {metrics['pooled_test_macro_f1']:.4f}, "
              f"{metrics['ft']['n_clients_fine_tuned']}/10 clients fine-tuned", flush=True)


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


def plateau_stats(history: list[dict]) -> dict:
    loss = [h["fed_val_loss"] for h in history]
    R = len(loss) - 1
    return {"rounds": R, "drop_last10pct": loss[R - R // 10] - loss[R], "drop_last20pct": loss[R - R // 5] - loss[R]}


def summarize(seeds: list[int]) -> dict:
    base = {m: {s: _load(BASELINE_DIR / f"{m}_seed{s}" / "metrics.json") for s in seeds}
            for m in ("majority_global", "majority_local", "local_only", "centralized")}
    missing = [f"{m}_seed{s}" for m, r in base.items() for s, v in r.items() if v is None]
    if missing:
        raise FileNotFoundError(f"baseline runs missing: {missing}")
    for m in STRONG_REFERENCES:   # optional AdamW references, included only when all seeds exist
        runs = {s: _load(BASELINE_DIR / f"{m}_seed{s}" / "metrics.json") for s in seeds}
        if all(runs.values()):
            base[m] = runs
    fl = {n: {s: _load(EXPERIMENT_DIR / f"{n}_seed{s}" / "metrics.json") for s in seeds} for n in [*CONFIGS, *FT_NAMES]}
    fl = {n: runs for n, runs in fl.items() if all(runs.values())}

    local, central = base["local_only"], base["centralized"]
    pooled = lambda runs, s: runs[s]["metrics"]["pooled_test_macro_f1"]
    rows = {}
    for name, runs in {**base, **fl}.items():
        per_seed = []
        for s in seeds:
            m = runs[s]["metrics"]
            f, lo, ce = pooled(runs, s), pooled(local, s), pooled(central, s)
            per_seed.append({"seed": s, "pooled_macro_f1": f, "d4_client_mean": m["d4"]["client_mean_macro_f1"],
                             "delta_vs_local": f - lo, "gap_recovery": (f - lo) / (ce - lo) if ce != lo else None,
                             "best_round": m.get("fl", {}).get("best_round"),
                             "best_fed_val_loss": m.get("fl", {}).get("best_fed_val_loss"),
                             **({"plateau": plateau_stats(m["history"])} if "history" in m else {})})
        rows[name] = {"per_seed": per_seed, **{k: _ms([p[k] for p in per_seed]) for k in
                      ("pooled_macro_f1", "d4_client_mean", "delta_vs_local", "gap_recovery", "best_round",
                       "best_fed_val_loss")}}
        if "history" in runs[seeds[0]]["metrics"]:
            d10 = _ms([p["plateau"]["drop_last10pct"] for p in per_seed])
            rows[name]["plateau"] = {"rounds": per_seed[0]["plateau"]["rounds"], "drop_last10pct": d10,
                                     "drop_last20pct": _ms([p["plateau"]["drop_last20pct"] for p in per_seed]),
                                     "verdict": "plateaued" if d10["mean"] < PLATEAU_TOL else "budget-limited"}

    # Headline configs: lowest mean best federated val loss over seeds. Never test metrics.
    def pick(names):
        cands = {n: rows[n]["best_fed_val_loss"]["mean"] for n in names if n in rows}
        return min(cands, key=cands.get) if cands else None
    headline = {"fedavg": pick([n for n in CONFIGS if n.startswith("fedavg")]),
                "fedprox": pick([n for n in CONFIGS if n.startswith("fedprox")])}
    prox = [n for n in CONFIGS if n.startswith("fedprox") and n in rows]
    best_prox_by_test = max(prox, key=lambda n: rows[n]["pooled_macro_f1"]["mean"]) if prox else None

    for name in rows:
        if name in STRONG_REFERENCES:
            rows[name]["selection"] = "reference (AdamW)"
        elif name in base:
            rows[name]["selection"] = "reference"
        elif name in headline.values():
            rows[name]["selection"] = "headline (val loss)"
        elif name in FT_NAMES:
            rows[name]["selection"] = "per-client val loss"
        else:
            rows[name]["selection"] = "grid member"
    appendix = None
    if best_prox_by_test and best_prox_by_test != headline["fedprox"]:
        appendix = {"row": best_prox_by_test, "reason": "best FedProx mu by TEST macro-F1 - appendix only"}

    per_client = {}
    for label, name in [("fedavg", headline["fedavg"]), ("fedprox", headline["fedprox"]),
                        *[(n, n) for n in FT_NAMES]]:
        if name is None or name not in fl:
            continue
        per_client[label] = {"config": name, "clients": {}}
        for c in sorted(local[seeds[0]]["metrics"]["per_client_test_accuracy"]):
            acc_d, f1_d = [], []
            for s in seeds:
                mf, ml = fl[name][s]["metrics"], local[s]["metrics"]
                acc_d.append(mf["per_client_test_accuracy"][c] - ml["per_client_test_accuracy"][c])
                a, b = mf["d4"]["per_client"][c]["macro_f1"], ml["d4"]["per_client"][c]["macro_f1"]
                f1_d.append(None if a is None or b is None else a - b)
            per_client[label]["clients"][c] = {"delta_accuracy": _ms(acc_d), "delta_d4_macro_f1": _ms(f1_d)}

    paired = {}
    if PAIRED_FL in fl:
        x = np.array([pooled(fl[PAIRED_FL], s) for s in seeds])
        for ref in ("local_only", "centralized", *STRONG_REFERENCES):
            if ref in base:
                y = np.array([pooled(base[ref], s) for s in seeds])
                paired[ref] = {**_ms(x - y), "p_paired_t": float(stats.ttest_rel(x, y).pvalue)}

    ft_info = {n: _ms([fl[n][s]["metrics"]["ft"]["n_clients_fine_tuned"] for s in seeds]) for n in FT_NAMES if n in fl}
    comm = {n: fl[n][seeds[0]]["metrics"]["fl"] for n in CONFIGS if n in fl}
    summary = {"seeds": seeds, "rows": rows, "headline": headline, "appendix_only": appendix,
               "headline_selection": "min mean best federated val loss over seeds",
               "per_client_delta_vs_local": per_client, "ft_clients_fine_tuned": ft_info,
               f"paired_delta_{PAIRED_FL}_vs": paired,
               "communication": {n: {k: c[k] for k in ("n_params", "bytes_per_round", "total_model_bytes",
                                                         "one_off_histogram_bytes")} for n, c in comm.items()}}
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_summary(summary)
    return summary


def print_summary(sm: dict) -> None:
    rows, seeds, hl = sm["rows"], sm["seeds"], sm["headline"]
    print(f"\nAll methods — mean ± std over seeds {seeds}")
    print(f"{'method':20s} | {'selection':20s} | {'pooled macro-F1':17s} | {'D4 client-mean':17s} | "
          f"{'Δ vs local':17s} | {'gap recovery':13s} | best rnd")
    print("-" * 132)
    for name, r in rows.items():
        best_round = "" if r["best_round"] is None else f"{r['best_round']['mean']:.0f}"
        print(f"{name:20s} | {r['selection']:20s} | {_fmt(r['pooled_macro_f1']):17s} | {_fmt(r['d4_client_mean']):17s} | "
              f"{_fmt(r['delta_vs_local']):17s} | {_fmt(r['gap_recovery'], 2):13s} | {best_round}")
    if sm["appendix_only"]:
        a = sm["appendix_only"]
        print(f"APPENDIX ONLY: {a['row']} — {a['reason']}")

    order = [o for o in ["majority_global", "local_only", hl.get("fedavg"), hl.get("fedprox"), *FT_NAMES,
                         "centralized"] if o and o in rows]
    print(f"\nPooled test macro-F1 per seed")
    print(f"{'seed':5s} | " + " | ".join(f"{o:18s}" for o in order))
    for i, s in enumerate(seeds):
        print(f"{s:<5d} | " + " | ".join(f"{rows[o]['per_seed'][i]['pooled_macro_f1']:<18.4f}" for o in order))
    print(f"{'mean':5s} | " + " | ".join(f"{_fmt(rows[o]['pooled_macro_f1']):18s}" for o in order))

    print("\nConvergence (mean federated val loss drop over the final 10% / 20% of rounds)")
    for name, r in rows.items():
        if "plateau" in r:
            p = r["plateau"]
            print(f"  {name:20s} R={p['rounds']:3d}  last10% {p['drop_last10pct']['mean']:+.4f}  "
                  f"last20% {p['drop_last20pct']['mean']:+.4f}  best round {r['best_round']['mean']:.0f}  -> {p['verdict']}")

    for label, pc in sm["per_client_delta_vs_local"].items():
        print(f"\nPer-client Δ vs local-only — {pc['config']} (mean ± std over seeds)")
        for c, d in pc["clients"].items():
            print(f"  {c:38s} Δacc {_fmt(d['delta_accuracy'], 3):15s}  ΔD4-F1 {_fmt(d['delta_d4_macro_f1'], 3)}")
    paired = sm.get(f"paired_delta_{PAIRED_FL}_vs", {})
    if paired:
        print(f"\nPaired Δ pooled macro-F1: {PAIRED_FL} minus reference (mean ± std over seeds, paired t-test)")
        for ref, d in paired.items():
            print(f"  vs {ref:20s} {d['mean']:+.4f} ± {d['std']:.4f}   p = {d['p_paired_t']:.4f}")
    for n, cell in sm["ft_clients_fine_tuned"].items():
        print(f"{n}: clients that chose to fine-tune (epoch > 0): {_fmt(cell, 1)} of 10")

    for name, c in sm["communication"].items():
        print(f"Communication {name:20s}: {c['n_params']:,} params, {c['bytes_per_round'] / 2**20:.2f} MiB/round, "
              f"{c['total_model_bytes'] / 2**20:.1f} MiB total + {c['one_off_histogram_bytes']} B label histograms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Silo 1 FedAvg / FedProx / FedAvg+FT experiments.")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--lr-sweep", action="store_true")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    # The 100k-parameter head is kernel-launch bound on GPU: CPU measured ~3.4x faster per round (RTX 3050).
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads per worker")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes over (config, seed) jobs")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.summarize:
        summarize(args.seeds)
        return
    git = require_clean_tree(args.allow_dirty)
    if args.lr_sweep:
        print(json.dumps(lr_sweep(args.device, git, args.workers, args.threads), indent=1))
        return
    if args.finetune:
        for seed in args.seeds:
            finetune(seed, args.device, git)
        summarize(args.seeds)
        return

    choice = _load(SWEEP_DIR / "choice.json")
    if choice is None:
        raise FileNotFoundError("run `python -m experiments.silo1_fl --lr-sweep` first")
    if choice["training_git"]["commit"] != git["commit"]:
        print(f"note: lr chosen at commit {choice['training_git']['commit'][:7]}, running at {git['commit'][:7]}")
    jobs = []
    for name in args.configs:
        cfg = CONFIGS[name]
        sel = choice[f"E{cfg['local_epochs']}"]
        if sel["rounds"] != cfg["rounds"]:
            raise ValueError(f"{name}: lr was swept at {sel['rounds']} rounds, config runs {cfg['rounds']}")
        for seed in args.seeds:
            jobs.append({"label": f"{name}_seed{seed}", "run_dir": str(EXPERIMENT_DIR / f"{name}_seed{seed}"),
                         "cfg": cfg, "lr": sel["lr"], "seed": seed, "device": args.device, "threads": args.threads,
                         "git": git})
    jobs.sort(key=lambda j: -j["cfg"]["rounds"] * j["cfg"]["local_epochs"])   # longest first for the pool
    run_jobs(jobs, args.workers)
    summarize(args.seeds)


if __name__ == "__main__":
    main()
