"""
Silo 1 client-level DP-FedAvg (src/fl/dp.py) on the FedAvg E=1 base, with an Opacus RDP accountant.

PRIVACY UNIT = ONE CLIENT (disaster event). With K = 10 clients, client-level ε is necessarily large: subsampling
amplification needs small sampling rates, and q < 0.5 with 10 clients leaves many rounds nearly empty. Large ε here is
the expected, honest finding for client-level DP at this scale — not a bug.

Grid: z (noise multiplier) in {0.5, 1, 2}, C (clip norm) in {0.1, 1.0}, q (client_fraction, Poisson rate) in
{0.5, 1.0}; 100 rounds, E = 1, client lr = the FedAvg E=1 lr (results/silo1_fl/lr_sweep/choice.json). Plus no-noise
references (z = 0: clipping + unweighted mean + local class weights + Poisson sampling, ε = ∞) for each (C, q), which
isolate the cost of the noise from the cost of the DP pipeline itself.

Inside the DP boundary: only clipped, noised model updates leave a client. Class weights are therefore CLIENT-LOCAL
(the federated label-count sum used by non-private FedAvg would be outside the accountant).
Outside the guarantee (stated, not hidden): the client lr was tuned on non-private runs, and the secondary
"val-selected checkpoint" reads private validation losses. The PRIMARY result is the final-round model.

Writes results/silo1_dp/<config>_seed<k>/{metrics.json, predictions_final.json, predictions_selected.json}.

Usage:  python -m experiments.silo1_dp --workers 5 --threads 2
        python -m experiments.silo1_dp --summarize
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from experiments.silo1_baselines import EXPERIMENT_DIR as BASELINE_DIR
from experiments.silo1_baselines import compute_metrics
from experiments.silo1_fl import COMMON as FL_COMMON
from experiments.silo1_fl import EXPERIMENT_DIR as FL_DIR
from experiments.silo1_fl import SWEEP_DIR as FL_SWEEP_DIR
from experiments.silo1_fl import build_clients
from src.fl.dp import DEFAULT_DELTA
from src.fl.models import class_weights
from src.fl.partition import LABELS, load_partition
from src.fl.runner import run_fl
from src.fl.utils import RESULTS_DIR, require_clean_tree, write_metrics

EXPERIMENT_DIR = RESULTS_DIR / "silo1_dp"
SEEDS = [0, 1, 2, 3, 4]
N_CLASSES = len(LABELS)
ROUNDS, LOCAL_EPOCHS = 100, 1
Z_GRID, C_GRID, Q_GRID = (0.5, 1.0, 2.0), (0.1, 1.0), (0.5, 1.0)


def config_name(z: float, C: float, q: float) -> str:
    return f"dp_z{z:g}_C{C:g}_q{q:g}"


CONFIGS = {config_name(z, C, q): {"noise_multiplier": z, "clip": C, "client_fraction": q}
           for q in Q_GRID for C in C_GRID for z in (0.0, *Z_GRID)}


def fl_lr() -> float:
    return json.loads((FL_SWEEP_DIR / "choice.json").read_text(encoding="utf-8"))["E1"]["lr"]


def run_one(run_dir: Path, part, cfg: dict, lr: float, seed: int, device: str, git: dict) -> dict:
    clients, _ = build_clients(part, device)
    local_weights = [class_weights(c.y_train.cpu().numpy(), N_CLASSES) for c in clients]
    t0 = time.perf_counter()
    res = run_fl("fedavg", clients, ROUNDS, LOCAL_EPOCHS, lr, FL_COMMON["batch_size"], cfg["client_fraction"], seed,
                 client_class_weights=local_weights, n_classes=N_CLASSES, head=FL_COMMON["head"], device=device,
                 dp={"clip": cfg["clip"], "noise_multiplier": cfg["noise_multiplier"], "delta": DEFAULT_DELTA})
    runtime = time.perf_counter() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions_final.json").write_text(json.dumps(res["final_test_predictions"]), encoding="utf-8")
    (run_dir / "predictions_selected.json").write_text(json.dumps(res["test_predictions"]), encoding="utf-8")
    sel = compute_metrics(res["test_predictions"])
    rounds = res["history"][1:]
    metrics = {
        **compute_metrics(res["final_test_predictions"]),        # PRIMARY: final-round model (DP post-processing)
        "selected_checkpoint": {"note": "selected on client val losses: NOT covered by the reported ε",
                                "best_round": res["best_round"], "best_fed_val_loss": res["best_fed_val_loss"],
                                "pooled_test_macro_f1": sel["pooled_test_macro_f1"],
                                "d4_client_mean": sel["d4"]["client_mean_macro_f1"]},
        "dp": {"privacy_unit": "client (disaster event)", "noise_multiplier": cfg["noise_multiplier"],
               "clip": cfg["clip"], "sample_rate": cfg["client_fraction"], "delta": DEFAULT_DELTA,
               "rounds": ROUNDS, "epsilon_final": res["epsilon_final"],
               "accountant": "opacus.accountants.RDPAccountant, one step per round",
               "expected_clients_per_round": cfg["client_fraction"] * len(clients),
               "mean_clients_sampled": float(np.mean([h["dp"]["n_clients"] for h in rounds])),
               "empty_rounds": int(sum(h["dp"]["n_clients"] == 0 for h in rounds)),
               "mean_frac_clipped": float(np.nanmean([np.nan if h["dp"]["frac_clipped"] is None
                                                      else h["dp"]["frac_clipped"] for h in rounds])),
               "mean_delta_norm": float(np.nanmean([np.nan if h["dp"]["mean_delta_norm"] is None
                                                    else h["dp"]["mean_delta_norm"] for h in rounds])),
               "noise_std_per_coordinate": rounds[0]["dp"]["noise_std_per_coordinate"],
               "total_model_bytes": res["total_bytes"]},
        "history": res["history"],
        "training_git": git,
    }
    write_metrics(run_dir / "metrics.json", metrics, seed=seed,
                  config={"method": "client-level DP-FedAvg (McMahan et al. 2018)", **cfg, "rounds": ROUNDS,
                          "local_epochs": LOCAL_EPOCHS, "lr": lr, "lr_source": "FedAvg E=1 non-private lr sweep",
                          "batch_size": FL_COMMON["batch_size"], "head": FL_COMMON["head"],
                          "client_optimizer": FL_COMMON["client_optimizer"],
                          "class_weights": "client-local (inside the DP boundary)",
                          "aggregation": "unweighted mean of clipped deltas, fixed denominator q*K, Gaussian noise",
                          "client_sampling": "Poisson, rate q", "device": device, "labels": list(LABELS),
                          "split_file": f"data/splits/silo1_seed{seed}.json"},
                  split_sizes=part.split_sizes(), runtime_s=runtime)
    return metrics


def _job(job: dict) -> dict:
    if job.get("threads"):
        torch.set_num_threads(job["threads"])
    m = run_one(Path(job["run_dir"]), load_partition(job["seed"]), job["cfg"], job["lr"], job["seed"],
                job["device"], job["git"])
    return {"label": job["label"], "pooled": m["pooled_test_macro_f1"], "eps": m["dp"]["epsilon_final"]}


# ── Summary ────────────────────────────────────────────────────────────────

def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ms(vals):
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) == 0:
        return None
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}


def summarize(seeds: list[int]) -> dict:
    ref = {name: [_load(d / f"{name}_seed{s}" / "metrics.json")["metrics"]["pooled_test_macro_f1"] for s in seeds]
           for name, d in (("fedavg_E1", FL_DIR), ("local_only", BASELINE_DIR), ("centralized", BASELINE_DIR))}
    ref = {k: np.array(v) for k, v in ref.items()}
    rows = {}
    for name, cfg in CONFIGS.items():
        ms = [_load(EXPERIMENT_DIR / f"{name}_seed{s}" / "metrics.json") for s in seeds]
        if not all(ms):
            continue
        ms = [m["metrics"] for m in ms]
        f1 = np.array([m["pooled_test_macro_f1"] for m in ms])
        eps = {m["dp"]["epsilon_final"] for m in ms}
        assert len(eps) == 1, f"{name}: ε differs across seeds {eps}"   # ε depends only on (z, q, rounds)
        eps = eps.pop()
        rows[name] = {
            **cfg, "epsilon": eps,
            "pooled_macro_f1": _ms(f1), "d4_client_mean": _ms([m["d4"]["client_mean_macro_f1"] for m in ms]),
            "delta_vs_fedavg_E1": {**_ms(f1 - ref["fedavg_E1"]), "p": float(stats.ttest_rel(f1, ref["fedavg_E1"]).pvalue)},
            "delta_vs_local_only": {**_ms(f1 - ref["local_only"]), "p": float(stats.ttest_rel(f1, ref["local_only"]).pvalue)},
            "selected_ckpt_pooled_macro_f1": _ms([m["selected_checkpoint"]["pooled_test_macro_f1"] for m in ms]),
            "mean_frac_clipped": float(np.mean([m["dp"]["mean_frac_clipped"] for m in ms])),
            "mean_delta_norm": float(np.mean([m["dp"]["mean_delta_norm"] for m in ms])),
            "mean_clients_sampled": float(np.mean([m["dp"]["mean_clients_sampled"] for m in ms])),
            "empty_rounds": int(sum(m["dp"]["empty_rounds"] for m in ms)),
        }
    summary = {"seeds": seeds, "delta": DEFAULT_DELTA, "rounds": ROUNDS, "privacy_unit": "client (disaster event)",
               "references": {k: _ms(v) for k, v in ref.items()}, "rows": rows,
               "note": "PRIMARY metric = final-round model. epsilon=None means no noise (epsilon = infinity)."}
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_summary(summary)
    return summary


def print_summary(sm: dict) -> None:
    f = lambda c, d=4: "n/a" if c is None else f"{c['mean']:.{d}f} ± {c['std']:.{d}f}"
    print(f"\nClient-level DP-FedAvg, FedAvg E=1 base, {sm['rounds']} rounds, δ = {sm['delta']:g}, seeds {sm['seeds']}")
    print("Primary = final-round model. References: " + ", ".join(f"{k} {f(v)}" for k, v in sm["references"].items()))
    print(f"{'q':>4} {'C':>4} {'z':>4} | {'ε (δ=1e-5)':>11} | {'pooled macro-F1':17s} | {'D4 client-mean':17s} | "
          f"{'Δ vs FedAvg (p)':22s} | {'Δ vs local (p)':22s} | {'val-sel. F1*':17s} | clip%")
    print("-" * 150)
    for name, r in sm["rows"].items():
        eps = "∞ (no noise)" if r["epsilon"] is None else f"{r['epsilon']:.1f}"
        dv, dl = r["delta_vs_fedavg_E1"], r["delta_vs_local_only"]
        print(f"{r['client_fraction']:>4g} {r['clip']:>4g} {r['noise_multiplier']:>4g} | {eps:>11} | "
              f"{f(r['pooled_macro_f1']):17s} | {f(r['d4_client_mean']):17s} | "
              f"{dv['mean']:+.4f} (p={dv['p']:.3f})     | {dl['mean']:+.4f} (p={dl['p']:.3f})     | "
              f"{f(r['selected_ckpt_pooled_macro_f1']):17s} | {100 * r['mean_frac_clipped']:.0f}%")
    print("* val-selected checkpoint reads private validation losses: NOT covered by ε (secondary only).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Silo 1 client-level DP-FedAvg grid.")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.seeds)
        return
    git = require_clean_tree(args.allow_dirty)
    lr = fl_lr()
    jobs = [{"label": f"{n}_seed{s}", "run_dir": str(EXPERIMENT_DIR / f"{n}_seed{s}"), "cfg": CONFIGS[n], "lr": lr,
             "seed": s, "device": args.device, "threads": args.threads, "git": git}
            for n in args.configs for s in args.seeds]
    jobs.sort(key=lambda j: -j["cfg"]["client_fraction"])     # q = 1 runs are the slowest; start them first
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for fut in as_completed([pool.submit(_job, j) for j in jobs]):
            r = fut.result()
            eps = "inf" if r["eps"] is None else f"{r['eps']:.1f}"
            print(f"  done {r['label']:28s} pooled macro-F1 {r['pooled']:.4f}  ε={eps}", flush=True)
    summarize(args.seeds)


if __name__ == "__main__":
    main()
