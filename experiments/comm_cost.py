"""
Communication cost of every federated method in both silos, next to the cost of centralising the raw data instead.

FL bytes (no compression, float32 on the wire, as the runs implement them):
  per round = (state tensors sent) x (clients that round) x 2   (global model down + client update up)
  Silo 1: read from each committed run (metrics.fl / metrics.dp: n_params, bytes_per_round, total_model_bytes);
          bytes to the SELECTED round = bytes_per_round x best_round (DP: the final round is the primary model).
  Silo 2: the full U-Net state_dict travels (weights + BatchNorm running stats; num_batches_tracked as int64),
          sizes measured from the model itself; 9 clients every round; rounds from the committed runs.
  FedAvg + local fine-tuning adds nothing: the clients fine-tune the model they already received.
Centralised alternative: every client uploads its raw train + val data once (centralized training needs both).
  Silo 1: UTF-8 bytes of the tweet texts.  Silo 2: on-disk bytes of the xBD files of those pairs
  (pre + post PNG images and the label JSON); stage-1 training reads only pre image + target, stated as such.

Writes results/comm_cost/summary.json and paper/tables/comm_cost.md (generated from the JSON).
Usage:  python -m experiments.comm_cost
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.fl.utils import REPO_ROOT, RESULTS_DIR, git_info

OUT_DIR = RESULTS_DIR / "comm_cost"
TABLE = REPO_ROOT / "paper" / "tables" / "comm_cost.md"
SILO1_SEEDS, SILO2_SEEDS = [0, 1, 2, 3, 4], [0, 1, 2]
SILO1_FL = ("fedavg_E1", "fedavg_E5_R200", "fedprox_mu0.001_E1", "fedprox_mu0.01_E1", "fedprox_mu0.1_E1")
MB = 1e6


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ms(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def silo1() -> dict:
    from src.fl.partition import load_partition
    rows = {}
    for name in SILO1_FL:
        ms = [_load(RESULTS_DIR / "silo1_fl" / f"{name}_seed{s}" / "metrics.json")["metrics"] for s in SILO1_SEEDS]
        fl = [m["fl"] for m in ms]
        rows[name] = {"n_params": fl[0]["n_params"], "clients_per_round": fl[0]["clients_per_round"],
                      "bytes_per_round": fl[0]["bytes_per_round"], "rounds": len(ms[0]["history"]) - 1,
                      "total_bytes": _ms([f["total_model_bytes"] for f in fl]),
                      "bytes_to_selected_round": _ms([f["bytes_per_round"] * f["best_round"] for f in fl]),
                      "one_off_bytes": fl[0]["one_off_histogram_bytes"],
                      "one_off_note": "federated label-count sum for global class weights"}
    dp_summary = _load(RESULTS_DIR / "silo1_dp" / "summary.json") or {"rows": {}}
    for name in dp_summary["rows"]:
        ms = [_load(RESULTS_DIR / "silo1_dp" / f"{name}_seed{s}" / "metrics.json") for s in SILO1_SEEDS]
        if not all(ms):
            continue
        dp = [m["metrics"]["dp"] for m in ms]
        rows[name] = {"rounds": dp[0]["rounds"], "expected_clients_per_round": dp[0]["expected_clients_per_round"],
                      "total_bytes": _ms([d["total_model_bytes"] for d in dp]),
                      "bytes_to_selected_round": _ms([d["total_model_bytes"] for d in dp]),
                      "selected_note": "primary DP model = final round"}
    raw = {}
    part = load_partition(0)
    for c in part.split_sizes():
        raw[c] = int(sum(len(t.encode("utf-8")) for split in ("train", "val") for t in part.rows(c, split)["text"]))
    return {"fl": rows, "centralize_raw_bytes": {"total": sum(raw.values()), "by_client": raw,
                                                 "what": "UTF-8 tweet text, train + val, split seed 0"}}


def silo2() -> dict:
    import torch
    from experiments.silo2_loc import COMMON
    from src.fl.config import xbd_split_dir
    from src.fl.models import UNetResNet
    from src.fl.partition_xbd import load_xbd_partition
    state = UNetResNet(COMMON["encoder"], pretrained=False).state_dict()
    sent = sum(v.numel() * v.element_size() for v in state.values())
    n_float = sum(v.numel() for v in state.values() if torch.is_floating_point(v))
    rows = {}
    for name in ("fedavg", "fedavg_ft"):
        ms = [_load(RESULTS_DIR / "silo2_loc" / f"{name}_seed{s}" / "metrics.json") for s in SILO2_SEEDS]
        if not all(ms):
            continue
        src = [m["metrics"] if name == "fedavg" else _load(Path(m["metrics"]["source"]["fedavg_metrics"]))["metrics"]
               for m in ms]
        rounds = len(src[0]["history"])
        clients = len(ms[0]["metrics"]["test_per_event"])
        per_round = sent * clients * 2
        rows[name] = {"state_bytes": sent, "float_values": n_float, "clients_per_round": clients,
                      "bytes_per_round": per_round, "rounds": rounds, "total_bytes": _ms([per_round * rounds] * len(ms)),
                      "bytes_to_selected_round": _ms([per_round * m["selection"]["best_round"] for m in src])}
        if name == "fedavg_ft":
            rows[name]["note"] = "the FedAvg rounds only; local fine-tuning sends nothing"
    d = xbd_split_dir("train")
    part = load_xbd_partition(0)
    raw = {}
    for c, splits in part["clients"].items():
        ids = [p for s in ("train", "val") for p in splits[s]]
        raw[c] = int(sum((d / "images" / f"{p}_{ph}_disaster.png").stat().st_size for p in ids for ph in ("pre", "post"))
                     + sum((d / "labels" / f"{p}_post_disaster.json").stat().st_size
                           + (d / "labels" / f"{p}_pre_disaster.json").stat().st_size for p in ids))
    return {"fl": rows, "centralize_raw_bytes": {"total": sum(raw.values()), "by_client": raw,
                                                 "what": "xBD pre + post PNG + label JSONs, train + val, split seed 0; "
                                                         "stage 1 alone reads pre image + target"}}


def table(summary: dict) -> str:
    lines = ["# Communication cost (generated by experiments/comm_cost.py from results/comm_cost/summary.json)", "",
             "| Silo | Method | Rounds | Clients/round | MB per round | Total MB | MB to selected round |",
             "|---|---|---|---|---|---|---|"]
    for silo in ("silo1", "silo2"):
        s = summary[silo]
        for name, r in s["fl"].items():
            cpr = r.get("clients_per_round", r.get("expected_clients_per_round"))
            per = f"{r['bytes_per_round'] / MB:.1f}" if "bytes_per_round" in r else "Poisson"
            lines.append(f"| {silo[-1]} | {name} | {r['rounds']} | {cpr:g} | {per} | "
                         f"{r['total_bytes']['mean'] / MB:,.0f} | {r['bytes_to_selected_round']['mean'] / MB:,.0f} |")
        lines.append(f"| {silo[-1]} | centralise raw data (once) | — | all | — | "
                     f"{s['centralize_raw_bytes']['total'] / MB:,.1f} | — |")
    lines += ["", "MB = 10^6 bytes, float32, no compression; mean over seeds. Down + up both counted. "
                  "Raw-data rows: train + val of every client, split seed 0."]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = {"silo1": silo1(), "silo2": silo2(), "git": git_info(),
               "definitions": {"bytes_per_round": "state bytes x clients x 2 (down + up)",
                               "bytes_to_selected_round": "rounds up to the checkpoint that is reported"}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(table(summary), encoding="utf-8")
    print(table(summary))


if __name__ == "__main__":
    main()
