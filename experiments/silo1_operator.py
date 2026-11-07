"""
Silo 1 operator metrics: what a response coordinator sees for the classes that trigger an action.

Macro-F1 averages all 9 classes; an operator cares whether reports of injured people, trapped people, damage,
evacuations and aid requests are FOUND (recall) and whether flagged tweets are real (precision). This script reads the
saved test predictions of the committed Silo 1 runs (no training) and reports, per method, mean ± std over seeds 0-4:

  - pooled per-class precision / recall / F1 / support (all 9 classes), via src/fl/utils.per_class_report;
  - an "actionable" view: the ACTIONABLE classes only, plus their macro-precision / macro-recall and the
    micro-recall of "any actionable tweet predicted as SOME actionable class" (triage recall);
  - per-event recall of each actionable class (seed mean), only where the event has >= MIN_SUPPORT true examples.

precision@k needs per-tweet class probabilities; the committed runs saved predicted labels only
(predictions.json = y_true / y_pred), so precision@k is reported as missing, not estimated.

Writes results/silo1_operator/summary.json.   Usage:  python -m experiments.silo1_operator
"""

from __future__ import annotations

import json

import numpy as np

from experiments.silo1_embed_cache import require_single_cache
from src.fl.partition import LABEL_SHORT, LABELS
from src.fl.utils import RESULTS_DIR, git_info, per_class_report

SEEDS = [0, 1, 2, 3, 4]
OUT_DIR = RESULTS_DIR / "silo1_operator"
MIN_SUPPORT = 3
LABEL_IDS = list(range(len(LABELS)))

# Classes whose tweets call for a responder action (rescue, medical, shelter, repair, aid logistics). Excluded:
# caution_and_advice (warnings broadcast TO the public), sympathy, other_useful_information, not_related.
ACTIONABLE = ("injured_dead", "missing", "displaced", "infra_damage", "donation")

METHODS = {   # label -> results dir / run-name prefix (all committed, test predictions saved)
    "local_only": "silo1_baselines/local_only",
    "centralized": "silo1_baselines/centralized",
    "centralized_adamw": "silo1_baselines/centralized_adamw",
    "fedavg_E1": "silo1_fl/fedavg_E1",
    "fedavg_E5_R200": "silo1_fl/fedavg_E5_R200",
    "fedprox_mu0.01_E1": "silo1_fl/fedprox_mu0.01_E1",
    "fedavg_E1+ft1": "silo1_fl/ft1_fedavg_E1",
    "fedavg_E1+ft2": "silo1_fl/ft2_fedavg_E1",
}


def _ms(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def seed_report(predictions: dict) -> dict:
    y_true = np.concatenate([p["y_true"] for p in predictions.values()])
    y_pred = np.concatenate([p["y_pred"] for p in predictions.values()])
    per_class = per_class_report(y_true, y_pred, LABEL_IDS, LABEL_SHORT)
    act = [LABEL_SHORT.index(c) for c in ACTIONABLE]
    is_act_true, is_act_pred = np.isin(y_true, act), np.isin(y_pred, act)
    per_event = {}
    for event, p in predictions.items():
        yt, yp = np.asarray(p["y_true"]), np.asarray(p["y_pred"])
        per_event[event] = {c: float((yp[yt == k] == k).mean()) for c, k in zip(ACTIONABLE, act)
                            if (yt == k).sum() >= MIN_SUPPORT}
    return {
        "per_class": per_class,
        "actionable_macro_precision": float(np.mean([per_class[c]["precision"] for c in ACTIONABLE])),
        "actionable_macro_recall": float(np.mean([per_class[c]["recall"] for c in ACTIONABLE])),
        "triage_recall": float(is_act_pred[is_act_true].mean()),          # actionable -> some actionable class
        "triage_precision": float(is_act_true[is_act_pred].mean()),
        "per_event_recall": per_event,
    }


def summarize_method(prefix: str) -> dict:
    reps = []
    for s in SEEDS:
        run = RESULTS_DIR / f"{prefix}_seed{s}"
        rep = seed_report(json.loads((run / "predictions.json").read_text(encoding="utf-8")))
        committed = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["metrics"]["pooled_per_class"]
        for c in LABEL_SHORT:            # recomputed from predictions == the run's own committed report
            if any(abs(rep["per_class"][c][k] - committed[c][k]) > 1e-12 for k in ("precision", "recall", "f1")):
                raise AssertionError(f"{run.name} {c}: per-class report disagrees with metrics.json")
        reps.append(rep)
    per_class = {c: {k: _ms([r["per_class"][c][k] for r in reps]) for k in ("precision", "recall", "f1")}
                 | {"support": reps[0]["per_class"][c]["support"]} for c in LABEL_SHORT}
    events = sorted({e for r in reps for e in r["per_event_recall"]})
    per_event = {e: {c: _ms([r["per_event_recall"][e][c] for r in reps if c in r["per_event_recall"].get(e, {})])
                     for c in ACTIONABLE if any(c in r["per_event_recall"].get(e, {}) for r in reps)}
                 for e in events}
    return {"per_class": per_class,
            **{k: _ms([r[k] for r in reps]) for k in ("actionable_macro_precision", "actionable_macro_recall",
                                                       "triage_recall", "triage_precision")},
            "per_event_recall": per_event, "source": f"results/{prefix}_seed<k>/predictions.json"}


def main() -> None:
    cache_sha = require_single_cache(
        {f"{p}_seed{s}": json.loads((RESULTS_DIR / f"{p}_seed{s}" / "metrics.json").read_text(encoding="utf-8"))
         for p in METHODS.values() for s in SEEDS}, "silo1_operator summary")
    summary = {
        "seeds": SEEDS, "embedding_cache_sha256": cache_sha, "actionable_classes": list(ACTIONABLE),
        "min_support_per_event": MIN_SUPPORT,
        "precision_at_k": {"status": "missing", "reason": "committed runs saved predicted labels only "
                                                          "(y_true / y_pred), no class probabilities"},
        "definitions": {"triage_recall": "share of actionable test tweets predicted as any actionable class",
                        "triage_precision": "share of tweets predicted actionable that are truly actionable",
                        "support": "pooled test examples of the class (identical across methods for a seed; "
                                   "seed 0 shown)"},
        "methods": {m: summarize_method(p) for m, p in METHODS.items()},
        "git": git_info(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{'method':20s} " + " ".join(f"{c:>13s}" for c in ACTIONABLE) + "   triage R / P")
    for m, d in summary["methods"].items():
        cells = " ".join(f"{d['per_class'][c]['precision']['mean']:.2f}/{d['per_class'][c]['recall']['mean']:.2f}   "
                         for c in ACTIONABLE)
        print(f"{m:20s} {cells}  {d['triage_recall']['mean']:.3f} / {d['triage_precision']['mean']:.3f}")
    print("cells = precision/recall (mean over seeds); precision@k: missing (no saved probabilities)")


if __name__ == "__main__":
    main()
