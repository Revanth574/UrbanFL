"""Seeds, metrics, and run logging shared by every UrbanFL experiment (CLAUDE.md integrity rules 1, 4, 7)."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU + all CUDA devices) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Metrics ────────────────────────────────────────────────────────────────

def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    """Rows = true class, columns = predicted class, in the order of `labels`."""
    pos = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in pos and p in pos:
            cm[pos[t], pos[p]] += 1
    return cm


def per_class_f1(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    """F1 per class in `labels` order; 0.0 where precision+recall is undefined (sklearn zero_division=0)."""
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> float:
    """Unweighted mean F1 over an EXPLICIT label list (classes absent from y_true still count, as 0 or not)."""
    return float(per_class_f1(y_true, y_pred, labels).mean())


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def per_class_report(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int],
                     names: Sequence[str] | None = None) -> dict:
    """{class_name: {precision, recall, f1, support}} computed from the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    pred_pos, support = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = per_class_f1(y_true, y_pred, labels)
    names = names or [str(lab) for lab in labels]
    return {n: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for n, p, r, f, s in zip(names, precision, recall, f1, support)}


def d4_metrics(y_true_by_client: dict[str, Sequence[int]],
               y_pred_by_client: dict[str, Sequence[int]],
               labels: Sequence[int],
               min_support: int = 3) -> dict:
    """
    Decision D4 (docs/AUDIT.md section 5): the Silo 1 headline metrics.

    Primary   — pooled test macro-F1: concatenate every client's test labels and predictions (the union of all
                client test sets) and compute macro-F1 over the EXPLICIT `labels` list (all 9 classes).
    Secondary — per-client macro-F1 restricted to the classes that have >= `min_support` examples in THAT
                client's y_true, reported together with how many classes qualified.

    Args:
        y_true_by_client: client name -> true class ids of that client's test set.
        y_pred_by_client: client name -> predicted class ids, same keys and lengths as y_true_by_client.
        labels: the full ordered label list (class ids), e.g. list(range(9)).
        min_support: minimum test examples for a class to count in a client's secondary macro-F1.

    Returns:
        {
          "pooled_macro_f1": float,
          "per_client": {
              client: {"macro_f1": float | None,      # None when no class qualifies
                       "n_qualifying_classes": int,
                       "qualifying_classes": list[int]},
              ...
          },
          "client_mean_macro_f1": float | None,       # mean over clients whose macro_f1 is not None
        }

    Use `macro_f1` / `per_class_f1` above as building blocks.
    """
    import numpy as np
    from sklearn.metrics import f1_score

    labels = [int(c) for c in labels]
    all_true, all_pred = [], []
    per_client = {}

    for client, y_true in y_true_by_client.items():
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred_by_client[client], dtype=int)
        if len(yt) != len(yp):
            raise ValueError(f"client {client}: {len(yt)} labels vs {len(yp)} predictions")
        all_true.append(yt)
        all_pred.append(yp)

        # Qualification depends ONLY on true test labels, never on predictions.
        qualifying = [c for c in labels if int((yt == c).sum()) >= min_support]
        if qualifying:
            client_f1 = float(f1_score(yt, yp, labels=qualifying,
                                       average="macro", zero_division=0))
        else:
            client_f1 = None  # not computable — do NOT turn this into 0.0

        per_client[client] = {
            "macro_f1": client_f1,
            "n_qualifying_classes": len(qualifying),
            "qualifying_classes": qualifying,
        }

    # Primary: pooled macro-F1 over the EXPLICIT full label list.
    pooled = float(f1_score(np.concatenate(all_true), np.concatenate(all_pred),
                            labels=labels, average="macro", zero_division=0))

    scored = [v["macro_f1"] for v in per_client.values() if v["macro_f1"] is not None]
    client_mean = float(np.mean(scored)) if scored else None

    return {
        "pooled_macro_f1": pooled,
        "per_client": per_client,
        "client_mean_macro_f1": client_mean,
    }


# ── Run logging ────────────────────────────────────────────────────────────

def git_info() -> dict:
    """HEAD commit and whether CODE differs from it. results/ is excluded: experiments write there by design."""
    def _git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": _git("rev-parse", "HEAD") or None,
                "dirty": bool(_git("status", "--porcelain", "--", ".", ":(exclude)results"))}
    except OSError:
        return {"commit": None, "dirty": None}


def require_clean_tree(allow_dirty: bool = False) -> dict:
    """Return git_info(); refuse to run (rule 4) if code differs from HEAD, unless explicitly allowed."""
    info = git_info()
    if info["dirty"] and not allow_dirty:
        raise RuntimeError("Working tree has uncommitted code changes; commit first or pass --allow-dirty "
                           "(results from a dirty tree cannot be traced to a commit).")
    return info


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_metrics(path: Path | str, metrics: dict, *, seed: int, config: dict,
                  split_sizes: dict, runtime_s: float) -> Path:
    """
    Write `metrics` plus the provenance CLAUDE.md rule 4 requires (git commit + dirty flag, seed, full config,
    split sizes, wall-clock runtime) to `path` (a metrics.json inside results/<experiment>/<run_id>/).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "seed": seed,
        "config": config,
        "split_sizes": split_sizes,
        "runtime_s": round(runtime_s, 3),
        "git": git_info(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
    }
    path.write_text(json.dumps(_jsonable(payload), indent=1), encoding="utf-8")
    return path
