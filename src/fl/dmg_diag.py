"""
Stage-2 damage diagnostics (pre-registered D1, docs/STAGE2_PROTOCOL.md) on the official scored pixels.

The official damage score (src/fl/xbd_metrics.py) only looks at post-target building pixels with a damage label
(dt in 1..4), with the prediction dp = (argmax + 1) * stage-1 mask (0 = not localised). collect_scored() keeps exactly
those pixels: their true class, the stage-1 mask and the 4 damage logits (float32, as the scorer's argmax sees them),
plus the localisation TP / FP / FN of every pixel. Everything else here is numpy on those arrays:

  damage_counts / harmonic   per-class TP / FP / FN and the harmonic mean, identical to XView2Scorer
  confusion                  rows true class 1..4, columns predicted 0 (not localised) .. 4
  mean_softmax               mean softmax vector over a set of pixels
  fit_offsets                per-class additive logit offsets (no-damage fixed at 0) maximising the damage
                             harmonic mean by coordinate ascent over OFFSET_GRID; fitted on validation only
"""

from __future__ import annotations

import numpy as np
import torch

from src.fl.dmg_eval import dmg_batch_to_io
from src.fl.xbd_data import DAMAGE_CLASSES
from src.fl.xbd_metrics import EPS, _f1

N_CLASSES = len(DAMAGE_CLASSES)
OFFSET_GRID = np.round(np.arange(-4.0, 8.0 + 1e-9, 0.25), 2)   # includes 0, so fitting can never lower val HM
MAX_CYCLES = 4
CHUNK = 1 << 21


@torch.no_grad()
def collect_scored(loc_model, dmg_model, loader, prefix: str, clients: list[str], device, amp: bool = True) -> dict:
    """Scored pixels of every image of `clients` under `prefix` (same forward pass as dmg_eval.xview2_evaluate)."""
    device = torch.device(device)
    amp = amp and device.type == "cuda"
    loc_model.eval()
    dmg_model.eval()
    dts, lps, logits, loc, n_img = [], [], [], np.zeros(3, dtype=np.int64), 0
    for _, batch in loader.run([(prefix + c, 0, 0) for c in clients]):
        x, _ = dmg_batch_to_io(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            lp = (torch.sigmoid(loc_model(x[:, :3]).float()).squeeze(1) > 0.5).cpu().numpy()
            z = dmg_model(x).float().cpu().numpy()
        for i in range(x.shape[0]):
            pre_t, post_t = batch["pre_target"][i].numpy() > 0, batch["post_target"][i].numpy()
            loc += [np.sum(lp[i] & pre_t), np.sum(lp[i] & ~pre_t), np.sum(~lp[i] & pre_t)]
            scored = post_t > 0
            dts.append(post_t[scored].astype(np.int8))
            lps.append(lp[i][scored])
            logits.append(z[i][:, scored].T.copy())
            n_img += 1
    cat = lambda a, shape, dt: np.concatenate(a) if a else np.zeros(shape, dt)   # noqa: E731
    return {"dt": cat(dts, (0,), np.int8), "lp": cat(lps, (0,), bool),
            "logits": cat(logits, (0, N_CLASSES), np.float32), "loc_tp_fp_fn": loc.tolist(), "n_images": n_img}


def predict(logits: np.ndarray, lp: np.ndarray, offsets=None) -> np.ndarray:
    """dp on the scored pixels: (argmax(logits + offsets) + 1) * mask, 0 = not localised."""
    out = np.zeros(len(lp), dtype=np.int8)
    b = np.zeros(N_CLASSES, np.float32) if offsets is None else np.asarray(offsets, np.float32)
    for s in range(0, len(lp), CHUNK):
        z = logits[s:s + CHUNK] + b
        out[s:s + CHUNK] = (z.argmax(1) + 1) * lp[s:s + CHUNK]
    return out


def damage_counts(dt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(4, 3) TP / FP / FN per class c = 1..4, one-vs-rest on the scored pixels (official definition)."""
    joint = np.bincount(dt.astype(np.int64) * (N_CLASSES + 1) + pred.astype(np.int64),
                        minlength=(N_CLASSES + 1) ** 2).reshape(N_CLASSES + 1, N_CLASSES + 1)
    rows = []
    for c in range(1, N_CLASSES + 1):
        tp = joint[c, c]
        rows.append([tp, joint[:, c].sum() - tp, joint[c, :].sum() - tp])
    return np.asarray(rows, dtype=np.int64)


def harmonic(counts: np.ndarray) -> float:
    return N_CLASSES / sum(1.0 / (_f1(*row) + EPS) for row in counts)


def per_class_f1(counts: np.ndarray) -> dict:
    return {c: _f1(*row) for c, row in zip(DAMAGE_CLASSES, counts)}


def confusion(dt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """(4, 5): rows true class 1..4, columns predicted 0 (not localised), 1..4; pixel counts."""
    joint = np.bincount(dt.astype(np.int64) * (N_CLASSES + 1) + pred.astype(np.int64),
                        minlength=(N_CLASSES + 1) ** 2).reshape(N_CLASSES + 1, N_CLASSES + 1)
    return joint[1:]


def mean_softmax(logits: np.ndarray, mask: np.ndarray) -> list[float] | None:
    """Mean softmax probability vector over the pixels where `mask` is true (None if there are none)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    total = np.zeros(N_CLASSES)
    for s in range(0, len(idx), CHUNK):
        z = logits[idx[s:s + CHUNK]].astype(np.float64)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        total += (p / p.sum(1, keepdims=True)).sum(0)
    return (total / len(idx)).tolist()


def fit_offsets(logits: np.ndarray, lp: np.ndarray, dt: np.ndarray, grid=OFFSET_GRID,
                max_cycles: int = MAX_CYCLES) -> tuple[np.ndarray, dict]:
    """Coordinate ascent: for minor, major, destroyed in turn, the grid value maximising the damage harmonic mean
    with the other offsets fixed (no-damage stays 0; a candidate replaces the current value only if strictly better).
    Stops after a cycle without change or after max_cycles. Returns (offsets, trace)."""
    b = np.zeros(N_CLASSES, np.float32)
    best = harmonic(damage_counts(dt, predict(logits, lp, b)))
    start, cycles = best, 0
    for cycles in range(1, max_cycles + 1):
        changed = False
        for k in range(1, N_CLASSES):
            for v in grid:
                if v == b[k]:
                    continue
                trial = b.copy()
                trial[k] = v
                h = harmonic(damage_counts(dt, predict(logits, lp, trial)))
                if h > best + 1e-12:
                    best, b, changed = h, trial, True
        if not changed:
            break
    at_edge = [DAMAGE_CLASSES[k] for k in range(1, N_CLASSES) if b[k] in (grid[0], grid[-1])]
    return b, {"val_harmonic_before": start, "val_harmonic_after": best, "cycles": cycles,
               "grid": [float(grid[0]), float(grid[-1]), float(grid[1] - grid[0])], "offset_at_grid_edge": at_edge}
