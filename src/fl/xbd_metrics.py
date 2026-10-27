"""
Official xView2 score, accumulated over images (decision X1):

  score = 0.3 * localisation F1 + 0.7 * harmonic mean of the four per-class damage F1s

- Localisation F1: binary building-vs-background F1 over all pixels (prediction vs PRE target).
- Damage F1s: one-vs-rest F1 for each of no / minor / major / destroyed, computed only on scored building pixels
  (ground-truth damage != IGNORE_INDEX, i.e. POST-target buildings, un-classified excluded).
- Harmonic mean: 4 / sum(1 / (F1_c + 1e-6)), as in the xView2 scoring code (a class with F1 = 0 drives it to ~0).
"""

from __future__ import annotations

import numpy as np

from src.fl.xbd_data import DAMAGE_CLASSES, IGNORE_INDEX

EPS = 1e-6


def _f1(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


class XView2Scorer:
    def __init__(self):
        self.loc = np.zeros(3)                         # tp, fp, fn
        self.dmg = np.zeros((len(DAMAGE_CLASSES), 3))  # per class tp, fp, fn

    def update_loc(self, pred, gt) -> None:
        pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
        self.loc += [np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)]

    def update_damage(self, pred, gt) -> None:
        pred, gt = np.asarray(pred), np.asarray(gt)
        valid = gt != IGNORE_INDEX
        p, g = pred[valid], gt[valid]
        for c in range(len(DAMAGE_CLASSES)):
            self.dmg[c] += [np.sum((p == c) & (g == c)), np.sum((p == c) & (g != c)), np.sum((p != c) & (g == c))]

    def compute(self) -> dict:
        loc_f1 = _f1(*self.loc)
        per_class = [_f1(*row) for row in self.dmg]
        harmonic = len(per_class) / sum(1.0 / (f + EPS) for f in per_class)
        return {"loc_f1": loc_f1, "damage_f1": dict(zip(DAMAGE_CLASSES, per_class)),
                "damage_f1_harmonic": harmonic, "score": 0.3 * loc_f1 + 0.7 * harmonic}
