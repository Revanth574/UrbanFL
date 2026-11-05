"""
Official xView2 score (decision X1), reproducing xview2_metrics.py of github.com/DIUx-xView/xView2_scoring
(RowPairCalculator.get_row_pair + XviewMetrics), TP / FP / FN summed over ALL images before any F1:

  score = 0.3 * localisation F1 + 0.7 * harmonic mean of the four per-class damage F1s

Per image, with lp / dp = localisation / damage prediction and lt / dt = localisation / damage target:
  lp_b, lt_b, dt_b = (x > 0) for x in (lp, lt, dt)          # building masks
  dp = dp * lp_b                                           # no damage credit where NO BUILDING IS PREDICTED
  dp, dt = dp[dt_b == 1], dt[dt_b == 1]                    # damage scored only on target-damage building pixels
  localisation: binary TP / FN / FP of lp_b vs lt_b over all pixels;  damage: one-vs-rest c = 1..4 on (dp, dt).
Harmonic mean: 4 / sum((F1_c + 1e-6)^-1).

Our files: lt = the PRE target (> 0, so un-classified buildings count as buildings), dt = the POST target as stored
(1..4 = no / minor / major / destroyed; it paints un-classified buildings 0, so they fall outside dt_b == 1).
Predictions: lp = stage-1 mask (0/1), dp = stage-2 class index 0..3 + 1. Consequences for the damage F1:
  - only ground-truth damage-building pixels are scored (predicting damage on background is never penalised);
  - a damage-building pixel that stage 1 MISSES has dp = 0: a false negative of its true class, FP of no class.
The training-time IGNORE_INDEX mask (xbd_data.damage_target) is a loss choice only; scoring uses the raw post target.
"""

from __future__ import annotations

import numpy as np

from src.fl.xbd_data import DAMAGE_CLASSES

EPS = 1e-6


def _f1(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


def official_damage_inputs(loc_pred, dmg_class, post_target) -> tuple[np.ndarray, np.ndarray]:
    """(dp, dt) on the scored pixels: dp in 0..4 (0 = no building predicted), dt in 1..4."""
    lp_b = np.asarray(loc_pred) > 0
    dp = (np.asarray(dmg_class).astype(np.int64) + 1) * lp_b
    dt = np.asarray(post_target).astype(np.int64)
    scored = dt > 0
    return dp[scored], dt[scored]


class XView2Scorer:
    def __init__(self):
        self.loc = np.zeros(3)                         # tp, fp, fn
        self.dmg = np.zeros((len(DAMAGE_CLASSES), 3))  # per class tp, fp, fn

    def update_loc(self, pred, gt) -> None:
        pred, gt = np.asarray(pred) > 0, np.asarray(gt) > 0
        self.loc += [np.sum(pred & gt), np.sum(pred & ~gt), np.sum(~pred & gt)]

    def update(self, loc_pred, pre_target, dmg_class, post_target) -> None:
        """One image, official definition. loc_pred 0/1 (H,W); pre_target / post_target as stored; dmg_class 0..3."""
        self.update_loc(loc_pred, pre_target)
        dp, dt = official_damage_inputs(loc_pred, dmg_class, post_target)
        for c in range(1, len(DAMAGE_CLASSES) + 1):
            self.dmg[c - 1] += [np.sum((dp == c) & (dt == c)), np.sum((dp == c) & (dt != c)),
                                np.sum((dp != c) & (dt == c))]

    def merge(self, other: "XView2Scorer") -> "XView2Scorer":
        self.loc += other.loc
        self.dmg += other.dmg
        return self

    def counts(self) -> dict:
        return {"loc_tp_fp_fn": self.loc.astype(int).tolist(),
                "damage_tp_fp_fn": {c: row.astype(int).tolist() for c, row in zip(DAMAGE_CLASSES, self.dmg)}}

    @classmethod
    def from_counts(cls, counts: dict) -> "XView2Scorer":
        s = cls()
        s.loc = np.asarray(counts["loc_tp_fp_fn"], dtype=float)
        s.dmg = np.asarray([counts["damage_tp_fp_fn"][c] for c in DAMAGE_CLASSES], dtype=float)
        return s

    def compute(self) -> dict:
        loc_f1 = _f1(*self.loc)
        per_class = [_f1(*row) for row in self.dmg]
        harmonic = len(per_class) / sum(1.0 / (f + EPS) for f in per_class)
        return {"loc_f1": loc_f1, "damage_f1": dict(zip(DAMAGE_CLASSES, per_class)),
                "damage_f1_harmonic": harmonic, "score": 0.3 * loc_f1 + 0.7 * harmonic}
