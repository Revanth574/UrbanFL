# Legacy prototype results — INVALID

`training_metrics.json` was written by the minor-project script `experiments/train_and_evaluate.py` (now deprecated).
None of its numbers are valid (docs/AUDIT.md §4A–4B):

- `silo3_bpr_r2 = 1.0` came from a hardcoded formula, `0.85 + 0.03 · min(n_road_types, 5)`.
- `fl_privacy_budget` is a hand-made sum with no privacy accountant.
- `silo1_*` / `silo2_*` curves come from silent fallback toy models (the real models raised on construction), trained
  on an unseeded split with no test set.

Kept only as evidence for the audit. Every valid result lives in `results/silo1_*/` with commit hash and
`dirty: false` provenance in each `metrics.json`. `experiments/make_figures.py` never reads this folder.
