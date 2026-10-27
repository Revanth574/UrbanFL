# Legacy paper draft — results INVALID

`urbanfl_paper.tex` is the minor-project draft. Its numerical results come from the prototype and are not valid:

- Table `tab:fl` (divergence, heterogeneity index, sync events, final ε) is produced by random number draws in
  `src/orchestrator.py` and a hand-decremented ε, not by any training run.
- Privacy statements rely on `src/federated/privacy_analysis.py`, which is not a DP mechanism and has no accountant.
- The Silo 1 / Silo 2 models it describes were never actually trained (silent fallbacks to toy models).

See `docs/AUDIT.md` sections 4A–4B for line-level details. Kept for reference (related work, problem framing) only.
Do not cite any number from it. The major-project paper is rebuilt from `results/*/metrics.json`.
