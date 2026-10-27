# UrbanFL — Claude Code Project Rules

## What this project is
Privacy-preserving federated learning (FL) for disaster response. Major-project rebuild of the minor-project prototype.
Core research setup: federation happens WITHIN each modality, across clients that are disaster events
(each event = one agency holding its own data). Text (Silo 1) and imagery (Silo 2) are separate federations.
Silo 3 (routing/BPR) is a deterministic decision layer, not an FL client.

## Non-negotiable integrity rules
1. Every number that appears in results, figures, the paper, or the dashboard must be produced by code in this repo
   and written to `results/<experiment>/<run_id>/metrics.json`. Never hardcode a metric. Never return a constant as
   an "accuracy", "R²", or "epsilon".
2. No `random` / `np.random` values may be shown as model outputs. Simulation randomness is allowed only for
   scenario generation (incident locations, timing) and must be seeded and labelled as simulated.
3. Privacy budgets (ε) come only from a real accountant (Opacus `RDPAccountant`). Never decrement ε by hand.
4. Every experiment run logs: git commit hash, seed, full config, dataset split sizes, wall-clock time.
5. If something cannot be computed honestly, report it as missing — do not estimate or fill it in.
6. Always report baselines next to FL results: centralized (upper bound) and local-only (lower bound).
7. Use macro-F1 as the primary classification metric (classes are imbalanced); accuracy is secondary.

## Environment
- Windows 11, PowerShell. Use `python -m ...` and pathlib; no bash-only syntax in scripts.
- GPU: RTX 3050 4 GB VRAM. Use fp16/AMP, small batches, cached embeddings where possible.
- Python 3.10+, PyTorch 2.x. FL loop is written in plain PyTorch (no Flower/Ray — Ray is unreliable on Windows).
- Deterministic seeds: `src/fl/utils.py::set_seed(seed)` sets python, numpy, torch, cuda.

## Layout (new code)
```
src/fl/
  partition.py      # event-based client partitioning + per-client train/val/test splits
  models.py         # MLP head for Silo 1 embeddings; fixed Siamese U-Net for Silo 2
  algorithms.py     # FedAvg, FedProx (later: SCAFFOLD, FedHetero)
  dp.py             # client-level DP-FedAvg: update clipping + Gaussian noise + Opacus RDP accountant
  runner.py         # generic FL round loop, evaluation, logging
  utils.py          # seeds, metrics (macro-F1, IoU), json logging, git hash
experiments/
  silo1_embed_cache.py   # encode texts once with a pretrained encoder, save .npy
  silo1_baselines.py     # centralized + local-only
  silo1_fl.py            # FL experiments (FedAvg / FedProx / DP)
  make_figures.py        # figures ONLY from results/*.json
tests/                   # pytest; run `python -m pytest -q` before every commit
```

## Legacy code
`src/federated/*`, `src/orchestrator.py`, and `experiments/train_and_evaluate.py::fl_aggregation` are the
minor-project prototype. Do not extend them. Mark them deprecated; the dashboard will be rewired to `src/fl` later.
Do not touch `frontend/` unless asked.

## Workflow
- Small commits with clear messages. Run tests before committing.
- Before writing a new module, state the plan in 3–5 lines and wait for approval on anything that changes this file's rules.
