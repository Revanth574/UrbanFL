# UrbanFL — Review 1 Plan (Tue 27 Oct → Sat 31 Oct)

CLAUDE.md is in the repo root. Run each prompt in order; do not move on until its "Done when" checks pass.

Review 1 goal: a real, reproducible FL result on Silo 1 (text) with honest baselines, plus a fixed Silo 2 model,
plus a clear account of what was wrong in the prototype and how it was fixed.

**Seeds: 0, 1, 2, 3, 4 for every experiment** (changed from 0–2 on 27 Oct). Report mean ± sample std over 5 seeds.

---

## TUESDAY — Prompt 0: Audit and data inventory  ✅ done → `docs/AUDIT.md`

Read-only audit: missing data adapters, inventory of `data/` (format, rows, label column, event column, per-event
label distribution), xBD samples (pairs, size, masks), and every hardcoded / random reported metric.

Outcome: CrisisNLP CrowdFlower v2, 10 humanitarian-taxonomy events → 10 FL clients. xBD has 10 pairs and no masks.

Approved decisions (AUDIT.md §5):
- **D1** Drop Ebola + MERS (different taxonomy).
- **D2** Keep `Chile_Earthquake_cl` (Spanish) and `_en` as two clients.
- **D3** Normalise (strip `RT @user:`, URLs, mentions, lowercase, collapse whitespace) before grouping; keep one
  copy of label-consistent groups; drop label-conflicting groups entirely; log both counts.
- **D4** Primary = pooled test macro-F1 over the union of all client test sets with an explicit 9-label list.
  Secondary = per-client macro-F1 over classes with ≥ 3 test examples in that client, plus the qualifying class
  count. Implemented by the project author in `src/fl/utils.py::d4_metrics`.
- **D5** Use the raw TSVs, never `crisis_reports.json`.

---

## WEDNESDAY — Prompt 1: Silo 1 data + embedding cache + baselines  ✅ done

Label = humanitarian information type (NOT disaster type). Clients = disaster events.

1. `src/fl/partition.py`: load TSVs, shared label set (documented mapping), D1/D3, drop events < 200 rows,
   stratified 70/15/15 split inside each client → `data/splits/silo1_seed{seed}.json`.
2. `experiments/silo1_embed_cache.py`: `paraphrase-multilingual-MiniLM-L12-v2`, GPU fp16, batch 64 →
   `data/cache/silo1_emb.npy` + id→row index, reused.
3. `src/fl/models.py`: `MLPHead` (Linear→GELU→Dropout(0.2)→Linear) + class-weighted CE.
4. `src/fl/utils.py`: `set_seed`, metrics, `write_metrics` (git hash, seed, config, split sizes, runtime).
5. `experiments/silo1_baselines.py`: centralized, local-only, majority. **Seeds 0–4**, mean ± std →
   `results/silo1_baselines/`.
6. `tests/`: disjoint splits, test sets never used in training, metrics match sklearn, D4 spec test.

Done when: centralized ≫ majority, local-only < centralized, results JSON for 5 seeds.

---

## THURSDAY — Prompt 2: Real FL core (FedAvg, FedProx)

1. `src/fl/algorithms.py`: `fedavg_aggregate(global_state, client_states, client_sizes)` (sample-weighted);
   FedProx local loss = CE + (μ/2)‖w − w_global‖². Aggregated weights are loaded back into the global model.
2. `src/fl/runner.py`: `run_fl(algorithm, clients, rounds, local_epochs, lr, batch_size, client_fraction, seed)`.
   Per round: sample clients, broadcast, local train, aggregate, evaluate. Log train loss, val metrics, bytes,
   round time. Final: evaluate the selected checkpoint on every client's test set.
3. `experiments/silo1_fl.py`: FedAvg and FedProx μ ∈ {0.001, 0.01, 0.1}; local_epochs ∈ {1, 5}; 100 rounds;
   client_fraction 1.0; **seeds 0–4**.
4. Sanity tests: single client + local_epochs=1 FedAvg == plain SGD; global weights change between rounds.
5. Results under `results/silo1_fl/`.

Amendments (27 Oct):
- **Federated checkpoint selection.** Each client reports val loss on its own val set; the server selects the
  round with the lowest client-size-weighted mean val loss. Pooled val macro-F1 may be logged for plots but is
  never used for selection or early stopping.
- Reuse existing splits + embedding cache; same `MLPHead` and class-weighting scheme as centralized. Global class
  weights only via a federated count sum, otherwise each client's own weights (state which and why).
- Report per seed, then mean ± std over seeds 0–4: pooled test macro-F1, D4 client-mean, paired delta vs
  local-only, gap recovery = (FL − local) / (central − local).
- Per-client delta vs local-only for FedAvg and best FedProx (Review 1 scatter plot).
- Log bytes communicated per round (head parameters × clients × 2 directions).

Table: Majority | Local-only | FedAvg | FedProx (best μ) | Centralized — test macro-F1 mean ± std over 5 seeds.

Consolidated clean rerun (27 Oct, all Review 1 numbers from one clean commit):
- Baselines use the FL client recipe: plain SGD via single-client `run_fl`, lr grid {0.01, 0.03, 0.1, 0.3, 1.0}
  chosen on seed 0 by validation LOSS (local-only: client-val-size-weighted), checkpoint = lowest val loss,
  100 epochs (= passes of FL E=1 × 100 rounds).
- FL: FedAvg E=1, FedProx E=1 (μ ∈ {0.001, 0.01, 0.1}) at 100 rounds; FedAvg E=5 at 200 rounds (lr swept at
  200 rounds); FedProx E=5 dropped. Check E=5 plateaus by round 200, else report it as budget-limited.
- Optional FedAvg+FT: each client fine-tunes the selected global model for ≤1 / ≤2 local epochs, keeping the
  epoch (0 = no fine-tuning) with the lowest loss on its own val set.
- Every metrics.json: clean commit, dirty = false, for training and metrics. Commit results/ JSON; push `review1`.
- Any row selected by test score is marked "appendix only".

Done when: sanity tests pass, FL sits between local-only and centralized (or a clear explanation), convergence
curve comes from logged JSON.

## THURSDAY — Prompt 3: Prototype cleanup (honesty pass)  ✅ done (commit da3ca78)

For every item in AUDIT.md §4: remove hardcoded metrics (return None / "not computed"); label simulated values in
`api/state.py` and remove fake model outputs (`nlpConfidence`, `envMatch`) with a TODO to wire `src/fl` inference;
DEPRECATED headers on `src/federated/*.py`, `src/orchestrator.py`, `fl_aggregation()`; remove dead adapter imports.
Backend must still start (`uvicorn api.main:app`); run pytest; show a diff summary.

---

## FRIDAY — Prompt 4: DP-FedAvg with a real accountant (if Thursday finished)

Client-level DP-FedAvg (McMahan et al. 2018) in `src/fl/dp.py`: clip each client delta to L2 norm C; server adds
N(0, (zC/m)²); `opacus.accountants.RDPAccountant`, one step per round, noise_multiplier = z,
sample_rate = client_fraction; report ε at δ = 1e-5 every round. CLIENT-level (event-level) privacy, not record-level.
Grid: z ∈ {0.5, 1.0, 2.0}, C ∈ {0.1, 1.0}, client_fraction ∈ {0.5, 1.0}, 100 rounds, **seeds 0–4**.
Privacy-utility table + plot (macro-F1 vs final ε) from logged JSON. Large ε with few clients is an honest finding.

(`from opacus.accountants import RDPAccountant` verified in the venv, opacus 1.6.0.)

Note: if FL uses federated class-count sums, those histograms are outside the DP accountant — either disclose,
use secure aggregation, or switch to client-local class weights for DP runs.

## FRIDAY — Prompt 5: Fix Silo 2 model + figures

1. Siamese U-Net: pretrained ResNet18 encoder shared pre/post, per-scale fusion [f_pre, f_post, |f_pre − f_post|],
   skip connections, 4-stage decoder, 1-channel mask; BCEWithLogits + Dice; AMP.
2. Train/eval on local xBD samples with a held-out split; IoU and F1; label as a small-sample sanity check.
   **Blocked:** no masks locally — needs xView2 label JSONs (AUDIT.md §3).
3. ✅ `experiments/make_figures.py` (figures a–e done): all figures only from `results/*.json` (baseline bars, FL convergence with
   mean ± std band, per-client local-only vs FedAvg scatter, DP privacy-utility, Silo 2 masks) → `paper/figures/`,
   300 DPI.

Friday evening: Review 1 slides. Saturday morning: dry run only.

---

## Review 1 slide outline (10–12 slides)

1. Title, team, guide
2. Problem: disaster data is siloed across agencies; it can't be pooled for privacy and legal reasons
3. Minor-project recap and audit findings — what was simulated or incorrect, and why cross-modality averaging was
   ill-posed (present as a strength: found and fixed)
4. Revised formulation: federation within each modality, clients = disaster events/agencies
5. Architecture diagram (Silo 1 text FL, Silo 2 imagery FL, Silo 3 decision layer, dashboard)
6. Literature survey and gaps (FedAvg, FedProx, SCAFFOLD, DP-FedAvg, FedProto; crisis NLP; xBD)
7. Dataset and partitioning: events, clients, label distribution per client (non-IID evidence)
8. Results: Majority / Local-only / FedAvg / FedProx / Centralized table (5 seeds)
9. Convergence curve + per-client gains
10. DP results (if done) or the DP design + accountant
11. Plan for Reviews 2–3: LoRA fine-tuning, SCAFFOLD, FedHetero on cross-modal prototypes, leave-one-event-out,
    Silo 2 federation on xBD events, dashboard wired to real models
12. Questions

## If you fall behind
Priority: Prompt 0 → 1 → 2 → slides. Prompts 3, 4, 5 can be presented as "in progress".
A real FedAvg-vs-baselines table on Silo 1 is enough for a strong Review 1.
