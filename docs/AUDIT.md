# UrbanFL — Prototype Audit (Review 1, Prompt 0)

Date: 2026-10-27 · Scope: read-only audit, no code changed · Repo state: `8d95ec2` (cloned from
`github.com/Revanth574/UrbanFL`; the original local working copy is no longer on this machine).

## TL;DR

| # | Finding | Impact |
|---|---|---|
| 1 | The five `data.*` adapter modules imported by `train_and_evaluate.py` do not exist — not in git history, not on any drive. | Prototype training script cannot run as committed. |
| 2 | Both "real" models silently fell back to toy models during the reported run (exceptions swallowed). Even if loaded, the XLM-R encoder is randomly initialised and frozen. | Reported Silo 1 acc (0.49) and Silo 2 IoU (0.05) are from toy fallbacks, not from the models described in the paper. |
| 3 | `fl_aggregation()` performs no aggregation; ε is a hand-made sum; the paper's FL table is the statistics of `random.random()`. | No FL result in the prototype is real. |
| 4 | Silo 3 BPR R² is a formula (`0.85 + 0.03·min(n,5)`) → reported as **1.0**. | Fabricated metric. |
| 5 | CrisisNLP CrowdFlower v2 is in the repo, organised **one folder per event** — a natural FL partition. 10 natural-disaster events, all ≥ 1,262 rows, share one 9-class humanitarian label set. | Silo 1 FL is unblocked. **10 clients.** |
| 6 | xBD: only 10 pre/post pairs, **no masks / per-building polygons**. | Silo 2 training (Prompt 5) is blocked until xBD labels are downloaded. |

---

## 1. Missing data adapters

| Import (file:line) | Status |
|---|---|
| `data.silo1_data_adapter` (`experiments/train_and_evaluate.py:30`, `experiments/run_operational.py:86`) | **Missing** |
| `data.silo2_data_adapter` (`train_and_evaluate.py:31`) | **Missing** |
| `data.silo3_data_adapter` (`train_and_evaluate.py:32`) | **Missing** |
| `data.multi_hazard_adapter` (`train_and_evaluate.py:33`) | **Missing** |
| `src.data.multi_hazard_adapter` (`experiments/test_new_modules.py:335`) | **Missing** |
| `data.real_data_loader` (`run_operational.py:173`) | Exists, but at `src/data/real_data_loader.py` — wrong import path |

Searched: whole repo, all git revisions (`git log --all --name-only`), all of `C:\` (excluding Windows/Program
Files/AppData/anaconda), other drives (none present), `Downloads/`. WSL (Ubuntu-24.04) exists but cannot start
(`HCS_E_SERVICE_NOT_AVAILABLE`), so its filesystem was not searched.

Other broken `from src.… import` targets (modules that do not exist): `src.utils.core`,
`src.federated.communication`, `src.federated.concept_drift`, `src.command.report_generator`,
`src.medical.patient_distributor`, `src.public.alert_system`, `src.public.citizen_platform`,
`src.recovery.recovery_tracker`. Also `experiments/convergence_analysis.py:14` imports `AdaptiveDPEngine` and
`GradientSpatialDecomposer` from `src/federated/privacy_analysis.py`, which defines neither.

**Recommendation:** do not recreate the adapters. Prompt 3 should remove the dead imports; new code reads the
raw TSVs directly in `src/fl/partition.py`.

---

## 2. Text datasets

### 2.1 CrisisNLP CrowdFlower v2 (primary — use this)

Path: `data/real/CrisisNLP_labeled_data_crowdflower_v2/CrisisNLP_labeled_data_crowdflower/<event>/*.tsv`
(ignore the `__MACOSX/` resource-fork copies).

- Format: TSV, UTF-8, columns `tweet_id` (quoted string), `tweet_text`, `label`. Read with `quoting=3` (QUOTE_NONE).
- **Event column:** not a column — the event is the **folder name** (e.g. `2015_Nepal_Earthquake_en`).
- Labels: ≥3 CrowdFlower workers agreed per tweet (Imran et al., LREC 2016 — cite this).
- 22,099 rows total, 0 duplicate tweet_ids, 0 exact duplicate texts (within or across events).

| Event (folder) | Rows | Taxonomy | Majority class (share) |
|---|---:|---|---|
| 2013_Pakistan_eq | 1,881 | humanitarian-9 | other_useful_information (38%) |
| 2014_California_Earthquake | 1,701 | humanitarian-9 | other_useful_information (51%) |
| 2014_Chile_Earthquake_cl *(Spanish)* | 1,585 | humanitarian-9 | donation_needs… (35%) |
| 2014_Chile_Earthquake_en | 1,932 | humanitarian-9 | other_useful_information (32%) |
| 2014_Hurricane_Odile_Mexico_en | 1,262 | humanitarian-9 | infrastructure_and_utilities_damage (36%) |
| 2014_India_floods | 1,820 | humanitarian-9 | injured_or_dead_people (49%) |
| 2014_Pakistan_floods | 1,769 | humanitarian-9 | other_useful_information (35%) |
| 2014_Philippines_Typhoon_Hagupit_en | 2,010 | humanitarian-9 | other_useful_information (36%) |
| 2015_Cyclone_Pam_en | 2,004 | humanitarian-9 | other_useful_information (27%) |
| 2015_Nepal_Earthquake_en | 3,003 | humanitarian-9 | donation_needs… (30%) |
| 2014_ebola_cf | 1,774 | **disease-8** | other_useful_information (28%) |
| 2014_Middle_East_Respiratory_Syndrome_en | 1,358 | **disease-8** | other_useful_information (34%) |

**Events with ≥ 200 rows on the shared humanitarian label set: 10** (all of them have ≥ 1,262). Ebola and MERS use
a different, epidemic-specific taxonomy (symptoms / transmission / prevention / treatment / deaths / affected
people) whose labels do not overlap the humanitarian set except `other_useful_information` and
`not_related_or_irrelevant`.

#### Per-event label distribution (humanitarian-9 events, row counts)

| label | Pak_eq | Calif | Chile_cl | Chile_en | Odile | India_fl | Pak_fl | Hagupit | Pam | Nepal | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| caution_and_advice | 69 | 72 | 46 | 246 | 57 | 41 | 51 | 348 | 92 | 35 | 1,057 |
| displaced_people_and_evacuations | 12 | **4** | 26 | 63 | 93 | 27 | 85 | 130 | 48 | 85 | 573 |
| donation_needs_or_offers_or_volunteering_services | 299 | 77 | 549 | 10 | 195 | 45 | 463 | 113 | 363 | 887 | 3,001 |
| infrastructure_and_utilities_damage | 28 | 305 | 37 | 26 | 450 | 66 | 86 | 92 | 212 | 129 | 1,431 |
| injured_or_dead_people | 326 | 183 | 417 | 116 | 31 | 893 | 220 | 66 | 142 | 344 | 2,738 |
| missing_trapped_or_found_people | **5** | **6** | 38 | **6** | 28 | 14 | 103 | **8** | 18 | 187 | 413 |
| not_related_or_irrelevant | 312 | 106 | 161 | 342 | 43 | 498 | 25 | 232 | 496 | 176 | 2,391 |
| other_useful_information | 724 | 868 | 122 | 619 | 221 | 206 | 618 | 731 | 540 | 638 | 5,287 |
| sympathy_and_emotional_support | 106 | 80 | 189 | 504 | 144 | 30 | 118 | 290 | 93 | 522 | 2,076 |
| **rows** | 1,881 | 1,701 | 1,585 | 1,932 | 1,262 | 1,820 | 1,769 | 2,010 | 2,004 | 3,003 | **18,967** |

This is strongly non-IID (label skew): e.g. India floods is 49% casualty reports, Odile 36% infrastructure,
California 51% "other". Good evidence for slide 7.

**Rare-class warning:** with a 70/15/15 split, classes with ≤ 8 examples in a client (bold above) leave 0–1 test
examples. Per-client macro-F1 will be noisy for those clients.

#### Near-duplicates
Exact-text dedup removes nothing. After normalising (strip leading `RT @user:`, URLs, @mentions, punctuation,
lower-case): **355 near-duplicate rows within events**, 358 overall, and **76 near-duplicate groups carry
conflicting labels**. These are mostly retweets; if left in, copies of the same tweet can land in train and test of
the same client (leakage).

### 2.2 `data/real/crisis_reports.json` (derived — do not use for training)

- JSON dict: `data_provenance`, `total_reports` (22,095), `categories`, `events`, `reports` (list of
  `{text, category, original_label, event, tweet_id, source_type, data_provenance}`).
- It is a **re-export of the same 12 CrisisNLP events** (4 rows fewer than the raw TSVs), with a remapped
  9-way `category`. The mapping is lossy and partly wrong: all disease labels — including `deaths_reports` —
  are dumped into `general_info`, so "casualty" excludes epidemic deaths.
- The prototype trained on **CrisisNLP + crisis_reports.json together** (`train_and_evaluate.py:141`), i.e. the
  same tweets twice; dedup was by the first 100 characters only (`:155`).

### 2.3 Other files under `data/` (not text-classification data)
Istanbul city-simulation inputs (`istanbul_*.json`, `istanbul_roads.gpickle`, population GeoTIFF), Palu road network
(`palu_real/`), USGS earthquakes, dashboard snapshots (`command_center/`, `reports/`). None are labelled text.
HumAID (mentioned at `train_and_evaluate.py:140`) is **not** present.

---

## 3. xBD samples

| Item | Finding |
|---|---|
| Image pairs | **10 pre/post pairs** in `frontend/public/xbd/` — 5 × `guatemala-volcano`, 5 × `hurricane-florence` |
| Image format | PNG, 1024 × 1024, RGB |
| Masks / label JSONs | **None.** `data/real/xbd/` (where the full set lived) is git-ignored and absent from this machine |
| `data/real/xbd_damage_annotations.json` | Aggregate stats over the full xBD train set (22,068 label files, 850,736 buildings, 19 events, damage distribution) + `sample_labels`: **one** polygon per image for the 5 sample images — not enough to rasterise a mask |
| `data/real/xbd_geotransforms.json` | Geotransforms for 22,068 images (all 19 events) — confirms the full set was downloaded once |

**Consequence for Prompt 5:** a Siamese U-Net cannot be trained or evaluated without per-building polygons. Options:
(a) re-download the xBD/xView2 label JSONs (registration at xview2.org; labels are small relative to imagery — a
subset of events is enough for a sanity check), or (b) present Silo 2 as "architecture fixed, training pending data".

---

## 4. Hardcoded, random, or fabricated metrics

### 4A. Fabricated metrics — must be removed (CLAUDE.md rules 1, 2, 3)

| Location | What it does |
|---|---|
| `experiments/train_and_evaluate.py:696-702` `_evaluate_bpr_accuracy` | Returns `0.85 + 0.03·min(n_types, 5)`; comment admits "without actual paired observations". Produced **R² = 1.0** in `results/training_metrics.json`. |
| `experiments/train_and_evaluate.py:707-795` `fl_aggregation` | Never aggregates: weights `w1, w2` (`:771-773`) are computed and unused, nothing is loaded back into any model. Missing grads replaced by `torch.randn_like(param)*0.001` (`:749`, `:760`). ε per round = `(1/n_params)/σ` summed linearly (`:765-768`) — not a DP accountant. Also averages across modalities (text model vs image model), which is ill-posed. |
| `src/orchestrator.py:49-63` | Client "accuracy" = `0.7 + random()*0.25`, latency random, divergence = `0.1 + random()*0.4`, ε decremented by hand (`-0.02`), "convergence" = closed-form `1 - exp(-0.1 t)`. |
| `src/federated/privacy_analysis.py:30-37` | Noise scaled by `2/ε`, ε decremented by hand (`-0.05`) per call — not a DP mechanism. |
| `src/federated/fedhetero.py:19-40` | Weights computed from the fake accuracies above. |
| `api/state.py:369-370` | `nlpConfidence = 0.5 + random()*0.5`, `envMatch = random() > 0.5` shown to the dashboard as model outputs. |
| `api/state.py:682-685` (and `:393`) | `fl_round` ticks on a wall-clock timer; ε decremented by `0.015` per tick. No training behind it. |
| `api/state.py:456` | `total_reports_ingested = tick_count * 2` — fabricated count. |
| `experiments/convergence_analysis.py:19-48` | "FedAvg vs FedProx vs FedNova vs SCAFFOLD vs FedHetero" curves = `base/(r+1)` + per-algorithm penalty constant + noise. No training. |
| `experiments/convergence_analysis.py:51-77` | Privacy-utility "Pareto frontier" = `0.92 - k/ε` with hand-chosen `k` and a hardcoded "55% noise reduction". |
| `experiments/convergence_analysis.py:80-118` | Gradient-inversion test on `torch.randn` gradients. |
| `experiments/ablation_study.py:110-145` | Ablation accuracies = `1 - (1/(r+1) + Σ hardcoded penalties) + noise`. |
| `experiments/system_baselines.py:18-80` | Baseline table typed in by hand (`f1_incident: 0.92, 0.68, 0.79, 0.86, 0.88`; `ε=7.5/4.2/1.62`). |
| `experiments/run_full_pipeline.py:185, 194, 206-207` | "Local training" = random gradients; drift detector fed `np.random.dirichlet`; logged `train_loss = 1/(r+1)+noise`, `accuracy = min(0.95, 0.5+0.03r)`. |
| `paper/urbanfl_paper.tex:248-272` (Table `tab:fl`) | Divergence 0.31 ± 0.12 = mean/std of `U(0.1, 0.5)` (0.30 / 0.115) from the orchestrator RNG; "final ε = 0.10" = floor of the hand decrement; "37% island-mode skips" not reproducible from code (`island_mode=False` in orchestrator). |
| `results/training_metrics.json` | `silo3_bpr_r2 = 1.0` (fabricated, above); `fl_privacy_budget = 3.6e-5` (meaningless formula). |

### 4B. Real numbers produced by broken pipelines (not fabricated, but invalid)

| Location | Problem |
|---|---|
| `train_and_evaluate.py:200-207` | `CrisisClassifier` has no `is_functional` attribute → `AttributeError` is swallowed → **silently trains `LightweightNLP`** (byte `Embedding(256,64)` + MLP). |
| `src/silo1/nlp_model.py:14-24` | Even when it loads, `XLMRobertaModel(config)` is **randomly initialised** (no `from_pretrained`) and then **frozen** — a classifier head on random features. |
| `train_and_evaluate.py:475` | `SiameseChangeDetector(in_channels=…)` — the class has no `in_channels` arg → `TypeError` swallowed → **silently trains `LightweightCD`**. |
| `src/silo2/change_detection.py:13` | `pretrained=False` default; no skip connections; decodes from 1/32 resolution. |
| `train_and_evaluate.py:165` | Train/val split with **unseeded** `np.random.permutation`; no test set; no per-event split. |
| `results/training_metrics.json` | Silo 1 val acc flat at 0.47–0.49 over 15 epochs; Silo 2 val IoU 0.05. Consistent with the toy fallbacks. |

### 4C. Simulation randomness — allowed, but must be seeded and labelled "simulated" (rule 2)

Unseeded (`random.*` / `np.random.*` global state):
`api/state.py:264-265, 297-299, 347, 364, 492, 556-560, 575, 590, 635, 655`;
`src/medical/hospital_network.py:10-13, 69, 74`; `src/resources/resource_manager.py:9-11`;
`src/environment/env_monitor.py:42-43`; `src/infrastructure/grid_monitor.py:70-71`;
`src/dashboard/situation_monitor.py:78-80`; `experiments/stress_test.py:56-61`;
`experiments/run_full_pipeline.py:257-268`; `experiments/test_new_modules.py:58`.

Seeded but not labelled as simulated: `api/routers/dispatch.py:56, 124, 302` (incl. random `en_route`/`on_scene`
status, `:182`); `api/routers/zones.py:47-48`; `api/state.py:203`; `src/dashboard/damage_assessment.py:129`.

Non-reproducible seed: `api/routers/analytics.py:68` seeds with `hash(zone_id)`, which is salted per process for
strings — casualty/trapped estimates (`:70-71`) change on every restart and are displayed as numbers.

### 4D. Benign
`experiments/convergence_analysis.py:120-170` (`computational_profile`) uses random tensors only to time
operations — fine, as long as it is not presented as a model result.

---

## 5. Decisions needed before Prompt 1

| # | Question | Recommendation |
|---|---|---|
| D1 | Ebola + MERS use a different taxonomy. | **Drop them** from Silo 1 FL (10 clients remain). Mention as future work (cross-taxonomy FL). |
| D2 | `Chile_Earthquake_cl` (Spanish) and `_en` are the same event, different tweet sets. | **Keep as two clients** — two language communities is a realistic "two agencies" setting, and the multilingual encoder handles Spanish. State it explicitly. |
| D3 | Exact-text dedup is a no-op. | **Normalised near-dup removal** (strip RT prefix/URLs/mentions) *before* splitting; drop the 76 label-conflicting groups. |
| D4 | Per-client macro-F1 with 0–1 test examples of rare classes. | Report per-client macro-F1 over labels **present in that client's test set**, plus pooled macro-F1 over all 9 classes as the headline. |
| D5 | Which source file. | Raw TSVs, never `crisis_reports.json`. |

## 6. Environment check
RTX 3050 Laptop, 4 GB, driver 617.14. System Python 3.13 already has `torch 2.11.0+cu126` with CUDA available;
Python 3.11 and Anaconda are also installed. `requirements.txt` lacks `sentence-transformers`, `opacus`, `pytest`.
