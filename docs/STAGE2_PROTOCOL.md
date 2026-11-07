# Silo 2 stage 2 (damage classification): protocol

The protocol below is the one the stage-2 campaign runs with: commit `c47923f`, launched 2026-11-07 07:36 in the
`UrbanFL-gpu` worktree. The source of truth is the module docstring and the `PROTOCOL` block of
`experiments/silo2_dmg.py` at that commit, also written into every run's `metrics.json`. This page restates it and
records the decisions made around it.

## Fixed before launch (commit c47923f)

- **Task.** Predict one of 4 damage classes (no / minor / major / destroyed) on building pixels from a pre/post image
  pair (xView2). Background and un-classified buildings are ignored in the loss.
- **Score.** The official xView2 score: 0.3 × localisation F1 + 0.7 × harmonic mean of the 4 damage-class F1s.
  - Localisation masks come from the method's own stage-1 model(s).
  - Damage is scored on post-target building pixels, with the prediction set to 0 wherever stage 1 predicts no
    building. A building that stage 1 misses therefore counts as a false negative for its true damage class.
- **Model.** `SiameseDamageNet`: the method's own stage-1 U-Net body, shared between the pre and post images, plus a
  new 1×1 head with 4 outputs on the 64-channel features.
  - Each method starts from its own stage-1 weights.
  - The new head is seeded with the run seed.
- **Loss.** Class-weighted cross-entropy on scored pixels.
  - Weights follow the Silo 1 rule, computed on damage-pixel counts of the training split, capped at 10.
  - Centralized uses pooled counts.
  - FedAvg and FedAvg+FT use the federated sum of the clients' 4-bin histograms.
  - Local-only uses each client's own counts.
- **Training pairs.** Only pairs with at least one classified building. Val, test and held-out sets use all pairs.
- **Budgets.**
  - lr sweep: 10 epochs per lr.
  - Centralized and local-only: 30 epochs.
  - FedAvg: 30 rounds of 1 local epoch.
  - FedAvg+FT: up to 2 epochs, with a fresh AdamW at the sweep lr.
- **Validation.**
  - Every 2nd epoch or round, and always the final one, for every method and the sweep.
  - Checkpoint selection uses only the validated epochs or rounds.
  - FedAvg+FT also validates its start model (epoch 0), so its candidates are epochs 0 and 2.
- **Selection** never reads test data:
  - lr sweep: centralized, split seed 0, grid {1e-4, 3e-4, 1e-3}, lowest best pooled validation loss.
  - Centralized: lowest pooled validation loss.
  - Local-only: lowest loss on the client's own validation set.
  - FedAvg: lowest image-weighted federated validation loss.
  - FedAvg+FT: epoch 0 or 2, chosen by the client's own validation loss.
- **Resume.** A resume refuses any checkpoint whose signature differs. The signature covers method, client, seed, lr,
  epochs or rounds, start-weights path plus SHA-256, and commit.
- **Order.** lr sweep, then seed 0 (centralized, local-only, FedAvg, FedAvg+FT), then seed 1, then seed 2.

## lr sweep outcome (2026-11-07 09:30)

Source: `results/silo2_dmg/lr_sweep/choice.json`.

| lr | best val loss | epoch of best |
|---|---|---|
| **1e-4** | **0.9202** | 10 |
| 3e-4 | 0.9289 | 10 |
| 1e-3 | 0.9833 | 10 |

- The chosen lr is **1e-4**, which is at the edge of the grid (`chosen_lr_at_grid_edge: true`).
- The running campaign keeps lr 1e-4 for all three seeds.

## Grid-edge check (pre-registered)

Recorded on 2026-11-07, after the sweep above but before any seed result exists and before the extra point is run.
The rule, verbatim as decided:

> After all three seeds finish, run one extra sweep point at lr 3e-5 (centralized, split seed 0, 10 epochs, best val
> loss). If its best val loss beats 1e-4's (0.9202) by more than 0.03, rerun stage 2 at 3e-5; otherwise keep 1e-4
> and report the grid-edge check as done.

How the rule is applied:

- **Threshold.** Stage 2 is rerun at 3e-5 only if the best validation loss at 3e-5 is **below 0.8902**
  (0.9202 − 0.03, using the unrounded 1e-4 value from `choice.json`). A difference of exactly 0.03, or anything
  smaller, keeps 1e-4.
- **Same procedure as the sweep.** The extra point uses the same training code path, data, split, initialisation and
  validation schedule (every 2nd epoch plus the final one).
- **Output.** It writes its own `metrics.json` under `results/silo2_dmg/lr_sweep/`, recording its commit. Whichever
  way the check goes, the result is reported.

## Sensitivity and diagnostics (pre-registered)

Recorded on 2026-11-07 at about 15:50, after the seed-0 results and before seeds 1–2 land.

What we had seen: at seed 0, FedAvg scored 0.253 on test. Its damage harmonic mean was 0.021, because it almost never
predicts minor or major damage (minor F1 0.006). Its federated val loss was still falling at round 30, the round it
selected. The rule, verbatim as decided:

> After the campaign and the 3e-5 check, on split seed 0 only, labelled as sensitivity/diagnostics. The 30-round,
> lr 1e-4 headline is unchanged whatever they show. All are run and reported regardless of outcome:
> D1. FedAvg vs centralized diagnostics on the saved best models: test confusion matrix on scored building pixels;
>     mean softmax probability per class on true-minor and true-major pixels; a post-hoc per-class logit offset fitted
>     on validation only (maximising val damage HM), applied to test.
> S1. FedAvg, 60 rounds, lr 1e-4.
> S2. FedAvg, 30 rounds, lr 3e-4 (FedAvg's lr was not tuned separately in Silo 2, unlike Silo 1).

How these are run:

- **Order.** First the 3e-5 grid-edge check, then S1, then S2, then D1. One job at a time, after the campaign
  (including its summary step) has finished.
- **S1 and S2.** Except for the rounds (S1) or the lr (S2), they follow the campaign's FedAvg protocol exactly: same
  split, starting weights, class weights, E = 1, validation every 2nd round plus the final one, and selection by
  federated val loss.
  - They write to `results/silo2_dmg/sensitivity/`, so they never overwrite or mix with the headline runs.
  - Their checkpoint directories are separate from the headline runs'.
- **"Scored building pixels" (D1)** means the pixels the official score uses for damage: post-target building pixels
  with a damage label. A pixel where the method's stage-1 model predicts no building counts as predicted
  "not localised", as in the official score. The confusion matrix therefore has a "not localised" prediction column.
- **Logit offsets (D1).**
  - One additive offset per damage class, with no-damage fixed at 0.
  - Fitted only on the validation split to maximise the validation damage harmonic mean (pooled TP/FP/FN, as in the
    official score), then applied unchanged to test.
  - Test scores are reported both without and with the offsets.
