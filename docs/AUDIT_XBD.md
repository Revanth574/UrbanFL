# xBD training split — read-only audit (Silo 2, before Prompt 5)

Date: 2026-10-27 · Scope: read-only inventory, nothing trained · Script: one-off inventory over every image/label/target.

## Location and configuration

- Data lives **outside the repo** on this machine (8.0 GB extracted from `train_images_labels_targets.tar`, 8.38 GB).
- Code finds it via the `xbd_root` path setting (`src/fl/config.py`), resolved in this order:
  1. environment variable `URBANFL_XBD_ROOT`;
  2. the user config file `%USERPROFILE%\.urbanfl\paths.json` (or the file named by `URBANFL_CONFIG`), key `xbd_root`;
  3. otherwise a clear error explaining how to set it.
  The split folder is `<xbd_root>/train`. No absolute path is committed; the value on this machine is set in the user
  config file, outside the repository.

## Files

| Folder | Files | Format |
|---|---:|---|
| `images/` | 5,598 | PNG, RGB, **1024 × 1024** (all files checked; metadata agrees) |
| `labels/` | 5,598 | JSON: building polygons (`features.xy`, WKT in pixel coordinates) + metadata (event, disaster type, GSD, sensor) |
| `targets/` | 5,598 | PNG masks, 1024 × 1024 (`*_target.png`) |

- Filenames: `<event>_<8-digit id>_{pre,post}_disaster[.png|.json|_target.png]`; every file parsed, none malformed.
- **2,799 complete pre/post pairs** (each with both images, both label files and both targets).
- Target encoding (checked on a sample of 1 in 25 targets):
  - **pre targets: {0, 1}** = background / building footprint;
  - **post targets: {0, 1, 2, 3, 4}** = background / no-damage / minor / major / destroyed.
- **How un-classified buildings are rasterised** (every one of the 2,993 un-classified polygons in 683 images was
  filled from its WKT and the target pixels under it tallied; classified polygons from a 1-in-20 sample as control):

  | Polygon class | Pre target under polygon | Post target under polygon |
  |---|---|---|
  | no / minor / major / destroyed | 97–98% = 1 (building) | 97–98% = the matching value 1 / 2 / 3 / 4 |
  | **un-classified** | **97% = 1 (building)** | **99.95% = 0 (background)** |

  The remaining 2–3% are polygon-edge pixels (rasteriser differences). So un-classified buildings are buildings for
  localisation but are painted as background in the damage target. The loader therefore keeps them as buildings in
  the localisation target and re-rasterises them to `ignore_index` (255) in the damage target (decision X2).
- Relation to files already in the repo: the 10 sample pairs in `frontend/public/xbd/` are **not** in this train split
  (none of the 20 filenames exist here), so dashboard demo images cannot leak into training. The aggregate statistics
  in `data/real/xbd_damage_annotations.json` (19 events, 22,068 label files) describe the full xBD release; this train
  split has 10 events.

## Per-event inventory (10 events)

| Event | Type | Images | Pre/post pairs | Post images with ≥1 building | Buildings (post) | GSD (m) |
|---|---|---:|---:|---:|---:|---:|
| guatemala-volcano | volcano | 36 | 18 | 17 | 856 | 1.392 |
| hurricane-florence | flooding | 638 | 319 | 306 | 6,446 | 2.031 |
| hurricane-harvey | flooding | 638 | 319 | 300 | 23,014 | 2.602 |
| hurricane-matthew | wind | 476 | 238 | 223 | 13,939 | 1.996 |
| hurricane-michael | wind | 686 | 343 | 334 | 22,686 | 2.089 |
| mexico-earthquake | earthquake | 242 | 121 | 121 | 32,271 | 2.068 |
| midwest-flooding | flooding | 558 | 279 | 244 | 8,756 | 1.716 |
| palu-tsunami | tsunami | 226 | 113 | 111 | 31,394 | 2.429 |
| santa-rosa-wildfire | fire | 452 | 226 | 219 | 12,950 | 1.86 |
| socal-fire | fire | 1,646 | 823 | 408 | 10,475 | 2.28 |
| **Total** | | **5,598** | **2,799** | **2,283** | **162,787** | |

## Damage-class distribution per event (post-disaster building polygons)

| Event | no-damage | minor-damage | major-damage | destroyed | un-classified |
|---|---:|---:|---:|---:|---:|
| guatemala-volcano | 639 (74.7%) | 9 (1.1%) | 9 (1.1%) | 25 (2.9%) | 174 (20.3%) |
| hurricane-florence | 4,689 (72.7%) | 132 (2.1%) | 1,245 (19.3%) | 54 (0.8%) | 326 (5.1%) |
| hurricane-harvey | 11,423 (49.6%) | 2,663 (11.6%) | 8,238 (35.8%) | 401 (1.7%) | 289 (1.3%) |
| hurricane-matthew | 2,515 (18.0%) | 6,548 (47.0%) | 1,945 (14.0%) | 2,147 (15.4%) | 784 (5.6%) |
| hurricane-michael | 14,588 (64.3%) | 5,207 (22.9%) | 1,902 (8.4%) | 757 (3.3%) | 232 (1.0%) |
| mexico-earthquake | 32,066 (99.4%) | 110 (0.3%) | 18 (0.1%) | 2 (0.0%) | 75 (0.2%) |
| midwest-flooding | 8,128 (92.8%) | 149 (1.7%) | 119 (1.4%) | 75 (0.9%) | 285 (3.2%) |
| palu-tsunami | 25,455 (81.1%) | 1 (0.0%) | 571 (1.8%) | 4,966 (15.8%) | 401 (1.3%) |
| santa-rosa-wildfire | 9,285 (71.7%) | 78 (0.6%) | 63 (0.5%) | 3,471 (26.8%) | 53 (0.4%) |
| socal-fire | 8,638 (82.5%) | 83 (0.8%) | 51 (0.5%) | 1,329 (12.7%) | 374 (3.6%) |
| **Total** | **117,426 (72.1%)** | **14,980 (9.2%)** | **14,161 (8.7%)** | **13,227 (8.1%)** | **2,993 (1.8%)** |

## Observations that matter for Prompt 5

1. **Heavy imbalance:** 72% of buildings are no-damage; minor / major / destroyed are 8–9% each. Pixel-level
   imbalance is worse still, since most pixels are background.
2. **Strongly non-IID across events** (events = natural FL clients again): mexico-earthquake is 99.4% no-damage;
   hurricane-matthew is 47% minor; palu-tsunami and santa-rosa-wildfire are dominated by destroyed vs intact; Harvey is
   36% major. Damage type also differs (flood water vs fire vs structural collapse).
3. **Client sizes are uneven:** socal-fire has 823 pairs, but **415 of its post images (50%) contain no building
   at all**. Guatemala-volcano has only 18 pairs (17 with buildings) and 20% un-classified buildings: too small for
   a 70/15/15 split to give a meaningful test set.
4. **Ground sampling distance varies 1.4–2.6 m** across events, so building scale differs per client.
5. **Memory:** 1024 × 1024 pairs do not fit a ResNet18 Siamese U-Net at useful batch sizes on 4 GB; training will
   need crops (e.g. 512 or 256) with AMP, and evaluation by tiling.

## Decisions (approved 27 Oct; training is Review 2 work)

| # | Decision |
|---|---|
| X1 | xView2 task in two stages: (1) building localisation from the pre image; (2) 4-class damage from pre + post, scored on building pixels. Report the official xView2 score = 0.3 · localisation F1 + 0.7 · harmonic mean of the four damage-class F1s. 512 crops, ResNet18/34 encoder, AMP. |
| X2 | Un-classified buildings → `ignore_index` in the damage target (rasterisation reported above). |
| X3 | guatemala-volcano (18 pairs) held out entirely as an unseen-disaster test set. |
| X4 | Per-event 70/15/15 split by pair id, seeded, saved to `data/splits/silo2_seed{s}.json` (seeds 0–4). |
| X5 | Building-free pairs down-sampled to 25% in train only; val / test keep all. |

## Splits (`src/fl/partition_xbd.py`, identical sizes for every seed)

- 9 event clients + 1 held-out event. Splits are stratified on "pair has ≥ 1 building", so val / test keep each
  event's share of building-free tiles. 516 of the 2,799 pairs are building-free; 268 of them are dropped from train.
- Sizes: train 1,681 · val 416 · test 416 · held-out (guatemala-volcano) 18 · dropped building-free train pairs 268.

| Event | train | dropped (empty) | val | test |
|---|---:|---:|---:|---:|
| hurricane-florence | 217 | 6 | 48 | 48 |
| hurricane-harvey | 214 | 9 | 48 | 48 |
| hurricane-matthew | 160 | 8 | 35 | 35 |
| hurricane-michael | 236 | 5 | 51 | 51 |
| mexico-earthquake | 85 | 0 | 18 | 18 |
| midwest-flooding | 177 | 18 | 42 | 42 |
| palu-tsunami | 78 | 1 | 17 | 17 |
| santa-rosa-wildfire | 155 | 3 | 34 | 34 |
| socal-fire | 359 | 218 | 123 | 123 |

### Spatial leakage (caveat for X4)

xBD tiles of one event are neighbouring 1024 × 1024 crops of one area. Using the per-image geotransforms already in
the repo (`data/real/xbd_geotransforms.json`, available for every pair), the seed-0 test set was checked against that
event's train tiles (Chebyshev distance between tile centres, in tile widths):

| Event | Test tiles | Train tile adjacent (≤ 1 tile) | Overlapping (< 0.5 tile) |
|---|---:|---:|---:|
| hurricane-florence | 48 | 40% | 2% |
| hurricane-harvey | 48 | 52% | 2% |
| hurricane-matthew | 35 | 97% | 0% |
| hurricane-michael | 51 | 61% | 2% |
| mexico-earthquake | 18 | 61% | 0% |
| midwest-flooding | 42 | 12% | 0% |
| palu-tsunami | 17 | 94% | 0% |
| santa-rosa-wildfire | 34 | 59% | 6% |
| socal-fire | 123 | 67% | 7% |
| **All** | **416** | **58%** | **3%** |

So 58% of test tiles touch a train tile, and 3% overlap one by more than half. Within-event test scores are therefore
likely optimistic (shared buildings at tile borders, same streets, same lighting). The held-out guatemala-volcano set
has no such leakage and is the honest generalisation number.

### Spatially blocked split — the default since 28 Oct (branch `major`)

The split above is kept as `random_pair` (`data/splits/silo2_random_pair_seed{s}.json`, regenerated and checked
identical to the Review 1 files). The default `spatial_block` split (`data/splits/silo2_seed{s}.json`) works per event:

1. Tiles whose centres are within **1.5** tile widths (Chebyshev, from the geotransforms) are linked; connected
   components are blocks. Most events break into dozens to hundreds of isolated clusters, but some do not (socal-fire
   is one 820-tile component), so components larger than the event's test target are cut into 8 × 8-tile grid cells.
2. Whole blocks go to test (≈ 15% of tiles).
3. Every other tile within 1.5 tile widths of a test tile goes to **val** (a buffer ring), so no train tile can touch a
   test tile even where a component was cut.
4. More whole blocks fill val to ≈ 15%; the rest is train. X5 down-sampling then applies to train as before.

**Test-train adjacency: 0.0% for all five seeds** (was 58%); the real-data test checks that every test tile is more
than 1.5 tile widths from every train tile. Test-val contact exists where val is the buffer, which is allowed.

| Seed | train | dropped building-free (train) | val | test | train+dropped / val / test |
|---|---:|---:|---:|---:|---|
| 0 | 1,715 | 264 | 414 | 388 | 71.2% / 14.9% / 14.0% |
| 1 | 1,727 | 257 | 415 | 382 | 71.3% / 14.9% / 13.7% |
| 2 | 1,744 | 258 | 396 | 383 | 72.0% / 14.2% / 13.8% |
| 3 | 1,698 | 264 | 421 | 398 | 70.6% / 15.1% / 14.3% |
| 4 | 1,747 | 260 | 401 | 373 | 72.2% / 14.4% / 13.4% |

Seed 0 per event (val ring = buffer tiles forced into val):

| Event | train (+dropped) | val | test | blocks after cutting (largest) | val ring |
|---|---|---:|---:|---|---:|
| hurricane-florence | 217 (+2) | 53 | 47 | 93 (34) | 0 |
| hurricane-harvey | 226 (+10) | 42 | 41 | 109 (44) | 0 |
| hurricane-matthew | 159 (+7) | 34 | 38 | 26 (36) | 21 |
| hurricane-michael | 249 (+4) | 46 | 44 | 43 (48) | 10 |
| mexico-earthquake | 88 (+0) | 17 | 16 | 48 (22) | 5 |
| midwest-flooding | 186 (+20) | 37 | 36 | 135 (14) | 0 |
| palu-tsunami | 77 (+1) | 17 | 18 | 20 (18) | 3 |
| santa-rosa-wildfire | 152 (+4) | 32 | 38 | 50 (40) | 11 |
| socal-fire | 361 (+216) | 136 | 110 | 34 (48) | 25 |

Per-event test sets, all seeds — test pairs (building-free share); ⚠ = fewer than 20 test pairs or more than 50%
building-free test pairs:

| Event | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| hurricane-florence | 47 (4%) | 42 (2%) | 43 (0%) | 50 (16%) | 41 (2%) |
| hurricane-harvey | 41 (7%) | 41 (0%) | 49 (4%) | 41 (0%) | 42 (5%) |
| hurricane-matthew | 38 (8%) | 32 (6%) | 36 (0%) | 34 (18%) | 33 (21%) |
| hurricane-michael | 44 (2%) | 52 (4%) | 45 (13%) | 59 (7%) | 50 (2%) |
| mexico-earthquake | 16 (0%) ⚠ | 18 (0%) ⚠ | 16 (0%) ⚠ | 18 (0%) ⚠ | 16 (0%) ⚠ |
| midwest-flooding | 36 (3%) | 36 (22%) | 36 (19%) | 40 (25%) | 36 (17%) |
| palu-tsunami | 18 (0%) ⚠ | 15 (0%) ⚠ | 17 (6%) ⚠ | 18 (0%) ⚠ | 16 (0%) ⚠ |
| santa-rosa-wildfire | 38 (3%) | 29 (3%) | 31 (0%) | 30 (0%) | 29 (7%) |
| socal-fire | 110 (68%) ⚠ | 117 (56%) ⚠ | 110 (70%) ⚠ | 108 (53%) ⚠ | 110 (62%) ⚠ |
| guatemala-volcano (held out) | 18 (6%) | 18 (6%) | 18 (6%) | 18 (6%) | 18 (6%) |

- mexico-earthquake and palu-tsunami are below 20 test pairs on every seed: the events have only 121 / 113 pairs, so
  15% is ≈ 17. Mexico's tiles are dense urban (32k buildings), so the pixel counts are large but the tile-level
  variance of per-event F1 is high; report these two per-event numbers with that caveat.
- socal-fire test sets are 53–70% building-free: the event is 50% building-free and, under spatial blocks, empty tiles
  cluster geographically. For localisation F1 (pooled pixel F1) empty tiles contribute only false positives.
- Possible remedies, not applied: a larger test fraction for small events, or stratifying block assignment on
  building presence.

Trade-off: fractions are approximate (blocks are indivisible), and the split is no longer stratified on building
presence. Test areas are geographically distinct from training areas within each event, so within-event test scores
measure spatial generalisation rather than interpolation between neighbouring tiles.

## Data loader and scorer (no training yet)

- Splits: `load_xbd_partition(seed)` returns the spatial_block split; `load_xbd_partition(seed, "random_pair")` the Review 1 split.
- `src/fl/xbd_data.py::XBDPairDataset(split_dir, pair_ids, stage)`, `stage` "loc" or "damage". Train items are random
  512 crops with identical flips / rot90 on every array, reproducible per (seed, epoch, index); eval items are full
  1024 images. Images are ImageNet-normalised; damage targets are class 0–3 on post-target buildings, 255 elsewhere
  and on un-classified buildings. `client_datasets(...)` builds one dataset per event (the FL clients).
- `src/fl/xbd_metrics.py::XView2Scorer` accumulates pixel counts over images and returns localisation F1, the four
  damage F1s, their harmonic mean (xView2 form, 4 / Σ 1/(F1 + 1e-6)) and the score 0.3 · loc + 0.7 · harmonic.
- Tests: `tests/test_xbd.py` (synthetic xBD tree with known polygons; real-data checks skip if the path is unset).
