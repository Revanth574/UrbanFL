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
  - "un-classified" buildings have **no separate value** in the post targets (images that contain them still show only
    0–4). They are most likely painted as background; this needs a per-pixel check against the polygons before training.
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

## Decisions needed before Prompt 5 (not started — awaiting approval)

| # | Question | Options |
|---|---|---|
| X1 | Target for the "fixed Silo 2 model" | (a) binary building / change mask from the targets (as written in Prompt 5); (b) building localisation (pre) + 4-class damage (post), the xView2 task |
| X2 | Un-classified buildings | verify how they are rasterised, then ignore them in the loss (recommended) or treat as background |
| X3 | Guatemala-volcano (18 pairs) | drop as a client; merge into a held-out evaluation set; or keep with a pair-level split noting the tiny test set |
| X4 | Splits | per-event 70/15/15 by pair id (pre and post always together), seeded, saved like the Silo 1 splits |
| X5 | Empty images (esp. socal-fire) | keep all, or down-sample building-free pairs during training only (never in val/test) |
