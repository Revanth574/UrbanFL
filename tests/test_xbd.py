"""Silo 2 (xBD): partition (X3-X5), two-stage targets (X1, X2), augmentation, xView2 scorer, real-data checks."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score

from src.fl import partition_xbd as PX
from src.fl.xbd_data import (IGNORE_INDEX, XBDPairDataset, damage_target, loc_target, random_crop_flip,
                             rasterize_unclassified)
from src.fl.xbd_metrics import XView2Scorer

SIZE = 64
SUBTYPE_VALUE = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4, "un-classified": 0}


def _square(x, y, s=10):
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y)]


def _wkt(pts):
    return "POLYGON ((" + ", ".join(f"{a} {b}" for a, b in pts) + "))"


def make_synthetic_xbd(root: Path, events: dict[str, tuple[int, int]]) -> Path:
    """events: name -> (pairs with buildings, building-free pairs). Each building pair has 4 classified + 1
    un-classified building."""
    split = root / "train"
    for d in ("images", "labels", "targets"):
        (split / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for ev, (n_b, n_e) in events.items():
        for k in range(n_b + n_e):
            pid = f"{ev}_{k:08d}"
            feats = []
            if k < n_b:
                for j, sub in enumerate(["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]):
                    feats.append((sub, _square(2 + 12 * j, 20)))
            for phase in ("pre", "post"):
                Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)).save(
                    split / "images" / f"{pid}_{phase}_disaster.png")
                tgt = Image.new("L", (SIZE, SIZE), 0)
                draw = ImageDraw.Draw(tgt)
                xy = []
                for sub, pts in feats:
                    draw.polygon(pts, fill=1 if phase == "pre" else SUBTYPE_VALUE[sub])
                    props = {"feature_type": "building", "uid": f"{pid}-{sub}"}
                    if phase == "post":
                        props["subtype"] = sub
                    xy.append({"properties": props, "wkt": _wkt(pts)})
                tgt.save(split / "targets" / f"{pid}_{phase}_disaster_target.png")
                (split / "labels" / f"{pid}_{phase}_disaster.json").write_text(
                    json.dumps({"features": {"xy": xy}, "metadata": {"width": SIZE, "height": SIZE}}), encoding="utf-8")
    return split


@pytest.fixture(scope="module")
def xbd(tmp_path_factory):
    root = tmp_path_factory.mktemp("xbd")
    return make_synthetic_xbd(root, {"ev-a": (16, 12), "ev-b": (14, 6), "guatemala-volcano": (3, 1)})


def test_scan_pairs(xbd):
    pairs = PX.scan_pairs(xbd)
    assert len(pairs) == 52
    assert sum(PX.has_buildings(r) for r in pairs.values()) == 33
    assert pairs["ev-a_00000000"]["n_unclassified"] == 1


def test_partition_heldout_disjoint_and_complete(xbd):
    part = PX.build_xbd_partition(0, xbd)
    assert set(part["heldout"]) == {"guatemala-volcano"} and len(part["heldout"]["guatemala-volcano"]) == 4
    assert "guatemala-volcano" not in part["clients"]
    pairs = PX.scan_pairs(xbd)
    for ev, c in part["clients"].items():
        groups = [set(c[k]) for k in ("train", "train_dropped_empty", "val", "test")]
        assert sum(len(g) for g in groups) == len(set().union(*groups))           # pairwise disjoint
        assert set().union(*groups) == {p for p, r in pairs.items() if r["event"] == ev}


def test_empty_downsampling_train_only(xbd):
    part = PX.build_xbd_partition(0, xbd)
    pairs = PX.scan_pairs(xbd)
    for ev, c in part["clients"].items():
        empty_train_all = [p for p in c["train"] + c["train_dropped_empty"] if not PX.has_buildings(pairs[p])]
        kept = [p for p in c["train"] if not PX.has_buildings(pairs[p])]
        assert len(kept) == int(np.ceil(0.25 * len(empty_train_all)))
        assert all(not PX.has_buildings(pairs[p]) for p in c["train_dropped_empty"])
    # val/test keep building-free pairs (stratified: ev-a has 12 empties -> at least one in val and in test)
    c = part["clients"]["ev-a"]
    assert any(not PX.has_buildings(pairs[p]) for p in c["val"])
    assert any(not PX.has_buildings(pairs[p]) for p in c["test"])


def test_seeds_differ(xbd):
    a, b = PX.build_xbd_partition(0, xbd), PX.build_xbd_partition(1, xbd)
    assert a["clients"]["ev-a"]["test"] != b["clients"]["ev-a"]["test"]


def test_loc_target_counts_unclassified_as_building(xbd):
    ds = XBDPairDataset(xbd, ["ev-a_00000000"], "loc", train=False)
    item = ds[0]
    assert item["pre"].shape == (3, SIZE, SIZE) and "post" not in item
    pre_t = np.asarray(Image.open(xbd / "targets" / "ev-a_00000000_pre_disaster_target.png"))
    assert np.array_equal(item["target"].numpy(), (pre_t > 0).astype(np.int64))
    assert item["target"][25, 4 * 12 + 7] == 1                               # inside the un-classified square


def test_damage_target_mapping_and_ignore(xbd):
    item = XBDPairDataset(xbd, ["ev-a_00000000"], "damage", train=False)[0]
    t = item["target"].numpy()
    for j, cls in enumerate([0, 1, 2, 3]):                                    # classified squares -> class 0..3
        assert t[25, 2 + 12 * j + 5] == cls
    assert t[25, 2 + 12 * 4 + 5] == IGNORE_INDEX                              # un-classified -> ignored
    assert t[5, 5] == IGNORE_INDEX                                            # background -> ignored
    assert item["post"].shape == (3, SIZE, SIZE)
    assert np.array_equal(item["building"].numpy(), t != IGNORE_INDEX)


def test_unit_target_helpers():
    post = np.array([[0, 1, 2], [3, 4, 0]])
    uncl = np.array([[False, False, True], [False, False, False]])
    assert damage_target(post, uncl).tolist() == [[255, 0, 255], [2, 3, 255]]
    assert loc_target(np.array([[0, 1], [1, 0]])).tolist() == [[0, 1], [1, 0]]


def test_rasterize_unclassified_matches_polygon(xbd):
    m = rasterize_unclassified(xbd / "labels" / "ev-a_00000000_post_disaster.json", (SIZE, SIZE))
    assert m[25, 55] and not m[25, 5] and m.sum() > 80


def test_augmentation_identical_across_arrays_and_reproducible(xbd):
    a = np.arange(SIZE * SIZE * 3).reshape(SIZE, SIZE, 3)
    out = random_crop_flip([a, a.copy(), a[..., 0].copy()], 32, np.random.default_rng(3))
    assert out[0].shape == (32, 32, 3) and np.array_equal(out[0], out[1]) and np.array_equal(out[0][..., 0], out[2])
    ds = XBDPairDataset(xbd, ["ev-a_00000001"], "damage", train=True, crop=32, seed=5)
    e0a, e0b = ds[0], ds[0]
    assert e0a["target"].shape == (32, 32) and e0a["pre"].shape == (3, 32, 32)
    assert np.array_equal(e0a["pre"], e0b["pre"])                             # same (seed, epoch, index) -> same
    ds.set_epoch(1)
    assert not np.array_equal(e0a["pre"], ds[0]["pre"])                       # new epoch -> new augmentation


def test_xview2_score_matches_sklearn():
    rng = np.random.default_rng(0)
    loc_gt, loc_pred = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    dmg_gt = rng.integers(0, 4, 500)
    dmg_gt[rng.random(500) < 0.3] = IGNORE_INDEX
    dmg_pred = rng.integers(0, 4, 500)
    s = XView2Scorer()
    s.update_loc(loc_pred[:200], loc_gt[:200]); s.update_loc(loc_pred[200:], loc_gt[200:])   # accumulates
    s.update_damage(dmg_pred[:300], dmg_gt[:300]); s.update_damage(dmg_pred[300:], dmg_gt[300:])
    r = s.compute()
    assert r["loc_f1"] == pytest.approx(f1_score(loc_gt, loc_pred))
    v = dmg_gt != IGNORE_INDEX
    per = f1_score(dmg_gt[v], dmg_pred[v], labels=[0, 1, 2, 3], average=None)
    assert list(r["damage_f1"].values()) == pytest.approx(list(per))
    harm = 4 / sum(1 / (f + 1e-6) for f in per)
    assert r["score"] == pytest.approx(0.3 * r["loc_f1"] + 0.7 * harm)


# ── real data (skipped unless the xBD path is configured and the split files exist) ─────────────────────────

def _real():
    from src.fl.config import ConfigError, xbd_split_dir
    try:
        d = xbd_split_dir("train")
    except ConfigError:
        pytest.skip("xBD path not configured")
    if not PX.split_path(0).exists():
        pytest.skip("silo2 split files not generated")
    return d


def test_real_split_file_consistent_with_disk():
    d = _real()
    part = PX.load_xbd_partition(0)
    assert part["heldout_events"] == ["guatemala-volcano"] and len(part["heldout"]["guatemala-volcano"]) == 18
    ids = [p for c in part["clients"].values() for k in ("train", "train_dropped_empty", "val", "test") for p in c[k]]
    ids += part["heldout"]["guatemala-volcano"]
    assert len(ids) == len(set(ids)) == 2799
    for pid in ids[::97]:
        assert (d / "images" / f"{pid}_post_disaster.png").exists()


def test_real_unclassified_pixels_are_ignored_in_damage_but_buildings_in_loc():
    d = _real()
    part = PX.load_xbd_partition(0)
    pid = next(p for p in part["heldout"]["guatemala-volcano"]
               if '"un-classified"' in (d / "labels" / f"{p}_post_disaster.json").read_text(encoding="utf-8"))
    uncl = rasterize_unclassified(d / "labels" / f"{pid}_post_disaster.json", (1024, 1024))
    dmg = XBDPairDataset(d, [pid], "damage")[0]["target"].numpy()
    loc = XBDPairDataset(d, [pid], "loc")[0]["target"].numpy()
    assert (dmg[uncl] == IGNORE_INDEX).all()
    assert loc[uncl].mean() > 0.9                                             # pre target paints them as buildings
