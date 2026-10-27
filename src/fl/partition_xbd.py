"""
Silo 2 (xBD) event-based client partitioning. Decisions X3-X5 are recorded in docs/AUDIT_XBD.md.

- Clients are disaster events (one event = one agency's imagery), as in Silo 1.
- X3: guatemala-volcano (18 pairs) is held out ENTIRELY as an unseen-disaster test set; it never trains or validates.
- X4: inside every other event, a seeded 70/15/15 train/val/test split BY PAIR ID (a pair's pre and post image, labels
  and targets always move together), stratified on "pair has at least one building" so val/test keep the event's share
  of building-free tiles. Saved to data/splits/silo2_seed{seed}.json.
  Caveat: xBD tiles of one event are neighbouring crops of one area, so a random tile split can place adjacent tiles in
  train and test (spatial leakage; quantified in docs/AUDIT_XBD.md).
- X5: building-free pairs are down-sampled to 25% in TRAIN only (the dropped ids are recorded); val/test keep all.

The data location comes from src/fl/config.py (never stored in the split file or the repo).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from src.fl.partition import REPO_ROOT, stratified_split

SPLITS_DIR = REPO_ROOT / "data" / "splits"
HELDOUT_EVENTS = ("guatemala-volcano",)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
EMPTY_TRAIN_KEEP = 0.25
_LABEL = re.compile(r"^(?P<pair>(?P<event>.+)_\d{8})_(?P<phase>pre|post)_disaster\.json$")


def scan_pairs(split_dir: Path) -> dict[str, dict]:
    """pair_id -> {event, n_buildings_pre, n_buildings_post, n_unclassified}; only complete pairs (both phases with
    image, label and target)."""
    pairs: dict[str, dict] = {}
    for p in sorted((split_dir / "labels").glob("*_disaster.json")):
        m = _LABEL.match(p.name)
        if not m:
            raise ValueError(f"unexpected label filename {p.name}")
        feats = json.loads(p.read_text(encoding="utf-8"))["features"]["xy"]
        b = [f for f in feats if f["properties"].get("feature_type") == "building"]
        rec = pairs.setdefault(m["pair"], {"event": m["event"], "phases": set()})
        rec["phases"].add(m["phase"])
        rec[f"n_buildings_{m['phase']}"] = len(b)
        if m["phase"] == "post":
            rec["n_unclassified"] = sum(f["properties"].get("subtype") == "un-classified" for f in b)
    complete = {}
    for pid, rec in pairs.items():
        files = [split_dir / "images" / f"{pid}_{ph}_disaster.png" for ph in ("pre", "post")]
        files += [split_dir / "targets" / f"{pid}_{ph}_disaster_target.png" for ph in ("pre", "post")]
        if rec["phases"] == {"pre", "post"} and all(f.exists() for f in files):
            rec.pop("phases")
            complete[pid] = rec
    return complete


def has_buildings(rec: dict) -> bool:
    return rec.get("n_buildings_pre", 0) > 0 or rec.get("n_buildings_post", 0) > 0


def build_xbd_partition(seed: int, split_dir: Path) -> dict:
    pairs = scan_pairs(split_dir)
    rng = np.random.default_rng(seed)
    clients, heldout = {}, {}
    for event in sorted({r["event"] for r in pairs.values()}):
        ids = sorted(pid for pid, r in pairs.items() if r["event"] == event)
        if event in HELDOUT_EVENTS:
            heldout[event] = ids
            continue
        flag = np.array([int(has_buildings(pairs[i])) for i in ids])
        tr, va, te = stratified_split(flag, rng, SPLIT_FRACTIONS)
        train = [ids[i] for i in tr]
        empty = [p for p in train if not has_buildings(pairs[p])]
        keep_n = math.ceil(EMPTY_TRAIN_KEEP * len(empty))
        kept_empty = set(rng.permutation(empty)[:keep_n].tolist()) if empty else set()
        clients[event] = {
            "train": [p for p in train if has_buildings(pairs[p]) or p in kept_empty],
            "train_dropped_empty": sorted(set(empty) - kept_empty),
            "val": [ids[i] for i in va],
            "test": [ids[i] for i in te],
        }
    stats = {
        "pairs_total": len(pairs),
        "pairs_heldout": sum(len(v) for v in heldout.values()),
        "pairs_building_free": sum(not has_buildings(r) for r in pairs.values()),
        "train_empty_dropped": sum(len(c["train_dropped_empty"]) for c in clients.values()),
        "split_sizes": {e: {k: len(v) for k, v in c.items()} for e, c in clients.items()},
        "heldout_sizes": {e: len(v) for e, v in heldout.items()},
    }
    return {"seed": seed, "xbd_split": split_dir.name, "split_fractions": SPLIT_FRACTIONS,
            "heldout_events": list(HELDOUT_EVENTS), "empty_train_keep": EMPTY_TRAIN_KEEP,
            "stratified_on": "pair has >= 1 building", "stats": stats, "clients": clients, "heldout": heldout}


def split_path(seed: int) -> Path:
    return SPLITS_DIR / f"silo2_seed{seed}.json"


def save_xbd_partition(part: dict) -> Path:
    path = split_path(part["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(part, indent=1), encoding="utf-8")
    return path


def load_xbd_partition(seed: int) -> dict:
    path = split_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m src.fl.partition_xbd`")
    return json.loads(path.read_text(encoding="utf-8"))


def pair_ids(part: dict, split: str, events: list[str] | None = None) -> list[str]:
    """Pair ids for split in {'train','val','test','heldout'} across the given events (default: all)."""
    if split == "heldout":
        return [p for e, ids in part["heldout"].items() if events is None or e in events for p in ids]
    return [p for e, c in part["clients"].items() if events is None or e in events for p in c[split]]


def main() -> None:
    from src.fl.config import xbd_split_dir
    ap = argparse.ArgumentParser(description="Build and save Silo 2 (xBD) client splits.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--xbd-split", default="train")
    args = ap.parse_args()
    split_dir = xbd_split_dir(args.xbd_split)
    for seed in args.seeds:
        part = build_xbd_partition(seed, split_dir)
        print(f"seed {seed}: {len(part['clients'])} clients, {part['stats']['pairs_heldout']} held-out pairs, "
              f"{part['stats']['train_empty_dropped']} building-free train pairs dropped -> {save_xbd_partition(part)}")
    print(json.dumps(part["stats"], indent=1))


if __name__ == "__main__":
    main()
