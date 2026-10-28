"""
Silo 2 (xBD) event-based client partitioning. Decisions X3-X5 are recorded in docs/AUDIT_XBD.md.

- Clients are disaster events (one event = one agency's imagery), as in Silo 1.
- X3: guatemala-volcano (18 pairs) is held out ENTIRELY as an unseen-disaster test set; it never trains or validates.
- X4: inside every other event, a seeded 70/15/15 train/val/test split. A pair's pre and post image, labels and targets
  always move together. Two strategies:
    "spatial_block" (DEFAULT) -> data/splits/silo2_seed{seed}.json
        Tiles are linked when their centres are within LINK_TILES tile widths (Chebyshev distance, from the per-image
        geotransforms); the connected components of that graph are spatial blocks, and components too large for the
        test target are cut into grid cells. Whole blocks go to test; every other tile within LINK_TILES of a test
        tile goes to val (a buffer ring); more whole blocks fill val; the rest is train (≈ 70/15/15 by tile count).
        Hence no test tile is adjacent to a train tile. Fractions are approximate: blocks are indivisible.
    "random_pair" -> data/splits/silo2_random_pair_seed{seed}.json
        The Review 1 split: random by pair id, stratified on "pair has at least one building". 58% of its test tiles
        touch a train tile (spatial leakage, docs/AUDIT_XBD.md).
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
GEOTRANSFORMS = REPO_ROOT / "data" / "real" / "xbd_geotransforms.json"
HELDOUT_EVENTS = ("guatemala-volcano",)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
EMPTY_TRAIN_KEEP = 0.25
STRATEGIES = ("spatial_block", "random_pair")
LINK_TILES = 1.5        # tiles closer than this (in tile widths) share a block: margin beyond plain adjacency
ADJACENT_TILES = 1.05   # "adjacent" for the reported leakage rate (edge or corner contact), as in AUDIT_XBD.md
BLOCK_TOLERANCE = 0.15  # accept a split's tile count within ±15% of its target when packing whole blocks
SUBBLOCK_TILES = 8      # components larger than the test target are cut into 8 x 8-tile grid cells
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


def load_geotransforms(path: Path = GEOTRANSFORMS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tile_centres(ids: list[str], geotransforms: dict) -> np.ndarray:
    """Tile centres in units of tile widths (per-axis median tile size of the event), from the pre-image geotransform."""
    if not ids:
        return np.empty((0, 2))
    missing = [p for p in ids if f"{p}_pre_disaster.png" not in geotransforms]
    if missing:
        raise KeyError(f"no geotransform for {len(missing)} pairs (e.g. {missing[:3]})")
    t = np.array([geotransforms[f"{p}_pre_disaster.png"][0] for p in ids], dtype=float)   # GDAL 6-tuple
    w, h = np.median(np.abs(t[:, 1])) * 1024, np.median(np.abs(t[:, 5])) * 1024
    return np.c_[(t[:, 0] + 512 * t[:, 1]) / w, (t[:, 3] + 512 * t[:, 5]) / h]


def chebyshev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2)


def spatial_blocks(xy: np.ndarray, link: float = LINK_TILES) -> np.ndarray:
    """Connected components of the graph 'centres within link tile widths'. Returns a block label per tile."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    _, labels = connected_components(csr_matrix(chebyshev(xy, xy) <= link), directed=False)
    return labels


def adjacency_rate(xy_test: np.ndarray, xy_other: np.ndarray, adjacent: float = ADJACENT_TILES) -> float | None:
    """Fraction of test tiles with at least one other-split tile within `adjacent` tile widths."""
    if len(xy_test) == 0:
        return None
    if len(xy_other) == 0:
        return 0.0
    return float((chebyshev(xy_test, xy_other).min(axis=1) <= adjacent).mean())


def pack_blocks(sizes: dict[int, int], target: float, rng: np.random.Generator) -> list[int]:
    """Greedy: visit blocks in random order, take a block if it keeps the total <= target*(1+tol); stop once the total
    reaches target*(1-tol). Returns the chosen block labels (may undershoot when blocks are large)."""
    chosen, total = [], 0
    for b in rng.permutation(sorted(sizes)).tolist():
        if total >= target * (1 - BLOCK_TOLERANCE):
            break
        if total + sizes[b] <= target * (1 + BLOCK_TOLERANCE):
            chosen.append(b)
            total += sizes[b]
    return chosen


def subdivide_large_blocks(xy: np.ndarray, labels: np.ndarray, max_size: float, rng: np.random.Generator) -> np.ndarray:
    """Split every block with more than max_size tiles into SUBBLOCK_TILES x SUBBLOCK_TILES grid cells (random grid
    offset). Neighbouring cells of one block DO touch; split_event_spatial buffers test cells with a val ring."""
    out, next_label = labels.copy(), labels.max() + 1
    for b in np.unique(labels):
        members = np.flatnonzero(labels == b)
        if len(members) <= max_size:
            continue
        origin = xy[members].min(axis=0) - rng.uniform(0, SUBBLOCK_TILES, size=2)
        cells = np.floor((xy[members] - origin) / SUBBLOCK_TILES).astype(int)
        for cell in np.unique(cells, axis=0):
            out[members[(cells == cell).all(axis=1)]] = next_label
            next_label += 1
    return out


def split_event_spatial(ids: list[str], geotransforms: dict, rng: np.random.Generator) -> tuple[list, list, list, dict]:
    """
    1. blocks = connected components (link LINK_TILES); components too big for the test target are cut into grid cells;
    2. whole blocks -> test (≈ 15% of tiles);
    3. every non-test tile within LINK_TILES of a test tile -> val (buffer ring), so no train tile can touch test;
    4. further whole blocks -> val until ≈ 15%; everything else -> train.
    """
    n = len(ids)
    _, f_val, f_test = SPLIT_FRACTIONS
    xy = tile_centres(ids, geotransforms)
    components = spatial_blocks(xy)
    labels = subdivide_large_blocks(xy, components, f_test * n, rng)
    sizes = {int(b): int(c) for b, c in zip(*np.unique(labels, return_counts=True))}

    test_blocks = set(pack_blocks(sizes, f_test * n, rng))
    is_test = np.isin(labels, list(test_blocks))
    near_test = (chebyshev(xy, xy[is_test]).min(axis=1) <= LINK_TILES) if is_test.any() else np.zeros(n, bool)
    is_ring = near_test & ~is_test

    remaining = {b: int(((labels == b) & ~is_test & ~is_ring).sum()) for b in sizes if b not in test_blocks}
    remaining = {b: c for b, c in remaining.items() if c > 0}
    val_blocks = set(pack_blocks(remaining, max(0.0, f_val * n - is_ring.sum()), rng))
    is_val = is_ring | (np.isin(labels, list(val_blocks)) & ~is_test)

    idx = {"test": np.flatnonzero(is_test).tolist(), "val": np.flatnonzero(is_val).tolist(),
           "train": np.flatnonzero(~is_test & ~is_val).tolist()}
    info = {"n_components": int(components.max() + 1), "largest_component": int(np.bincount(components).max()),
            "n_blocks_after_subdivision": len(sizes), "largest_block": max(sizes.values()),
            "blocks": {"test": len(test_blocks), "val_packed": len(val_blocks)}, "val_ring_tiles": int(is_ring.sum()),
            "adjacency_test_to_train": adjacency_rate(xy[idx["test"]], xy[idx["train"]]),
            "adjacency_test_to_val": adjacency_rate(xy[idx["test"]], xy[idx["val"]])}
    return idx["train"], idx["val"], idx["test"], info


def build_xbd_partition(seed: int, split_dir: Path, strategy: str = "spatial_block",
                        geotransforms: dict | None = None) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    pairs = scan_pairs(split_dir)
    if strategy == "spatial_block" and geotransforms is None:
        geotransforms = load_geotransforms()
    rng = np.random.default_rng(seed)
    clients, heldout, block_info = {}, {}, {}
    for event in sorted({r["event"] for r in pairs.values()}):
        ids = sorted(pid for pid, r in pairs.items() if r["event"] == event)
        if event in HELDOUT_EVENTS:
            heldout[event] = ids
            continue
        if strategy == "random_pair":
            flag = np.array([int(has_buildings(pairs[i])) for i in ids])
            tr, va, te = stratified_split(flag, rng, SPLIT_FRACTIONS)
        else:
            tr, va, te, block_info[event] = split_event_spatial(ids, geotransforms, rng)
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
    if strategy == "spatial_block":
        stats["blocks"] = block_info
        n_test = {e: len(c["test"]) for e, c in clients.items()}
        stats["adjacency_test_to_train_overall"] = (
            sum(block_info[e]["adjacency_test_to_train"] * n for e, n in n_test.items() if n)
            / max(1, sum(n_test.values())))
    extra = ({"stratified_on": "pair has >= 1 building"} if strategy == "random_pair" else
             {"block_link_tiles": LINK_TILES, "adjacent_tiles": ADJACENT_TILES, "block_tolerance": BLOCK_TOLERANCE})
    return {"seed": seed, "strategy": strategy, "xbd_split": split_dir.name, "split_fractions": SPLIT_FRACTIONS,
            "heldout_events": list(HELDOUT_EVENTS), "empty_train_keep": EMPTY_TRAIN_KEEP, **extra,
            "stats": stats, "clients": clients, "heldout": heldout}


def split_path(seed: int, strategy: str = "spatial_block") -> Path:
    name = f"silo2_seed{seed}.json" if strategy == "spatial_block" else f"silo2_{strategy}_seed{seed}.json"
    return SPLITS_DIR / name


def save_xbd_partition(part: dict) -> Path:
    path = split_path(part["seed"], part.get("strategy", "random_pair"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(part, indent=1), encoding="utf-8")
    return path


def load_xbd_partition(seed: int, strategy: str = "spatial_block") -> dict:
    path = split_path(seed, strategy)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m src.fl.partition_xbd --strategy {strategy}`")
    part = json.loads(path.read_text(encoding="utf-8"))
    if part.get("strategy", "random_pair") != strategy:
        raise ValueError(f"{path} holds strategy {part.get('strategy')!r}, expected {strategy!r}")
    return part


def pair_ids(part: dict, split: str, events: list[str] | None = None) -> list[str]:
    """Pair ids for split in {'train','val','test','heldout'} across the given events (default: all)."""
    if split == "heldout":
        return [p for e, ids in part["heldout"].items() if events is None or e in events for p in ids]
    return [p for e, c in part["clients"].items() if events is None or e in events for p in c[split]]


def main() -> None:
    from src.fl.config import xbd_split_dir
    ap = argparse.ArgumentParser(description="Build and save Silo 2 (xBD) client splits.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--strategy", default="spatial_block", choices=STRATEGIES)
    ap.add_argument("--xbd-split", default="train")
    args = ap.parse_args()
    split_dir = xbd_split_dir(args.xbd_split)
    for seed in args.seeds:
        part = build_xbd_partition(seed, split_dir, args.strategy)
        adj = part["stats"].get("adjacency_test_to_train_overall")
        msg = f"seed {seed} [{args.strategy}]: {len(part['clients'])} clients, {part['stats']['pairs_heldout']} held-out"
        msg += f" pairs, {part['stats']['train_empty_dropped']} building-free train pairs dropped"
        msg += "" if adj is None else f", test-train adjacency {adj:.1%}"
        print(f"{msg} -> {save_xbd_partition(part)}")
    print(json.dumps({k: v for k, v in part["stats"].items() if k != "blocks"}, indent=1))


if __name__ == "__main__":
    main()
