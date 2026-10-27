"""
Silo 1 event-based client partitioning for CrisisNLP CrowdFlower v2.

Clients are disaster events (one TSV folder = one agency's data). The label is the humanitarian
information type, never the disaster type. Decisions D1-D5 are recorded in docs/AUDIT.md section 5.

Pipeline (deterministic given `seed`):
  1. Load raw TSVs, drop epidemic events with a different taxonomy (D1).
  2. Map labels onto the shared 9-label set (identity mapping for the 10 humanitarian events; documented in
     LABEL_MAP so it is explicit and checked).
  3. Normalise text and remove near-duplicates GLOBALLY (D3): label-consistent groups keep one copy,
     label-conflicting groups are dropped entirely. Grouping is global, not per event, so a tweet can never
     sit in one client's train set and another client's test set.
  4. Drop clients with < MIN_CLIENT_ROWS rows.
  5. Per client, stratified 70/15/15 train/val/test split.

Rows are identified by `tweet_id` everywhere (splits, embedding cache), never by positional index.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CRISISNLP_ROOT = (REPO_ROOT / "data" / "real" / "CrisisNLP_labeled_data_crowdflower_v2"
                  / "CrisisNLP_labeled_data_crowdflower")
SPLITS_DIR = REPO_ROOT / "data" / "splits"

# D1: epidemic events use a disease-specific 8-label taxonomy that does not overlap the humanitarian set.
EXCLUDED_EVENTS = ("2014_ebola_cf", "2014_Middle_East_Respiratory_Syndrome_en")

# Explicit, ordered label list. Index in this tuple == integer class id used by every model and metric.
LABELS = (
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "donation_needs_or_offers_or_volunteering_services",
    "infrastructure_and_utilities_damage",
    "injured_or_dead_people",
    "missing_trapped_or_found_people",
    "not_related_or_irrelevant",
    "other_useful_information",
    "sympathy_and_emotional_support",
)
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

# Raw CrowdFlower label -> shared label. All 10 humanitarian events already use these exact strings, so the
# mapping is the identity; any raw label not in this map makes load_events() fail loudly instead of guessing.
LABEL_MAP = {name: name for name in LABELS}

# Short names for tables and figures only.
LABEL_SHORT = ("caution", "displaced", "donation", "infra_damage", "injured_dead",
               "missing", "irrelevant", "other_info", "sympathy")

MIN_CLIENT_ROWS = 200
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

_RT_PREFIX = re.compile(r"^\s*rt\s+@\w+:?\s*", re.IGNORECASE)
_URL = re.compile(r"(?:https?://|www\.)\S*", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """D3 normalisation: strip (repeated) leading 'RT @user:', URLs, @mentions; lowercase; collapse whitespace."""
    t = text
    while True:
        stripped = _RT_PREFIX.sub("", t, count=1)
        if stripped == t:
            break
        t = stripped
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    return _WHITESPACE.sub(" ", t.lower()).strip()


def load_events(root: Path = CRISISNLP_ROOT) -> tuple[pd.DataFrame, dict]:
    """Load all humanitarian-taxonomy events. Returns (df[tweet_id, event, text, label, y], stats)."""
    frames = []
    for tsv in sorted(root.glob("*/*.tsv")):
        df = pd.read_csv(tsv, sep="\t", dtype=str, quoting=3, encoding="utf-8")
        df["event"] = tsv.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CrisisNLP TSVs under {root}")
    raw = pd.concat(frames, ignore_index=True)
    raw["tweet_id"] = raw["tweet_id"].str.strip("'")

    excluded = raw["event"].isin(EXCLUDED_EVENTS)
    df = raw.loc[~excluded].copy()
    unknown = sorted(set(df["label"]) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"Labels outside LABEL_MAP: {unknown}")
    df["label"] = df["label"].map(LABEL_MAP)
    df["y"] = df["label"].map(LABEL_TO_ID).astype(int)
    df = df.rename(columns={"tweet_text": "text"})[["tweet_id", "event", "text", "label", "y"]]
    if df["tweet_id"].duplicated().any():
        raise ValueError("Duplicate tweet_ids: rows would be ambiguous in splits and cache")

    stats = {"raw_rows": int(len(raw)), "excluded_events": list(EXCLUDED_EVENTS),
             "excluded_event_rows": int(excluded.sum())}
    return df.reset_index(drop=True), stats


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    D3 near-duplicate removal over normalised text, across all events.
    Rows whose normalised text is empty (URL/mention-only tweets) carry no text signal and would all collapse
    into one group, so they are dropped and counted separately.
    """
    df = df.copy()
    df["norm"] = df["text"].map(normalize_text)
    empty = df["norm"] == ""
    df = df.loc[~empty]

    # Deterministic keeper: first row by (event, tweet_id).
    df = df.sort_values(["event", "tweet_id"], kind="mergesort")
    group_size = df.groupby("norm")["norm"].transform("size")
    n_labels = df.groupby("norm")["y"].transform("nunique")

    conflicting = n_labels > 1
    dup_consistent = (group_size > 1) & ~conflicting
    keep = ~conflicting & ~(dup_consistent & df.duplicated("norm", keep="first"))

    stats = {
        "empty_after_normalization_dropped": int(empty.sum()),
        "dup_groups_label_consistent": int(df.loc[dup_consistent, "norm"].nunique()),
        "dup_rows_removed_label_consistent": int((dup_consistent & ~keep).sum()),
        "dup_groups_label_conflicting": int(df.loc[conflicting, "norm"].nunique()),
        "dup_rows_removed_label_conflicting": int(conflicting.sum()),
        "dup_groups_cross_event": int((df.loc[group_size > 1].groupby("norm")["event"].nunique() > 1).sum()),
    }
    return df.loc[keep].drop(columns="norm").reset_index(drop=True), stats


def stratified_split(y: np.ndarray, rng: np.random.Generator,
                     fractions: tuple[float, float, float] = SPLIT_FRACTIONS) -> tuple[np.ndarray, ...]:
    """
    Per-class split into (train, val, test) positions. Each class with n >= 3 gets at least one val and one test
    example; rare classes are never all pushed into test. Written by hand because sklearn's stratified splitter
    refuses classes with < 2 members.
    """
    _, f_val, f_test = fractions
    train, val, test = [], [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n = len(idx)
        n_test = int(round(n * f_test))
        n_val = int(round(n * f_val))
        if n >= 3:
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        else:
            n_test, n_val = 0, 0
        test.extend(idx[:n_test])
        val.extend(idx[n_test:n_test + n_val])
        train.extend(idx[n_test + n_val:])
    return tuple(np.sort(np.asarray(part, dtype=int)) for part in (train, val, test))


@dataclass
class Partition:
    seed: int
    df: pd.DataFrame                      # deduplicated rows of retained clients
    clients: dict[str, dict[str, list]]   # event -> {"train"|"val"|"test": [tweet_id, ...]}
    stats: dict

    def rows(self, client: str, split: str) -> pd.DataFrame:
        ids = self.clients[client][split]
        return self.df.set_index("tweet_id").loc[ids].reset_index()

    def split_sizes(self) -> dict:
        return {c: {s: len(ids) for s, ids in parts.items()} for c, parts in self.clients.items()}


def build_partition(seed: int, root: Path = CRISISNLP_ROOT) -> Partition:
    df, load_stats = load_events(root)
    df, dedup_stats = deduplicate(df)

    counts = df["event"].value_counts()
    small = sorted(counts[counts < MIN_CLIENT_ROWS].index)
    df = df.loc[~df["event"].isin(small)].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    clients = {}
    for event in sorted(df["event"].unique()):
        sub = df.loc[df["event"] == event].reset_index(drop=True)
        tr, va, te = stratified_split(sub["y"].to_numpy(), rng)
        clients[event] = {name: sub.loc[pos, "tweet_id"].tolist()
                          for name, pos in (("train", tr), ("val", va), ("test", te))}

    stats = {**load_stats, **dedup_stats, "events_dropped_min_rows": small,
             "min_client_rows": MIN_CLIENT_ROWS, "final_rows": int(len(df)), "n_clients": len(clients)}
    return Partition(seed=seed, df=df, clients=clients, stats=stats)


def split_path(seed: int) -> Path:
    return SPLITS_DIR / f"silo1_seed{seed}.json"


def save_partition(part: Partition) -> Path:
    path = split_path(part.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": part.seed, "labels": list(LABELS), "label_map": LABEL_MAP,
               "split_fractions": SPLIT_FRACTIONS, "stats": part.stats,
               "split_sizes": part.split_sizes(), "clients": part.clients}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_partition(seed: int, root: Path = CRISISNLP_ROOT) -> Partition:
    """Load the saved split for `seed` (creating it on first use) and re-attach texts/labels from the TSVs."""
    path = split_path(seed)
    if not path.exists():
        save_partition(build_partition(seed, root))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload["labels"]) != LABELS:
        raise ValueError(f"{path} was written with a different label list")
    df, _ = load_events(root)
    ids = [tid for parts in payload["clients"].values() for s in parts.values() for tid in s]
    df = df.set_index("tweet_id").loc[ids].reset_index()
    return Partition(seed=seed, df=df, clients=payload["clients"], stats=payload["stats"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and save Silo 1 client splits.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    for seed in args.seeds:
        part = build_partition(seed)
        path = save_partition(part)
        print(f"seed {seed}: {part.stats['n_clients']} clients, {part.stats['final_rows']} rows -> {path}")
    print(json.dumps(part.stats, indent=1))


if __name__ == "__main__":
    main()
