import json

import numpy as np
import pandas as pd
import pytest

from src.fl import partition as P


@pytest.fixture(scope="module")
def part0():
    return P.build_partition(seed=0)


def test_normalize_text():
    assert P.normalize_text("RT @abc: RT @d_e: Quake HITS  http://t.co/x @fema now") == "quake hits now"
    assert P.normalize_text("Bridge down www.x.com/a\tbad") == "bridge down bad"
    assert P.normalize_text("http://t.co/x @only") == ""


def test_deduplicate_keeps_one_consistent_copy_and_drops_conflicts():
    df = pd.DataFrame({
        "tweet_id": ["1", "2", "3", "4", "5", "6"],
        "event": ["a", "a", "b", "a", "b", "a"],
        "text": ["Roads closed http://t.co/1", "RT @x: roads closed", "Pray for Nepal",
                 "pray for nepal", "unique tweet", "http://t.co/only"],
        "label": ["l0", "l0", "l1", "l2", "l3", "l0"],
        "y": [0, 0, 1, 2, 3, 0],
    })
    out, stats = P.deduplicate(df)
    assert sorted(out["tweet_id"]) == ["1", "5"]           # 1 kept over 2; 3/4 conflict; 6 empty
    assert stats["dup_groups_label_consistent"] == 1
    assert stats["dup_rows_removed_label_consistent"] == 1
    assert stats["dup_groups_label_conflicting"] == 1
    assert stats["dup_rows_removed_label_conflicting"] == 2
    assert stats["empty_after_normalization_dropped"] == 1


def test_stratified_split_rare_classes():
    y = np.array([0] * 100 + [1] * 4 + [2] * 2 + [3])
    tr, va, te = P.stratified_split(y, np.random.default_rng(0))
    assert len(set(tr) | set(va) | set(te)) == len(y)
    for cls, min_eval in ((0, 15), (1, 1)):
        assert (y[te] == cls).sum() >= min_eval and (y[va] == cls).sum() >= min_eval
    assert (y[te] == 2).sum() == 0 and (y[tr] == 3).sum() == 1   # n < 3 stays in train


def test_excluded_events_and_min_rows(part0):
    assert not set(P.EXCLUDED_EVENTS) & set(part0.clients)
    assert len(part0.clients) == 10
    for parts in part0.clients.values():
        assert sum(len(v) for v in parts.values()) >= P.MIN_CLIENT_ROWS


def test_splits_disjoint_within_and_across_clients(part0):
    seen = {}
    for client, parts in part0.clients.items():
        sets = {s: set(ids) for s, ids in parts.items()}
        assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
            and not (sets["val"] & sets["test"])
        for s, ids in sets.items():
            for tid in ids:
                assert tid not in seen, f"{tid} in {seen.get(tid)} and {(client, s)}"
                seen[tid] = (client, s)
    assert len(seen) == len(part0.df)


def test_no_near_duplicate_crosses_splits(part0):
    norm = part0.df.set_index("tweet_id")["text"].map(P.normalize_text)
    assert norm.is_unique


def test_saved_split_matches_rebuild(part0):
    path = P.split_path(0)
    if not path.exists():
        pytest.skip("split file not generated yet")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["clients"] == part0.clients
    assert saved["labels"] == list(P.LABELS)


def test_seeds_give_different_splits():
    a, b = P.build_partition(0), P.build_partition(1)
    client = sorted(a.clients)[0]
    assert a.clients[client]["test"] != b.clients[client]["test"]
