"""
Encode every Silo 1 tweet once with a frozen pretrained multilingual sentence encoder and cache the result.

Output (git-ignored via `cache/`):
  data/cache/silo1_emb.npy         float32 [n_rows, 384]
  data/cache/silo1_emb_index.json  {"model": ..., "tweet_ids": [...], ...}  row i of the .npy <-> tweet_ids[i]

All rows of the retained events are encoded BEFORE deduplication, so changing dedup or split settings never
invalidates the cache. Texts are encoded as written (URLs and mentions included); normalisation is used only for
duplicate detection. Rerunning is a no-op when the cache already covers the same model and tweet ids.

Usage:  python -m experiments.silo1_embed_cache [--force]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from src.fl.partition import REPO_ROOT, load_events

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = REPO_ROOT / "data" / "cache"
EMB_PATH = CACHE_DIR / "silo1_emb.npy"
INDEX_PATH = CACHE_DIR / "silo1_emb_index.json"


def load_embeddings(tweet_ids: list[str]) -> np.ndarray:
    """Return cached embeddings for `tweet_ids` in that order. Fails if the cache is missing or incomplete."""
    if not (EMB_PATH.exists() and INDEX_PATH.exists()):
        raise FileNotFoundError("Embedding cache missing: run `python -m experiments.silo1_embed_cache`")
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    row_of = {tid: i for i, tid in enumerate(index["tweet_ids"])}
    missing = [t for t in tweet_ids if t not in row_of]
    if missing:
        raise KeyError(f"{len(missing)} tweet ids not in embedding cache (e.g. {missing[:3]}); rebuild with --force")
    emb = np.load(EMB_PATH, mmap_mode="r")
    return np.array(emb[[row_of[t] for t in tweet_ids]], dtype=np.float32)  # writable copy, detached from mmap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--force", action="store_true", help="re-encode even if a matching cache exists")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    df, _ = load_events()
    tweet_ids = df["tweet_id"].tolist()

    if not args.force and EMB_PATH.exists() and INDEX_PATH.exists():
        index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if index["model"] == MODEL_NAME and index["tweet_ids"] == tweet_ids:
            print(f"Cache up to date: {EMB_PATH} ({len(tweet_ids)} rows). Use --force to rebuild.")
            return

    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "cuda":
        model = model.half()

    t0 = time.perf_counter()
    emb = model.encode(df["text"].tolist(), batch_size=args.batch_size, convert_to_numpy=True,
                       show_progress_bar=True, normalize_embeddings=False)
    runtime = time.perf_counter() - t0
    emb = emb.astype(np.float32)
    if not np.isfinite(emb).all():
        raise FloatingPointError("Non-finite values in fp16 embeddings")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)
    INDEX_PATH.write_text(json.dumps({
        "model": MODEL_NAME, "dim": int(emb.shape[1]), "n_rows": int(emb.shape[0]),
        "device": device, "dtype_during_encoding": "float16" if device == "cuda" else "float32",
        "batch_size": args.batch_size, "encode_seconds": round(runtime, 2),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
        "tweet_ids": tweet_ids,
    }), encoding="utf-8")
    print(f"Encoded {emb.shape} on {device} in {runtime:.1f}s -> {EMB_PATH}")


if __name__ == "__main__":
    main()
