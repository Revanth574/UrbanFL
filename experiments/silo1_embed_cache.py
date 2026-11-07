"""
Encode every Silo 1 tweet once with a frozen pretrained multilingual sentence encoder and cache the result.

Output (git-ignored via `cache/`):
  data/cache/silo1_emb.npy         float32 [n_rows, 384]
  data/cache/silo1_emb_index.json  {"model": ..., "tweet_ids": [...], ...}  row i of the .npy <-> tweet_ids[i]

All rows of the retained events are encoded BEFORE deduplication, so changing dedup or split settings never
invalidates the cache. Texts are encoded as written (URLs and mentions included); normalisation is used only for
duplicate detection. Rerunning is a no-op when the cache already covers the same model and tweet ids.

Usage:  python -m experiments.silo1_embed_cache [--force]

Canonical cache. Encoding is not bit-reproducible across devices / torch builds (a CPU fp32 re-encode differs from the
CUDA fp16 original by up to 2.4e-3), so exactly one cache file is canonical: the Review 1 encoding, SHA-256
CANONICAL_EMB_SHA256. load_embeddings() refuses any other file. Every Silo 1 run records the cache it loaded
(silo1_environment), and every summary / figure refuses to combine runs whose cache hashes differ or are unrecorded
(require_single_cache). Other encodings are kept only as named backups (data/cache/backup_*), never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from src.fl.partition import REPO_ROOT, load_events

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = REPO_ROOT / "data" / "cache"
EMB_PATH = CACHE_DIR / "silo1_emb.npy"
INDEX_PATH = CACHE_DIR / "silo1_emb_index.json"
# Review 1 encoding (CUDA, fp16, batch 64, 2026-10-27). Change only deliberately, together with rerunning Silo 1.
CANONICAL_EMB_SHA256 = "e83ffb635c83db331489bf4125051fbf509ef526963d74c775cc35eeecde9914"
_LOADED: dict | None = None            # the cache this process loaded (recorded in every run config)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load_embeddings(tweet_ids: list[str]) -> np.ndarray:
    """Return cached embeddings for `tweet_ids` in that order. Fails if the cache is missing, incomplete or not the
    canonical file, or if it changed since this process first loaded it."""
    global _LOADED
    if not (EMB_PATH.exists() and INDEX_PATH.exists()):
        raise FileNotFoundError("Embedding cache missing: run `python -m experiments.silo1_embed_cache`")
    sha = _sha256(EMB_PATH)
    if sha != CANONICAL_EMB_SHA256:
        raise ValueError(f"{EMB_PATH} has SHA-256 {sha[:12]}..., not the canonical Silo 1 cache "
                         f"{CANONICAL_EMB_SHA256[:12]}... Restore the canonical file (backups live in "
                         f"data/cache/backup_*); results from another encoding are not comparable.")
    if _LOADED is not None and _LOADED["sha256"] != sha:
        raise RuntimeError("embedding cache changed while this process was running")
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    _LOADED = {"path": str(EMB_PATH.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha,
               "index_sha256": _sha256(INDEX_PATH), "model": index["model"],
               "encode_device": index.get("device"), "dtype_during_encoding": index.get("dtype_during_encoding")}
    row_of = {tid: i for i, tid in enumerate(index["tweet_ids"])}
    missing = [t for t in tweet_ids if t not in row_of]
    if missing:
        raise KeyError(f"{len(missing)} tweet ids not in embedding cache (e.g. {missing[:3]}); rebuild with --force")
    emb = np.load(EMB_PATH, mmap_mode="r")
    return np.array(emb[[row_of[t] for t in tweet_ids]], dtype=np.float32)  # writable copy, detached from mmap


def silo1_environment(device) -> dict:
    """What every Silo 1 run config records: the embedding cache this process loaded (None if it loaded none),
    torch version and the training device."""
    return {"embedding_cache": dict(_LOADED) if _LOADED else None, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "device": str(device)}


def cache_sha_of(run: dict) -> str | None:
    """Embedding-cache SHA-256 recorded in a metrics.json payload, or None if the run recorded none."""
    cache = ((run.get("config") or {}).get("environment") or {}).get("embedding_cache") or {}
    return cache.get("sha256")


def require_single_cache(runs: dict, what: str) -> str:
    """`runs` maps a name (e.g. a run directory) to its metrics.json payload. Returns the one cache hash they share;
    raises ValueError if any run recorded no hash or if the hashes differ, naming the offending runs."""
    by_sha: dict[str | None, list[str]] = {}
    for name, run in runs.items():
        by_sha.setdefault(cache_sha_of(run), []).append(str(name))
    unrecorded = by_sha.pop(None, [])
    if unrecorded:
        raise ValueError(f"{what}: {len(unrecorded)} run(s) record no embedding-cache hash (e.g. {unrecorded[:3]}); "
                         "rerun them on the canonical cache before combining them with other runs.")
    if len(by_sha) > 1:
        groups = "; ".join(f"{sha[:12]}...: {len(v)} run(s), e.g. {v[0]}" for sha, v in by_sha.items())
        raise ValueError(f"{what}: refusing to combine runs made with different embedding caches ({groups}).")
    if not by_sha:
        raise ValueError(f"{what}: no runs to combine")
    return next(iter(by_sha))


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
    emb_path, index_path = EMB_PATH, INDEX_PATH
    if EMB_PATH.exists():          # never overwrite an existing cache: a re-encode is kept only as a named backup
        tag = f"{device}_{'fp16' if device == 'cuda' else 'fp32'}_{time.strftime('%Y%m%d')}"
        emb_path, index_path = CACHE_DIR / f"backup_{tag}_silo1_emb.npy", CACHE_DIR / f"backup_{tag}_silo1_emb_index.json"
    np.save(emb_path, emb)
    if _sha256(emb_path) != CANONICAL_EMB_SHA256:
        print(f"WARNING: {emb_path.name} is not the canonical cache; load_embeddings() will not use it.")
    index_path.write_text(json.dumps({
        "model": MODEL_NAME, "dim": int(emb.shape[1]), "n_rows": int(emb.shape[0]),
        "device": device, "dtype_during_encoding": "float16" if device == "cuda" else "float32",
        "batch_size": args.batch_size, "encode_seconds": round(runtime, 2),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
        "tweet_ids": tweet_ids,
    }), encoding="utf-8")
    print(f"Encoded {emb.shape} on {device} in {runtime:.1f}s -> {emb_path}")


if __name__ == "__main__":
    main()
