"""
Checkpoint / resume for long training runs (local GPU or Colab), so an interruption costs at most N steps.

A checkpoint holds everything needed to continue EXACTLY where training stopped: model, optimizer, AMP GradScaler,
LR scheduler (if any), epoch, step within the epoch, global step, and the python / numpy / torch (CPU + CUDA) RNG
states. Mid-epoch resume is exact because the trainer's sample order and augmentations are functions of
(seed, epoch, index) rather than of a running RNG (see src/fl/seg_train.py).

Writes are atomic (temporary file + os.replace), the newest `keep_last` checkpoints are kept, and a `signature` (run
config) stored in every checkpoint must match on resume, so a run never silently resumes from a different setup.

Directory resolution (resolve_checkpoint_dir): explicit argument > env URBANFL_CHECKPOINT_DIR > "checkpoint_dir" in
the user config (~/.urbanfl/paths.json) > <repo>/checkpoints (git-ignored); the run name is appended.
On Colab, point it at Google Drive, e.g. URBANFL_CHECKPOINT_DIR=/content/drive/MyDrive/urbanfl_ckpt.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

from src.fl.config import ConfigError, data_path
from src.fl.utils import REPO_ROOT

_NAME = re.compile(r"^ckpt_step(\d{9})\.pt$")


def resolve_checkpoint_dir(explicit: str | Path | None, run_name: str) -> Path:
    if explicit:
        base = Path(explicit)
    else:
        try:
            base = data_path("checkpoint_dir", must_exist=False)
        except ConfigError:
            base = REPO_ROOT / "checkpoints"
    return base.expanduser() / run_name


def save_atomic(obj, path: str | Path) -> Path:
    """torch.save via a temporary file + os.replace (e.g. best-model weights next to the checkpoints)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(self, directory: str | Path, every_n_steps: int, keep_last: int = 2, signature: dict | None = None):
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        self.dir = Path(directory)
        self.every = every_n_steps
        self.keep_last = keep_last
        self.signature = json.loads(json.dumps(signature or {}, default=str))   # normalised, JSON-comparable

    def due(self, global_step: int) -> bool:
        return global_step % self.every == 0

    def checkpoints(self) -> list[Path]:
        if not self.dir.exists():
            return []
        return sorted((p for p in self.dir.iterdir() if _NAME.match(p.name)), key=lambda p: p.name)

    def latest(self) -> Path | None:
        found = self.checkpoints()
        return found[-1] if found else None

    def save(self, state: dict, global_step: int) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"ckpt_step{global_step:09d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({**state, "global_step": global_step, "signature": self.signature}, tmp)
        os.replace(tmp, path)                                   # atomic: a crash never leaves a half-written ckpt
        for old in self.checkpoints()[:-self.keep_last]:
            old.unlink()
        return path

    def prune_resume_checkpoints(self) -> int:
        """Delete every resume checkpoint (ckpt_step*.pt, *.tmp) once a run is complete; other files such as
        best_model.pt and done.json stay. Returns the number of bytes freed."""
        freed = 0
        if self.dir.exists():
            for p in list(self.dir.iterdir()):
                if _NAME.match(p.name) or p.name.endswith(".pt.tmp"):
                    freed += p.stat().st_size
                    p.unlink()
        return freed

    def load_latest(self, map_location: str | torch.device = "cpu") -> dict | None:
        path = self.latest()
        if path is None:
            return None
        state = torch.load(path, map_location=map_location, weights_only=False)
        if state.get("signature") != self.signature:
            raise ValueError(f"checkpoint {path} was written by a different run configuration; refusing to resume.\n"
                             f"  saved:   {state.get('signature')}\n  current: {self.signature}")
        return state
