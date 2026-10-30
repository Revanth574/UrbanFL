"""
Full-resolution evaluation for Silo 2 stage 1 (building localisation), per client, in one pass:
  - mean BCE+Dice loss per image (the checkpoint-selection signal; size-weighted across clients for FL),
  - pixel TP / FP / FN at probability 0.5, so localisation F1 can be pooled over any set of clients or reported
    per event (the xView2 localisation F1 is the pooled pixel F1).
One persistent loader serves every evaluation of a run (workers start once).
"""

from __future__ import annotations

import numpy as np
import torch

from src.fl.client_loader import PersistentClientLoader
from src.fl.models import bce_dice_loss


def loc_confusion(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple[int, int, int]:
    pred = torch.sigmoid(logits.float()).squeeze(1) > threshold
    gt = target.bool()
    return int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


class SegEvaluator:
    def __init__(self, datasets: dict, batch_size: int = 2, num_workers: int = 0, device: str = "cpu",
                 amp: bool = True, loader: PersistentClientLoader | None = None, prefix: str = ""):
        """Pass a shared `loader` whose keys are prefix + client to reuse one worker pool for a whole run."""
        self.datasets, self.prefix = datasets, prefix
        self.device = torch.device(device)
        self.amp = amp and self.device.type == "cuda"
        self.loader = loader or PersistentClientLoader({prefix + c: d for c, d in datasets.items()}, batch_size,
                                                       num_workers, seed_of=0, pin_memory=self.device.type == "cuda")

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, clients: list[str] | None = None) -> dict[str, dict]:
        """{client: {"loss": mean per-image loss, "n_images", "tp", "fp", "fn", "f1"}} for the chosen clients."""
        model.eval()
        clients = sorted(self.datasets) if clients is None else clients
        acc = {c: np.zeros(5) for c in clients}                  # loss_sum, n_images, tp, fp, fn
        n = len(self.prefix)
        for key, batch in self.loader.run([(self.prefix + c, 0, 0) for c in clients]):   # eval sets ignore epoch
            c = key[n:]
            x = batch["pre"].to(self.device, non_blocking=True)
            y = batch["target"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                logits = model(x)
            for i in range(x.shape[0]):                          # per-image loss: images weigh equally
                acc[c][0] += float(bce_dice_loss(logits[i:i + 1].float(), y[i:i + 1]))
            acc[c][1] += x.shape[0]
            acc[c][2:] += loc_confusion(logits, y)
        out = {}
        for c, (ls, n, tp, fp, fn) in acc.items():
            out[c] = {"loss": ls / n if n else None, "n_images": int(n), "tp": int(tp), "fp": int(fp), "fn": int(fn),
                      "f1": f1_from_counts(tp, fp, fn)}
        return out


def weighted_loss(per_client: dict[str, dict]) -> float:
    """Size-weighted mean loss over clients (weights = number of images), i.e. the pooled per-image mean."""
    n = sum(r["n_images"] for r in per_client.values())
    return sum(r["loss"] * r["n_images"] for r in per_client.values()) / n


def pooled_f1(per_client: dict[str, dict]) -> float:
    return f1_from_counts(*(sum(r[k] for r in per_client.values()) for k in ("tp", "fp", "fn")))
