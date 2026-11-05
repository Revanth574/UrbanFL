"""
Silo 2 stage 2 (damage) losses and evaluation.

Loss: class-weighted cross-entropy over the 4 damage classes on scored building pixels only (background and
un-classified buildings = IGNORE_INDEX). Weighted mean over pixels exactly as torch's weighted CE, except that a batch
with no scored pixel (a crop without buildings) gives 0 instead of NaN.

Selection signal (DamageValEvaluator): per-image weighted CE on the full-size val images that contain scored pixels,
averaged over those images (FedAvg: image-weighted across clients, as in stage 1). Images without buildings carry no
damage signal and are skipped (n_images counts the scored images).

Test score (xview2_evaluate): the official xView2 score (src/fl/xbd_metrics.py) of a (stage-1 localisation model,
stage-2 damage model) pair, plus one record per building of the post label file, grouped by pair id: the majority vote
of the scored damage prediction dp = (argmax + 1) * localisation mask over the building's pixels (0 = not localised,
-1 = no own pixel because a later polygon covers it) and the share of its pixels stage 1 localised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.fl.client_loader import PersistentClientLoader
from src.fl.xbd_data import DAMAGE_CLASSES, IGNORE_INDEX, building_instances, post_buildings
from src.fl.xbd_metrics import XView2Scorer

PRED_NAMES = ("not-localised", *DAMAGE_CLASSES)        # per-building prediction codes 0..4


def dmg_batch_to_io(batch: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 2: cat(pre, post) -> 4-class damage target with IGNORE_INDEX."""
    x = torch.cat([batch["pre"], batch["post"]], dim=1)
    return x.to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)


def weighted_ce_loss(weights: torch.Tensor):
    """loss(logits (B,4,H,W), target (B,H,W)) = class-weighted CE mean over scored pixels; 0 if there are none."""
    w_cpu = torch.as_tensor(weights, dtype=torch.float32)

    def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = w_cpu.to(logits.device)
        total = F.cross_entropy(logits.float(), target, weight=w, ignore_index=IGNORE_INDEX, reduction="sum")
        valid = target != IGNORE_INDEX
        denom = w[target[valid]].sum()
        return total / denom.clamp_min(1e-12) if valid.any() else total * 0.0

    return loss


class DamageValEvaluator:
    """Same interface as seg_eval.SegEvaluator (evaluate -> {client: {"loss", "n_images"}}) for stage 2."""

    def __init__(self, datasets: dict, weights_of, device, loader: PersistentClientLoader, prefix: str,
                 amp: bool = True):
        self.datasets, self.weights_of, self.prefix, self.loader = datasets, weights_of, prefix, loader
        self.device = torch.device(device)
        self.amp = amp and self.device.type == "cuda"

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, clients: list[str] | None = None) -> dict[str, dict]:
        model.eval()
        clients = sorted(self.datasets) if clients is None else clients
        acc = {c: [0.0, 0] for c in clients}
        n = len(self.prefix)
        for key, batch in self.loader.run([(self.prefix + c, 0, 0) for c in clients]):
            c = key[n:]
            x, y = dmg_batch_to_io(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                logits = model(x)
            loss_fn = weighted_ce_loss(self.weights_of(c))
            for i in range(x.shape[0]):
                if (y[i] != IGNORE_INDEX).any():
                    acc[c][0] += float(loss_fn(logits[i:i + 1], y[i:i + 1]))
                    acc[c][1] += 1
        # a client without scored val images gets loss 0 with weight n_images = 0 (weighted_loss stays defined)
        return {c: {"loss": ls / k if k else 0.0, "n_images": int(k)} for c, (ls, k) in acc.items()}


def _building_records(label_path: Path, dp: np.ndarray, lp: np.ndarray) -> dict[str, list]:
    buildings = post_buildings(label_path)
    inst = building_instances(buildings, dp.shape)
    flat_i, flat_d, flat_l = inst.ravel(), dp.ravel(), lp.ravel()
    order = np.argsort(flat_i, kind="stable")
    bounds = np.searchsorted(flat_i[order], np.arange(1, len(buildings) + 2))
    rows = []
    for k, b in enumerate(buildings):
        idx = order[bounds[k]:bounds[k + 1]]
        if len(idx) == 0:                                  # fully covered by a later polygon (overlap)
            rows.append((b["uid"], b["subtype"], -1, 0, 0.0))
            continue
        votes = np.bincount(flat_d[idx], minlength=len(PRED_NAMES))
        rows.append((b["uid"], b["subtype"], int(votes.argmax()), int(len(idx)), round(float(flat_l[idx].mean()), 3)))
    cols = list(zip(*rows)) if rows else [()] * 5
    return dict(zip(("uid", "true", "pred", "n_px", "frac_localised"), [list(c) for c in cols]))


@torch.no_grad()
def xview2_evaluate(loc_model: torch.nn.Module, dmg_model: torch.nn.Module, loader: PersistentClientLoader,
                    prefix: str, clients: list[str], split_dir: Path, device, amp: bool = True,
                    with_buildings: bool = True) -> dict[str, dict]:
    """{client: {"counts": XView2Scorer counts, **score, "buildings": columnar records}} for the given clients."""
    device = torch.device(device)
    amp = amp and device.type == "cuda"
    loc_model.eval()
    dmg_model.eval()
    scorers = {c: XView2Scorer() for c in clients}
    recs, n_img = {c: {} for c in clients}, {c: 0 for c in clients}
    n = len(prefix)
    for key, batch in loader.run([(prefix + c, 0, 0) for c in clients]):
        c = key[n:]
        x, _ = dmg_batch_to_io(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            lp = (torch.sigmoid(loc_model(x[:, :3]).float()).squeeze(1) > 0.5).cpu().numpy()
            dcls = dmg_model(x).float().argmax(1).cpu().numpy()
        n_img[c] += len(batch["pair_id"])
        for i, pid in enumerate(batch["pair_id"]):
            pre_t, post_t = batch["pre_target"][i].numpy(), batch["post_target"][i].numpy()
            scorers[c].update(lp[i], pre_t, dcls[i], post_t)
            if with_buildings:
                dp = ((dcls[i] + 1) * lp[i]).astype(np.int64)
                recs[c][pid] = _building_records(Path(split_dir) / "labels" / f"{pid}_post_disaster.json", dp,
                                                 lp[i].astype(np.float32))
    out = {}
    for c in clients:
        out[c] = {"counts": scorers[c].counts(), **scorers[c].compute(), "n_images": n_img[c]}
        if with_buildings:
            out[c]["buildings"] = recs[c]                 # {pair_id: {uid, true, pred, n_px, frac_localised}}
    return out


def pooled_score(per_client: dict[str, dict]) -> dict:
    """xView2 score of the union of clients (TP / FP / FN summed first, as the official code does)."""
    s = XView2Scorer()
    for r in per_client.values():
        s.merge(XView2Scorer.from_counts(r["counts"]))
    return s.compute()
