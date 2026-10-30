"""Silo 2 stage-1 evaluation and segmentation FedAvg (resume equality, aggregation, selection)."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.fl.checkpoint import CheckpointManager
from src.fl.models import bce_dice_loss
from src.fl.seg_eval import SegEvaluator, f1_from_counts, loc_confusion, pooled_f1, weighted_loss
from src.fl.seg_fl import SegFLConfig, run_seg_fedavg
from src.fl.xbd_data import XBDPairDataset
from tests.test_checkpoint import TinySeg, ToySegDataset
from tests.test_xbd import make_synthetic_xbd


def test_loc_confusion_and_f1():
    logits = torch.tensor([[[[5.0, -5.0], [5.0, -5.0]]]])          # predicts left column
    target = torch.tensor([[[1, 1], [0, 0]]])
    assert loc_confusion(logits, target) == (1, 1, 1)
    assert f1_from_counts(1, 1, 1) == pytest.approx(0.5)


class ZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.full((x.shape[0], 1, *x.shape[2:]), -10.0) + 0 * self.p


def test_evaluator_counts_on_synthetic_xbd(tmp_path):
    split = make_synthetic_xbd(tmp_path, {"ev-a": (3, 1), "ev-b": (2, 0)})
    ids = {e: sorted(p.name.split("_pre_")[0] for p in (split / "images").glob(f"{e}_*_pre_disaster.png"))
           for e in ("ev-a", "ev-b")}
    ev = SegEvaluator({e: XBDPairDataset(split, ids[e], "loc") for e in ids}, batch_size=2, num_workers=0)
    out = ev.evaluate(ZeroModel())
    for e in ids:
        building_px = sum(int((np.asarray(Image.open(split / "targets" / f"{p}_pre_disaster_target.png")) > 0).sum())
                          for p in ids[e])
        assert out[e]["tp"] == 0 and out[e]["fp"] == 0 and out[e]["fn"] == building_px
        assert out[e]["n_images"] == len(ids[e]) and out[e]["loss"] > 0
    assert pooled_f1(out) == 0.0
    w = weighted_loss(out)
    assert min(r["loss"] for r in out.values()) <= w <= max(r["loss"] for r in out.values())
    assert ev.evaluate(ZeroModel(), clients=["ev-b"]).keys() == {"ev-b"}


def toy_fl(tmp, model, max_rounds=None, rounds=3):
    train_sets = {"c1": ToySegDataset(n=6, seed=1), "c2": ToySegDataset(n=4, seed=2)}
    val = SegEvaluator({"c1": ToySegDataset(n=3, seed=9), "c2": ToySegDataset(n=2, seed=8)}, batch_size=2)
    cfg = SegFLConfig(rounds=rounds, local_epochs=1, batch_size=2, lr=1e-2, amp=False, num_workers=0, seed=4,
                      max_rounds_this_call=max_rounds)
    mgr = CheckpointManager(tmp, every_n_steps=1, keep_last=2, signature={"toy": True})
    return run_seg_fedavg(model, train_sets, val, cfg, bce_dice_loss, "cpu", ckpt=mgr)


def fresh(seed):
    torch.manual_seed(seed)
    return TinySeg()


def test_seg_fedavg_resume_equals_uninterrupted(tmp_path):
    ref = toy_fl(tmp_path / "a", fresh(0))
    first = toy_fl(tmp_path / "b", fresh(0), max_rounds=1)                 # "interrupted" after round 1
    assert len(first["history"]) == 1
    resumed = toy_fl(tmp_path / "b", fresh(77))                            # different init must be overwritten
    assert resumed["resumed_from_round"] == 1 and len(resumed["history"]) == 3
    for k, v in ref["final_state"].items():
        assert torch.equal(v, resumed["final_state"][k]), k
    assert resumed["best_round"] == ref["best_round"]


def test_seg_fedavg_selects_lowest_federated_val_loss(tmp_path):
    res = toy_fl(tmp_path, fresh(0), rounds=4)
    losses = [h["fed_val_loss"] for h in res["history"]]
    assert res["best_round"] == int(np.argmin(losses)) + 1
    h = res["history"][res["best_round"] - 1]
    n = sum(v["n_images"] for v in h["client_val"].values())
    assert h["fed_val_loss"] == pytest.approx(sum(v["loss"] * v["n_images"] for v in h["client_val"].values()) / n)
