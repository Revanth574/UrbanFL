"""Checkpoint/resume: an interrupted + resumed run must equal an uninterrupted one; manager hygiene; U-Net shapes."""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.fl.checkpoint import CheckpointManager, resolve_checkpoint_dir
from src.fl.models import UNetResNet, bce_dice_loss
from src.fl.seg_train import EpochPermutationSampler, TrainConfig, train
from src.fl.utils import set_seed


class ToySegDataset(torch.utils.data.Dataset):
    """Items depend on (seed, epoch, index) like XBDPairDataset, including an epoch-dependent 'augmentation'."""

    def __init__(self, n=10, size=16, seed=0):
        self.n, self.size, self.seed, self.epoch = n, size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(1000 * i + 7)
        img = torch.rand(3, self.size, self.size, generator=g)
        tgt = (img[0] > 0.5).long()
        if np.random.default_rng([self.seed, self.epoch, i]).random() < 0.5:   # epoch-dependent flip
            img, tgt = img.flip(-1), tgt.flip(-1)
        return {"pre": img, "target": tgt}


class TinySeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Dropout2d(0.3),
                                 nn.Conv2d(8, 1, 1))

    def forward(self, x):
        return self.net(x)


def fresh_model(seed):
    set_seed(seed)
    return TinySeg()


CFG = TrainConfig(epochs=2, batch_size=2, lr=1e-2, amp=False, num_workers=0, seed=3, log_every=1)


def run(tmp, model, on_step=None, every=3):
    mgr = CheckpointManager(tmp, every_n_steps=every, keep_last=2, signature={"cfg": "toy", "seed": CFG.seed})
    stats = train(model, ToySegDataset(seed=CFG.seed), CFG, bce_dice_loss, "cpu", ckpt=mgr, on_step=on_step)
    return model, stats, mgr


def test_interrupted_and_resumed_run_equals_uninterrupted(tmp_path):
    ref, ref_stats, _ = run(tmp_path / "a", fresh_model(0))
    assert ref_stats["global_step"] == 10                                   # 2 epochs x 5 steps

    class Crash(Exception):
        pass

    def crash_at_5(step, loss):
        if step == 5:
            raise Crash                                                     # after step 5, before any step-5 save

    with pytest.raises(Crash):
        run(tmp_path / "b", fresh_model(0), on_step=crash_at_5)
    resumed, stats, mgr = run(tmp_path / "b", fresh_model(123))           # different init: must be overwritten
    assert stats["resumed_from_step"] == 3                                  # last checkpoint before the crash
    assert stats["global_step"] == 10
    for k, v in ref.state_dict().items():
        assert torch.equal(v, resumed.state_dict()[k]), k                   # bit-identical (incl. BN stats)


def test_manager_keeps_last_and_writes_atomically(tmp_path):
    mgr = CheckpointManager(tmp_path, every_n_steps=1, keep_last=2, signature={"a": 1})
    for s in (1, 2, 3, 4):
        mgr.save({"x": torch.tensor(s)}, s)
    names = [p.name for p in mgr.checkpoints()]
    assert names == ["ckpt_step000000003.pt", "ckpt_step000000004.pt"]
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())
    assert mgr.load_latest()["x"].item() == 4


def test_resume_refuses_different_signature(tmp_path):
    CheckpointManager(tmp_path, 1, signature={"lr": 1e-3}).save({"x": 1}, 1)
    with pytest.raises(ValueError, match="different run configuration"):
        CheckpointManager(tmp_path, 1, signature={"lr": 3e-4}).load_latest()


def test_checkpoint_dir_resolution(tmp_path, monkeypatch):
    assert resolve_checkpoint_dir(tmp_path / "x", "run1") == tmp_path / "x" / "run1"
    monkeypatch.setenv("URBANFL_CHECKPOINT_DIR", str(tmp_path / "env"))
    assert resolve_checkpoint_dir(None, "run1") == tmp_path / "env" / "run1"
    monkeypatch.delenv("URBANFL_CHECKPOINT_DIR")
    monkeypatch.setenv("URBANFL_CONFIG", str(tmp_path / "absent.json"))
    assert resolve_checkpoint_dir(None, "run1").parts[-2:] == ("checkpoints", "run1")


def test_epoch_sampler_resume_offset():
    full = EpochPermutationSampler(10, seed=1, epoch=2).order()
    assert list(EpochPermutationSampler(10, seed=1, epoch=2, start=4)) == full[4:]
    assert full != EpochPermutationSampler(10, seed=1, epoch=3).order()


@pytest.mark.parametrize("encoder", ["resnet18", "resnet34"])
def test_unet_output_shape(encoder):
    m = UNetResNet(encoder, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.zeros(1, 3, 64, 96)).shape == (1, 1, 64, 96)
