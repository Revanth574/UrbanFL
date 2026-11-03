"""
GPU resume: a checkpoint written during CUDA training is restored in a FRESH process. Every RNG state (python, numpy,
torch CPU, CUDA) must match the saved one, weights and optimizer moments must land on the GPU, and training must
continue from the checkpoint step. Skipped without a CUDA device.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

REPO = Path(__file__).resolve().parents[1]

COMMON = textwrap.dedent("""
    import json, random, sys
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    import numpy as np, torch
    from src.fl.checkpoint import CheckpointManager, rng_state, set_rng_state
    from src.fl.models import bce_dice_loss
    from src.fl.seg_train import TrainConfig, train
    from src.fl.utils import set_seed
    from tests.test_checkpoint import TinySeg, ToySegDataset
    CFG = TrainConfig(epochs=2, batch_size=2, lr=1e-2, amp=True, num_workers=0, seed=3, log_every=1)
    def mgr(d):
        return CheckpointManager(d, every_n_steps=3, keep_last=2, signature={{"cfg": "cuda-toy"}})
    def model():
        set_seed(0)
        return TinySeg()
""")

FIRST = COMMON + textwrap.dedent("""
    out = Path(sys.argv[1])
    class Crash(Exception):
        pass
    def crash(step, loss):
        if step == 5:
            raise Crash
    try:
        train(model(), ToySegDataset(seed=3), CFG, bce_dice_loss, "cuda", ckpt=mgr(out / "ckpt"), on_step=crash)
    except Crash:
        pass
    ref = train(model(), ToySegDataset(seed=3), CFG, bce_dice_loss, "cuda", ckpt=mgr(out / "ref"))
    (out / "ref.json").write_text(json.dumps({{"losses": ref["loss_log"]}}))
""")

SECOND = COMMON + textwrap.dedent("""
    out = Path(sys.argv[1])
    m = mgr(out / "ckpt")
    state = m.load_latest()
    saved = state["rng"]
    set_rng_state(saved)
    now = rng_state()
    checks = {{
        "python": now["python"] == saved["python"],
        "numpy": all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                     for a, b in zip(now["numpy"], saved["numpy"])),
        "torch_cpu": torch.equal(now["torch"], saved["torch"].cpu()),
        "cuda": all(torch.equal(a, b.cpu()) for a, b in zip(now["cuda"], saved["cuda"])),
        "saved_rng_tensors_on_cpu": saved["torch"].device.type == "cpu"
                                    and all(s.device.type == "cpu" for s in saved["cuda"]),
    }}
    net = TinySeg()                                    # deliberately different init: must be overwritten
    st = train(net, ToySegDataset(seed=3), CFG, bce_dice_loss, "cuda", ckpt=m)   # train() asserts optimizer devices
    checks["params_on_gpu"] = all(p.device.type == "cuda" for p in net.parameters())
    (out / "resumed.json").write_text(json.dumps({{"checks": checks, "resumed_from_step": st["resumed_from_step"],
                                                   "global_step": st["global_step"], "losses": st["loss_log"]}}))
""")


def run(code: str, out: Path) -> None:
    script = out / "script.py"
    script.write_text(code.format(repo=str(REPO)), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script), str(out)], cwd=REPO, capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]


def test_gpu_checkpoint_restores_in_a_fresh_process(tmp_path):
    run(FIRST, tmp_path)                               # process 1: GPU training, crash after step 5 (ckpt at 3)
    run(SECOND, tmp_path)                              # process 2: fresh interpreter, resume on the GPU
    res = json.loads((tmp_path / "resumed.json").read_text())
    ref = json.loads((tmp_path / "ref.json").read_text())
    assert all(res["checks"].values()), res["checks"]
    assert res["resumed_from_step"] == 3 and res["global_step"] == 10
    ref_by_step = {r["step"]: r["loss"] for r in ref["losses"]}
    for r in res["losses"]:                            # continues the same trajectory (cuDNN may jitter slightly)
        assert r["step"] > 3
        assert r["loss"] == pytest.approx(ref_by_step[r["step"]], rel=1e-3, abs=1e-4)
