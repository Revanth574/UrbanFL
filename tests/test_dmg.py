import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.fl.checkpoint import CheckpointManager
from src.fl.dmg_eval import _building_records, weighted_ce_loss
from src.fl.models import SiameseDamageNet, UNetResNet
from src.fl.xbd_data import IGNORE_INDEX


def test_weighted_ce_equals_torch_and_is_zero_on_empty_crop():
    torch.manual_seed(0)
    w = torch.tensor([0.3, 2.0, 3.0, 4.0])
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    y = torch.randint(0, 4, (2, 8, 8))
    y[0, :4] = IGNORE_INDEX
    ref = F.cross_entropy(logits, y, weight=w, ignore_index=IGNORE_INDEX)
    assert weighted_ce_loss(w)(logits, y).item() == pytest.approx(ref.item(), rel=1e-6)
    empty = torch.full((2, 8, 8), IGNORE_INDEX)
    loss = weighted_ce_loss(w)(logits, empty)
    assert loss.item() == 0.0 and not torch.isnan(loss)
    loss.backward()                                        # still differentiable (zero gradient)
    assert torch.all(logits.grad == 0)


def test_siamese_reuses_every_stage1_weight_and_shares_the_body():
    stage1 = UNetResNet("resnet18", pretrained=False)
    with torch.no_grad():
        for p in stage1.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    state = stage1.state_dict()
    net = SiameseDamageNet.from_stage1(state)
    body = net.body.state_dict()
    assert all(torch.equal(body[k], v) for k, v in state.items() if not k.startswith("head."))
    out = net(torch.randn(1, 6, 64, 64))
    assert out.shape == (1, 4, 64, 64)
    # one body for both images: swapping pre and post swaps the two halves of the head's input
    x = torch.randn(1, 6, 64, 64)
    f = lambda a: torch.cat([net.body(a[:, :3]), net.body(a[:, 3:])], 1)   # noqa: E731
    assert torch.allclose(f(x)[:, :32], f(torch.cat([x[:, 3:], x[:, :3]], 1))[:, 32:])


def _git(commit="c" * 40):
    return {"commit": commit, "dirty": False}


def test_stage2_checkpoint_dirs_are_distinct_per_method_and_client(tmp_path):
    from experiments.silo2_dmg import ckpt_manager
    dirs = {(m, c): ckpt_manager(f"{m}_seed0/{c}", _git(), {}, str(tmp_path)).dir
            for m in ("local_only", "fedavg_ft") for c in ("hurricane-michael", "palu-tsunami")}
    dirs.update({(m, None): ckpt_manager(f"{m}_seed0", _git(), {}, str(tmp_path)).dir
                 for m in ("centralized", "fedavg")})
    assert len({d.resolve() for d in dirs.values()}) == len(dirs)          # every full path distinct
    lo = ckpt_manager("local_only_seed0/hurricane-michael", _git(), {}, str(tmp_path))
    ft = ckpt_manager("fedavg_ft_seed0/hurricane-michael", _git(), {}, str(tmp_path))
    assert lo.dir.name == ft.dir.name and lo.label != ft.label             # same leaf, distinct log labels


def test_resume_refuses_any_signature_mismatch(tmp_path):
    from experiments.silo2_dmg import ckpt_manager, run_signature
    w1, w2 = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save({"w": torch.zeros(3)}, w1)
    torch.save({"w": torch.ones(3)}, w2)
    base = dict(method="fedavg_ft", seed=0, init_path=str(w1), client="hurricane-michael", lr=3e-4, epochs=2)
    mk = lambda sig, git=_git(): ckpt_manager("fedavg_ft_seed0/hurricane-michael", git, sig, str(tmp_path / "ck"))  # noqa: E731
    mk(run_signature(**base)).save({"x": 1}, 2)
    assert mk(run_signature(**base)).load_latest()["x"] == 1                # identical signature resumes
    w1_copy = tmp_path / "a_copy.pt"
    w1_copy.write_bytes(w1.read_bytes())
    for change in ({"method": "local_only"}, {"client": "palu-tsunami"}, {"seed": 1}, {"init_path": str(w2)},
                   {"init_path": str(w1_copy)}):                            # same bytes, other path: also refused
        with pytest.raises(ValueError, match="different run configuration"):
            mk(run_signature(**{**base, **change})).load_latest()
    saved_dir = mk(run_signature(**base)).dir
    with pytest.raises(ValueError, match="different run configuration"):    # only the commit differs
        CheckpointManager(saved_dir, 1, signature={**run_signature(**base), "commit": "d" * 40}).load_latest()
    torch.save({"w": torch.full((3,), 2.0)}, w1)                            # start weights changed in place
    with pytest.raises(ValueError, match="different run configuration"):
        mk(run_signature(**base)).load_latest()


def test_validation_schedule():
    from experiments.silo2_dmg import n_validations, validate_now
    assert [e for e in range(1, 31) if validate_now(e, 30)] == list(range(2, 31, 2))
    assert [e for e in range(1, 11) if validate_now(e, 10)] == [2, 4, 6, 8, 10]
    assert [e for e in range(1, 4) if validate_now(e, 3)] == [2, 3]       # odd total: the last is always validated
    assert [e for e in range(1, 3) if validate_now(e, 2)] == [2]
    assert n_validations(30) == 15 and n_validations(1) == 1


def test_building_record_majority_vote(tmp_path):
    import json
    lab = {"features": {"xy": [
        {"wkt": "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))", "properties": {"uid": "a", "subtype": "destroyed"}},
        {"wkt": "POLYGON ((6 6, 9 6, 9 9, 6 9, 6 6))", "properties": {"uid": "b", "subtype": "no-damage"}}]}}
    p = tmp_path / "x_post_disaster.json"
    p.write_text(json.dumps(lab))
    dp = np.zeros((10, 10), dtype=np.int64)
    dp[0:5, 0:5] = 4
    dp[0:2, 0:5] = 3                                       # minority of building a's pixels
    lp = (dp > 0).astype(np.float32)
    r = _building_records(p, dp, lp)
    assert r["uid"] == ["a", "b"] and r["true"] == ["destroyed", "no-damage"]
    assert r["pred"] == [4, 0]                             # a: majority destroyed; b: never localised -> 0
    assert r["frac_localised"][1] == 0.0
