import numpy as np
import pytest
import torch
import torch.nn.functional as F

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
