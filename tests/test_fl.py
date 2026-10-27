"""FL core sanity tests on small synthetic clients (CPU)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.fl.algorithms import fedavg_aggregate, local_train
from src.fl.models import MLPHead
from src.fl.runner import ClientData, client_generator, run_fl
from src.fl.utils import set_seed

N_CLASSES, DIM, HEAD = 3, 16, {"hidden": 8, "dropout": 0.2}


def make_client(name: str, n: int, shift: float, seed: int) -> ClientData:
    g = torch.Generator().manual_seed(seed)

    def split(k):
        X = torch.randn(k, DIM, generator=g) + shift
        y = ((X[:, 0] > shift).long() + (X[:, 1] > shift).long()).clamp(max=N_CLASSES - 1)
        return X, y

    return ClientData(name, *split(n), *split(n // 3), *split(n // 3))


@pytest.fixture
def clients():
    return [make_client("a", 90, 0.0, 1), make_client("b", 150, 0.5, 2), make_client("c", 60, -0.5, 3)]


def states_equal(s1, s2):
    return all(torch.equal(s1[k], s2[k]) for k in s1)


def test_fedavg_aggregate_weighted_average():
    g = {"w": torch.zeros(2), "n": torch.tensor(0)}
    out = fedavg_aggregate(g, [{"w": torch.tensor([1.0, 0.0]), "n": torch.tensor(5)},
                               {"w": torch.tensor([4.0, 3.0]), "n": torch.tensor(7)}], [1, 2])
    assert torch.allclose(out["w"], torch.tensor([3.0, 2.0]))
    assert out["n"].item() == 5          # integer buffers are not averaged


def test_single_client_fedavg_equals_plain_sgd():
    c = make_client("solo", 100, 0.0, 7)
    rounds, lr, bs, seed = 4, 0.1, 16, 3
    captured = {}
    res = run_fl("fedavg", [c], rounds, 1, lr, bs, 1.0, seed, n_classes=N_CLASSES, head=HEAD,
                 on_round_end=lambda t, s: captured.setdefault(t, s))

    # Plain SGD written out independently: same init seed, same shuffling generator, `rounds` epochs.
    set_seed(seed)
    model = MLPHead(DIM, N_CLASSES, **HEAD)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    gen, ce = client_generator(seed, 0), nn.CrossEntropyLoss()
    model.train()
    for epoch in range(rounds):
        perm = torch.randperm(len(c.y_train), generator=gen)
        for start in range(0, len(perm), bs):
            idx = perm[start:start + bs]
            opt.zero_grad()
            ce(model(c.X_train[idx]), c.y_train[idx]).backward()
            opt.step()
    assert states_equal(res["final_state"], model.state_dict())
    assert states_equal(captured[rounds], model.state_dict())


def test_global_weights_change_every_round(clients):
    captured = {}
    run_fl("fedavg", clients, 3, 1, 0.1, 16, 1.0, 0, n_classes=N_CLASSES, head=HEAD,
           on_round_end=lambda t, s: captured.setdefault(t, s))
    for t in (1, 2):
        assert not states_equal(captured[t], captured[t + 1])


def test_fedprox_mu_zero_equals_fedavg(clients):
    a = run_fl("fedavg", clients, 3, 2, 0.1, 16, 1.0, 0, n_classes=N_CLASSES, head=HEAD)
    b = run_fl("fedprox", clients, 3, 2, 0.1, 16, 1.0, 0, mu=0.0, n_classes=N_CLASSES, head=HEAD)
    assert states_equal(a["final_state"], b["final_state"])


def test_fedprox_limits_client_drift():
    c = make_client("x", 120, 1.0, 5)
    set_seed(0)
    start = MLPHead(DIM, N_CLASSES, **HEAD)
    init = {k: v.clone() for k, v in start.state_dict().items()}

    def drift(mu):
        model = MLPHead(DIM, N_CLASSES, **HEAD)
        model.load_state_dict(init)
        set_seed(1)
        state, _ = local_train(model, c.X_train, c.y_train, nn.CrossEntropyLoss(), epochs=20, lr=0.5,
                               batch_size=16, generator=torch.Generator().manual_seed(0), mu=mu)
        return sum(((state[k] - init[k]) ** 2).sum().item() for k in init)

    assert drift(1.0) < drift(0.0)


def test_checkpoint_is_argmin_federated_val_loss(clients):
    res = run_fl("fedavg", clients, 8, 1, 0.3, 16, 1.0, 0, n_classes=N_CLASSES, head=HEAD)
    losses = [h["fed_val_loss"] for h in res["history"][1:]]
    assert res["best_round"] == int(np.argmin(losses)) + 1
    n_val = np.array([len(c.y_val) for c in clients])
    h = res["history"][res["best_round"]]
    expected = sum(h["client_val_loss"][c.name] * n for c, n in zip(clients, n_val)) / n_val.sum()
    assert h["fed_val_loss"] == pytest.approx(expected)


def test_partial_participation_and_bytes(clients):
    res = run_fl("fedavg", clients, 4, 1, 0.1, 16, 2 / 3, 0, n_classes=N_CLASSES, head=HEAD)
    assert all(len(h["sampled"]) == 2 for h in res["history"][1:])
    assert res["bytes_per_round"] == res["n_params"] * 4 * 2 * 2
