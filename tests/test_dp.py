"""Client-level DP-FedAvg: clipping, noise calibration, accountant, and runner integration."""

import numpy as np
import pytest
import torch

from src.fl.dp import ClientLevelAccountant, clip_delta, delta_norm, dp_aggregate, poisson_sample
from src.fl.runner import run_fl
from tests.test_fl import DIM, HEAD, N_CLASSES, make_client, states_equal


def test_clip_delta_bounds_whole_model_norm():
    d = {"a": torch.full((3,), 2.0), "b": torch.full((4,), 1.0)}          # norm sqrt(12 + 4) = 4
    clipped, norm, was = clip_delta(d, 1.0)
    assert norm == pytest.approx(4.0) and was
    assert delta_norm(clipped) == pytest.approx(1.0, rel=1e-6)
    assert torch.allclose(clipped["a"] / clipped["b"][0], d["a"] / d["b"][0])   # direction kept
    same, _, was2 = clip_delta(d, 10.0)
    assert not was2 and all(torch.equal(same[k], d[k]) for k in d)


def test_dp_aggregate_no_noise_is_unweighted_mean_of_clipped_deltas():
    g = {"w": torch.zeros(4)}
    states = [{"w": torch.tensor([0.1, 0.0, 0.0, 0.0])}, {"w": torch.tensor([0.0, 3.0, 4.0, 0.0])}]  # norms 0.1, 5
    new, st = dp_aggregate(g, states, clip=1.0, noise_multiplier=0.0, expected_clients=2,
                           generator=torch.Generator().manual_seed(0))
    expected = (torch.tensor([0.1, 0, 0, 0]) + torch.tensor([0, 0.6, 0.8, 0])) / 2
    assert torch.allclose(new["w"], expected)
    assert st["frac_clipped"] == 0.5


def test_dp_noise_std_matches_z_C_over_m():
    g = {"w": torch.zeros(200_000)}
    z, C, m = 1.5, 0.2, 4.0
    new, st = dp_aggregate(g, [], clip=C, noise_multiplier=z, expected_clients=m,
                           generator=torch.Generator().manual_seed(1))
    assert st["noise_std_per_coordinate"] == pytest.approx(z * C / m)
    assert new["w"].std().item() == pytest.approx(z * C / m, rel=0.01)      # noise added even with 0 clients


def test_accountant_matches_opacus_and_is_monotone():
    from opacus.accountants import RDPAccountant
    ours, ref = ClientLevelAccountant(1.0, 0.5), RDPAccountant()
    eps = []
    for _ in range(20):
        ours.step(); ref.step(noise_multiplier=1.0, sample_rate=0.5)
        eps.append(ours.epsilon())
    assert eps[-1] == pytest.approx(ref.get_epsilon(delta=1e-5))
    assert all(b > a for a, b in zip(eps, eps[1:]))                        # ε grows with rounds
    more_noise = ClientLevelAccountant(2.0, 0.5)
    for _ in range(20):
        more_noise.step()
    assert more_noise.epsilon() < eps[-1]                                  # more noise -> smaller ε
    with pytest.raises(ValueError):
        ClientLevelAccountant(0.0, 0.5)


def test_poisson_sampling_rate():
    rng = np.random.default_rng(0)
    counts = [len(poisson_sample(rng, 10, 0.5)) for _ in range(4000)]
    assert np.mean(counts) == pytest.approx(5.0, abs=0.1)
    assert min(counts) == 0 or min(counts) < 5                              # sizes vary (not fixed-size)
    assert poisson_sample(rng, 10, 1.0) == list(range(10))


def test_dp_without_noise_or_clipping_equals_fedavg_for_equal_clients():
    # equal client sizes -> unweighted mean == sample-weighted mean; huge C -> no clipping; z = 0 -> no noise
    clients = [make_client(n, 90, s, i) for i, (n, s) in enumerate((("a", 0.0), ("b", 0.4), ("c", -0.4)))]
    a = run_fl("fedavg", clients, 3, 1, 0.1, 16, 1.0, 0, n_classes=N_CLASSES, head=HEAD)
    b = run_fl("fedavg", clients, 3, 1, 0.1, 16, 1.0, 0, n_classes=N_CLASSES, head=HEAD,
               dp={"clip": 1e9, "noise_multiplier": 0.0})
    for k in a["final_state"]:
        assert torch.allclose(a["final_state"][k], b["final_state"][k], atol=1e-6)
    assert b["epsilon_final"] is None


def test_dp_run_reports_epsilon_each_round():
    clients = [make_client(n, 60, s, i) for i, (n, s) in enumerate((("a", 0.0), ("b", 0.4)))]
    res = run_fl("fedavg", clients, 4, 1, 0.1, 16, 1.0, 0, n_classes=N_CLASSES, head=HEAD,
                 dp={"clip": 0.5, "noise_multiplier": 1.0})
    eps = [h["dp"]["epsilon"] for h in res["history"][1:]]
    assert all(b > a for a, b in zip(eps, eps[1:]))
    assert res["epsilon_final"] == pytest.approx(eps[-1])
    assert "final_test_predictions" in res and set(res["final_test_predictions"]) == {"a", "b"}
