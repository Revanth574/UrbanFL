"""
Client-level DP-FedAvg (McMahan, Ramage, Talwar & Zhang, "Learning Differentially Private Recurrent Language
Models", ICLR 2018) with an Opacus RDP accountant.

PRIVACY UNIT = ONE CLIENT (one disaster event / agency, i.e. all of its tweets at once). The reported ε bounds what
the sequence of released global models reveals about whether any single CLIENT took part. It is NOT record-level
(per-tweet) privacy, and with only K = 10 clients the achievable ε is large — an expected property of client-level
DP at this scale, not a bug.

Per round t (implemented in src/fl/runner.py when `dp` is passed to run_fl):
  1. Poisson client sampling: each client joins independently with probability q (= client_fraction). This is the
     sampling scheme the RDP accountant analyses (a fixed-size "m of K" draw would be a different mechanism).
  2. Each sampled client trains locally from w_t and returns delta_k = w_k - w_t.
  3. delta_k is clipped to L2 norm C over the WHOLE delta (all parameters concatenated).
  4. Server: w_{t+1} = w_t + (sum_k clip(delta_k) + N(0, (z*C)^2 I)) / m, with the FIXED denominator m = q*K
     (expected number of participants), so each client's influence is at most C/m and the per-coordinate noise std is
     z*C/m. Clients are weighted equally: sample-size weighting would give large clients more sensitivity.
     Noise is added even in a round where no client was sampled.
  5. RDPAccountant.step(noise_multiplier=z, sample_rate=q) exactly once per round; ε(δ) is read after every round.

Outside the guarantee (reported, not hidden): checkpoint selection on client validation losses and hyperparameters
tuned on non-private runs both use private data that the accountant does not cover. The experiment script therefore
reports the FINAL-round model as the DP result and the val-selected checkpoint only as a secondary number.
"""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np
import torch

DEFAULT_DELTA = 1e-5


def poisson_sample(rng: np.random.Generator, n_clients: int, q: float) -> list[int]:
    """Each client independently with probability q (q = 1 -> everyone). May return an empty list."""
    if q >= 1.0:
        return list(range(n_clients))
    return [k for k in range(n_clients) if rng.random() < q]


def delta_norm(delta: dict[str, torch.Tensor]) -> float:
    """L2 norm of the whole update, all floating-point tensors concatenated."""
    return math.sqrt(sum(float((v.double() ** 2).sum()) for v in delta.values() if torch.is_floating_point(v)))


def clip_delta(delta: dict[str, torch.Tensor], clip: float) -> tuple[dict[str, torch.Tensor], float, bool]:
    """Scale the update so its whole-model L2 norm is <= clip. Returns (clipped, original norm, was_clipped)."""
    norm = delta_norm(delta)
    scale = min(1.0, clip / (norm + 1e-12))
    return {k: v * scale for k, v in delta.items()}, norm, scale < 1.0


def dp_aggregate(global_state: dict[str, torch.Tensor], client_states: Sequence[dict[str, torch.Tensor]],
                 clip: float, noise_multiplier: float, expected_clients: float,
                 generator: torch.Generator) -> tuple[dict[str, torch.Tensor], dict]:
    """
    w_new = w + (sum of clipped client deltas + N(0, (z*C)^2)) / m,  m = expected_clients (fixed denominator).
    Noise is drawn on the CPU from `generator` (reproducible) and moved to the parameters' device.
    """
    if expected_clients <= 0:
        raise ValueError("expected_clients must be positive")
    sum_delta = {k: torch.zeros_like(v) for k, v in global_state.items() if torch.is_floating_point(v)}
    norms, clipped = [], 0
    for state in client_states:
        delta = {k: state[k] - global_state[k] for k in sum_delta}
        delta, norm, was_clipped = clip_delta(delta, clip)
        norms.append(norm)
        clipped += int(was_clipped)
        for k in sum_delta:
            sum_delta[k] += delta[k]

    sigma = noise_multiplier * clip
    new_state = {}
    for k, v in global_state.items():
        if k not in sum_delta:                      # integer buffers: unchanged
            new_state[k] = v.clone()
            continue
        noise = torch.randn(v.shape, generator=generator, dtype=v.dtype).to(v.device) * sigma if sigma > 0 else 0.0
        new_state[k] = v + (sum_delta[k] + noise) / expected_clients
    return new_state, {"n_clients": len(client_states), "mean_delta_norm": float(np.mean(norms)) if norms else None,
                       "frac_clipped": clipped / len(client_states) if client_states else None,
                       "noise_std_per_coordinate": sigma / expected_clients}


class ClientLevelAccountant:
    """Thin wrapper around opacus.accountants.RDPAccountant: one step per FL round."""

    def __init__(self, noise_multiplier: float, sample_rate: float, delta: float = DEFAULT_DELTA):
        from opacus.accountants import RDPAccountant
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be > 0 for a finite ε (z = 0 means no privacy)")
        self.noise_multiplier, self.sample_rate, self.delta = noise_multiplier, sample_rate, delta
        self._acct = RDPAccountant()
        self.steps = 0

    def step(self) -> None:
        self._acct.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sample_rate)
        self.steps += 1

    def epsilon(self) -> float:
        with warnings.catch_warnings():               # opacus warns about optimal-order edge cases
            warnings.simplefilter("ignore")
            return float(self._acct.get_epsilon(delta=self.delta))
