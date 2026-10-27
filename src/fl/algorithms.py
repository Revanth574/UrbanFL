"""
Federated optimisation primitives: client update (FedAvg / FedProx local objective) and server aggregation.

FedAvg  — McMahan et al., AISTATS 2017: clients run local SGD from the global weights; the server replaces the
          global weights with the sample-size-weighted average of the returned client weights.
FedProx — Li et al., MLSys 2020: identical, except each client minimises CE + (mu/2)·||w - w_global||², which
          limits client drift on heterogeneous (non-IID) clients. mu = 0 reduces exactly to FedAvg.

Clients use plain SGD (no momentum, no weight decay): optimizer state would otherwise have to be reset or
communicated every round, and plain SGD is what the FedAvg == SGD sanity test is defined against.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

ALGORITHMS = ("fedavg", "fedprox")


def fedavg_aggregate(global_state: dict[str, torch.Tensor],
                     client_states: Sequence[dict[str, torch.Tensor]],
                     client_sizes: Sequence[int]) -> dict[str, torch.Tensor]:
    """Sample-size-weighted average of client state_dicts: sum_k (n_k / n) * w_k, key by key."""
    if not client_states or len(client_states) != len(client_sizes):
        raise ValueError("need one size per client state, and at least one client")
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("client sizes must sum to a positive number")
    new_state = {}
    for key, ref in global_state.items():
        stacked = torch.stack([state[key].to(ref.dtype) for state in client_states])
        if not torch.is_floating_point(ref):     # integer buffers (e.g. BatchNorm counters): not averaged
            new_state[key] = stacked[0].clone()
            continue
        weights = torch.tensor([n / total for n in client_sizes], dtype=ref.dtype, device=ref.device)
        new_state[key] = (stacked * weights.view(-1, *([1] * ref.dim()))).sum(dim=0)
    return new_state


def prox_term(model: nn.Module, global_params: Sequence[torch.Tensor], mu: float) -> torch.Tensor:
    """(mu / 2) * ||w - w_global||² over all trainable parameters."""
    sq = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
    return 0.5 * mu * sq


def local_train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, *,
                epochs: int, lr: float, batch_size: int, generator: torch.Generator,
                mu: float = 0.0) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Client update. `model` must already hold the broadcast global weights. Minibatch order comes from the
    client's own persistent `generator` (CPU), so a client's data order is reproducible and independent of
    which other clients were sampled. Returns (detached state_dict copy, stats).
    """
    global_params = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    # Loss sums stay on-device; one .item() at the end avoids a host sync per step.
    ce_sum = torch.zeros((), device=X.device)
    prox_sum = torch.zeros((), device=X.device)
    n_seen, steps = 0, 0
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=generator).to(X.device)
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            ce = criterion(model(X[idx]), y[idx])
            loss = ce
            if global_params is not None:
                prox = prox_term(model, global_params, mu)
                loss = ce + prox
                prox_sum += prox.detach() * len(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ce_sum += ce.detach() * len(idx)
            n_seen += len(idx)
            steps += 1
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return state, {"train_ce": ce_sum.item() / n_seen, "prox": prox_sum.item() / n_seen, "steps": steps}
