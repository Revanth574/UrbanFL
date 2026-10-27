"""
Generic synchronous FL round loop (plain PyTorch, single process; clients are simulated sequentially).

Round t:
  1. server samples m = max(1, round(client_fraction * K)) clients (seeded numpy RNG);
  2. broadcasts the global weights; each sampled client runs local_train (FedAvg or FedProx objective);
  3. server aggregates with fedavg_aggregate and LOADS the result into the global model;
  4. every client evaluates the new global model on its OWN val set and reports (val loss, n_val);
  5. server computes the client-size-weighted mean val loss — the ONLY checkpoint-selection signal.

Pooled val macro-F1 is logged for plots but never used for selection or stopping (in a real deployment the server
never sees client predictions). All rounds run; the round with the lowest federated val loss is the checkpoint
that is evaluated on every client's test set.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.fl.algorithms import ALGORITHMS, fedavg_aggregate, local_train
from src.fl.dp import DEFAULT_DELTA, ClientLevelAccountant, dp_aggregate, poisson_sample
from src.fl.models import MLPHead
from src.fl.utils import d4_metrics, macro_f1, set_seed

BYTES_PER_PARAM = 4  # float32 on the wire


@dataclass
class ClientData:
    name: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


def client_generator(seed: int, client_index: int) -> torch.Generator:
    """Per-client minibatch-order RNG, persistent across rounds (also used by the FedAvg == SGD test)."""
    return torch.Generator(device="cpu").manual_seed(seed * 10_000 + client_index)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> tuple[float, np.ndarray]:
    model.eval()
    logits = model(X)
    return criterion(logits, y).item(), logits.argmax(dim=1).cpu().numpy()


def run_fl(algorithm: str, clients: list[ClientData], rounds: int, local_epochs: int, lr: float,
           batch_size: int, client_fraction: float, seed: int, *, mu: float = 0.0,
           class_weights: torch.Tensor | None = None, n_classes: int, head: dict | None = None,
           device: str | torch.device = "cpu",
           on_round_end: Callable[[int, dict[str, torch.Tensor]], None] | None = None,
           dp: dict | None = None, client_class_weights: list[torch.Tensor] | None = None) -> dict:
    """
    Returns {"history": [...per round...], "best_round", "best_fed_val_loss", "best_state", "final_state",
             "test_predictions" (best-val checkpoint), "final_test_predictions" (last round),
             "n_params", "bytes_per_round", "total_bytes", "epsilon_final"}.
    `class_weights` (if given) define the CE used both for local training and for the reported val loss;
    `client_class_weights` (one tensor per client) override it per client.
    `dp` = {"clip": C, "noise_multiplier": z, "delta": δ} switches to client-level DP-FedAvg (src/fl/dp.py): Poisson
    client sampling with rate client_fraction, whole-delta clipping, unweighted mean with fixed denominator q*K,
    Gaussian noise, and an RDP accountant stepped once per round (z = 0: clipping only, no privacy, ε = ∞).
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    if algorithm == "fedavg" and mu != 0.0:
        raise ValueError("FedAvg has mu = 0; use algorithm='fedprox'")

    set_seed(seed)
    in_dim = clients[0].X_train.shape[1]
    global_model = MLPHead(in_dim, n_classes, **(head or {})).to(device)
    local_model = copy.deepcopy(global_model)       # deepcopy: consumes no RNG, unlike a fresh init
    if client_class_weights is not None:
        criteria = [nn.CrossEntropyLoss(weight=w.to(device)) for w in client_class_weights]
    else:
        criteria = [nn.CrossEntropyLoss(weight=None if class_weights is None else class_weights.to(device))] * len(clients)

    n_params = sum(p.numel() for p in global_model.parameters())
    K = len(clients)
    m = max(1, int(round(client_fraction * K)))
    bytes_per_round = n_params * BYTES_PER_PARAM * m * 2   # broadcast down + update up (expected, for DP)
    sampler = np.random.default_rng(seed)
    accountant, noise_gen, expected_clients = None, None, client_fraction * K
    if dp is not None:
        if dp["noise_multiplier"] > 0:
            accountant = ClientLevelAccountant(dp["noise_multiplier"], client_fraction, dp.get("delta", DEFAULT_DELTA))
        noise_gen = torch.Generator(device="cpu").manual_seed(seed * 7_919 + 17)
    generators = [client_generator(seed, i) for i in range(K)]
    n_val = np.array([len(c.y_val) for c in clients], dtype=float)

    def eval_round(round_idx: int) -> dict:
        val_losses, preds = [], {}
        for c, crit in zip(clients, criteria):
            loss, preds[c.name] = evaluate(global_model, c.X_val, c.y_val, crit)
            val_losses.append(loss)
        val_losses = np.array(val_losses)
        y_true = {c.name: c.y_val.cpu().numpy() for c in clients}
        entry = {
            "round": round_idx,
            "fed_val_loss": float((val_losses * n_val).sum() / n_val.sum()),
            "client_val_loss": dict(zip([c.name for c in clients], val_losses.tolist())),
            # logged for plots only — never read by the selection logic below
            "pooled_val_macro_f1": macro_f1(np.concatenate(list(y_true.values())),
                                            np.concatenate(list(preds.values())), list(range(n_classes))),
        }
        try:
            entry["val_client_mean_macro_f1"] = d4_metrics(y_true, preds, list(range(n_classes)))[
                "client_mean_macro_f1"]
        except NotImplementedError:
            entry["val_client_mean_macro_f1"] = None
        return entry

    history = [{**eval_round(0), "train_ce": None, "prox": None, "sampled": [], "bytes": 0, "round_time_s": 0.0}]
    best = {"round": None, "loss": float("inf"), "state": None}

    for t in range(1, rounds + 1):
        t0 = time.perf_counter()
        if dp is not None:
            sampled = poisson_sample(sampler, K, client_fraction)
        else:
            sampled = sorted(sampler.choice(K, size=m, replace=False).tolist())
        global_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

        states, sizes, stats = [], [], []
        for i in sampled:
            c = clients[i]
            local_model.load_state_dict(global_state)
            state, st = local_train(local_model, c.X_train, c.y_train, criteria[i], epochs=local_epochs, lr=lr,
                                    batch_size=batch_size, generator=generators[i], mu=mu)
            states.append(state)
            sizes.append(len(c.y_train))
            stats.append(st)

        dp_stats = {}
        if dp is not None:
            new_state, dp_stats = dp_aggregate(global_state, states, dp["clip"], dp["noise_multiplier"],
                                               expected_clients, noise_gen)
            if accountant is not None:
                accountant.step()
                dp_stats["epsilon"] = accountant.epsilon()
            global_model.load_state_dict(new_state)
        else:
            global_model.load_state_dict(fedavg_aggregate(global_state, states, sizes))
        if on_round_end is not None:
            on_round_end(t, {k: v.detach().clone() for k, v in global_model.state_dict().items()})

        entry = eval_round(t)
        w = np.array(sizes, dtype=float) / sum(sizes) if sizes else np.array([])
        entry.update({
            "train_ce": float(sum(wi * s["train_ce"] for wi, s in zip(w, stats))) if stats else None,
            "prox": float(sum(wi * s["prox"] for wi, s in zip(w, stats))) if stats else None,
            "sampled": [clients[i].name for i in sampled],
            "bytes": n_params * BYTES_PER_PARAM * len(sampled) * 2,   # down + up for the clients actually sampled
            "round_time_s": time.perf_counter() - t0,
            **({"dp": dp_stats} if dp is not None else {}),
        })
        history.append(entry)

        if entry["fed_val_loss"] < best["loss"]:
            best = {"round": t, "loss": entry["fed_val_loss"],
                    "state": {k: v.detach().clone() for k, v in global_model.state_dict().items()}}

    final_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    def predict_tests() -> dict:
        out = {}
        for c, crit in zip(clients, criteria):
            _, pred = evaluate(global_model, c.X_test, c.y_test, crit)
            out[c.name] = {"y_true": c.y_test.cpu().numpy().tolist(), "y_pred": pred.tolist()}
        return out

    final_test_predictions = predict_tests()          # last-round model (the DP-clean release)
    global_model.load_state_dict(best["state"])
    test_predictions = predict_tests()                # best federated-val-loss checkpoint

    return {"history": history, "best_round": best["round"], "best_fed_val_loss": best["loss"],
            "best_state": best["state"], "final_state": final_state, "test_predictions": test_predictions,
            "final_test_predictions": final_test_predictions,
            "n_params": n_params, "clients_per_round": m, "bytes_per_round": bytes_per_round,
            "total_bytes": sum(h["bytes"] for h in history),
            "epsilon_final": accountant.epsilon() if accountant is not None else None}
