"""
FedAvg for Silo 2 segmentation (stage 1 localisation), sharing one persistent DataLoader across all clients.

Round r (E local epochs):
  1. every client starts from the global weights, runs E local epochs of AdamW (fresh optimizer state each round: client
     optimizer state is never communicated) with AMP, on batches from the shared loader (client c, local epoch
     r*E + e, sample order epoch_order(n_c, seed_of(c), r*E + e));
  2. server: sample-size-weighted fedavg_aggregate of the client weights (BatchNorm running statistics are averaged
     too; integer counters are taken from the first client);
  3. every client evaluates the new global model on its OWN val set; the server selects the round with the lowest
     image-weighted mean val loss (pooled per-image loss). Val F1 is logged, never used for selection.
Checkpoints are written at round boundaries (every `ckpt_every_rounds`): global weights, best weights so far, history,
RNG states; a resumed run continues from the last completed round (an interruption costs at most one round).
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from src.fl.algorithms import fedavg_aggregate
from src.fl.checkpoint import CheckpointManager, rng_state, set_rng_state
from src.fl.client_loader import PersistentClientLoader
from src.fl.seg_eval import SegEvaluator, pooled_f1, weighted_loss
from src.fl.seg_train import loc_batch_to_io


@dataclass
class SegFLConfig:
    rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    num_workers: int = 4
    seed: int = 0
    max_rounds_this_call: int | None = None     # stop early (timing measurements, tests)


def client_seed(seed: int, index: int) -> int:
    return seed * 100 + index


def run_seg_fedavg(model: nn.Module, train_sets: dict, evaluator: SegEvaluator, cfg: SegFLConfig, loss_fn,
                   device: str | torch.device, ckpt: CheckpointManager | None = None,
                   loader: PersistentClientLoader | None = None, prefix: str = "") -> dict:
    """`loader` (optional): a shared PersistentClientLoader whose entries prefix + client serve the train sets with
    batch size cfg.batch_size and sampler seed client_seed(cfg.seed, client index)."""
    device = torch.device(device)
    use_amp = cfg.amp and device.type == "cuda"
    clients = sorted(train_sets)
    sizes = {c: len(train_sets[c]) for c in clients}
    index = {c: i for i, c in enumerate(clients)}
    own_loader = loader is None
    if own_loader:
        loader = PersistentClientLoader({prefix + c: d for c, d in train_sets.items()}, cfg.batch_size,
                                        cfg.num_workers, seed_of=lambda k: client_seed(cfg.seed, index[k[len(prefix):]]),
                                        pin_memory=device.type == "cuda", loader_seed=cfg.seed * 1_000_003 + 11)
    model.to(device)
    local = copy.deepcopy(model)
    start_round, history, best = 0, [], {"round": None, "loss": float("inf"), "state": None}

    state = ckpt.load_latest(map_location=device) if ckpt else None
    if state is not None:
        model.load_state_dict(state["global"])
        set_rng_state(state["rng"])
        start_round, history, best = state["round"], state["history"], state["best"]

    def save(completed_rounds: int) -> None:
        if ckpt is not None:
            ckpt.save({"global": model.state_dict(), "rng": rng_state(), "round": completed_rounds,
                       "history": history, "best": best, "fl_config": asdict(cfg)}, completed_rounds)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rounds_done_now = 0
    for r in range(start_round, cfg.rounds):
        if cfg.max_rounds_this_call is not None and rounds_done_now >= cfg.max_rounds_this_call:
            break
        t0 = time.perf_counter()
        global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        schedule = [(prefix + c, r * cfg.local_epochs + e, 0) for c in clients for e in range(cfg.local_epochs)]
        states, client_loss, current, opt, scaler = {}, {}, None, None, None
        loss_sum, n_steps = 0.0, 0
        for key, batch in loader.run(schedule):
            c = key[len(prefix):]
            if c != current:
                if current is not None:
                    states[current] = {k: v.detach().clone() for k, v in local.state_dict().items()}
                    client_loss[current] = loss_sum / max(1, n_steps)
                current, loss_sum, n_steps = c, 0.0, 0
                local.load_state_dict(global_state)
                local.train()
                opt = torch.optim.AdamW(local.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            x, y = loc_batch_to_io(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = local(x)
            loss = loss_fn(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
            n_steps += 1
        states[current] = {k: v.detach().clone() for k, v in local.state_dict().items()}
        client_loss[current] = loss_sum / max(1, n_steps)
        t_train = time.perf_counter() - t0

        model.load_state_dict(fedavg_aggregate(global_state, [states[c] for c in clients], [sizes[c] for c in clients]))
        t1 = time.perf_counter()
        val = evaluator.evaluate(model)
        fed_val_loss = weighted_loss(val)
        t_eval = time.perf_counter() - t1
        entry = {"round": r + 1, "fed_val_loss": fed_val_loss, "pooled_val_loc_f1_logged_only": pooled_f1(val),
                 "client_val": {c: {"loss": v["loss"], "n_images": v["n_images"]} for c, v in val.items()},
                 "client_train_loss": client_loss, "train_s": t_train, "eval_s": t_eval,
                 "round_s": time.perf_counter() - t0}
        history.append(entry)
        if fed_val_loss < best["loss"]:
            best = {"round": r + 1, "loss": fed_val_loss,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        rounds_done_now += 1
        save(r + 1)
    if own_loader:
        loader.close()
    stats = {"history": history, "best_round": best["round"], "best_fed_val_loss": best["loss"],
             "best_state": best["state"], "final_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "resumed_from_round": None if state is None else start_round}
    if device.type == "cuda":
        stats["peak_vram_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return stats
