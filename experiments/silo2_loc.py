"""
Silo 2 stage 1 — building localisation (pre image -> building mask), same protocol as Silo 1.

  lr sweep     : centralized, split seed 0, LR_GRID, SWEEP_EPOCHS epochs each, chosen by best pooled val loss
  centralized  : union of all clients' train sets, EPOCHS epochs, checkpoint = epoch with lowest pooled val loss
  local_only   : one model per client on its own train set, EPOCHS epochs, checkpoint = lowest loss on ITS OWN val set
  fedavg       : 9 clients, E = 1, ROUNDS rounds, checkpoint = round with lowest image-weighted federated val loss
Seeds 0-2; seed s uses the spatial_block split silo2_seed{s}.json and seeds the model/data order with s.
Model: U-Net (ResNet18, ImageNet weights), 512 crops, batch 4, AMP, AdamW (lr from the sweep, wd 1e-4), BCE + Dice.
Reported: test localisation F1 (pixel F1 at p = 0.5) pooled over all event test sets and per event, held-out
guatemala-volcano F1, paired deltas across seeds. Selection never reads test data.
Local-only has no single model: its pooled test F1 sums each client's TP/FP/FN on its own test set, and its held-out
F1 is reported per client model and as their mean.
Checkpoints: resolve_checkpoint_dir -> C:/data/checkpoints/silo2_loc/<run>_<commit> (configurable); runs resume.

Usage:  python -m experiments.silo2_loc --lr-sweep
        python -m experiments.silo2_loc --method centralized local_only fedavg --seeds 0 1 2
        python -m experiments.silo2_loc --summarize
        python -m experiments.silo2_loc --measure-round       # timing only: one FedAvg round, no results claims
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

from src.fl.checkpoint import CheckpointManager, resolve_checkpoint_dir, save_atomic
from src.fl.config import xbd_split_dir
from src.fl.models import UNetResNet, bce_dice_loss
from src.fl.partition_xbd import load_xbd_partition, pair_ids
from src.fl.client_loader import PersistentClientLoader
from src.fl.seg_eval import SegEvaluator, pooled_f1, weighted_loss
from src.fl.seg_fl import SegFLConfig, client_seed, run_seg_fedavg
from src.fl.seg_train import TrainConfig, train
from src.fl.utils import RESULTS_DIR, require_clean_tree, set_seed, write_metrics
from src.fl.xbd_data import XBDPairDataset

EXPERIMENT_DIR = RESULTS_DIR / "silo2_loc"
SWEEP_DIR = EXPERIMENT_DIR / "lr_sweep"
METHODS = ("centralized", "local_only", "fedavg")
LR_GRID = [1e-4, 3e-4, 1e-3]
SWEEP_EPOCHS, EPOCHS, ROUNDS, LOCAL_EPOCHS = 10, 30, 30, 1
COMMON = {"encoder": "resnet18", "pretrained": "ImageNet", "crop": 512, "batch_size": 4, "eval_batch_size": 2,
          "amp": True, "num_workers": 4, "weight_decay": 1e-4, "loss": "BCEWithLogits + soft Dice",
          "optimizer": "AdamW", "ckpt_every_steps": 200, "split": "spatial_block"}


class Data:
    """Per-client train / val / test datasets for split seed `seed`, plus the held-out event."""

    def __init__(self, seed: int, limit: int | None = None):
        self.dir = xbd_split_dir("train")
        self.part = load_xbd_partition(seed)
        cut = (lambda ids: ids[:limit]) if limit else (lambda ids: ids)
        c = self.part["clients"]
        self.clients = sorted(c)
        self.train = {e: XBDPairDataset(self.dir, cut(c[e]["train"]), "loc", train=True, crop=COMMON["crop"], seed=seed)
                      for e in self.clients}
        self.val = {e: XBDPairDataset(self.dir, cut(c[e]["val"]), "loc") for e in self.clients}
        self.test = {e: XBDPairDataset(self.dir, cut(c[e]["test"]), "loc") for e in self.clients}
        self.heldout = {e: XBDPairDataset(self.dir, cut(ids), "loc") for e, ids in self.part["heldout"].items()}
        self.all_train_ids = [p for e in self.clients for p in cut(c[e]["train"])]
        self.seed = seed

    def split_sizes(self) -> dict:
        return {e: {"train": len(self.train[e]), "val": len(self.val[e]), "test": len(self.test[e])} for e in self.clients}

    def hub(self, method: str, device) -> "Hub":
        return Hub(self, method, device)


class Hub:
    """
    ONE persistent DataLoader (one worker pool) for a whole run: keys train/<client> (or train/all for centralized),
    val/<client>, test/<client>, heldout/<event>. Each worker process loads torch's CUDA libraries, so on Windows
    several pools alive at once exhausted the commit limit (WinError 1455). Close it when the run ends.
    Train sampler seeds follow the method: centralized and local-only use the run seed; FedAvg uses client_seed().
    """

    def __init__(self, data: Data, method: str, device):
        self.data, self.device = data, torch.device(device)
        sets, bs = {}, {}
        if method == "centralized":
            sets["train/all"] = XBDPairDataset(data.dir, data.all_train_ids, "loc", train=True, crop=COMMON["crop"],
                                               seed=data.seed)
        else:
            sets.update({f"train/{e}": d for e, d in data.train.items()})
        bs.update({k: COMMON["batch_size"] for k in sets})
        for split, group in (("val", data.val), ("test", data.test), ("heldout", data.heldout)):
            for e, d in group.items():
                sets[f"{split}/{e}"] = d
                bs[f"{split}/{e}"] = COMMON["eval_batch_size"]
        index = {e: i for i, e in enumerate(data.clients)}

        def seed_of(key: str) -> int:
            if method == "fedavg" and key.startswith("train/"):
                return client_seed(data.seed, index[key.split("/", 1)[1]])
            return data.seed

        self.sets = sets
        self.loader = PersistentClientLoader(sets, bs, COMMON["num_workers"], seed_of=seed_of,
                                             pin_memory=self.device.type == "cuda",
                                             loader_seed=data.seed * 1_000_003 + 13)

    def evaluator(self, which: str) -> SegEvaluator:
        group = {"val": self.data.val, "test": self.data.test, "heldout": self.data.heldout}[which]
        return SegEvaluator(group, device=self.device, loader=self.loader, prefix=f"{which}/")

    def close(self) -> None:
        self.loader.close()


def new_model(seed: int) -> UNetResNet:
    set_seed(seed)
    return UNetResNet(COMMON["encoder"], pretrained=True)


def ckpt_manager(run: str, git: dict, signature: dict, base: str | None) -> CheckpointManager:
    d = resolve_checkpoint_dir(base, f"silo2_loc/{run}_{git['commit'][:7]}")
    return CheckpointManager(d, COMMON["ckpt_every_steps"], keep_last=2, signature={**signature, "commit": git["commit"]})


def test_report(model, hub: Hub) -> dict:
    model.to(hub.device)
    test = hub.evaluator("test").evaluate(model)
    held = hub.evaluator("heldout").evaluate(model)
    return {"test_per_event": test, "test_pooled_loc_f1": pooled_f1(test), "heldout": held,
            "heldout_loc_f1": pooled_f1(held)}


def run_centralized(data: Data, lr: float, epochs: int, device, git, ckpt_base, tag: str) -> dict:
    run = f"{tag}_seed{data.seed}"
    model = new_model(data.seed)
    hub = data.hub("centralized", device)
    try:
        val_eval = hub.evaluator("val")
        mgr = ckpt_manager(run, git, {"method": "centralized", "lr": lr, "epochs": epochs, "seed": data.seed},
                           ckpt_base)
        best_path = mgr.dir / "best_model.pt"

        def on_epoch_end(epoch, m, extra):
            val = val_eval.evaluate(m)
            entry = {"epoch": epoch + 1, "val_loss": weighted_loss(val),
                     "pooled_val_loc_f1_logged_only": pooled_f1(val)}
            extra.setdefault("history", []).append(entry)
            if entry["val_loss"] < extra.get("best_loss", float("inf")):
                extra.update(best_loss=entry["val_loss"], best_epoch=entry["epoch"])
                save_atomic({k: v.detach().cpu() for k, v in m.state_dict().items()}, best_path)

        cfg = TrainConfig(epochs=epochs, batch_size=COMMON["batch_size"], lr=lr, weight_decay=COMMON["weight_decay"],
                          amp=True, num_workers=COMMON["num_workers"], seed=data.seed)
        st = train(model, hub.sets["train/all"], cfg, bce_dice_loss, device, ckpt=mgr, on_epoch_end=on_epoch_end,
                   loader=hub.loader, loader_key="train/all")
        model.load_state_dict(torch.load(best_path, map_location=device))
        return {"selection": {"best_epoch": st["extra"]["best_epoch"], "best_val_loss": st["extra"]["best_loss"]},
                "history": st["extra"]["history"], **test_report(model, hub),
                "timing": {k: st[k] for k in ("wall_s", "data_wait_s", "compute_s")}, "checkpoint_dir": str(mgr.dir)}
    finally:
        hub.close()


def run_local_only(data: Data, lr: float, epochs: int, device, git, ckpt_base, tag: str) -> dict:
    hub = data.hub("local_only", device)
    try:
        val_eval, test_eval, held_eval = hub.evaluator("val"), hub.evaluator("test"), hub.evaluator("heldout")
        per_client, test_counts, held_by_client, wall = {}, {}, {}, 0.0
        for e in data.clients:
            model = new_model(data.seed)
            mgr = ckpt_manager(f"{tag}_seed{data.seed}/{e}", git,
                               {"method": "local_only", "client": e, "lr": lr, "epochs": epochs, "seed": data.seed},
                               ckpt_base)
            best_path = mgr.dir / "best_model.pt"

            def on_epoch_end(epoch, m, extra, e=e, best_path=best_path):
                loss = val_eval.evaluate(m, clients=[e])[e]["loss"]          # own val set only
                extra.setdefault("history", []).append({"epoch": epoch + 1, "val_loss": loss})
                if loss < extra.get("best_loss", float("inf")):
                    extra.update(best_loss=loss, best_epoch=epoch + 1)
                    save_atomic({k: v.detach().cpu() for k, v in m.state_dict().items()}, best_path)

            cfg = TrainConfig(epochs=epochs, batch_size=COMMON["batch_size"], lr=lr,
                              weight_decay=COMMON["weight_decay"], amp=True, num_workers=COMMON["num_workers"],
                              seed=data.seed)
            st = train(model, data.train[e], cfg, bce_dice_loss, device, ckpt=mgr, on_epoch_end=on_epoch_end,
                       loader=hub.loader, loader_key=f"train/{e}")
            wall += st["wall_s"]
            model.load_state_dict(torch.load(best_path, map_location=device))
            test_counts[e] = test_eval.evaluate(model, clients=[e])[e]
            held_by_client[e] = pooled_f1(held_eval.evaluate(model))
            per_client[e] = {"best_epoch": st["extra"]["best_epoch"], "best_val_loss": st["extra"]["best_loss"],
                             "history": st["extra"]["history"], "checkpoint_dir": str(mgr.dir)}
        return {"selection": per_client, "test_per_event": test_counts, "test_pooled_loc_f1": pooled_f1(test_counts),
                "heldout_loc_f1_by_client_model": held_by_client,
                "heldout_loc_f1": float(np.mean(list(held_by_client.values()))),
                "heldout_note": "mean over the 9 client models (local-only has no single model)",
                "timing": {"wall_s": wall}}
    finally:
        hub.close()


def run_fedavg(data: Data, lr: float, rounds: int, device, git, ckpt_base, tag: str,
               max_rounds_this_call: int | None = None) -> dict:
    model = new_model(data.seed)
    hub = data.hub("fedavg", device)
    try:
        cfg = SegFLConfig(rounds=rounds, local_epochs=LOCAL_EPOCHS, batch_size=COMMON["batch_size"], lr=lr,
                          weight_decay=COMMON["weight_decay"], num_workers=COMMON["num_workers"], seed=data.seed,
                          max_rounds_this_call=max_rounds_this_call)
        mgr = ckpt_manager(f"{tag}_seed{data.seed}", git, {"method": "fedavg", "lr": lr, "rounds": rounds,
                                                            "E": LOCAL_EPOCHS, "seed": data.seed}, ckpt_base)
        mgr.every = 1                                                    # checkpoint every round
        st = run_seg_fedavg(model, data.train, hub.evaluator("val"), cfg, bce_dice_loss, device, ckpt=mgr,
                            loader=hub.loader, prefix="train/")
        out = {"selection": {"best_round": st["best_round"], "best_fed_val_loss": st["best_fed_val_loss"]},
               "history": st["history"], "checkpoint_dir": str(mgr.dir),
               "timing": {"round_s": [h["round_s"] for h in st["history"]]}}
        if st["best_state"] is not None and max_rounds_this_call is None:
            model.load_state_dict(st["best_state"])
            out.update(test_report(model, hub))
        return out
    finally:
        hub.close()


# ── driver ─────────────────────────────────────────────────────────────────

def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_run(run_dir: Path, method: str, res: dict, data: Data, lr: float, git: dict, runtime: float, extra_cfg: dict):
    write_metrics(run_dir / "metrics.json", {**res, "training_git": git}, seed=data.seed,
                  config={**COMMON, **extra_cfg, "method": method, "lr": lr,
                          "split_file": f"data/splits/silo2_seed{data.seed}.json"},
                  split_sizes=data.split_sizes(), runtime_s=runtime)


def lr_sweep(device, git, ckpt_base, limit=None, out_dir=SWEEP_DIR) -> dict:
    data = Data(0, limit)
    losses = {}
    for lr in LR_GRID:
        t0 = time.perf_counter()
        res = run_centralized(data, lr, SWEEP_EPOCHS, device, git, ckpt_base, tag=f"sweep_lr{lr:g}")
        write_run(out_dir / f"centralized_lr{lr:g}_seed0", "centralized", res, data, lr, git,
                  time.perf_counter() - t0, {"epochs": SWEEP_EPOCHS})
        losses[lr] = res["selection"]["best_val_loss"]
        print(f"  sweep lr={lr:g}: best val loss {losses[lr]:.4f} @ epoch {res['selection']['best_epoch']}", flush=True)
    best = min(losses, key=losses.get)
    choice = {"lr": best, "grid": LR_GRID, "val_loss_by_lr": {str(k): v for k, v in losses.items()},
              "chosen_lr_at_grid_edge": best in (LR_GRID[0], LR_GRID[-1]), "epochs": SWEEP_EPOCHS,
              "selection": "centralized, split seed 0, min best pooled val loss; no test data", "training_git": git}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    return choice


def summarize(seeds: list[int]) -> dict:
    runs = {m: {s: _load(EXPERIMENT_DIR / f"{m}_seed{s}" / "metrics.json") for s in seeds} for m in METHODS}
    runs = {m: {s: r["metrics"] for s, r in rs.items() if r} for m, rs in runs.items()}
    ms = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
    out = {"seeds": seeds, "methods": {}, "paired": {}}
    for m, rs in runs.items():
        if not rs:
            continue
        events = sorted(next(iter(rs.values()))["test_per_event"])
        out["methods"][m] = {
            "test_pooled_loc_f1": ms([r["test_pooled_loc_f1"] for r in rs.values()]),
            "heldout_loc_f1": ms([r["heldout_loc_f1"] for r in rs.values()]),
            "test_loc_f1_per_event": {e: ms([r["test_per_event"][e]["f1"] for r in rs.values()]) for e in events}}
    for a, b in (("fedavg", "local_only"), ("fedavg", "centralized"), ("centralized", "local_only")):
        common = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
        if len(common) >= 2:
            x = np.array([runs[a][s]["test_pooled_loc_f1"] for s in common])
            y = np.array([runs[b][s]["test_pooled_loc_f1"] for s in common])
            out["paired"][f"{a}_minus_{b}"] = {**ms(x - y), "p_paired_t": float(sstats.ttest_rel(x, y).pvalue),
                                              "seeds": common}
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


def measure_round(device, git, ckpt_base) -> None:
    """Timing only: one FedAvg round (9 clients, E = 1) with the persistent loader + one federated val evaluation."""
    data = Data(0)
    t0 = time.perf_counter()
    res = run_fedavg(data, 3e-4, ROUNDS, device, git, ckpt_base, tag="measure_round", max_rounds_this_call=1)
    h = res["history"][0]
    metrics = {"note": "Timing of ONE FedAvg round; no accuracy claims (val numbers are not reported).",
               "round_s": h["round_s"], "train_s": h["train_s"], "eval_s": h["eval_s"],
               "train_pairs": sum(len(d) for d in data.train.values()),
               "val_images": sum(len(d) for d in data.val.values()),
               "process_wall_s_incl_worker_startup": time.perf_counter() - t0,
               "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
               "training_git": git}
    out = write_metrics(RESULTS_DIR / "silo2_feasibility" / "fl_round_resnet18_crop512_bs4" / "metrics.json", metrics,
                        seed=0, config={**COMMON, "lr": 3e-4, "local_epochs": LOCAL_EPOCHS, "clients": data.clients,
                                        "loader": "one persistent DataLoader + client-aware batch sampler"},
                        split_sizes=data.split_sizes(), runtime_s=h["round_s"])
    print(f"wrote {out}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "training_git"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Silo 2 stage 1 (localisation) experiments.")
    ap.add_argument("--lr-sweep", action="store_true")
    ap.add_argument("--method", nargs="+", choices=METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--measure-round", action="store_true")
    ap.add_argument("--checkpoint-dir", default=None, help="default: checkpoint_dir from the user config")
    ap.add_argument("--smoke", type=int, default=None, help="limit pairs per client/split (code check only)")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.seeds)
        return
    git = require_clean_tree(args.allow_dirty)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # smoke runs (code check on a few pairs) never write into the repo
    out_root = EXPERIMENT_DIR if args.smoke is None else Path(tempfile.gettempdir()) / "urbanfl_smoke_silo2_loc"
    if args.measure_round:
        measure_round(device, git, args.checkpoint_dir)
        return
    if args.lr_sweep:
        print(json.dumps(lr_sweep(device, git, args.checkpoint_dir, args.smoke, out_root / "lr_sweep"), indent=1))
        return
    choice = _load(out_root / "lr_sweep" / "choice.json")
    if choice is None:
        raise FileNotFoundError("run the lr sweep first (--lr-sweep)")
    lr = choice["lr"]
    epochs, rounds = (EPOCHS, ROUNDS) if args.smoke is None else (1, 1)
    for seed in args.seeds:
        data = Data(seed, args.smoke)
        for method in args.method or []:
            t0 = time.perf_counter()
            if method == "centralized":
                res = run_centralized(data, lr, epochs, device, git, args.checkpoint_dir, "centralized")
            elif method == "local_only":
                res = run_local_only(data, lr, epochs, device, git, args.checkpoint_dir, "local_only")
            else:
                res = run_fedavg(data, lr, rounds, device, git, args.checkpoint_dir, "fedavg")
            write_run(out_root / f"{method}_seed{seed}", method, res, data, lr, git, time.perf_counter() - t0,
                      {"epochs": epochs} if method != "fedavg" else {"rounds": rounds, "local_epochs": LOCAL_EPOCHS})
            print(f"seed {seed} {method}: test pooled loc F1 {res['test_pooled_loc_f1']:.4f}, "
                  f"held-out {res['heldout_loc_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
