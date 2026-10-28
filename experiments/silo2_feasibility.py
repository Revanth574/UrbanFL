"""
Silo 2 feasibility on the local GPU: one epoch of stage-1 building localisation (ResNet U-Net, 512 crops, AMP) on the
TRAIN split, plus the timings needed to project full runs. Produces NO accuracy claims: the train loss is logged only
as a sanity signal, and val/test data are not touched (eval cost is timed on train images).

Measured: peak VRAM, seconds per epoch, data-wait vs compute time, DataLoader start-up (paid per client per round in
FL), full-resolution eval time per image, FedAvg aggregation time for this model, checkpoint size / save time.
Projected (arithmetic on the measurements, labelled as projections):
  centralized  = epochs x (epoch time + val eval)
  FL, 9 clients, E local epochs, R rounds = R x (E x epoch time + 9 x loader start-up + aggregation + federated val eval)

Usage:  python -m experiments.silo2_feasibility [--encoder resnet18] [--batch-size 4] [--num-workers 4]
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from src.fl.algorithms import fedavg_aggregate
from src.fl.checkpoint import CheckpointManager, resolve_checkpoint_dir
from src.fl.config import xbd_split_dir
from src.fl.models import UNetResNet, bce_dice_loss
from src.fl.partition_xbd import load_xbd_partition, pair_ids
from src.fl.seg_train import TrainConfig, train
from src.fl.utils import RESULTS_DIR, require_clean_tree, set_seed, write_metrics
from src.fl.xbd_data import XBDPairDataset

EXPERIMENT_DIR = RESULTS_DIR / "silo2_feasibility"


def time_loader_startup(ds, batch_size: int, num_workers: int, repeats: int = 3) -> float:
    """Seconds from DataLoader creation to the first batch (worker spawn + first decode), averaged."""
    times = []
    for r in range(repeats):
        t0 = time.perf_counter()
        next(iter(DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)))
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


@torch.no_grad()
def time_full_res_eval(model, ds, device, num_workers: int, n_images: int) -> float:
    """Seconds per full-resolution (1024 x 1024) image: load + AMP forward, batch 1, eval mode."""
    model.eval()
    sub = torch.utils.data.Subset(ds, list(range(min(n_images, len(ds)))))
    loader = DataLoader(sub, batch_size=1, num_workers=num_workers)
    it = iter(loader)
    first = next(it)                                              # exclude worker start-up and cuDNN warm-up
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        model(first["pre"].to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0, n = time.perf_counter(), 0
    for batch in it:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            model(batch["pre"].to(device)).float().sigmoid()
        n += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / max(1, n)


def time_aggregation(model, n_clients: int = 9) -> float:
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.perf_counter()
    fedavg_aggregate(state, [state] * n_clients, [1] * n_clients)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--encoder", default="resnet18", choices=list(UNetResNet.ENCODERS))
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--eval-images", type=int, default=40)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    git = require_clean_tree(args.allow_dirty)
    if not torch.cuda.is_available():
        raise SystemExit("feasibility is about the local GPU; CUDA is not available")
    device = torch.device("cuda")
    run_id = f"loc_{args.encoder}_crop{args.crop}_bs{args.batch_size}"

    split_dir = xbd_split_dir("train")
    part = load_xbd_partition(args.split_seed)                    # spatial_block (default)
    train_ids = pair_ids(part, "train")
    n_val = len(pair_ids(part, "val"))
    clients = {e: len(c["train"]) for e, c in part["clients"].items()}
    set_seed(0)
    ds = XBDPairDataset(split_dir, train_ids, "loc", train=True, crop=args.crop, seed=0)
    model = UNetResNet(args.encoder, pretrained=True)
    n_params = sum(p.numel() for p in model.parameters())

    cfg = TrainConfig(epochs=1, batch_size=args.batch_size, lr=args.lr, amp=True, num_workers=args.num_workers,
                      seed=0, log_every=25)
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, f"silo2_feasibility_{run_id}_{git['commit'][:7]}")
    signature = {"run": run_id, "split_seed": args.split_seed, "lr": args.lr, "commit": git["commit"]}
    mgr = CheckpointManager(ckpt_dir, every_n_steps=args.checkpoint_every, keep_last=2, signature=signature)
    stats = train(model, ds, cfg, bce_dice_loss, device, ckpt=mgr)
    if stats["resumed_from_step"] is not None:
        raise SystemExit(f"resumed from an existing checkpoint in {ckpt_dir}: timings would be partial; "
                         "delete it or pass another --checkpoint-dir")
    epoch_s = stats["wall_s"]

    loader_startup_s = time_loader_startup(XBDPairDataset(split_dir, train_ids[:64], "loc", train=True,
                                                          crop=args.crop), args.batch_size, args.num_workers)
    eval_s = time_full_res_eval(model, XBDPairDataset(split_dir, train_ids, "loc", train=False), device,
                                args.num_workers, args.eval_images)
    agg_s = time_aggregation(model, len(clients))
    latest = mgr.latest()
    t0 = time.perf_counter()
    mgr.save(torch.load(latest, weights_only=False), stats["global_step"] + 1)   # time one save of the same state
    ckpt_save_s = time.perf_counter() - t0

    val_eval_s = n_val * eval_s
    proj = {
        "centralized_30_epochs_h": 30 * (epoch_s + val_eval_s) / 3600,
        "centralized_30_epochs_train_only_h": 30 * epoch_s / 3600,
    }
    for rounds in (30, 50):
        for e in (1, 2):
            per_round = e * epoch_s + len(clients) * loader_startup_s + agg_s + val_eval_s
            proj[f"fl_9clients_E{e}_R{rounds}_h"] = rounds * per_round / 3600
    metrics = {
        "note": "Feasibility timings only. Train loss is a sanity signal, not a result; no val/test data were used.",
        "model": {"arch": f"U-Net, {args.encoder} encoder (ImageNet weights)", "params": n_params},
        "train_pairs": len(train_ids), "steps_per_epoch": stats["global_step"],
        "peak_vram_allocated_mib": stats["peak_vram_allocated_mib"],
        "peak_vram_reserved_mib": stats["peak_vram_reserved_mib"],
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "epoch_seconds": epoch_s, "data_wait_seconds": stats["data_wait_s"], "compute_seconds": stats["compute_s"],
        "seconds_per_step": epoch_s / stats["global_step"],
        "dataloader_startup_seconds": loader_startup_s,
        "full_res_eval_seconds_per_image": eval_s, "val_images": n_val, "val_eval_seconds": val_eval_s,
        "fedavg_aggregation_seconds_9_clients": agg_s,
        "checkpoint_mib": latest.stat().st_size / 2**20, "checkpoint_save_seconds": ckpt_save_s,
        "clients_train_pairs": clients,
        "projections_hours": proj,
        "projection_formulae": {
            "centralized": "epochs x (epoch_seconds + val_images x eval_s_per_image)",
            "fl": "rounds x (E x epoch_seconds + 9 x dataloader_startup + aggregation + val_images x eval_s_per_image)",
        },
        "train_loss_log_sanity_only": stats["loss_log"],
        "training_git": git,
    }
    out = write_metrics(EXPERIMENT_DIR / run_id / "metrics.json", metrics, seed=0,
                        config={**vars(args), "train_config": cfg.__dict__, "split_strategy": part["strategy"],
                                "checkpoint_dir": str(ckpt_dir)},
                        split_sizes=part["stats"]["split_sizes"], runtime_s=epoch_s)
    print(f"wrote {out}")
    for k in ("peak_vram_allocated_mib", "peak_vram_reserved_mib", "epoch_seconds", "seconds_per_step",
              "data_wait_seconds", "compute_seconds", "dataloader_startup_seconds", "full_res_eval_seconds_per_image",
              "val_eval_seconds", "fedavg_aggregation_seconds_9_clients", "checkpoint_mib", "checkpoint_save_seconds"):
        print(f"  {k:40s} {metrics[k]:.2f}")
    for k, v in proj.items():
        print(f"  projection {k:32s} {v:.2f} h")


if __name__ == "__main__":
    main()
