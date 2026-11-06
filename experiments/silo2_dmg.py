"""
Silo 2 stage 2 — damage classification (pre + post image -> 4 damage classes on building pixels), stage-1 protocol.

Task (xView2, decisions X1/X2): no / minor / major / destroyed on building pixels; un-classified buildings and
background are ignored in the loss. Score = official xView2 score (src/fl/xbd_metrics.py): 0.3 x localisation F1 of
the method's STAGE-1 model + 0.7 x harmonic mean of the 4 damage F1s, damage scored on post-target building pixels
with dp = 0 wherever stage 1 predicts no building.

Model: SiameseDamageNet — the method's own stage-1 U-Net body shared between pre and post, new 1x1 head on the
concatenated features. Every method starts from ITS OWN stage-1 weights, so no method borrows from another:
  centralized : stage-1 centralized (same seed)              local_only : client k from stage-1 local-only client k
  fedavg      : stage-1 FedAvg global (same seed)            fedavg_ft  : client k fine-tunes the stage-2 FedAvg model
Localisation masks for scoring come from the same method's stage-1 model(s): local_only client k uses its own
stage-1 model, fedavg_ft client k its stage-1 fedavg_ft model.

Loss: class-weighted cross-entropy on scored pixels (src/fl/dmg_eval.py). Class weights = the Silo 1 rule
(models.class_weights_from_counts: total / (n_present x count_c), absent classes 0) on DAMAGE PIXEL counts of the
training split, capped at WEIGHT_CAP: centralized = pooled counts; fedavg / fedavg_ft = federated sum of the clients'
4-bin pixel histograms (one-off upload, same numbers as pooled); local_only = the client's own counts. The cap never
binds on pooled counts (max ~4.8); it stops a client's near-absent class (e.g. palu minor-damage, 0.01% of pixels ->
raw weight ~3,500) from dominating that client's loss. Raw and capped weights are both reported (--class-stats).
Training pairs: pairs with >= 1 classified building (building-free pairs give no damage gradient); val / test /
held-out: all pairs (localisation F1 needs them). Selection never reads test data:
  lr sweep    : centralized, split seed 0, LR_GRID, SWEEP_EPOCHS, min best pooled val loss
  centralized : EPOCHS, epoch with lowest pooled val loss      local_only : EPOCHS, lowest loss on its own val set
  fedavg      : E = 1, ROUNDS, lowest image-weighted federated val loss
  fedavg_ft   : epoch 0..FT_EPOCHS (0 = FedAvg unchanged) by the client's own val loss, fresh AdamW, same lr
Outputs per run: metrics.json (scores, counts, selection) + buildings_test.json / buildings_heldout.json (one record
per building of the post labels: true subtype, predicted class by majority vote, 0 = not localised).

Usage:  python -m experiments.silo2_dmg --class-stats --seeds 0 1 2
        python -m experiments.silo2_dmg --measure                 # timing + projection only
        python -m experiments.silo2_dmg --lr-sweep
        python -m experiments.silo2_dmg --method centralized local_only fedavg fedavg_ft --seeds 0 1 2
        python -m experiments.silo2_dmg --summarize
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

from experiments.silo2_loc import COMMON as LOC_COMMON
from experiments.silo2_loc import NOISY_MIN_TEST_PAIRS, PROTOCOL as LOC_PROTOCOL, SENSITIVITY_EXCLUDE
from experiments.silo2_loc import completed, prune_run
from src.fl.checkpoint import CheckpointManager, resolve_checkpoint_dir, save_atomic
from src.fl.client_loader import PersistentClientLoader
from src.fl.config import xbd_split_dir
from src.fl.dmg_eval import DamageValEvaluator, dmg_batch_to_io, pooled_score, weighted_ce_loss, xview2_evaluate
from src.fl.models import SiameseDamageNet, UNetResNet, class_weights_from_counts
from src.fl.partition_xbd import load_xbd_partition
from src.fl.seg_eval import weighted_loss
from src.fl.seg_fl import SegFLConfig, client_seed, run_seg_fedavg
from src.fl.seg_train import TrainConfig, train
from src.fl.utils import RESULTS_DIR, require_clean_tree, set_seed, write_metrics
from src.fl.xbd_data import DAMAGE_CLASSES, XBDPairDataset, damage_pixel_counts, has_classified_building

EXPERIMENT_DIR = RESULTS_DIR / "silo2_dmg"
STAGE1_DIR = RESULTS_DIR / "silo2_loc"
METHODS = ("centralized", "local_only", "fedavg", "fedavg_ft")
LR_GRID = [1e-4, 3e-4, 1e-3]
SWEEP_EPOCHS, EPOCHS, ROUNDS, LOCAL_EPOCHS, FT_EPOCHS = 10, 30, 30, 1, 2
WEIGHT_CAP = 10.0
CLASS_WEIGHTS = (f"models.class_weights_from_counts on damage-pixel counts of the training split (Silo 1 rule), "
                 f"capped at {WEIGHT_CAP:g}")
PROTOCOL = {**LOC_PROTOCOL,
            "init": "each method from its own stage-1 weights (see module docstring); new head seeded with the run seed",
            "class_weights": {"rule": CLASS_WEIGHTS, "centralized": "pooled counts",
                              "fedavg_and_fedavg_ft": "federated sum of client 4-bin pixel histograms",
                              "local_only": "own counts (absent classes weight 0)"},
            "fedavg_ft_optimizer": "fresh AdamW per client, lr = sweep lr; loss weights = federated weights"}
COMMON = {**{k: v for k, v in LOC_COMMON.items() if k not in ("loss", "protocol")},
          "model": "SiameseDamageNet(stage-1 UNetResNet body shared pre/post, 1x1 head on 64 ch)",
          "loss": "class-weighted cross-entropy on scored building pixels (background, un-classified ignored)",
          "train_pairs": "pairs with >= 1 classified building", "protocol": PROTOCOL}


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ── data ───────────────────────────────────────────────────────────────────

class DmgData:
    """Per-client damage-stage datasets for split seed `seed` (train: pairs with a classified building)."""

    def __init__(self, seed: int, limit: int | None = None):
        self.dir, self.seed = xbd_split_dir("train"), seed
        part = load_xbd_partition(seed)
        cut = (lambda ids: ids[:limit]) if limit else (lambda ids: ids)
        c = part["clients"]
        self.clients = sorted(c)
        lab = lambda p: self.dir / "labels" / f"{p}_post_disaster.json"        # noqa: E731
        with ThreadPoolExecutor(8) as ex:
            self.train_ids = {e: cut([p for p, ok in zip(c[e]["train"], ex.map(has_classified_building,
                                                                               map(lab, c[e]["train"]))) if ok])
                              for e in self.clients}
        self.train = {e: XBDPairDataset(self.dir, self.train_ids[e], "damage", train=True, crop=COMMON["crop"],
                                        seed=seed) for e in self.clients}
        self.val = {e: XBDPairDataset(self.dir, cut(c[e]["val"]), "damage") for e in self.clients}
        self.test = {e: XBDPairDataset(self.dir, cut(c[e]["test"]), "damage") for e in self.clients}
        self.heldout = {e: XBDPairDataset(self.dir, cut(ids), "damage") for e, ids in part["heldout"].items()}
        self.all_train_ids = [p for e in self.clients for p in self.train_ids[e]]
        self.n_train_pairs_before_filter = {e: len(cut(c[e]["train"])) for e in self.clients}
        self._counts = None

    def class_counts(self) -> dict[str, list[int]]:
        if self._counts is None:
            with ThreadPoolExecutor(8) as ex:
                res = ex.map(lambda e: damage_pixel_counts(self.dir, self.train_ids[e]), self.clients)
                self._counts = {e: r.tolist() for e, r in zip(self.clients, res)}
        return self._counts

    def weights(self, client: str | None = None) -> torch.Tensor:
        """Federated / pooled weights (client=None) or one client's own weights."""
        return self.raw_weights(client).clamp(max=WEIGHT_CAP)

    def raw_weights(self, client: str | None = None) -> torch.Tensor:
        counts = self.class_counts()
        return class_weights_from_counts(np.sum(list(counts.values()), axis=0) if client is None else counts[client])

    def split_sizes(self) -> dict:
        return {e: {"train": len(self.train[e]), "train_before_building_filter": self.n_train_pairs_before_filter[e],
                    "val": len(self.val[e]), "test": len(self.test[e])} for e in self.clients}


class DmgHub:
    """One persistent DataLoader per run (see silo2_loc.Hub): train/<client> or train/all, val/, test/, heldout/."""

    def __init__(self, data: DmgData, method: str, device):
        self.data, self.device = data, torch.device(device)
        sets, bs = {}, {}
        if method == "centralized":
            sets["train/all"] = XBDPairDataset(data.dir, data.all_train_ids, "damage", train=True,
                                               crop=COMMON["crop"], seed=data.seed)
        else:
            sets.update({f"train/{e}": d for e, d in data.train.items()})
        bs.update({k: COMMON["batch_size"] for k in sets})
        for split, group in (("val", data.val), ("test", data.test), ("heldout", data.heldout)):
            for e, d in group.items():
                sets[f"{split}/{e}"], bs[f"{split}/{e}"] = d, COMMON["eval_batch_size"]
        index = {e: i for i, e in enumerate(data.clients)}

        def seed_of(key: str) -> int:
            if method == "fedavg" and key.startswith("train/"):
                return client_seed(data.seed, index[key.split("/", 1)[1]])
            return data.seed

        self.sets = sets
        self.loader = PersistentClientLoader(sets, bs, COMMON["num_workers"], seed_of=seed_of,
                                             pin_memory=self.device.type == "cuda",
                                             loader_seed=data.seed * 1_000_003 + 17)

    def val_evaluator(self, weights_of) -> DamageValEvaluator:
        return DamageValEvaluator(self.data.val, weights_of, self.device, self.loader, "val/")

    def score(self, loc_model, dmg_model, which: str, clients: list[str], with_buildings: bool = True) -> dict:
        return xview2_evaluate(loc_model.to(self.device), dmg_model.to(self.device), self.loader, f"{which}/",
                               clients, self.data.dir, self.device, with_buildings=with_buildings)

    def close(self) -> None:
        self.loader.close()


# ── stage-1 sources ────────────────────────────────────────────────────────

def stage1_source(method: str, seed: int, root: Path, client: str | None = None) -> dict:
    m = _load(root / f"{method}_seed{seed}" / "metrics.json")
    if m is None:
        raise FileNotFoundError(f"stage-1 {method} seed {seed} has no results under {root}")
    d = m["metrics"]["checkpoint_dir"] if client is None else m["metrics"]["selection"][client]["checkpoint_dir"]
    return {"stage1_metrics": str(root / f"{method}_seed{seed}" / "metrics.json"),
            "best_model": str(Path(d) / "best_model.pt"), "training_commit": m["metrics"]["training_git"]["commit"]}


def load_unet(path: str) -> UNetResNet:
    m = UNetResNet(COMMON["encoder"], pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m


def new_dmg_model(seed: int, stage1_path: str) -> SiameseDamageNet:
    set_seed(seed)                                         # seeds the new head
    return SiameseDamageNet.from_stage1(torch.load(stage1_path, map_location="cpu"), COMMON["encoder"])


def ckpt_manager(run: str, git: dict, signature: dict, base: str | None) -> CheckpointManager:
    d = resolve_checkpoint_dir(base, f"silo2_dmg/{run}_{git['commit'][:7]}")
    return CheckpointManager(d, COMMON["ckpt_every_steps"], keep_last=2, signature={**signature, "commit": git["commit"]})


# ── reporting ──────────────────────────────────────────────────────────────

def _strip(per_client: dict) -> tuple[dict, dict]:
    """(scores without building records, building records by client)."""
    return ({c: {k: v for k, v in r.items() if k != "buildings"} for c, r in per_client.items()},
            {c: r["buildings"] for c, r in per_client.items() if "buildings" in r})


def report_test(test: dict) -> dict:
    per_event = {e: {**v, "noisy": v["n_images"] < NOISY_MIN_TEST_PAIRS} for e, v in test.items()}
    pooled = pooled_score(test)
    excl = pooled_score({e: v for e, v in test.items() if e not in SENSITIVITY_EXCLUDE})
    return {"test_xview2_score": pooled["score"], "test_pooled": pooled,
            "test_xview2_score_excl_socal_fire": excl["score"], "test_per_event": per_event,
            "noisy_events": sorted(e for e, v in per_event.items() if v["noisy"]),
            "reporting": {"headline": "test_xview2_score (official definition, TP/FP/FN pooled over all test images)",
                          "noisy_rule": f"< {NOISY_MIN_TEST_PAIRS} test pairs",
                          "sensitivity_excluded": list(SENSITIVITY_EXCLUDE)}}


def single_model_report(hub: DmgHub, loc_model, dmg_model) -> tuple[dict, dict]:
    test, b_test = _strip(hub.score(loc_model, dmg_model, "test", hub.data.clients))
    held, b_held = _strip(hub.score(loc_model, dmg_model, "heldout", sorted(hub.data.heldout)))
    return ({**report_test(test), "heldout": held, "heldout_xview2_score": pooled_score(held)["score"]},
            {"test": b_test, "heldout": b_held})


# ── runs ───────────────────────────────────────────────────────────────────

def run_centralized(data: DmgData, lr: float, epochs: int, device, git, ckpt_base, tag: str, stage1_root: Path):
    src = stage1_source("centralized", data.seed, stage1_root)
    model = new_dmg_model(data.seed, src["best_model"])
    w = data.weights()
    hub = DmgHub(data, "centralized", device)
    try:
        val_eval = hub.val_evaluator(lambda c: w)
        mgr = ckpt_manager(f"{tag}_seed{data.seed}", git, {"method": "centralized", "lr": lr, "epochs": epochs,
                                                           "seed": data.seed, "init": src["best_model"]}, ckpt_base)
        best_path = mgr.dir / "best_model.pt"

        def on_epoch_end(epoch, m, extra):
            loss = weighted_loss(val_eval.evaluate(m))
            extra.setdefault("history", []).append({"epoch": epoch + 1, "val_loss": loss})
            if loss < extra.get("best_loss", float("inf")):
                extra.update(best_loss=loss, best_epoch=epoch + 1)
                save_atomic({k: v.detach().cpu() for k, v in m.state_dict().items()}, best_path)
            print(f"EPOCH {tag}_seed{data.seed} {epoch + 1}/{epochs}: val_loss={loss:.4f} "
                  f"(best {extra['best_loss']:.4f} @ epoch {extra['best_epoch']})", flush=True)

        cfg = TrainConfig(epochs=epochs, batch_size=COMMON["batch_size"], lr=lr, weight_decay=COMMON["weight_decay"],
                          amp=True, num_workers=COMMON["num_workers"], seed=data.seed)
        st = train(model, hub.sets["train/all"], cfg, weighted_ce_loss(w), device, ckpt=mgr,
                   batch_to_io=dmg_batch_to_io, on_epoch_end=on_epoch_end, loader=hub.loader, loader_key="train/all")
        model.load_state_dict(torch.load(best_path, map_location=device))
        rep, buildings = single_model_report(hub, load_unet(src["best_model"]), model)
        return ({"selection": {"best_epoch": st["extra"]["best_epoch"], "best_val_loss": st["extra"]["best_loss"]},
                 "history": st["extra"]["history"], **rep, "class_weights": w.tolist(),
                 "class_pixel_counts": data.class_counts(), "stage1_source": src,
                 "timing": {k: st[k] for k in ("wall_s", "data_wait_s", "compute_s")},
                 "checkpoint_dir": str(mgr.dir)}, buildings)
    finally:
        hub.close()


def run_per_client(data: DmgData, lr: float, epochs: int, device, git, ckpt_base, tag: str, method: str,
                   init_of, loc_of, weights_of, epoch0: bool, sources: dict) -> tuple[dict, dict]:
    """local_only (epoch0=False) and fedavg_ft (epoch0=True): one model per client, own val set selects."""
    hub = DmgHub(data, "local_only", device)
    try:
        val_eval = hub.val_evaluator(weights_of)
        per_client, test, held_by_client, b_test, b_held, wall = {}, {}, {}, {}, {}, 0.0
        for e in data.clients:
            init_path = init_of(e)
            model = new_dmg_model(data.seed, init_path) if not epoch0 else _load_dmg(init_path)
            init_state = {k: v.clone() for k, v in model.state_dict().items()} if epoch0 else None
            mgr = ckpt_manager(f"{tag}_seed{data.seed}/{e}", git, {"method": method, "client": e, "lr": lr,
                                                                    "epochs": epochs, "seed": data.seed,
                                                                    "init": init_path}, ckpt_base)
            best_path, done_path = mgr.dir / "best_model.pt", mgr.dir / "done.json"
            if done_path.exists() and best_path.exists():
                info = json.loads(done_path.read_text(encoding="utf-8"))
                print(f"SKIP {method} client {e}: done.json found (best epoch {info['best_epoch']}), not retrained",
                      flush=True)
            else:
                start_loss = val_eval.evaluate(model.to(device), clients=[e])[e]["loss"] if epoch0 else None

                def on_epoch_end(epoch, m, extra, e=e, best_path=best_path, start_loss=start_loss,
                                 init_state=init_state):
                    if start_loss is not None and "history" not in extra:
                        extra.update(history=[{"epoch": 0, "val_loss": start_loss}], best_loss=start_loss, best_epoch=0)
                        save_atomic({k: v.detach().cpu() for k, v in init_state.items()}, best_path)
                    loss = val_eval.evaluate(m, clients=[e])[e]["loss"]
                    extra.setdefault("history", []).append({"epoch": epoch + 1, "val_loss": loss})
                    if loss < extra.get("best_loss", float("inf")):
                        extra.update(best_loss=loss, best_epoch=epoch + 1)
                        save_atomic({k: v.detach().cpu() for k, v in m.state_dict().items()}, best_path)
                    print(f"EPOCH {method} {e} {epoch + 1}/{epochs}: val_loss={loss:.4f} "
                          f"(best {extra['best_loss']:.4f} @ epoch {extra['best_epoch']})", flush=True)

                cfg = TrainConfig(epochs=epochs, batch_size=COMMON["batch_size"], lr=lr,
                                  weight_decay=COMMON["weight_decay"], amp=True, num_workers=COMMON["num_workers"],
                                  seed=data.seed)
                st = train(model, data.train[e], cfg, weighted_ce_loss(weights_of(e)), device, ckpt=mgr,
                           batch_to_io=dmg_batch_to_io, on_epoch_end=on_epoch_end, loader=hub.loader,
                           loader_key=f"train/{e}")
                info = {"best_epoch": st["extra"]["best_epoch"], "best_val_loss": st["extra"]["best_loss"],
                        "history": st["extra"]["history"], "wall_s": st["wall_s"], "training_git": git}
                done_path.write_text(json.dumps(info), encoding="utf-8")
                freed = mgr.prune_resume_checkpoints()
                print(f"DONE {method} client {e}: done.json written, pruned {freed / 2**20:.0f} MiB of resume "
                      f"checkpoints", flush=True)
            wall += info["wall_s"]
            model.to(device)
            model.load_state_dict(torch.load(best_path, map_location=device))
            loc = load_unet(loc_of(e))
            r = hub.score(loc, model, "test", [e])[e]
            b_test[e] = r.pop("buildings")
            test[e] = r
            h = hub.score(loc, model, "heldout", sorted(data.heldout))
            held, b_held[e] = _strip(h)
            held_by_client[e] = pooled_score(held)["score"]
            per_client[e] = {k: info[k] for k in ("best_epoch", "best_val_loss", "history")} | {
                "checkpoint_dir": str(mgr.dir), "init": init_path, "loc_model": loc_of(e),
                "class_weights": weights_of(e).tolist()}
        return ({"selection": per_client, **report_test(test), "heldout_xview2_by_client_model": held_by_client,
                 "heldout_xview2_score": float(np.mean(list(held_by_client.values()))),
                 "heldout_note": f"mean over the 9 client models ({method} has no single model)",
                 "class_pixel_counts": data.class_counts(), "sources": sources, "timing": {"wall_s": wall}},
                {"test": b_test, "heldout_by_client_model": b_held})
    finally:
        hub.close()


def _load_dmg(path: str) -> SiameseDamageNet:
    m = SiameseDamageNet(UNetResNet(COMMON["encoder"], pretrained=False), len(DAMAGE_CLASSES))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m


def run_local_only(data, lr, epochs, device, git, ckpt_base, tag, stage1_root):
    srcs = {e: stage1_source("local_only", data.seed, stage1_root, e) for e in data.clients}
    return run_per_client(data, lr, epochs, device, git, ckpt_base, tag, "local_only",
                          init_of=lambda e: srcs[e]["best_model"], loc_of=lambda e: srcs[e]["best_model"],
                          weights_of=lambda e: data.weights(e), epoch0=False, sources={"stage1_by_client": srcs})


def run_fedavg_ft(data, lr, ft_epochs, device, git, ckpt_base, tag, stage1_root, out_root):
    fed = _load(out_root / f"fedavg_seed{data.seed}" / "metrics.json")
    if fed is None:
        raise FileNotFoundError(f"stage-2 FedAvg seed {data.seed} must finish before its fine-tuning")
    fed_path = str(Path(fed["metrics"]["checkpoint_dir"]) / "best_model.pt")
    locs = {e: stage1_source("fedavg_ft", data.seed, stage1_root, e) for e in data.clients}
    w = data.weights()
    return run_per_client(data, lr, ft_epochs, device, git, ckpt_base, tag, "fedavg_ft",
                          init_of=lambda e: fed_path, loc_of=lambda e: locs[e]["best_model"], weights_of=lambda e: w,
                          epoch0=True, sources={"stage2_fedavg_best_model": fed_path,
                                                "stage2_fedavg_training_commit": fed["metrics"]["training_git"]["commit"],
                                                "stage1_fedavg_ft_by_client": locs})


def run_fedavg(data, lr, rounds, device, git, ckpt_base, tag, stage1_root, max_rounds_this_call=None):
    src = stage1_source("fedavg", data.seed, stage1_root)
    model = new_dmg_model(data.seed, src["best_model"])
    w = data.weights()
    hub = DmgHub(data, "fedavg", device)
    try:
        cfg = SegFLConfig(rounds=rounds, local_epochs=LOCAL_EPOCHS, batch_size=COMMON["batch_size"], lr=lr,
                          weight_decay=COMMON["weight_decay"], num_workers=COMMON["num_workers"], seed=data.seed,
                          max_rounds_this_call=max_rounds_this_call)
        mgr = ckpt_manager(f"{tag}_seed{data.seed}", git, {"method": "fedavg", "lr": lr, "rounds": rounds,
                                                            "E": LOCAL_EPOCHS, "seed": data.seed,
                                                            "init": src["best_model"]}, ckpt_base)
        mgr.every = 1
        st = run_seg_fedavg(model, data.train, hub.val_evaluator(lambda c: w), cfg, weighted_ce_loss(w), device,
                            ckpt=mgr, loader=hub.loader, prefix="train/", batch_to_io=dmg_batch_to_io)
        out = {"selection": {"best_round": st["best_round"], "best_fed_val_loss": st["best_fed_val_loss"]},
               "history": st["history"], "checkpoint_dir": str(mgr.dir), "class_weights": w.tolist(),
               "class_pixel_counts": data.class_counts(), "stage1_source": src,
               "timing": {"round_s": [h["round_s"] for h in st["history"]]}}
        buildings = {}
        if st["best_state"] is not None and max_rounds_this_call is None:
            save_atomic(st["best_state"], mgr.dir / "best_model.pt")
            model.load_state_dict(st["best_state"])
            rep, buildings = single_model_report(hub, load_unet(src["best_model"]), model)
            out.update(rep)
        return out, buildings
    finally:
        hub.close()


# ── driver ─────────────────────────────────────────────────────────────────

def write_run(run_dir: Path, method: str, res: dict, buildings: dict, data: DmgData, lr: float, git: dict,
              runtime: float, extra_cfg: dict) -> None:
    for name, recs in buildings.items():                    # building records: separate files (size)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"buildings_{name}.json").write_text(json.dumps(recs), encoding="utf-8")
    write_metrics(run_dir / "metrics.json", {**res, "protocol": PROTOCOL, "training_git": git}, seed=data.seed,
                  config={**COMMON, **extra_cfg, "method": method, "lr": lr,
                          "split_file": f"data/splits/silo2_seed{data.seed}.json"},
                  split_sizes=data.split_sizes(), runtime_s=runtime)


def lr_sweep(device, git, ckpt_base, stage1_root: Path, limit=None, out_dir=EXPERIMENT_DIR / "lr_sweep") -> dict:
    data = DmgData(0, limit)
    losses = {}
    for lr in LR_GRID:
        run_dir = out_dir / f"centralized_lr{lr:g}_seed0"
        res = completed(run_dir, git, allow_dirty=limit is not None)
        if res is None:
            t0 = time.perf_counter()
            res, b = run_centralized(data, lr, SWEEP_EPOCHS, device, git, ckpt_base, f"sweep_lr{lr:g}", stage1_root)
            write_run(run_dir, "centralized", res, {}, data, lr, git, time.perf_counter() - t0,
                      {"epochs": SWEEP_EPOCHS})
            prune_run(res)
        else:
            print(f"  sweep lr={lr:g}: already completed at this commit, reusing", flush=True)
        losses[lr] = res["selection"]["best_val_loss"]
        print(f"  sweep lr={lr:g}: best val loss {losses[lr]:.4f} @ epoch {res['selection']['best_epoch']}", flush=True)
    best = min(losses, key=losses.get)
    choice = {"lr": best, "grid": LR_GRID, "val_loss_by_lr": {str(k): v for k, v in losses.items()},
              "chosen_lr_at_grid_edge": best in (LR_GRID[0], LR_GRID[-1]), "epochs": SWEEP_EPOCHS,
              "selection": "centralized, split seed 0, min best pooled weighted-CE val loss; no test data",
              "training_git": git}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    return choice


def class_stats(seeds: list[int]) -> dict:
    """Damage-pixel counts and the resulting class weights per split seed (pooled = federated sum, and per client)."""
    out = {"rule": CLASS_WEIGHTS, "classes": list(DAMAGE_CLASSES), "seeds": {}}
    for s in seeds:
        d = DmgData(s)
        counts = d.class_counts()
        pooled = np.sum(list(counts.values()), axis=0)
        out["seeds"][str(s)] = {
            "pooled_pixel_counts": pooled.tolist(), "pooled_pixel_share": (pooled / pooled.sum()).tolist(),
            "pooled_weights": d.weights().tolist(), "pooled_weights_uncapped": d.raw_weights().tolist(),
            "by_client": {e: {"pixel_counts": counts[e], "weights": d.weights(e).tolist(),
                              "weights_uncapped": d.raw_weights(e).tolist(),
                              "train_pairs": len(d.train_ids[e]),
                              "train_pairs_before_building_filter": d.n_train_pairs_before_filter[e]}
                          for e in d.clients}}
    from src.fl.utils import git_info
    out["git"] = git_info()
    (EXPERIMENT_DIR / "class_stats").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "class_stats" / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def measure(device, git, ckpt_base, stage1_root: Path) -> dict:
    """Timing only: one centralized training epoch, its val pass, one FedAvg round, one test + held-out xView2
    evaluation; projects the campaign hours from these. No accuracy claims."""
    data = DmgData(0)
    src = stage1_source("centralized", 0, stage1_root)
    model, w = new_dmg_model(0, src["best_model"]), data.weights()
    hub = DmgHub(data, "centralized", device)
    try:
        cfg = TrainConfig(epochs=1, batch_size=COMMON["batch_size"], lr=3e-4, weight_decay=COMMON["weight_decay"],
                          amp=True, num_workers=COMMON["num_workers"], seed=0)
        st = train(model, hub.sets["train/all"], cfg, weighted_ce_loss(w), device, ckpt=None,
                   batch_to_io=dmg_batch_to_io, loader=hub.loader, loader_key="train/all")
        t0 = time.perf_counter()
        hub.val_evaluator(lambda c: w).evaluate(model)
        val_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        single_model_report(hub, load_unet(src["best_model"]), model)
        test_eval_s = time.perf_counter() - t0
    finally:
        hub.close()
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        fed, _ = run_fedavg(data, 3e-4, ROUNDS, device, git, tmp, "measure_round", stage1_root, max_rounds_this_call=1)
        fed_wall = time.perf_counter() - t0
    h = fed["history"][0]
    epoch_s = st["wall_s"] + val_s
    n_steps = st["global_step"]
    # local-only / fine-tuning: every client trains on its own pairs, so one pass over all clients ~ one epoch, plus
    # one own-val evaluation per client per epoch (~ one full val pass) and one test + held-out scoring per client
    proj = {
        "lr_sweep_h": len(LR_GRID) * (SWEEP_EPOCHS * epoch_s + test_eval_s) / 3600,
        "centralized_per_seed_h": (EPOCHS * epoch_s + test_eval_s) / 3600,
        "local_only_per_seed_h": (EPOCHS * epoch_s + len(data.clients) * test_eval_s) / 3600,
        "fedavg_per_seed_h": (ROUNDS * h["round_s"] + test_eval_s) / 3600,
        "fedavg_ft_per_seed_h": (FT_EPOCHS * epoch_s + val_s + len(data.clients) * test_eval_s) / 3600,
    }
    per_seed = sum(v for k, v in proj.items() if k.endswith("per_seed_h"))
    proj.update(per_seed_h=per_seed, seeds=3, total_h=proj["lr_sweep_h"] + 3 * per_seed,
                assumptions="epoch = measured train + val pass; local-only = EPOCHS pooled epochs + 9 client "
                            "test/held-out scorings (held-out scored by each client model, an over-estimate); "
                            "no sleep / crash time")
    metrics = {"note": "Timing only (seed-0 split, real data, RTX 3050); no accuracy claims.",
               "train_pairs": len(data.all_train_ids), "steps_per_epoch": n_steps,
               "epoch_train_s": st["wall_s"], "s_per_step": st["wall_s"] / max(1, n_steps),
               "data_wait_s": st["data_wait_s"], "compute_s": st["compute_s"], "val_pass_s": val_s,
               "test_and_heldout_scoring_s": test_eval_s, "fedavg_round_s": h["round_s"],
               "fedavg_round_train_s": h["train_s"], "fedavg_round_eval_s": h["eval_s"],
               "fedavg_process_wall_s": fed_wall, "peak_vram_allocated_mib": st.get("peak_vram_allocated_mib"),
               "fedavg_peak_vram_allocated_mib": fed.get("peak_vram_allocated_mib"),
               "protocol_counts": {"lr_grid": LR_GRID, "sweep_epochs": SWEEP_EPOCHS, "epochs": EPOCHS,
                                   "rounds": ROUNDS, "ft_epochs": FT_EPOCHS},
               "projection_hours": proj, "training_git": git}
    write_metrics(RESULTS_DIR / "silo2_feasibility" / "dmg_epoch_round_resnet18_crop512_bs4" / "metrics.json",
                  metrics, seed=0, config={**COMMON, "lr": 3e-4}, split_sizes=data.split_sizes(),
                  runtime_s=st["wall_s"] + val_s + test_eval_s + fed_wall)
    print(json.dumps({k: v for k, v in metrics.items() if k != "training_git"}, indent=1))
    return metrics


def summarize(seeds: list[int]) -> dict:
    runs = {m: {s: _load(EXPERIMENT_DIR / f"{m}_seed{s}" / "metrics.json") for s in seeds} for m in METHODS}
    runs = {m: {s: r["metrics"] for s, r in rs.items() if r} for m, rs in runs.items()}
    ms = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}  # noqa: E731
    out = {"seeds": seeds, "methods": {}, "paired": {}}
    for m, rs in runs.items():
        if not rs:
            continue
        out["methods"][m] = {
            "test_xview2_score (headline)": ms([r["test_xview2_score"] for r in rs.values()]),
            "test_loc_f1": ms([r["test_pooled"]["loc_f1"] for r in rs.values()]),
            "test_damage_f1_harmonic": ms([r["test_pooled"]["damage_f1_harmonic"] for r in rs.values()]),
            "test_damage_f1_per_class": {c: ms([r["test_pooled"]["damage_f1"][c] for r in rs.values()])
                                         for c in DAMAGE_CLASSES},
            "test_xview2_score_excl_socal_fire (sensitivity)":
                ms([r["test_xview2_score_excl_socal_fire"] for r in rs.values()]),
            "heldout_xview2_score": ms([r["heldout_xview2_score"] for r in rs.values()])}
    pairs = [(a, b) for a in METHODS for b in METHODS if a != b and METHODS.index(a) > METHODS.index(b)]
    for a, b in pairs:
        common = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
        if len(common) >= 2:
            x = np.array([runs[a][s]["test_xview2_score"] for s in common])
            y = np.array([runs[b][s]["test_xview2_score"] for s in common])
            out["paired"][f"{a}_minus_{b}"] = {**ms(x - y), "seeds": common,
                                               "p_paired_t": float(sstats.ttest_rel(x, y).pvalue)}
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Silo 2 stage 2 (damage) experiments.")
    ap.add_argument("--lr-sweep", action="store_true")
    ap.add_argument("--method", nargs="+", choices=METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--class-stats", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--stage1-root", default=None, help="default: results/silo2_loc (smoke: the stage-1 smoke dir)")
    ap.add_argument("--smoke", type=int, default=None, help="limit pairs per client/split (code check only)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--sweep-epochs", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    args = ap.parse_args()
    if args.summarize:
        summarize(args.seeds)
        return
    overrides = {k: v for k, v in (("epochs", args.epochs), ("rounds", args.rounds),
                                   ("sweep_epochs", args.sweep_epochs), ("ckpt_every", args.ckpt_every)) if v}
    if overrides and args.smoke is None:
        raise SystemExit(f"{sorted(overrides)} may only be overridden together with --smoke")
    global SWEEP_EPOCHS
    SWEEP_EPOCHS = args.sweep_epochs or SWEEP_EPOCHS
    COMMON["ckpt_every_steps"] = args.ckpt_every or COMMON["ckpt_every_steps"]
    git = require_clean_tree(args.allow_dirty)
    if args.class_stats:
        class_stats(args.seeds)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke_root = Path(tempfile.gettempdir())
    out_root = EXPERIMENT_DIR if args.smoke is None else smoke_root / "urbanfl_smoke_silo2_dmg"
    stage1_root = Path(args.stage1_root) if args.stage1_root else (
        STAGE1_DIR if args.smoke is None else smoke_root / "urbanfl_smoke_silo2_loc")
    if args.measure:
        measure(device, git, args.checkpoint_dir, stage1_root)
        return
    if args.lr_sweep:
        print(json.dumps(lr_sweep(device, git, args.checkpoint_dir, stage1_root, args.smoke, out_root / "lr_sweep"),
                         indent=1))
        return
    choice = _load(out_root / "lr_sweep" / "choice.json")
    if choice is None:
        raise FileNotFoundError("run the lr sweep first (--lr-sweep)")
    lr = choice["lr"]
    epochs, rounds = (EPOCHS, ROUNDS) if args.smoke is None else (args.epochs or 1, args.rounds or 1)
    ft_epochs = FT_EPOCHS if args.smoke is None else min(FT_EPOCHS, epochs)
    for seed in args.seeds:
        data = DmgData(seed, args.smoke)
        for method in args.method or []:
            run_dir = out_root / f"{method}_seed{seed}"
            if completed(run_dir, git, allow_dirty=args.smoke is not None):
                print(f"seed {seed} {method}: already completed at this commit, skipping", flush=True)
                continue
            t0 = time.perf_counter()
            ck = args.checkpoint_dir
            if method == "centralized":
                res, b = run_centralized(data, lr, epochs, device, git, ck, "centralized", stage1_root)
            elif method == "local_only":
                res, b = run_local_only(data, lr, epochs, device, git, ck, "local_only", stage1_root)
            elif method == "fedavg":
                res, b = run_fedavg(data, lr, rounds, device, git, ck, "fedavg", stage1_root)
            else:
                res, b = run_fedavg_ft(data, lr, ft_epochs, device, git, ck, "fedavg_ft", stage1_root, out_root)
            extra_cfg = {"fedavg": {"rounds": rounds, "local_epochs": LOCAL_EPOCHS},
                         "fedavg_ft": {"ft_epochs": ft_epochs}}.get(method, {"epochs": epochs})
            write_run(run_dir, method, res, b, data, lr, git, time.perf_counter() - t0, extra_cfg)
            prune_run(res)
            print(f"seed {seed} {method}: test xView2 score {res['test_xview2_score']:.4f} "
                  f"(loc F1 {res['test_pooled']['loc_f1']:.4f}, damage harmonic "
                  f"{res['test_pooled']['damage_f1_harmonic']:.4f}), held-out {res['heldout_xview2_score']:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
