"""
Silo 2 campaign: runs one experiment module step by step (--experiment silo2_loc = stage 1, the default;
silo2_dmg = stage 2) in the order

  lr sweep  ->  seed 0: its methods  ->  seed 1: ...  ->  seed 2: ...  ->  summary

stage 1: centralized, local_only, fedavg;  stage 2: centralized, local_only, fedavg, fedavg_ft

so a full comparison exists after seed 0. Every step is its own subprocess (a crash cannot take the campaign down).
On a crash the step is started once more; it resumes from its last checkpoint (same commit, same checkpoint dir).
If it crashes again it is SKIPPED and the campaign continues — except the lr sweep, whose lr every later step needs:
two sweep failures abort the campaign. Steps that already finished at this commit are skipped by silo2_loc itself,
so restarting the campaign after a reboot continues where it stopped.

Progress (campaign events + every step's own output) is appended to a log file, and a status JSON is rewritten after
every step; both live next to the checkpoints (default C:/data/checkpoints/silo2_loc/), outside the repo.

Usage:  python -m experiments.silo2_campaign [--experiment silo2_dmg]      (requires a clean, committed tree)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from src.fl.checkpoint import resolve_checkpoint_dir
from src.fl.utils import REPO_ROOT, require_clean_tree

SEEDS = (0, 1, 2)
METHODS = ("centralized", "local_only", "fedavg")
EXPERIMENT_METHODS = {"silo2_loc": METHODS, "silo2_dmg": ("centralized", "local_only", "fedavg", "fedavg_ft")}


# Pre-registered stage-2 follow-ups (docs/STAGE2_PROTOCOL.md), in their recorded order, after the campaign.
DMG_POST_STEPS = (("edge_check_lr3e-05", ["--edge-check"]), ("S1_fedavg_R60_lr1e-4", ["--sensitivity", "S1"]),
                  ("S2_fedavg_R30_lr3e-4", ["--sensitivity", "S2"]), ("D1_diagnostics", ["--diagnostics"]))


def post_steps(extra: list[str] | None = None) -> list[tuple[str, list[str]]]:
    return [(name, [*args, *(extra or [])]) for name, args in DMG_POST_STEPS]


def campaign_steps(seeds=SEEDS, extra: list[str] | None = None, methods=METHODS) -> list[tuple[str, list[str]]]:
    extra = extra or []
    steps = [("lr_sweep", ["--lr-sweep", *extra])]
    for s in seeds:
        steps += [(f"{m}_seed{s}", ["--method", m, "--seeds", str(s), *extra]) for m in methods]
    return steps


def run_campaign(steps: list[tuple[str, list[str]]], runner: Callable[[list[str]], int], log: Callable[[str], None],
                 status_path: Path | None = None, critical: tuple[str, ...] = ("lr_sweep",)) -> dict:
    """Run steps in order; one retry per step; skip after a second failure; abort if a critical step fails twice."""
    status = {"steps": {}, "aborted": False}

    def write_status():
        if status_path is not None:
            status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")

    for name, args in steps:
        for attempt in (1, 2):
            log(f"START {name} (attempt {attempt}): {' '.join(args)}")
            t0 = time.time()
            rc = runner(args)
            took = time.time() - t0
            if rc == 0:
                status["steps"][name] = {"result": "ok", "attempts": attempt, "seconds": round(took)}
                log(f"DONE  {name} in {took / 60:.1f} min")
                break
            log(f"CRASH {name} (attempt {attempt}) exit code {rc} after {took / 60:.1f} min"
                + (" -> resuming from the last checkpoint" if attempt == 1 else ""))
        else:
            status["steps"][name] = {"result": "skipped after 2 crashes", "attempts": 2}
            if name in critical:
                status["aborted"] = True
                log(f"ABORT {name} failed twice; later steps depend on it")
                write_status()
                return status
            log(f"SKIP  {name} failed twice; continuing with the next step")
        write_status()
    write_status()
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description="Silo 2 stage-1 campaign runner.")
    ap.add_argument("--experiment", default="silo2_loc", choices=sorted(EXPERIMENT_METHODS))
    ap.add_argument("--log-dir", default=None, help="default: <checkpoint_dir>/<experiment>")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", type=int, default=None, help="pass --smoke N --allow-dirty to every step (code check)")
    ap.add_argument("--checkpoint-dir", default=None, help="passed to every step (smoke tests: keep their "
                                                           "checkpoints away from the real ones)")
    ap.add_argument("--post", action="store_true", help="silo2_dmg only: the pre-registered follow-ups (3e-5 "
                                                        "grid-edge check, S1, S2, D1) instead of the campaign")
    ap.add_argument("--step-args", nargs=argparse.REMAINDER, default=[],
                    help="extra arguments for every step, smoke tests only (e.g. --epochs 3 --ckpt-every 5)")
    args = ap.parse_args()
    if args.step_args and args.smoke is None:
        raise SystemExit("--step-args is only allowed together with --smoke")
    if args.post and args.experiment != "silo2_dmg":
        raise SystemExit("--post is only defined for --experiment silo2_dmg")
    git = require_clean_tree(allow_dirty=args.smoke is not None)
    log_dir = Path(args.log_dir) if args.log_dir else resolve_checkpoint_dir(None, args.experiment)
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = ("post_" if args.post else "") + git["commit"][:7] + ("_smoke" if args.smoke is not None else "")
    log_path, status_path = log_dir / f"campaign_{tag}.log", log_dir / f"campaign_{tag}_status.json"
    extra = ["--smoke", str(args.smoke), "--allow-dirty", *args.step_args] if args.smoke is not None else []
    if args.checkpoint_dir:
        extra += ["--checkpoint-dir", args.checkpoint_dir]

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def runner(step_args: list[str]) -> int:
        with log_path.open("a", encoding="utf-8") as f:
            return subprocess.call([sys.executable, "-u", "-m", f"experiments.{args.experiment}", *step_args],
                                   cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)

    log(f"campaign {args.experiment}{' --post' if args.post else ''} start at commit {git['commit']} "
        f"(dirty={git['dirty']}); log {log_path}")
    if args.post:                     # independent follow-ups: none is critical, no summary step
        status = run_campaign(post_steps(extra), runner, log, status_path, critical=())
        log("post steps finished: " + json.dumps({k: v["result"] for k, v in status["steps"].items()}))
        return
    status = run_campaign(campaign_steps(args.seeds, extra, EXPERIMENT_METHODS[args.experiment]), runner, log,
                          status_path)
    if not status["aborted"] and args.smoke is None:
        rc = runner(["--summarize", "--seeds", *map(str, args.seeds)])
        log(f"summary written (exit code {rc})")
    log("campaign finished: " + json.dumps({k: v["result"] for k, v in status["steps"].items()}))


if __name__ == "__main__":
    main()
