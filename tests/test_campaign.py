"""Campaign order and crash policy: retry once, skip after a second crash, abort if the lr sweep fails twice."""

from experiments.silo2_campaign import campaign_steps, run_campaign


def test_order_sweep_then_seed_by_seed():
    names = [n for n, _ in campaign_steps()]
    assert names == ["lr_sweep",
                     "centralized_seed0", "local_only_seed0", "fedavg_seed0",
                     "centralized_seed1", "local_only_seed1", "fedavg_seed1",
                     "centralized_seed2", "local_only_seed2", "fedavg_seed2"]


def fake_runner(plan: dict):
    """plan: step name -> list of exit codes for successive attempts (default 0)."""
    calls = []

    def runner(args):
        name = "lr_sweep" if "--lr-sweep" in args else f"{args[args.index('--method') + 1]}_seed{args[-1]}"
        calls.append(name)
        codes = plan.get(name, [])
        return codes.pop(0) if codes else 0
    return runner, calls


def test_crash_once_resumes_and_continues(tmp_path):
    runner, calls = fake_runner({"local_only_seed0": [1]})
    st = run_campaign(campaign_steps(seeds=(0,)), runner, lambda m: None, tmp_path / "s.json")
    assert calls == ["lr_sweep", "centralized_seed0", "local_only_seed0", "local_only_seed0", "fedavg_seed0"]
    assert st["steps"]["local_only_seed0"] == {"result": "ok", "attempts": 2, "seconds": 0}
    assert (tmp_path / "s.json").exists()


def test_two_crashes_skip_and_continue():
    runner, calls = fake_runner({"centralized_seed0": [1, 1]})
    st = run_campaign(campaign_steps(seeds=(0, 1)), runner, lambda m: None)
    assert st["steps"]["centralized_seed0"]["result"] == "skipped after 2 crashes"
    assert calls.count("centralized_seed0") == 2 and calls[-1] == "fedavg_seed1" and not st["aborted"]


def test_sweep_failing_twice_aborts():
    runner, calls = fake_runner({"lr_sweep": [1, 1]})
    st = run_campaign(campaign_steps(seeds=(0,)), runner, lambda m: None)
    assert st["aborted"] and calls == ["lr_sweep", "lr_sweep"]
