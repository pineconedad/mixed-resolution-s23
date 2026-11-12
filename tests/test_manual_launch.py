"""Build-only batch checks. The study runner is stubbed before scripts import it."""
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from mixres.io import atomic_json, digest, read_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(pool_op="area", lift_op="bilinear", token_interp="bilinear",
                rmfa_layers="all", head_grid="lo", alignment_scope="global", position_scale="actual")


@pytest.fixture
def scripts(tmp_path, monkeypatch):
    stub = types.ModuleType("mixres.study_runner")
    stub.METHOD_DEFAULTS = DEFAULTS
    stub.StudyRunner = None
    stub.committed_hash = lambda path: "committed"
    stub.model_identity = lambda *args: {}
    monkeypatch.setitem(sys.modules, "mixres.study_runner", stub)
    monkeypatch.chdir(tmp_path)

    def load(name):
        spec = importlib.util.spec_from_file_location(f"manual_{name}", ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load, stub


def result(dataset, arm, failed=False):
    return dict(dataset=dataset, arm=arm, status="partial_or_failed" if failed else "completed",
                scenes_expected=["a", "b"], timing_comparable=True,
                per_scene=[dict(scene=name, status="failed" if failed and name == "a" else "completed",
                                auc3=3.0, latency_s=2.0) for name in ("a", "b")])


def frozen_inputs(model="pi3", precision="fp32"):
    path = Path(f"frozen/s23/{model}.yaml")
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(dict(precision=precision, **DEFAULTS)))
    (path.parent / "predictions.md").write_text("predictions")


def test_plan_counts_and_explicit_execute_model(scripts):
    load, _ = scripts
    launch = load("launch_study")
    assert len(launch.matrix("main")) == 270
    assert len(launch.matrix("extra")) == 150
    for phase, count in (("main", 270), ("extra", 150)):
        launch.main(["--phase", phase])
        assert read_json(f"artifacts/matrix-{phase}.json")["count"] == count
    with pytest.raises(SystemExit) as error:
        launch.main(["--phase", "main", "--execute"])
    assert error.value.code == 2


@pytest.mark.parametrize("phase,count,probes", [("main", 90, True), ("extra", 50, False)])
def test_manual_execution_loads_once_and_continues_reported_failures(scripts, phase, count, probes):
    load, stub = scripts
    launch = load("launch_study")
    frozen_inputs("da3")
    loads, calls = [], []

    class Runner:
        def __init__(self, *args, **kwargs):
            loads.append((args, kwargs))

        def run_cell(self, *args, **kwargs):
            calls.append((args, kwargs))
            return result(args[0], args[2], failed=True)

    stub.StudyRunner = Runner
    launch.main(["--phase", phase, "--execute", "--model", "da3"])
    assert loads == [(("da3", "checkpoints/da3"), {"precision": "fp32"})]
    assert len(calls) == count
    assert all(kwargs == dict(phase=phase, frozen_path="frozen/s23/da3.yaml", probes=probes,
                              save_preds=True, max_seconds=None, stop_file=None) for _, kwargs in calls)
    assert all(args[-1] == DEFAULTS for args, _ in calls)


@pytest.mark.parametrize("stop", ["deadline", "stop_file", "timeout"])
def test_manual_batch_limits(scripts, monkeypatch, stop):
    load, stub = scripts
    launch = load("launch_study")
    frozen_inputs()
    clock = [0.0]
    monkeypatch.setattr(launch.time, "monotonic", lambda: clock[0])
    remaining, loads = [], []

    class Runner:
        def __init__(self, *args, **kwargs):
            loads.append(args)
            clock[0] += 1

        def run_cell(self, *args, **kwargs):
            remaining.append(kwargs["max_seconds"])
            if stop == "timeout":
                raise TimeoutError("deadline")
            clock[0] += 3
            if stop == "stop_file":
                Path("STOP").touch()
            return result(args[0], args[2])

    stub.StudyRunner = Runner
    launch.main(["--phase", "main", "--execute", "--model", "pi3",
                 "--max-seconds", "7", "--stop-file", "STOP"])
    assert len(loads) == 1
    assert remaining == ([6, 3] if stop == "deadline" else [6])


def test_manual_preexisting_stop_does_not_load(scripts):
    load, _ = scripts
    launch = load("launch_study")
    frozen_inputs()
    Path("STOP").touch()
    launch.main(["--phase", "main", "--execute", "--model", "pi3", "--stop-file", "STOP"])


@pytest.mark.parametrize("script", ["launch_study", "02_dev_sweep"])
def test_model_load_counts_against_batch_deadline(scripts, monkeypatch, script):
    load, stub = scripts
    module = load(script)
    clock, calls = [0.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    class Runner:
        def __init__(self, *args, **kwargs):
            clock[0] = 5

        def run_cell(self, *args, **kwargs):
            calls.append(args)

    if script == "launch_study":
        stub.StudyRunner = Runner
        frozen_inputs()
        module.main(["--phase", "main", "--execute", "--model", "pi3", "--max-seconds", "4"])
    else:
        monkeypatch.setattr(module, "StudyRunner", Runner)
        payload = module.sweep(["eth3d"], "checkpoint", max_seconds=4)
        assert payload["status"] == "partial_or_failed"
        assert payload["stop_reason"] == "deadline"
    assert calls == []


def test_manual_rejects_bad_budget_and_other_interpreter(scripts):
    load, _ = scripts
    launch = load("launch_study")
    for option in (["--max-seconds", "nan"], ["--max-seconds", "0"], ["--python", "/other/env/python"]):
        with pytest.raises(SystemExit) as error:
            launch.main(["--phase", "main", "--execute", "--model", "pi3", *option])
        assert error.value.code == 2


@pytest.mark.parametrize("failed_arm,expected_ranked", [(None, 24), ("A2", 24), ("A4", 23)])
def test_sweep_persists_and_never_ranks_incomplete_configs(scripts, monkeypatch, failed_arm, expected_ranked):
    load, _ = scripts
    sweep = load("02_dev_sweep")
    loads, calls, snapshots = [], [], []

    class Runner:
        def __init__(self, *args):
            loads.append(args)

        def run_cell(self, dataset, budget, arm, seed, method, **kwargs):
            calls.append((dataset, arm))
            return result(dataset, arm, failed=arm == failed_arm and calls.count((dataset, arm)) == 1
                          and dataset == "scannetpp")

    monkeypatch.setattr(sweep, "StudyRunner", Runner)

    def publish(path, payload):
        snapshots.append(copy.deepcopy(payload))
        atomic_json(path, payload)

    monkeypatch.setattr(sweep, "atomic_json", publish)
    payload = sweep.sweep(["eth3d", "scannetpp", "7scenes"], "checkpoint")
    assert len(loads) == 1
    assert len(calls) == 78
    assert len(payload["cells"]) == 78
    assert len(payload["ranking"]) == expected_ranked
    assert payload["status"] == ("complete" if failed_arm is None else "partial_or_failed")
    assert all(set(row["dataset_auc3"]) == {"eth3d", "scannetpp", "7scenes"} for row in payload["ranking"])
    assert snapshots[0]["status"] == "partial_or_failed"
    assert any(len(snapshot["ranking"]) == 1 for snapshot in snapshots)
    assert read_json("runs/dev/ranking.json") == payload
    assert len(Path("runs/dev/ranking.csv").read_text().splitlines()) == expected_ranked + 1


@pytest.mark.parametrize("stop", ["deadline", "stop_file", "timeout"])
def test_sweep_batch_limits_preserve_completed_ranking(scripts, monkeypatch, stop):
    load, _ = scripts
    sweep = load("02_dev_sweep")
    clock, remaining, loads = [0.0], [], []
    monkeypatch.setattr(sweep.time, "monotonic", lambda: clock[0])

    class Runner:
        def __init__(self, *args):
            loads.append(args)
            clock[0] += 1

        def run_cell(self, dataset, budget, arm, seed, method, **kwargs):
            remaining.append(kwargs["max_seconds"])
            clock[0] += 2
            if len(remaining) == 4:
                if stop == "timeout":
                    raise TimeoutError("deadline")
                if stop == "stop_file":
                    Path("STOP").touch()
            return result(dataset, arm)

    monkeypatch.setattr(sweep, "StudyRunner", Runner)
    payload = sweep.sweep(["eth3d"], "checkpoint", max_seconds=9, stop_file="STOP")
    assert len(loads) == 1
    assert remaining == [8, 6, 4, 2]
    assert len(payload["ranking"]) == (1 if stop == "timeout" else 2)
    assert payload["status"] == "partial_or_failed"
    assert payload["stop_reason"] == ("stop_file" if stop == "stop_file" else "deadline")
    assert read_json("runs/dev/ranking.json") == payload


def test_sweep_preexisting_stop_and_invalid_limits(scripts, monkeypatch):
    load, _ = scripts
    sweep = load("02_dev_sweep")
    Path("STOP").touch()
    monkeypatch.setattr(sweep, "StudyRunner", None)
    payload = sweep.sweep(["eth3d"], "checkpoint", stop_file="STOP")
    assert payload["cells"] == []
    assert payload["status"] != "complete"
    for seconds in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            sweep.sweep(["eth3d"], "checkpoint", max_seconds=seconds)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "nonfinite", "timing"])
def test_completed_auc_refuses_bad_scene_sets(scripts, mutation):
    load, _ = scripts
    sweep = load("02_dev_sweep")
    cell = result("eth3d", "A4")
    if mutation == "missing":
        cell["per_scene"].pop()
    elif mutation == "duplicate":
        cell["per_scene"][1]["scene"] = "a"
    elif mutation == "failed":
        cell["per_scene"][0]["status"] = "failed"
    elif mutation == "nonfinite":
        cell["per_scene"][0]["latency_s"] = float("nan")
    else:
        cell["timing_comparable"] = False
    with pytest.raises(RuntimeError):
        sweep.completed_auc(cell)


def ranking_fixture(load):
    sweep = load("02_dev_sweep")
    datasets = ["eth3d", "scannetpp", "7scenes"]
    rows = [dict(method=method, method_id=digest(method)[:12], dataset_auc3=dict.fromkeys(datasets, 3.0),
                 pooled_auc3=3.0, mean_latency_s=2.0, status="complete") for method in sweep.configurations()]
    return dict(status="complete", model="pi3", seed=42, budget=4096, datasets=datasets, ranking=rows)


@pytest.mark.parametrize("mutation", ["few", "duplicate", "dataset", "pooled", "nan", "inf", "string",
                                     "bool", "negative_cost", "partial", "method", "cells", "batch_partial"])
def test_freeze_rejects_invalid_rankings_before_writes(scripts, mutation):
    load, _ = scripts
    freeze = load("03_freeze")
    payload = ranking_fixture(load)
    row = payload["ranking"][-1]
    if mutation == "few":
        payload["ranking"].pop()
    elif mutation == "duplicate":
        payload["ranking"][-1] = payload["ranking"][0]
    elif mutation == "dataset":
        row["dataset_auc3"].pop("scannetpp")
    elif mutation == "pooled":
        row["pooled_auc3"] = 4.0
    elif mutation in ("nan", "inf", "string", "bool", "negative_cost"):
        row["mean_latency_s"] = {"nan": float("nan"), "inf": float("inf"), "string": "2",
                                 "bool": True, "negative_cost": -1}[mutation]
    elif mutation == "partial":
        row["status"] = "partial_or_failed"
    elif mutation == "method":
        row["method"]["pool_op"] = "invalid"
        row["method_id"] = digest(row["method"])[:12]
    elif mutation == "cells":
        payload["cells"] = []
    else:
        payload["status"] = "partial_or_failed"
    # Deliberately allow nonfinite JSON to test the reader's validation.
    Path("ranking.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        freeze.freeze("ranking.json")
    assert not Path("frozen").exists()


def test_freeze_stable_winner_and_atomic_publication(scripts, monkeypatch):
    load, _ = scripts
    freeze = load("03_freeze")
    payload = ranking_fixture(load)
    payload["ranking"].reverse()
    expected = min(payload["ranking"], key=lambda row: row["method_id"])
    # Legacy full rankings have no per-config status or cell audit fields.
    for row in payload["ranking"]:
        row.pop("status")
    atomic_json("ranking.json", payload)
    for dataset in freeze.REQUIRED_DATASETS:
        for seed in (42, 1000, 2000, 3000):
            atomic_json(f"runs/views/s23/{dataset}/seed{seed}.json",
                        dict(dataset=dataset, seed=seed, budgets=[4096, 8192], scenes=[{}]))
    Path("configs/models").mkdir(parents=True)
    Path("configs/s23.yaml").write_text("study: s23\n")
    for model in ("pi3", "vggt", "da3"):
        Path(f"configs/models/{model}.yaml").write_text("checkpoint: test\ncheckpoint_revision: rev\ncode_commit: code\n")
        atomic_json(f"checkpoints/{model}/receipt.json", dict(checkpoint="test", revision="rev", sha256="a" * 64))
    original = freeze.write_new

    def fail_late(path, text):
        if Path(path).name == "da3.yaml":
            raise OSError("write failed")
        original(path, text)

    monkeypatch.setattr(freeze, "write_new", fail_late)
    with pytest.raises(OSError, match="write failed"):
        freeze.freeze("ranking.json")
    assert not Path("frozen/s23").exists()
    assert list(Path("frozen").iterdir()) == []
    monkeypatch.setattr(freeze, "write_new", original)
    freeze.freeze("ranking.json")
    for model in ("pi3", "da3", "vggt"):
        config = yaml.safe_load(Path(f"frozen/s23/{model}.yaml").read_text())
        assert config["selected_method_id"] == expected["method_id"]
    with pytest.raises(RuntimeError, match="overwrite"):
        freeze.freeze("ranking.json")
