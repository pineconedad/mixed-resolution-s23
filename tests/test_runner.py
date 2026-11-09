import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mixres.io import atomic_json, digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(tmp_path, monkeypatch):
    module = load_script("04_run")
    monkeypatch.chdir(tmp_path)
    config = {"study": "quick", "seed": 42, "model": "pi3", "dataset": "synthetic_test_only",
              "precision": "fp32", "views": "views.json", "data_root": "data", "arms": ["A1", "A2", "A3", "A4"],
              "budget_tokens": 4096, "warmup": 1, "timed_passes": 3,
              "pool_op": "area", "lift_op": "bilinear", "token_interp": "bilinear", "head_grid": "lo", "rmfa_layers": "all",
              "scenes": ["first", "second"], "bootstrap_resamples": 100}
    meta = {"git_commit": "test", "config_hash": digest(config)}
    scenes = []
    for name in config["scenes"]:
        views = []
        for i in range(2):
            path = Path("data") / f"{name}-{i}.png"
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (28, 28)).save(path)
            pose = np.eye(4)
            pose[0, 3] = i
            views.append({"image": path.name, "image_sha256": file_hash(path), "c2w": pose.tolist(),
                          "hw": [28, 28], "hi_hw": [28, 28], "sharp": True, "crop": [0, 0, 1, 1]})
        scenes.append({"scene": name, "views": views})
    atomic_json("views.json", {"scenes": scenes, "config_hash": digest(config)})
    atomic_json("runs/sanity/pi3-quick.json", {"status": "passed", "scope": "quick-pi3-fp32-T1", "provenance": meta})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda x: x)
    monkeypatch.setattr(module, "seed_everything", lambda _: None)
    monkeypatch.setattr(module, "provenance", lambda *args: meta)
    monkeypatch.setattr(module, "load_pi3", lambda *args: object())
    calls = []

    def fake_engine(adapter, images, views, arm, **kwargs):
        calls.append(arm)
        return [{"camera_poses": torch.tensor(view["c2w"], dtype=torch.float32), "points": torch.zeros(28, 28, 3)} for view in views], []

    monkeypatch.setattr(module, "run_engine", fake_engine)
    monkeypatch.setattr(module, "timed_inference", lambda forward, *_: (forward(), {"latency_s": 1, "peak_mem_gb": 1}))
    return module, config, meta, calls


def test_scene_resume_skips_only_verified_successes(runner):
    module, config, meta, calls = runner
    module.run(config, "unused-test-checkpoint")
    assert len(calls) == 8
    module.run(config, "unused-test-checkpoint")
    assert len(calls) == 8
    for arm in config["arms"]:
        result = read_json(f"runs/quick/pi3/synthetic_test_only/4096/{arm}/seed42/result.json")
        assert result["status"] == "completed"
        assert len(result["per_scene"]) == 2


def test_partial_checkpoint_survives_interruption_and_resumes(runner, monkeypatch):
    module, config, meta, calls = runner
    original = module.atomic_json
    writes = []

    def interrupted(path, record):
        original(path, record)
        if str(path).endswith("result.json"):
            writes.append(str(path))
            if len(writes) == 1:
                raise KeyboardInterrupt("Simulated process interruption after durable result")

    monkeypatch.setattr(module, "atomic_json", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.run(config, "unused-test-checkpoint")
    assert len(calls) == 1
    monkeypatch.setattr(module, "atomic_json", original)
    module.run(config, "unused-test-checkpoint")
    assert len(calls) == 8


def test_interruption_before_run_summary_rebuilds_summary(runner, monkeypatch):
    module, config, meta, calls = runner
    original = module.atomic_json

    def interrupted(path, record):
        if str(path).endswith("result.json"):
            raise KeyboardInterrupt("Interruption between scene and run summary")
        original(path, record)

    monkeypatch.setattr(module, "atomic_json", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.run(config, "unused-test-checkpoint")
    assert len(calls) == 1
    monkeypatch.setattr(module, "atomic_json", original)
    module.run(config, "unused-test-checkpoint")
    assert len(calls) == 8
    summary = read_json("runs/quick/pi3/synthetic_test_only/4096/A1/seed42/result.json")
    assert summary["status"] == "completed"
    assert len(summary["per_scene"]) == 2


def test_changed_provenance_cannot_overwrite_outputs(runner):
    module, config, meta, calls = runner
    module.run(config, "unused-test-checkpoint")
    meta["git_commit"] = "different-test-code"
    atomic_json("runs/sanity/pi3-quick.json", {"status": "passed", "scope": "quick-pi3-fp32-T1", "provenance": meta})
    with pytest.raises(RuntimeError, match="different provenance"):
        module.run(config, "unused-test-checkpoint")


def test_failed_scene_is_visible_and_retryable(runner, monkeypatch):
    module, config, meta, calls = runner
    original = module.run_engine
    failed = []

    def failing(*args, **kwargs):
        if args[3] == "A4" and not failed:
            failed.append(True)
            raise RuntimeError("Synthetic OOM; no resolution change allowed")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_engine", failing)
    module.run(config, "unused-test-checkpoint")
    aggregate = load_script("05_aggregate")
    aggregate.aggregate(config)
    summary = read_json("runs/quick/summary/snapshot.json")
    assert summary["scenes_paired"] == ["second"]
    assert summary["missing_or_failed"]["A4"] == ["first"]
    assert summary["recovery"] is None
    module.run(config, "unused-test-checkpoint")
    aggregate.aggregate(config)
    assert read_json("runs/quick/summary/snapshot.json")["scenes_paired"] == ["first", "second"]


def test_missing_gate_blocks_every_arm(runner):
    module, config, meta, calls = runner
    atomic_json("runs/sanity/pi3-quick.json", {"status": "failed", "provenance": meta})
    with pytest.raises(RuntimeError, match="01_sanity"):
        module.run(config, "unused-test-checkpoint")
    assert not calls


def test_session_limit_stops_between_evaluations(runner):
    module, config, meta, calls = runner
    module.run(config, "unused-test-checkpoint", max_seconds=0)
    assert not calls


def test_aggregate_refuses_mixed_metric_protocols(runner):
    module, config, meta, calls = runner
    module.run(config, "unused-test-checkpoint")
    path = Path("runs/quick/pi3/synthetic_test_only/4096/A4/seed42/result.json")
    record = read_json(path)
    record["metric_protocol"] = "different-test-protocol"
    atomic_json(path, record)
    with pytest.raises(ValueError, match="Refusing mixed"):
        load_script("05_aggregate").aggregate(config)


def test_frozen_run_not_silently_substituted(runner):
    module, config, meta, calls = runner
    with pytest.raises(RuntimeError, match="prerequisites"):
        module.run({**config, "study": "s23", "seed": 1000}, "unused-test-checkpoint")
