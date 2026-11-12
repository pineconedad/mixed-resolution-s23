import importlib.util
import json
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import mixres.study_runner as study
from mixres.io import digest, file_hash
from mixres.study_runner import materialize_views


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def cell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = study.StudyRunner.__new__(study.StudyRunner)
    runner.model = "pi3"
    runner.precision = "bf16"
    runner.warmup, runner.timed_passes = 1, 3
    runner.base_meta = dict(source_hash="source", model_identity={"weights": "hash"},
                            git_commit="old", dirty_tree=True, gpu_name="A40",
                            driver_version="1", cpu_threads=4, cuda_version="12", versions={})
    runner.adapter = SimpleNamespace(patch_size=14, special_count=5, embed_dim=4, mlp_ratio=4)
    view = dict(image="image", image_sha256="image-hash", c2w=torch.eye(4).tolist(),
                sharp=False, rho=2.0,
                sizes={"4096": dict(hw=[14, 28], hi_hw=[28, 56], uniform_hw=[28, 28])})
    manifest = dict(dataset="eth3d", seed=42, budgets=[4096],
                    scenes=[dict(scene="one", views=[view]), dict(scene="two", views=[view])])
    write_json("runs/views/s23/eth3d/seed42.json", manifest)
    monkeypatch.setattr(study, "file_hash", lambda path: "image-hash" if str(path) == "image" else file_hash(path))
    monkeypatch.setattr(study, "seed_everything", lambda seed: None)
    monkeypatch.setattr(study, "load_view", lambda *args: SimpleNamespace(cuda=lambda: None))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    calls = []

    def engine(*args, **kwargs):
        calls.append(1)
        return [{"camera_poses": torch.eye(4)}], [dict(total_tokens=7, frame_total_tokens=13)]

    monkeypatch.setattr(study, "run_engine", engine)
    monkeypatch.setattr(study, "timed_inference", lambda forward, *args: (forward(), {"latency_s": 1}))
    monkeypatch.setattr(study, "transformer_flops", lambda *args: {})
    monkeypatch.setattr(study, "score_poses_da3", lambda *args: {"auc3": 1})
    return runner, manifest, calls


def test_materialize_mix_and_uniform_views():
    scene = {"views": [{"frame": "x", "sharp": False, "rho": 2.0,
                         "sizes": {"4096": {"hw": [28, 42], "hi_hw": [56, 84],
                                             "uniform_hw": [42, 56]}}}]}
    mixed = materialize_views(scene, 4096, "A4")[0]
    uniform = materialize_views(scene, 4096, "A0")[0]
    assert mixed["hw"] == [28, 42] and mixed["hi_hw"] == [56, 84]
    assert uniform["hw"] == uniform["hi_hw"] == [42, 56]
    assert uniform["sharp"] and uniform["rho"] == 1.0


@pytest.mark.parametrize("arm,expected", [("A0", 9), ("A3", 13), ("A4", 7), ("A6", 13)])
def test_expected_tokens_from_sizes(cell, arm, expected):
    runner, manifest, _ = cell
    views = materialize_views(manifest["scenes"][0], 4096, arm)
    assert study.expected_global_tokens(views, arm, runner.adapter) == expected


def test_consistently_wrong_trace_fails_and_errors_are_not_retried(cell, monkeypatch):
    runner, _, calls = cell
    monkeypatch.setattr(study, "run_engine", lambda *args, **kwargs:
                        ([{"camera_poses": torch.eye(4)}], [dict(total_tokens=99)] * 2))
    result = runner.run_cell("eth3d", 4096, "A4", 42, {}, phase="dev")
    assert all("does not match view sizes" in row["error"] for row in result["per_scene"])
    monkeypatch.setattr(study, "run_engine", lambda *args, **kwargs: pytest.fail("retried failed scene"))
    resumed = runner.run_cell("eth3d", 4096, "A4", 42, {}, phase="dev")
    assert resumed["per_scene"] == result["per_scene"]


def test_resume_keeps_original_provenance_and_labels_timing_groups(cell):
    runner, manifest, calls = cell
    # Keep the full manifest identity while stopping after one completed scene.
    stop = Path("stop")
    original = study.score_poses_da3
    def score(*args):
        stop.touch()
        return original(*args)
    study.score_poses_da3 = score
    try:
        result = runner.run_cell("eth3d", 4096, "A4", 42, {}, phase="dev", stop_file=stop)
    finally:
        study.score_poses_da3 = original
    # A stop request during evaluation preserves a failed record rather than claiming completion.
    assert len(result["per_scene"]) == 1
    stop.unlink()
    runner.base_meta.update(git_commit="new", dirty_tree=False, gpu_name="4090")
    resumed = runner.run_cell("eth3d", 4096, "A4", 42, {}, phase="dev")
    assert len(calls) == 2
    assert resumed["per_scene"][0]["provenance"]["git_commit"] == "old"
    assert not resumed["timing_comparable"]
    assert len(resumed["timing_groups"]) == 2
    runner.base_meta["source_hash"] = "changed"
    with pytest.raises(RuntimeError, match="changed provenance"):
        runner.run_cell("eth3d", 4096, "A4", 42, {}, phase="dev")


def test_deadline_interrupts_whole_cell_work(cell, monkeypatch):
    runner, _, _ = cell
    monkeypatch.setattr(study, "load_view", lambda *args: time.sleep(1))
    start = time.monotonic()
    result = runner.run_cell("eth3d", 4096, "A4", 42, {}, phase="dev", max_seconds=0.05)
    assert time.monotonic() - start < 0.5
    assert len(result["per_scene"]) == 1
    assert "TimeoutError" in result["per_scene"][0]["error"]


def test_scene_seed_is_stable_after_resume(cell, monkeypatch):
    runner, _, _ = cell
    seeds = []
    monkeypatch.setattr(study, "seed_everything", seeds.append)
    runner.run_cell("eth3d", 4096, "A4", 42, {}, phase="dev")
    second_seed = seeds[-1]
    next(Path("runs/dev").rglob("two.json")).unlink()
    seeds.clear()
    runner.run_cell("eth3d", 4096, "A4", 42, {}, phase="dev")
    assert seeds == [42, second_seed]


def test_frozen_bindings(cell, monkeypatch):
    runner, manifest, _ = cell
    manifest["seed"] = 1000
    views = "runs/views/s23/eth3d/seed1000.json"
    write_json(views, manifest)
    Path("configs").mkdir()
    Path("configs/s23.yaml").write_text("study: s23\n")
    Path("frozen/s23").mkdir(parents=True)
    Path("frozen/s23/predictions.md").write_text("predictions\n")
    frozen = dict(study="s23", model="pi3", precision="bf16", **study.METHOD_DEFAULTS,
                  model_identity=runner.base_meta["model_identity"],
                  study_config_hash=file_hash("configs/s23.yaml"),
                  predictions_hash=file_hash("frozen/s23/predictions.md"),
                  view_hashes={"eth3d/seed1000": file_hash(views)})
    path = Path("frozen/s23/pi3.yaml")
    path.write_text(yaml.safe_dump(frozen))
    monkeypatch.setattr(study, "git_state", lambda: {"dirty_tree": False})
    monkeypatch.setattr(study, "committed_hash", file_hash)
    result = runner.run_cell("eth3d", 4096, "A4", 1000, {}, phase="main", frozen_path=path)
    assert result["status"] == "completed"
    assert result["provenance"]["frozen_hash"] == file_hash(path)
    for key in ("model_identity", "study_config_hash", "predictions_hash", "view_hashes", "model", "precision"):
        broken = {**frozen, key: "wrong"}
        path.write_text(yaml.safe_dump(broken))
        with pytest.raises(RuntimeError, match="identity mismatch|manifest hash mismatch"):
            runner.run_cell("eth3d", 4096, "A4", 1000, {}, phase="main", frozen_path=path)


def test_committed_hash_rejects_untracked_and_modified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("predictions.md").write_text("predictions")
    committed = None
    def git_output(command, **kwargs):
        if command[1] == "rev-parse":
            return str(tmp_path)
        assert command == ["git", "show", "HEAD:predictions.md"]
        if committed is None:
            raise subprocess.CalledProcessError(128, command)
        return committed
    monkeypatch.setattr(study.subprocess, "check_output", git_output)
    with pytest.raises(RuntimeError, match="must be committed"):
        study.committed_hash("predictions.md")
    committed = b"predictions"
    assert study.committed_hash("predictions.md") == file_hash("predictions.md")
    Path("predictions.md").write_text("changed")
    with pytest.raises(RuntimeError, match="differs from committed"):
        study.committed_hash("predictions.md")


def test_freeze_prevalidates_late_inputs(tmp_path, monkeypatch):
    module = script("03_freeze")
    monkeypatch.chdir(tmp_path)
    methods = list(script("02_dev_sweep").configurations())
    write_json("ranking.json", dict(status="complete", model="pi3", seed=42, budget=4096,
               datasets=sorted(module.REQUIRED_DATASETS), ranking=[dict(method=method,
               method_id=digest(method)[:12], pooled_auc3=1, mean_latency_s=1,
               dataset_auc3={dataset: 1 for dataset in module.REQUIRED_DATASETS}) for method in methods]))
    for dataset in module.REQUIRED_DATASETS:
        for seed in (42, 1000, 2000, 3000):
            write_json(f"runs/views/s23/{dataset}/seed{seed}.json",
                       dict(dataset=dataset, seed=seed, budgets=[4096, 8192], scenes=[{}]))
    Path("configs/models").mkdir(parents=True)
    Path("configs/s23.yaml").write_text("study: s23\n")
    monkeypatch.setattr(module, "model_identity", lambda *args: {})
    for model in ("pi3", "vggt"):
        Path(f"configs/models/{model}.yaml").write_text("checkpoint: test\ncheckpoint_revision: rev\ncode_commit: code\n")
        write_json(f"checkpoints/{model}/receipt.json", dict(checkpoint="test", revision="rev", sha256="a" * 64))
    with pytest.raises(FileNotFoundError):
        module.freeze("ranking.json")
    assert not Path("frozen").exists()
    Path("configs/models/da3.yaml").write_text("checkpoint: test\ncheckpoint_revision: rev\ncode_commit: code\n")
    write_json("checkpoints/da3/receipt.json", dict(checkpoint="test", revision="rev", sha256="a" * 64))
    original_write = module.write_new
    def fail_late(path, text):
        if Path(path).name == "da3.yaml":
            raise OSError("write failed")
        original_write(path, text)
    monkeypatch.setattr(module, "write_new", fail_late)
    with pytest.raises(OSError, match="write failed"):
        module.freeze("ranking.json")
    assert not Path("frozen/s23").exists()
    assert list(Path("frozen").iterdir()) == []
    monkeypatch.setattr(module, "write_new", original_write)
    module.freeze("ranking.json")
    assert len(list(Path("frozen/s23").iterdir())) == 4
    with pytest.raises(RuntimeError, match="overwrite"):
        module.freeze("ranking.json")


def test_probes_on_refused_before_cuda():
    result = subprocess.run([__import__("sys").executable, str(ROOT / "scripts/04_run.py"),
                             "--probes", "on"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires an S23 model and seed 1000" in result.stderr


def test_sanity_source_hash_ignores_unrelated_code(monkeypatch):
    hashes = {}
    monkeypatch.setattr(study, "file_hash", lambda path: hashes.get(str(path), "unchanged"))
    original = study.sanity_source_hash("pi3")
    hashes[str(ROOT / "mixres/study_runner.py")] = "runner edit"
    assert study.sanity_source_hash("pi3") == original
    hashes[str(ROOT / "mixres/rmfa/engine.py")] = "engine edit"
    assert study.sanity_source_hash("pi3") != original


def test_sanity_gate_rejects_legacy_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("configs/models").mkdir(parents=True)
    Path("configs/models/pi3.yaml").write_text("checkpoint: test\ncheckpoint_revision: rev\n")
    meta = dict(upstream_commit="upstream", checkpoint="test", checkpoint_revision="rev",
                checkpoint_sha256="weights", model_config_hash="config")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(study, "provenance", lambda *args: dict(meta))
    monkeypatch.setattr(study, "load_adapter", lambda *args: "adapter")
    identity = study.model_identity("pi3", meta)
    legacy = {key: value for key, value in identity.items()
              if key not in {"sanity_source_hash", "model_config_hash"}}
    gate = dict(status="passed", scope="s23-pi3-fp32-T1", test_precision="fp32", model_identity=legacy)
    write_json("runs/sanity/pi3-s23.json", gate)
    with pytest.raises(RuntimeError, match="sanity gate does not match"):
        study.StudyRunner("pi3", "weights")
    gate["model_identity"] = identity
    write_json("runs/sanity/pi3-s23.json", gate)
    assert study.StudyRunner("pi3", "weights").adapter == "adapter"
