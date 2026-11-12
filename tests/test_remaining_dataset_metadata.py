"""Synthetic metadata only. Never opens the study's held-out dataset roots."""
import ast
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mixres.data import benchmark


def test_remaining_metadata_access_native_records_and_archive_checks(tmp_path, monkeypatch, capsys):
    def script(name):
        spec = importlib.util.spec_from_file_location(name, benchmark.PROJECT_ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    downloader = script("download_benchmark")
    views = script("00_make_views")
    monkeypatch.setattr(downloader, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "unopened"
    # A missing committed prediction must fail before any synthetic dataset read.
    for dataset in ("scannetpp", "dtu", "hiroom"):
        with pytest.raises(PermissionError):
            benchmark.load_dataset_scene(dataset, root, "scene", project_root=tmp_path)
        with pytest.raises(PermissionError):
            benchmark.scene_names(dataset, root, project_root=tmp_path)
        with pytest.raises(PermissionError):
            downloader.main([dataset], root=root)
    assert not root.exists()
    downloader.main(["scannetpp", "dtu", "hiroom"], root=root, metadata_only=True)
    assert benchmark.HUB_REVISION in capsys.readouterr().out
    assert not root.exists()
    constants = benchmark._native_constants()
    assert len(constants["DTU_SCENES"]) == 22
    assert len(constants["SCANNETPP_SCENES"]) == 20

    receipt = tmp_path / "permission.json"
    receipt.write_text(json.dumps({"dataset": "scannetpp", "terms_accepted": True}))
    with pytest.raises(PermissionError):
        benchmark.require_dataset_access("scannetpp", permission_receipt=receipt)
    receipt.write_text(json.dumps({"dataset": "scannetpp", "permission_granted": True,
                                  "terms_accepted": True, "accepted_by": "synthetic test",
                                  "accepted_utc": "2026-11-11T00:00:00Z",
                                  "license_url": "https://example.invalid/terms",
                                  "evidence": "synthetic fixture, not real permission"}))
    benchmark.require_dataset_access("scannetpp", permission_receipt=receipt)

    predictions = tmp_path / "frozen/s23/predictions.md"
    predictions.parent.mkdir(parents=True)
    predictions.write_text("synthetic committed prediction\n")
    committed = predictions.read_bytes()
    real_run = subprocess.run
    git_state = {"present": False, "staged": False}

    def git_run(command, **kwargs):
        if kwargs.get("cwd") == tmp_path:
            if command[1] == "show":
                return SimpleNamespace(returncode=0 if git_state["present"] else 128,
                                       stdout=committed if git_state["present"] else b"")
            return SimpleNamespace(returncode=int("--cached" in command and git_state["staged"]))
        return real_run(command, **kwargs)

    monkeypatch.setattr(benchmark.subprocess, "run", git_run)
    with pytest.raises(PermissionError):
        benchmark.require_dataset_access("hiroom", tmp_path)
    git_state["present"] = True
    predictions.write_text("uncommitted edit")
    with pytest.raises(PermissionError):
        benchmark.require_dataset_access("hiroom", tmp_path)
    predictions.write_bytes(committed)
    git_state["staged"] = True
    with pytest.raises(PermissionError):
        benchmark.require_dataset_access("dtu", tmp_path)
    git_state["staged"] = False
    benchmark.require_dataset_access("hiroom", tmp_path)

    hiroom = tmp_path / "synthetic-hiroom"
    scene = "building/room/view"
    directory = hiroom / "data" / scene
    for name in ("image", "pose"):
        (directory / name).mkdir(parents=True)
    (hiroom / "selected_scene_list_val.txt").write_text(scene + "\n")
    np.save(directory / "cam_K.npy", np.diag([100.0, 100.0, 1.0]))
    for index in range(12):
        Image.new("RGB", (32, 24)).save(directory / "image" / f"{index:03}.png")
        pose = np.eye(4)
        pose[0, 3] = index
        np.save(directory / "pose" / f"{index:03}.npy", pose)
    assert benchmark.scene_names("hiroom", hiroom, project_root=tmp_path) == (scene,)
    manifest = benchmark.make_manifest("hiroom", hiroom, tmp_path, 1000, [4096])
    assert len(manifest["scenes"][0]["views"]) == 12
    view = manifest["scenes"][0]["views"][0]
    assert view["source_metadata"]["aux"]["gt_pcd_path"].endswith("building-room-view.ply")
    np.testing.assert_allclose(np.asarray(view["w2c"]) @ view["c2w"], np.eye(4))
    assert manifest["source"]["native_revision"] == benchmark.NATIVE_REVISION
    with pytest.raises(ValueError, match="held-out only"):
        benchmark.make_manifest("hiroom", hiroom, tmp_path, 42, [4096])

    # Run the pinned native DTU metadata methods without importing GPU/eval dependencies.
    source = benchmark.NATIVE_ROOT / "src/depth_anything_3/bench/datasets/dtu.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {"read_cam_file", "get_data", "_depth_mask_path"}]
    namespace = {"np": np, "os": os, "glob": glob, "Dict": dict}
    class AttrDict(dict):
        __getattr__ = dict.__getitem__
    namespace["Dict"] = AttrDict
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    native_class = type("SyntheticNativeDTU", (), {node.name: namespace[node.name] for node in methods})
    dtu = tmp_path / "synthetic-dtu"
    rgb = dtu / "Rectified/scan1"
    rgb.mkdir(parents=True)
    (dtu / "Cameras").mkdir()
    for index in range(34):
        Image.new("RGB", (32, 24)).save(rgb / f"rect_{index + 1:03}_3_r5000.png")
        (dtu / "Cameras" / f"{index:08}_cam.txt").write_text(
            "extrinsic\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n\nintrinsic\n100 0 16\n0 100 12\n0 0 1\n")
    native = native_class()
    native.data_root = str(dtu)

    def loader(dataset, root):
        if dataset == "dtu":
            return native
        image = tmp_path / "synthetic-scannetpp/scene/iphone.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24)).save(image)
        data = {"image_files": [str(image)], "extrinsics": [np.eye(4)], "intrinsics": [np.eye(3) * 2],
                "aux": {"ixt_raw_list": [np.eye(3)], "dist_list": [np.zeros(5)],
                        "roi_list": [(0, 0, 32, 24)], "cam_hw_list": [(24, 32)],
                        "gt_mesh_path": "synthetic.ply", "gt_depth_files": ["synthetic.png"]}}
        return SimpleNamespace(data_root=str(image.parents[1]), get_data=lambda scene: data)

    monkeypatch.setattr(benchmark, "_native_loader", loader)
    records = benchmark.load_dataset_scene("dtu", dtu, "scan1", project_root=tmp_path)
    assert len(records) == 34
    assert records[0]["image"].name == "rect_034_3_r5000.png"
    assert records[0]["source_metadata"]["aux"]["mask_files"].endswith("depth_visual_0033.png")
    assert benchmark.load_dataset_scene("dtu", dtu, "absent", project_root=tmp_path) == []
    scan = benchmark.load_dataset_scene("scannetpp", tmp_path / "synthetic-scannetpp", "scene",
                                        permission_receipt=receipt)[0]
    np.testing.assert_array_equal(scan["intrinsics"], np.eye(3))
    assert scan["source_metadata"]["aux"]["roi_list"] == [0, 0, 32, 24]
    assert "undistorted_intrinsics" in scan["source_metadata"]["aux"]

    # Exercise consistent drops and seed exclusion without writing any manifests.
    manifests = []
    monkeypatch.setattr(views, "scene_names", lambda *args, **kwargs: ("small", "large"))
    monkeypatch.setattr(views, "load_dataset_scene", lambda dataset, root, scene, **kwargs:
                        [None] * (23 if scene == "small" else 24))
    def make(dataset, root, project, seed, budgets, **kwargs):
        assert Path(root) == tmp_path
        assert kwargs["dropped_scenes"] == ["small"]
        assert kwargs["excluded"] == ({"large": ["dev"]} if seed != 42 else None)
        return {"scenes": [{"scene": "large", "views": [{"frame": "dev"}]}]}
    monkeypatch.setattr(views, "make_manifest", make)
    monkeypatch.setattr(views, "write_once", lambda path, manifest: manifests.append(manifest))
    views.main(["scannetpp"], [42, 1000], [4096], roots={"scannetpp": tmp_path}, permission_receipt=receipt)
    assert len(manifests) == 2

    archive = tmp_path / "synthetic.zip"
    archive.write_bytes(b"synthetic archive, not dataset contents")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setitem(downloader.FILES, "hiroom", (archive.stat().st_size, sha))
    calls = []
    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(archive)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(downloader.shutil, "disk_usage", lambda root: SimpleNamespace(free=10**12))
    downloader.main(["hiroom"], root=tmp_path / "archives")
    assert calls[0][2]["revision"] == benchmark.HUB_REVISION
    report = json.loads((tmp_path / "reports/data/development-downloads.json").read_text())
    assert report["datasets"]["hiroom"]["status"] == "verified"
    archive.write_bytes(b"x" * archive.stat().st_size)
    with pytest.raises(SystemExit, match="download failed"):
        downloader.main(["hiroom"], root=tmp_path / "archives")
    report = json.loads((tmp_path / "reports/data/development-downloads.json").read_text())
    assert report["datasets"]["hiroom"]["status"] == "failed"
