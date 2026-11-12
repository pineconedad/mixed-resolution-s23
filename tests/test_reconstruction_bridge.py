import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mixres.eval import benchmark_recon as bridge


@pytest.fixture
def saved(tmp_path):
    images, views = [], []
    for i in range(3):
        path = tmp_path / f"frame-{i}.color.png"
        Image.new("RGB", (6, 4)).save(path)
        images.append(str(path))
        views.append({"image": str(path), "crop": [0, 0, 1, 1],
                      "original_hw": [4, 6], "hw": [2, 3], "hi_hw": [4, 6]})
    full = SimpleNamespace(image_files=images,
                           extrinsics=np.repeat(np.eye(4)[None], 3, axis=0),
                           intrinsics=np.repeat(np.eye(3)[None], 3, axis=0), aux={})
    predictions = tmp_path / "predictions.npz"
    arrays = {}
    for i in range(3):
        pose = np.eye(4)
        pose[0, 3] = i + 1
        hw = views[i]["hw"] if i != 1 else views[i]["hi_hw"]
        arrays.update({f"camera_poses_{i}": pose,
                       f"intrinsics_{i}": np.array([[5, 0, 1], [0, 7, 1], [0, 0, 1]]),
                       f"depth_{i}": np.arange(np.prod(hw), dtype=np.float32).reshape(*hw, 1) + 1})
    np.savez(predictions, **arrays)
    native = SimpleNamespace(get_data=lambda scene: full)
    return tmp_path, native, views, predictions, arrays


@pytest.mark.parametrize("dataset", ["eth3d", "7scenes", "hiroom"])
def test_native_export_order_cameras_and_mixed_grids(saved, dataset, monkeypatch):
    root, native, views, predictions, arrays = saved
    # Synthetic fixtures do not open held-out data.
    monkeypatch.setattr(bridge, "require_dataset_access", lambda *args: None)
    # Saved order, not loader order. Predicted poses must not become GT poses.
    views = [views[2], views[0], views[1]]
    path = bridge.prepare_scene(native, dataset, "scene", views, predictions, root / "work", root)
    with np.load(path) as exported, np.load(root / "work/gt_meta.npz") as gt:
        assert exported["depth"].shape == (3, 4, 6)
        np.testing.assert_array_equal(exported["extrinsics"][:, 0, 3], [-1, -2, -3])
        np.testing.assert_array_equal(gt["image_files"], [view["image"] for view in views])
        np.testing.assert_array_equal(gt["extrinsics"][:, 0, 3], 0)
        np.testing.assert_array_equal(exported["intrinsics"][0, 0], [10, 0, 2])
        np.testing.assert_array_equal(exported["depth"][0],
                                      np.repeat(np.repeat(arrays["depth_0"][..., 0], 2, 0), 2, 1))


@pytest.mark.parametrize("problem,match", [
    ("missing_k", "GT intrinsics are never substituted"),
    ("local_only", "GT intrinsics are never substituted"),
    ("crop", "Cropped"), ("manifest", "materialized"),
    ("missing_image", "not in native"), ("bad_pose", "Invalid predicted"),
    ("extra_view", "view count mismatch"), ("grid", "Depth grid"),
])
def test_explicit_failures(saved, problem, match):
    root, native, views, predictions, arrays = saved
    if problem == "missing_k":
        del arrays["intrinsics_0"]
    elif problem == "local_only":
        arrays = {f"local_points_{i}": np.ones((2, 3, 3)) for i in range(3)}
    elif problem == "crop":
        views[0]["crop"] = [0, 0, 0.5, 1]
    elif problem == "manifest":
        views[0]["sizes"] = {}
    elif problem == "missing_image":
        views[0]["image"] = "absent.png"
    elif problem == "bad_pose":
        arrays["camera_poses_0"][0, 0] = 2
    elif problem == "extra_view":
        arrays["depth_3"] = np.ones((2, 3))
    elif problem == "grid":
        arrays["depth_0"] = np.ones((5, 6))
    np.savez(predictions, **arrays)
    with pytest.raises(ValueError, match=match):
        bridge.prepare_scene(native, "eth3d", "scene", views, predictions, root / "work", root)


def test_dtu_subset_preserves_native_masks_and_rejects_mixed_grids(saved, monkeypatch):
    root, native, views, predictions, arrays = saved
    monkeypatch.setattr(bridge, "require_dataset_access", lambda *args: None)
    native.data_root = str(root)
    for relative in ("Points/stl/stl001_total.ply", "SampleSet/mvs_data/ObsMask/ObsMask1_10.mat",
                     "SampleSet/mvs_data/ObsMask/Plane1.mat"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    full = native.get_data("scan1")
    full.aux["mask_files"] = []
    for i in range(3):
        mask = root / f"mask{i}.png"
        mask.touch()
        full.aux["mask_files"].append(str(mask))
    with pytest.raises(ValueError, match="equal predicted depth grids"):
        bridge.prepare_scene(native, "dtu", "scan1", views, predictions, root / "bad", root)
    arrays["depth_1"] = arrays["depth_0"].copy()
    np.savez(predictions, **arrays)
    path = bridge.prepare_scene(native, "dtu", "scan1", views[::-1], predictions, root / "good", root)
    assert native.get_data("scan1").aux["mask_files"] == full.aux["mask_files"][::-1]
    with np.load(path) as exported:
        np.testing.assert_array_equal(exported["depth"][0], arrays["depth_0"][..., 0])
        assert exported["intrinsics"][0, 0, 0] == 5


@pytest.mark.parametrize("dataset", ["eth3d", "scannetpp"])
def test_delegates_native_protocol_and_preserves_inference(saved, monkeypatch, permission, dataset):
    root, native, views, predictions, _ = saved
    calls = []

    def fuse(scene, result_path, fused, mode):
        calls.append(("fuse", scene, mode))
        with np.load(result_path) as exported:
            assert exported["depth"].dtype == np.float32
        assert (Path(result_path).parent.parent / "gt_meta.npz").exists()
        Path(fused).write_bytes(b"native point cloud")

    def evaluate(scene, fused):
        calls.append(("eval", scene))
        assert Path(fused).is_file()
        return {"fscore": np.float64(0.75)}

    native.fuse3d, native.eval3d = fuse, evaluate

    def load(dataset_name, data_root, upstream, **kwargs):
        assert dataset_name == dataset
        assert kwargs["project_root"] == root
        assert kwargs["permission_receipt"] == permission
        return native

    monkeypatch.setattr(bridge, "load_native_dataset", load)
    result = root / "result.json"
    result.write_text(json.dumps({"study": "s23", "dataset": dataset, "scenes_expected": ["scene"],
                                 "per_scene": [{"scene": "scene", "status": "completed",
                                                "views": views, "predictions_file": str(predictions)}]}))
    original, prediction_bytes = result.read_bytes(), predictions.read_bytes()
    report = bridge.reconstruct_result(result, data_root=root, project_root=root,
                                       permission_receipt=permission)
    assert report["status"] == "completed"
    assert report["per_scene"][0]["metrics"] == {"fscore": 0.75}
    assert calls == [("fuse", "scene", "recon_unposed"), ("eval", "scene")]
    assert result.read_bytes() == original
    assert predictions.read_bytes() == prediction_bytes
    assert (root / "recon.log").exists()
    assert json.loads((root / "recon.json").read_text()) == report
    with pytest.raises(FileExistsError):
        bridge.reconstruct_result(result, data_root=root)


def test_unavailable_dataset_and_missing_dependencies_are_clear(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Dataset unavailable"):
        bridge.load_native_dataset("unknown", tmp_path, tmp_path)
    with pytest.raises(FileNotFoundError, match="Dataset unavailable"):
        bridge.load_native_dataset("eth3d", tmp_path / "absent", tmp_path)
    monkeypatch.setattr(bridge.subprocess, "check_output",
                        lambda command, **kw: bridge.DA3_COMMIT if "rev-parse" in command else "")

    def missing(name):
        raise ImportError("No module named open3d")

    monkeypatch.setattr(bridge.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="dependencies unavailable.*open3d"):
        bridge.load_native_dataset("eth3d", tmp_path, tmp_path)


def test_failure_report_without_fake_metrics(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("Dataset unavailable")

    monkeypatch.setattr(bridge, "load_native_dataset", missing)
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"study": "s23", "dataset": "hiroom", "per_scene": []}))
    report = bridge.reconstruct_result(path, data_root=tmp_path)
    assert report["status"] == "partial_or_failed"
    assert "Dataset unavailable" in report["error"]
    assert report["per_scene"] == []


@pytest.fixture
def permission(tmp_path):
    receipt = tmp_path / "permission.json"
    receipt.write_text(json.dumps({
        "dataset": "scannetpp", "permission_granted": True, "terms_accepted": True,
        "accepted_by": "synthetic fixture", "accepted_utc": "2026-11-12T00:00:00Z",
        "license_url": "https://example.invalid/terms", "evidence": "synthetic data only",
    }))
    return receipt


@pytest.mark.parametrize("dataset,match", [
    ("scannetpp", "explicit license permission receipt"),
    ("dtu", "committed, unchanged"), ("hiroom", "committed, unchanged"),
])
def test_access_denied_before_dataset_or_native_access(tmp_path, monkeypatch, dataset, match):
    def forbidden(*args, **kwargs):
        pytest.fail("Dataset access or native import happened before the access gate")

    monkeypatch.setattr(bridge.importlib, "import_module", forbidden)
    monkeypatch.setattr(bridge.subprocess, "check_output", forbidden)
    with pytest.raises(PermissionError, match=match):
        bridge.load_native_dataset(dataset, tmp_path / "absent", tmp_path,
                                   project_root=tmp_path)
    with pytest.raises(PermissionError, match=match):
        bridge.prepare_scene(SimpleNamespace(get_data=forbidden), dataset, "scene",
                             [], tmp_path / "absent.npz", tmp_path / "work", tmp_path)
    assert not (tmp_path / "work").exists()


def test_scannet_loader_checks_receipt_and_uses_native_class(tmp_path, permission, monkeypatch):
    module = SimpleNamespace(
        __file__=str(tmp_path / "src/depth_anything_3/bench/datasets/scannetpp.py"),
        ScanNetPP=lambda: SimpleNamespace(_scene_cache={}))
    imports = []
    monkeypatch.setattr(bridge.subprocess, "check_output",
                        lambda command, **kw: bridge.DA3_COMMIT if "rev-parse" in command else "")
    monkeypatch.setattr(bridge.importlib, "import_module",
                        lambda name: imports.append(name) or module)
    monkeypatch.setattr(bridge.sys, "path", list(bridge.sys.path))
    native = bridge.load_native_dataset("scannetpp", tmp_path, tmp_path,
                                        project_root=tmp_path, permission_receipt=permission)
    assert imports == ["depth_anything_3.bench.datasets.scannetpp"]
    assert native.data_root == str(tmp_path)
    invalid = json.loads(permission.read_text())
    invalid["terms_accepted"] = False
    permission.write_text(json.dumps(invalid))
    with pytest.raises(PermissionError, match="Invalid ScanNet"):
        bridge.load_native_dataset("scannetpp", tmp_path / "absent", tmp_path,
                                   permission_receipt=permission)
    assert len(imports) == 1


def test_scannet_export_and_native_unposed_scaling(saved, permission):
    """Synthetic predictions; native prep with mocked resize/alignment, no scoring."""
    root, native, views, predictions, arrays = saved
    full = native.get_data("scene")
    # Different raw dimensions must not choose the exported model grid or K basis.
    Image.new("RGB", (9, 5)).save(views[1]["image"])
    views[1]["original_hw"] = [5, 9]
    full.intrinsics[:] = [[91, 0, 17], [0, 93, 19], [0, 0, 1]]
    full.aux.update(cam_hw_list=[(4, 6), (5, 9), (4, 6)],
                    roi_list=[(1, 1, 4, 2)] * 3,
                    gt_depth_files=full.image_files.copy())
    native._scene_cache = {"scene": full}
    cached_k = full.intrinsics.copy()
    ordered = views[::-1]
    path = bridge.prepare_scene(native, "scannetpp", "scene", ordered, predictions,
                                root / "work", root, permission_receipt=permission)
    with np.load(path) as exported, np.load(root / "work/gt_meta.npz") as gt:
        predicted = SimpleNamespace(**{key: exported[key].copy() for key in exported.files})
        gt_data = SimpleNamespace(**{key: gt[key].copy() for key in gt.files})
    assert predicted.depth.shape == (3, 768, 1024)
    for i in range(3):
        h, w = arrays[f"depth_{i}"].shape[:2]
        expected_k = arrays[f"intrinsics_{i}"].astype(np.float32)
        expected_k[0] *= 1024 / w
        expected_k[1] *= 768 / h
        np.testing.assert_allclose(predicted.intrinsics[i], expected_k)
        rows = np.arange(768) * h // 768
        cols = np.arange(1024) * w // 1024
        expected_depth = arrays[f"depth_{i}"][..., 0][rows[:, None], cols[None, :]]
        np.testing.assert_array_equal(predicted.depth[i], expected_depth)
    np.testing.assert_array_equal(gt_data.image_files, full.image_files[::-1])
    np.testing.assert_array_equal(gt_data.intrinsics, cached_k[::-1])
    np.testing.assert_array_equal(native._scene_cache["scene"].intrinsics, cached_k)
    np.testing.assert_array_equal(predicted.extrinsics[:, 0, 3], [-1, -2, -3])

    source = (Path(__file__).resolve().parents[1] / "third_party/Depth-Anything-3/src/"
              "depth_anything_3/bench/datasets/scannetpp.py")
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    prep = next(node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == "_prep_unposed")
    resize_calls = []

    def resize(depth, size, interpolation):
        resize_calls.append((depth.shape, size, interpolation))
        # Interpolation is OpenCV's responsibility; this test checks the native
        # dimension/K contract and mask order without optional heavy dependencies.
        return np.full(size[::-1], depth.flat[0], dtype=np.float32)

    namespace = {"np": np, "cv2": SimpleNamespace(resize=resize, INTER_NEAREST=0, INTER_LINEAR=1),
                 "Dict": SimpleNamespace,
                 "align_poses_umeyama": lambda gt, pred, **kw: (None, None, 2, pred)}
    exec(compile(ast.Module(body=[prep], type_ignores=[]), str(source), "exec"), namespace)
    masks = []
    mask = np.ones((768, 1024), dtype=bool)
    mask[0] = False
    prep_native = SimpleNamespace(input_h=768, input_w=1024,
                                 _load_gt_mask=lambda path: masks.append(path) or mask,
                                 _mask_invalid_depth=lambda depth, valid: np.where(valid, depth, 0))
    depths, intrinsics, extrinsics = namespace["_prep_unposed"](
        prep_native, predicted, gt_data,
        SimpleNamespace(aux=SimpleNamespace(**full.aux)), [2, 1, 0], scene="scene")
    assert depths.shape == (3, 768, 1024)
    assert resize_calls == [
        ((768, 1024), (6, 4), 0), ((4, 6), (1024, 768), 1),
        ((768, 1024), (9, 5), 0), ((5, 9), (1024, 768), 1),
        ((768, 1024), (6, 4), 0), ((4, 6), (1024, 768), 1),
    ]
    assert masks == full.aux["gt_depth_files"][::-1]
    assert np.all(depths[:, 0] == 0)
    assert np.any(depths[:, 1:] > 0)
    np.testing.assert_array_equal(intrinsics, predicted.intrinsics)
    np.testing.assert_array_equal(extrinsics, predicted.extrinsics)
    np.testing.assert_array_equal(full.intrinsics, cached_k)


def test_cli_forwards_explicit_permission_receipt(tmp_path, permission, monkeypatch, capsys):
    import runpy
    import sys

    result = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["07_reconstruction.py", str(result), "--data-root",
                                      str(tmp_path), "--permission-receipt", str(permission)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def reconstruct(path, **kwargs):
        assert path == result
        assert kwargs["permission_receipt"] == permission
        return {"status": "completed"}

    monkeypatch.setattr(bridge, "reconstruct_result", reconstruct)
    monkeypatch.setattr(sys, "path", list(sys.path))
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/07_reconstruction.py"),
                   run_name="__main__")
    assert "completed:" in capsys.readouterr().out
