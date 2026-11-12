"""Synthetic RGB only; no restricted dataset reads or native heavy imports."""

import ast
import json
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

from mixres.data import benchmark
from mixres.data.resize import SCANNETPP_HW, load_view, resize_image, transform_intrinsics
from mixres.io import file_hash
from mixres.rmfa.arms import ARMS
from mixres.rmfa.grid import budget_sizes


def test_scannet_native_preprocessing_and_manifest(tmp_path, monkeypatch):
    source = benchmark.NATIVE_ROOT / "src/depth_anything_3/bench/datasets/scannetpp.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "load_image")
    namespace = {"np": np, "cv2": cv2, "Dict": dict,
                 "imageio": SimpleNamespace(imread=lambda path: np.array(Image.open(path)))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    constants = ast.parse((benchmark.NATIVE_ROOT / "src/depth_anything_3/utils/constants.py").read_text())
    sizes = {node.targets[0].id: ast.literal_eval(node.value) for node in constants.body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in {"SCANNETPP_INPUT_H", "SCANNETPP_INPUT_W"}}
    assert SCANNETPP_HW == (sizes["SCANNETPP_INPUT_H"], sizes["SCANNETPP_INPUT_W"])

    image = tmp_path / "scene" / "iphone.png"
    image.parent.mkdir()
    pixels = np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image)
    raw_hash = file_hash(image)
    raw = np.array([[72, 0, 46.5], [0, 70, 30.5], [0, 0, 1]], dtype=np.float32)
    distortion = np.array([0.15, -0.05, 0.01, -0.02, 0], dtype=np.float32)
    optimal, _ = cv2.getOptimalNewCameraMatrix(raw, distortion, (96, 64), 1, (96, 64))
    # Nonzero origin and unequal scaling catch ROI/principal-point mistakes.
    roi = (3, 4, 85, 53)
    aux = {"ixt_raw_list": [raw], "dist_list": [distortion],
           "roi_list": [roi], "cam_hw_list": [(64, 96)]}
    data = {"image_files": [str(image)], "extrinsics": [np.eye(4)],
            "intrinsics": [optimal], "aux": aux}
    monkeypatch.setattr(benchmark, "_native_loader", lambda *args:
                        SimpleNamespace(data_root=str(tmp_path), get_data=lambda scene: data))
    monkeypatch.setattr(benchmark, "scene_names", lambda *args, **kwargs: ("scene",))
    receipt = tmp_path / "permission.json"
    receipt.write_text(json.dumps({
        "dataset": "scannetpp", "permission_granted": True, "terms_accepted": True,
        "accepted_by": "synthetic fixture", "accepted_utc": "2026-11-12T00:00:00Z",
        "license_url": "https://example.invalid/terms", "evidence": "synthetic data only",
    }))
    manifest = benchmark.make_manifest("scannetpp", tmp_path, tmp_path, 42, [256],
                                       n_views=1, n_sharp=1, permission_receipt=receipt)
    view = manifest["scenes"][0]["views"][0]
    assert view["original_hw"] == [64, 96]
    assert view["intrinsics_hw"] == [768, 1024]
    assert view["image_sha256"] == raw_hash
    assert manifest["source"]["permission_receipt_sha256"] == file_hash(receipt)
    np.testing.assert_array_equal(view["source_metadata"]["aux"]["ixt_raw_list"], raw)
    np.testing.assert_array_equal(view["source_metadata"]["aux"]["undistorted_intrinsics"], optimal)
    expected_k = optimal.astype(np.float64)
    expected_k[0, 2] -= roi[0]
    expected_k[1, 2] -= roi[1]
    expected_k[0] *= 1024 / roi[2]
    expected_k[1] *= 768 / roi[3]
    np.testing.assert_allclose(view["intrinsics"], expected_k)
    hi, _ = budget_sizes(256, 1, 0, 2.0, 4 / 3)
    assert view["sizes"]["256"]["hi_hw"] == list(hi)

    native = SimpleNamespace(_scene_cache={}, input_h=768, input_w=1024)
    expected = namespace["load_image"](native, str(image), 0, SimpleNamespace(**aux))
    full_view = {**view, "hw": [768, 1024], "hi_hw": [768, 1024]}
    tensor = torch.from_numpy(expected).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(load_view(image, full_view, "pixel"), tensor, rtol=0, atol=0)

    crop = [0.13, 0.19, 0.88, 0.91]
    x0, y0, x1, y1 = round(crop[0] * 1024), round(crop[1] * 768), round(crop[2] * 1024), round(crop[3] * 768)
    cropped = tensor[:, y0:y1, x0:x1]
    for sharp in (False, True):
        for entry in sorted({arm.entry for arm in ARMS.values()}):
            arm_view = {**view, "sharp": sharp, "crop": crop, "hw": [28, 42], "hi_hw": [56, 70]}
            if sharp or entry == "real":
                reference = resize_image(cropped, arm_view["hi_hw"], downsample=True)
            else:
                reference = resize_image(cropped, arm_view["hw"], downsample=True)
                if entry == "pixel":
                    reference = resize_image(reference, arm_view["hi_hw"], downsample=False)
            torch.testing.assert_close(load_view(image, arm_view, entry), reference, rtol=0, atol=0)
    cropped_k = expected_k.copy()
    cropped_k[0, 2] -= x0
    cropped_k[1, 2] -= y0
    cropped_k[0] *= 70 / (x1 - x0)
    cropped_k[1] *= 56 / (y1 - y0)
    np.testing.assert_allclose(transform_intrinsics(view["intrinsics"], view["intrinsics_hw"],
                                                  [56, 70], crop), cropped_k)

    # Other datasets still enter the existing PIL crop/resize path unchanged.
    plain = {"sharp": True, "crop": [0, 0, 1, 1], "hi_hw": [64, 96]}
    raw_tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(load_view(image, plain, "real"), raw_tensor, rtol=0, atol=0)
    assert file_hash(image) == raw_hash
