"""Bridge saved S23 predictions to the pinned DA3 recon_unposed protocol.

No inference is performed. Saved materialized views must contain image, crop,
original_hw, hw and hi_hw, in prediction order. A seed manifest with `sizes`
is not a substitute. Only full-image views are supported. Pi3 exports without
predicted intrinsics are unsupported: this bridge never fits K to GT.
"""

import copy
import importlib
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mixres.io import atomic_json, file_hash, read_json


DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DATASETS = {"eth3d": ("eth3d", "ETH3D"),
            "7scenes": ("sevenscenes", "SevenScenes"),
            "dtu": ("dtu", "DTU"), "hiroom": ("hiroom", "HiRoomDataset")}


def load_native_dataset(dataset, data_root, upstream, *, gt_root=None):
    """Import optional benchmark dependencies only when evaluation is requested."""
    if dataset not in DATASETS:
        raise ValueError(f"Dataset unavailable in this bridge: {dataset}")
    data_root, upstream = Path(data_root).resolve(), Path(upstream).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset unavailable: {data_root}")
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != DA3_COMMIT:
        raise RuntimeError(f"DA3 revision mismatch: expected {DA3_COMMIT}, got {commit}")
    if subprocess.check_output(
            ["git", "-C", str(upstream), "diff", "HEAD", "--", "src/depth_anything_3"],
            text=True):
        raise RuntimeError("DA3 source differs from the pinned revision")
    sys.path.insert(0, str(upstream / "src"))
    module, name = DATASETS[dataset]
    try:
        imported = importlib.import_module(f"depth_anything_3.bench.datasets.{module}")
    except ImportError as error:
        raise RuntimeError(f"Native DA3 benchmark dependencies unavailable: {error}") from error
    if not Path(imported.__file__).resolve().is_relative_to(upstream / "src"):
        raise RuntimeError("Imported DA3 is not from the pinned checkout")
    native = getattr(imported, name)()
    native.data_root = str(data_root)
    if dataset == "hiroom":
        if gt_root is None or not Path(gt_root).is_dir():
            raise FileNotFoundError("HiRoom requires an available --gt-root fused_pcd directory")
        native.gt_root_path = str(Path(gt_root).resolve())
    if dataset == "dtu":
        import torch

        if torch.cuda.is_available():
            raise RuntimeError("Native DTU auto-selects CUDA; run with CUDA_VISIBLE_DEVICES='' for CPU only")
    return native


def prepare_scene(native, dataset, scene, views, predictions, workdir, project_root):
    """Write native inputs without modifying predictions or using GT as predictions."""
    from PIL import Image

    if not views or any("sizes" in view for view in views):
        raise ValueError("Saved per-scene materialized views required, not a seed manifest")
    full = native.get_data(scene)
    lookup = {Path(path).resolve(): i for i, path in enumerate(full.image_files)}
    indices, images, original_sizes = [], [], []
    for view in views:
        if any(key not in view for key in ("image", "crop", "original_hw", "hw", "hi_hw")):
            raise ValueError("Incomplete saved materialized view")
        if list(view["crop"]) != [0, 0, 1, 1]:
            raise ValueError("Cropped views are unsupported by the native full-image protocol")
        image = (Path(project_root) / view["image"]).resolve()
        if image not in lookup:
            raise ValueError(f"Saved image not in native {dataset}/{scene} loader: {image}")
        if not image.is_file():
            raise FileNotFoundError(f"Dataset image unavailable: {image}")
        if "image_sha256" in view and file_hash(image) != view["image_sha256"]:
            raise ValueError(f"Saved image checksum mismatch: {image}")
        with Image.open(image) as opened:
            hw = (opened.height, opened.width)
        if tuple(view["original_hw"]) != hw:
            raise ValueError(f"Saved image dimensions mismatch: {image}")
        indices.append(lookup[image])
        images.append(str(full.image_files[lookup[image]]))
        original_sizes.append(hw)
    if len(set(indices)) != len(indices) or len(indices) < 3:
        raise ValueError("At least three distinct saved views are required for native alignment")
    if len(set(original_sizes)) != 1:
        raise ValueError("Native fusion requires equal original image dimensions")

    selected = copy.deepcopy(full)
    selected.image_files = images
    selected.extrinsics = np.asarray(full.extrinsics)[indices].copy()
    selected.intrinsics = np.asarray(full.intrinsics)[indices].copy()
    for key, value in full.aux.items():
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == len(full.image_files):
            selected.aux[key] = [value[i] for i in indices]
    for key in ("gt_mesh_path", "gt_pcd_path"):
        if key in full.aux and not Path(full.aux[key]).is_file():
            raise FileNotFoundError(f"Dataset GT unavailable: {full.aux[key]}")
    for key in ("gt_depth_files", "aliasing_mask_files", "mask_files"):
        for path in selected.aux.get(key, []):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Dataset mask/depth unavailable: {path}")
    if dataset == "dtu":
        scan = int(scene.removeprefix("scan"))
        for relative in (f"Points/stl/stl{scan:03}_total.ply",
                         f"SampleSet/mvs_data/ObsMask/ObsMask{scan}_10.mat",
                         f"SampleSet/mvs_data/ObsMask/Plane{scan}.mat"):
            if not (Path(native.data_root) / relative).is_file():
                raise FileNotFoundError(f"DTU evaluation GT unavailable: {relative}")

    depths, intrinsics, extrinsics = [], [], []
    with np.load(predictions, allow_pickle=False) as saved:
        expected_indices = set(range(len(views)))
        for field in ("camera_poses", "depth", "intrinsics"):
            actual = {int(key.removeprefix(field + "_")) for key in saved.files
                      if key.startswith(field + "_") and key.removeprefix(field + "_").isdigit()}
            if actual != expected_indices:
                raise ValueError(f"Predicted {field} missing or view count mismatch. Pi3 local_points/points "
                                 "without predicted K are unsupported; GT intrinsics are never substituted")
        for i, view in enumerate(views):
            pose = np.asarray(saved[f"camera_poses_{i}"], dtype=np.float64)
            k = np.array(saved[f"intrinsics_{i}"], dtype=np.float32, copy=True)
            depth = np.asarray(saved[f"depth_{i}"], dtype=np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if (pose.shape != (4, 4) or k.shape != (3, 3) or depth.ndim != 2
                    or not all(np.isfinite(a).all() for a in (pose, k, depth))
                    or not np.allclose(pose[3], [0, 0, 0, 1])
                    or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
                    or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3)
                    or k[0, 0] <= 0 or k[1, 1] <= 0
                    or not np.allclose(k[2], [0, 0, 1]) or not np.any(depth > 0)):
                raise ValueError(f"Invalid predicted camera/depth/intrinsics for view {i}")
            if depth.shape not in (tuple(view["hw"]), tuple(view["hi_hw"])):
                raise ValueError(f"Depth grid does not match saved materialized view {i}")
            if dataset != "dtu":
                # Native TSDF loaders resize each depth to the original image using
                # nearest sampling. Do that before stacking heterogeneous grids.
                h, w = original_sizes[i]
                k[0] *= w / depth.shape[1]
                k[1] *= h / depth.shape[0]
                # cv2.INTER_NEAREST uses floor(dst * src / dst_size), unlike
                # Pillow's half-pixel nearest convention.
                rows = np.arange(h) * depth.shape[0] // h
                cols = np.arange(w) * depth.shape[1] // w
                depth = depth[rows[:, None], cols[None, :]]
            depths.append(depth)
            intrinsics.append(k)
            extrinsics.append(np.linalg.inv(pose))
    if len({depth.shape for depth in depths}) != 1:
        raise ValueError("Native DTU requires equal predicted depth grids; mixed grids are unsupported")
    # DTU has no gt_meta hook. Restrict its loader to exactly the saved sample,
    # including the native mask order, while retaining its own fusion and eval.
    if dataset == "dtu":
        native.get_data = lambda requested: selected if requested == scene else full
    workdir = Path(workdir)
    export = workdir / "mini_npz"
    export.mkdir(parents=True)
    np.savez(workdir / "gt_meta.npz", extrinsics=selected.extrinsics,
             intrinsics=selected.intrinsics, image_files=np.asarray(images))
    path = export / "results.npz"
    np.savez(path, depth=np.stack(depths).astype(np.float32),
             intrinsics=np.stack(intrinsics), extrinsics=np.stack(extrinsics).astype(np.float32))
    return path


def reconstruct_result(result_path, *, data_root, views_path=None, project_root=".",
                       upstream="third_party/Depth-Anything-3", gt_root=None):
    """Evaluate one result cell, writing only recon.json and reconstruction logs.

    Views can be embedded as per_scene[].views or supplied as a JSON document
    with scenes: [{scene, views}]. No views are reconstructed from current config.
    Intermediate native exports and fused clouds are temporary.
    """
    result_path = Path(result_path).resolve()
    if result_path.name != "result.json":
        raise ValueError("Expected an inference result.json")
    result = read_json(result_path)
    if result.get("study") != "s23":
        raise ValueError("Only saved S23 results are supported")
    output = result_path.with_name("recon.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reconstruction: {output}")
    views_by_scene = {}
    if views_path is not None:
        document = read_json(views_path)
        for item in document["scenes"]:
            if item["scene"] in views_by_scene:
                raise ValueError("Duplicate scene in saved materialized views")
            views_by_scene[item["scene"]] = item["views"]
    report = {"protocol": "da3_native_recon_unposed", "da3_commit": DA3_COMMIT,
              "dataset": result["dataset"], "source_result": str(result_path),
              "source_sha256": file_hash(result_path), "per_scene": [],
              "materialized_views_sha256": file_hash(views_path) if views_path else None,
              "limitations": ["Full-image saved materialized views only; no seed-manifest reconstruction",
                              "Predicted depth, c2w and K required; no Pi3 intrinsic estimation",
                              "DTU mixed depth grids unsupported; native CPU fusion only",
                              "Native GT alignment and validity masks are part of recon_unposed",
                              "Scores describe the saved view subset, not the full DA3 sampling schedule"]}
    log_path = result_path.with_name("recon.log")
    with log_path.open("x") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            native = load_native_dataset(result["dataset"], data_root, upstream, gt_root=gt_root)
            records = result["per_scene"]
            if not records or len({r["scene"] for r in records}) != len(records):
                raise ValueError("Missing or duplicate inference scene records")
            for record in records:
                entry = {"scene": record["scene"], "status": "failed"}
                try:
                    if record.get("status") != "completed":
                        raise ValueError("Inference scene is not completed")
                    views = views_by_scene.get(record["scene"], record.get("views"))
                    if views is None:
                        raise ValueError("Missing saved per-scene materialized views; provide --views")
                    prediction = record.get("predictions_file")
                    if not prediction:
                        raise ValueError("Predictions were not saved for this inference scene")
                    prediction = (Path(project_root) / prediction).resolve()
                    if not prediction.is_file():
                        raise FileNotFoundError(f"Saved predictions unavailable: {prediction}")
                    entry["predictions_sha256"] = file_hash(prediction)
                    with TemporaryDirectory(prefix="da3-recon-") as temporary:
                        # A fresh instance avoids leaking DTU's sampled loader into another scene.
                        scene_native = copy.copy(native)
                        path = prepare_scene(scene_native, result["dataset"], record["scene"],
                                             views, prediction, temporary, project_root)
                        fused = str(Path(temporary) / "fused.ply")
                        scene_native.fuse3d(record["scene"], str(path), fused, "recon_unposed")
                        if not Path(fused).is_file() or Path(fused).stat().st_size == 0:
                            raise RuntimeError("Native fusion did not produce a point cloud")
                        metrics = {key: float(value) for key, value in
                                   scene_native.eval3d(record["scene"], fused).items()}
                        if not metrics or not all(np.isfinite(v) for v in metrics.values()):
                            raise ValueError("Native evaluation returned empty or nonfinite metrics")
                        entry.update(status="completed", metrics=metrics, n_views=len(views))
                except Exception as error:
                    entry["error"] = f"{type(error).__name__}: {error}"
                    traceback.print_exc()
                report["per_scene"].append(entry)
                log.flush()
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc()
    completed = {item["scene"] for item in report["per_scene"] if item["status"] == "completed"}
    expected = set(result.get("scenes_expected", [r["scene"] for r in result.get("per_scene", [])]))
    report["status"] = "completed" if expected and completed == expected and "error" not in report else "partial_or_failed"
    atomic_json(output, report)
    return report
