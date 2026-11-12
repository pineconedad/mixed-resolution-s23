"""Deterministic metadata loading for the preprocessed DA3-BENCH archives."""

from collections import OrderedDict
import ast
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from mixres.io import file_hash, read_json
from mixres.rmfa.grid import budget_sizes
from mixres.data.resize import (
    SCANNETPP_HW, SCANNETPP_PREPROCESSING, scannetpp_camera,
)


ETH3D_SCENES = (
    "courtyard", "delivery_area", "electro", "facade", "kicker", "office",
    "pipes", "playground", "relief", "relief_2", "terrains",
)
SEVEN_SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
ETH3D_FILTERS = {
    "delivery_area": {"711.JPG", "712.JPG", "713.JPG", "714.JPG"},
    "electro": {"9289.JPG", "9290.JPG", "9291.JPG", "9292.JPG", "9293.JPG", "9298.JPG"},
    "playground": {"587.JPG", "588.JPG", "589.JPG", "590.JPG", "591.JPG", "592.JPG"},
    "relief": {f"{index}.JPG" for index in range(427, 439)},
    "relief_2": {f"{index}.JPG" for index in range(458, 469)},
}
ARCHIVES = {
    "eth3d": {
        "sha256": "2a12b961e72566e194d707db02f2be65686ff12ad5f4347cb16b4c266b2d6917",
        "revision": "cb98821d5d4f721704f11bafde0403826b82b638",
        "aspect": 3 / 2,
    },
    "7scenes": {
        "sha256": "4fc12afe9922c975068abd8918b897580926a932b718ad7646f6822205ef5b88",
        "revision": "cb98821d5d4f721704f11bafde0403826b82b638",
        "aspect": 4 / 3,
    },
}
HUB_REVISION = "cb98821d5d4f721704f11bafde0403826b82b638"
NATIVE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = PROJECT_ROOT / "third_party/Depth-Anything-3"
# Public LFS release metadata, not hashes inferred from downloaded dataset files.
for _name, _size, _sha in (
    ("scannetpp", 10830641822, "01f8ce1996ac64ba1dc9770c4f52db139f91c5c1a0d8a98107ef38fe45490d6c"),
    ("dtu", 8870335043, "1cf441e062db6d76c153db2b62f37de238127fbb1f57fc32bb5d2a525979e382"),
    ("hiroom", 716098077, "c2e3afbdb86a6dbbf9434c214f4e910d8c13f3a11c7e14b414d08bafa1be52d5"),
):
    ARCHIVES[_name] = {"bytes": _size, "sha256": _sha, "revision": HUB_REVISION}


def require_dataset_access(dataset, project_root=None, permission_receipt=None):
    """Fail before imports, directory listings, or dataset file reads."""
    project_root = Path(project_root or PROJECT_ROOT)
    if dataset == "scannetpp":
        if permission_receipt is None:
            raise PermissionError("ScanNet++ requires an explicit license permission receipt")
        receipt = read_json(permission_receipt)
        if (receipt.get("dataset") != "scannetpp"
                or receipt.get("permission_granted") is not True
                or receipt.get("terms_accepted") is not True
                or not all(isinstance(receipt.get(key), str) and receipt[key].strip()
                           for key in ("accepted_by", "accepted_utc", "license_url", "evidence"))):
            raise PermissionError("Invalid ScanNet++ permission receipt; original permission is required")
    if dataset in {"dtu", "hiroom"}:
        relative = "frozen/s23/predictions.md"
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=project_root,
            capture_output=True, check=False,
        )
        path = project_root / relative
        if (committed.returncode or not committed.stdout.strip()
                or not path.is_file() or path.read_bytes() != committed.stdout):
            raise PermissionError("Held-out access requires committed, unchanged frozen/s23/predictions.md")
        for staged in ([], ["--cached"]):
            result = subprocess.run(
                ["git", "diff", "--quiet", *staged, "HEAD", "--", relative],
                cwd=project_root, check=False,
            )
            if result.returncode:
                raise PermissionError("Held-out predictions must match HEAD and the index")


def _native_constants():
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=NATIVE_ROOT, text=True).strip()
    clean = subprocess.run(
        ["git", "diff", "--quiet", NATIVE_REVISION, "--", "src/depth_anything_3"],
        cwd=NATIVE_ROOT, check=False,
    )
    if revision != NATIVE_REVISION or clean.returncode:
        raise RuntimeError("DA3 benchmark source does not match the pinned native revision")
    tree = ast.parse((NATIVE_ROOT / "src/depth_anything_3/utils/constants.py").read_text())
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"SCANNETPP_SCENES", "DTU_SCENES"}}


def _native_loader(dataset, root):
    _native_constants()
    source = NATIVE_ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module(f"depth_anything_3.bench.datasets.{dataset}")
    if Path(module.__file__).resolve() != source / f"depth_anything_3/bench/datasets/{dataset}.py":
        raise RuntimeError("Imported DA3 benchmark is not the pinned local source")
    cls = getattr(module, {"scannetpp": "ScanNetPP", "dtu": "DTU"}[dataset])
    loader = cls()
    loader.data_root = str(Path(root))
    return loader


def _metadata_value(value):
    if isinstance(value, dict):
        return {key: _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_native_scene(dataset, root, scene):
    if dataset == "hiroom":
        # Mirror pinned get_data without its import-time default-root scene-list read.
        _native_constants()
        scene_root = Path(root) / "data" / scene
        intrinsics = np.load(scene_root / "cam_K.npy").astype(np.float32)
        records = []
        for image in sorted((scene_root / "image").iterdir()):
            frame = image.name.split(".")[0]
            pose = scene_root / "pose" / f"{frame}.npy"
            if not pose.exists():
                continue
            w2c = np.load(pose).astype(np.float32)
            with Image.open(image) as opened:
                hw = [opened.height, opened.width]
            records.append({"frame": image.name, "image": image, "original_hw": hw,
                            "w2c": w2c, "c2w": np.linalg.inv(w2c), "intrinsics": intrinsics.copy(),
                            "source_metadata": {"native_index": len(records), "aux": {
                                "gt_pcd_path": str(Path(root) / "fused_pcd" / ("-".join(scene.split("/")[-3:]) + ".ply")),
                                "gt_depth_files": str(scene_root / "depth" / f"{frame}.png"),
                                "aliasing_mask_files": str(scene_root / "aliasing_mask" / f"{frame}.png"),
                            }}})
        return records
    if dataset == "dtu" and len(list((Path(root) / "Rectified" / scene).glob("*.png"))) < 34:
        # Native DTU requires reference view at sorted index 33.
        return []
    loader = _native_loader(dataset, root)
    data = loader.get_data(scene)
    records = []
    aux = _metadata_value(data["aux"])
    for index, filename in enumerate(data["image_files"]):
        image = Path(filename)
        with Image.open(image) as opened:
            hw = [opened.height, opened.width]
        w2c = np.asarray(data["extrinsics"][index])
        per_view = {key: value[index] if isinstance(value, list) else value
                    for key, value in aux.items()}
        intrinsics = data["intrinsics"][index]
        if dataset == "scannetpp":
            # Keep raw source provenance, but calibrate the pixels load_view supplies.
            per_view["undistorted_intrinsics"] = np.asarray(intrinsics).tolist()
            _, _, optimal = scannetpp_camera(per_view)
            x, y, w, h = per_view["roi_list"]
            intrinsics = np.array(optimal, dtype=np.float64, copy=True)
            intrinsics[0, 2] -= x
            intrinsics[1, 2] -= y
            intrinsics[0] *= SCANNETPP_HW[1] / w
            intrinsics[1] *= SCANNETPP_HW[0] / h
            per_view["preprocessing"] = SCANNETPP_PREPROCESSING
        records.append({"frame": image.relative_to(Path(loader.data_root)).as_posix(),
                        "image": image, "original_hw": hw, "w2c": w2c,
                        "c2w": np.linalg.inv(w2c), "intrinsics": np.asarray(intrinsics),
                         "source_metadata": {"native_index": index, "aux": per_view}})
        if dataset == "scannetpp":
            records[-1]["intrinsics_hw"] = list(SCANNETPP_HW)
    return records


def _quaternion_matrix(qw, qx, qy, qz):
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _eth3d_cameras(path):
    cameras = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8:
            cameras[parts[0]] = {
                "width": int(float(parts[2])), "height": int(float(parts[3])),
                "intrinsics": np.asarray([
                    [float(parts[4]), 0, float(parts[6])],
                    [0, float(parts[5]), float(parts[7])],
                    [0, 0, 1],
                ]),
            }
    return cameras


def _eth3d_poses(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines()
             if not line.startswith("#")]
    poses = OrderedDict()
    for line in lines[::2]:
        parts = line.split()
        if len(parts) < 10:
            continue
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = _quaternion_matrix(*map(float, parts[1:5]))
        w2c[:3, 3] = np.asarray(parts[5:8], dtype=np.float64)
        poses[parts[9]] = (parts[8], w2c)
    return poses


def load_eth3d(root, scene):
    scene_root = Path(root) / scene
    cameras = _eth3d_cameras(scene_root / "dslr_calibration_jpg/cameras.txt")
    poses = _eth3d_poses(scene_root / "dslr_calibration_jpg/images.txt")
    records = []
    for name, (camera_id, w2c) in poses.items():
        if any(name.endswith(suffix) for suffix in ETH3D_FILTERS.get(scene, set())):
            continue
        image = scene_root / "images" / name
        camera = cameras.get(camera_id)
        if image.exists() and camera is not None:
            records.append({"frame": name, "image": image, "w2c": w2c,
                            "c2w": np.linalg.inv(w2c), "intrinsics": camera["intrinsics"],
                            "original_hw": [camera["height"], camera["width"]]})
    return records


def load_7scenes(root, scene):
    sequence = "seq-02" if scene == "stairs" else "seq-01"
    scene_root = Path(root) / "7Scenes" / scene / sequence
    intrinsics = np.asarray([[585.0, 0, 320.0], [0, 585.0, 240.0], [0, 0, 1]])
    records = []
    for image in sorted(scene_root.glob("frame-*.color.png")):
        prefix = image.name.removesuffix(".color.png")
        pose = image.with_name(f"{prefix}.pose.txt")
        depth = image.with_name(f"{prefix}.depth.png")
        if not pose.exists() or not depth.exists():
            continue
        c2w = np.loadtxt(pose, dtype=np.float64)
        records.append({"frame": prefix, "image": image, "depth": depth,
                        "w2c": np.linalg.inv(c2w), "c2w": c2w,
                        "intrinsics": intrinsics, "original_hw": [480, 640]})
    return records


def load_dataset_scene(dataset, root, scene, *, project_root=None, permission_receipt=None):
    require_dataset_access(dataset, project_root, permission_receipt)
    if dataset == "eth3d":
        return load_eth3d(root, scene)
    if dataset == "7scenes":
        return load_7scenes(root, scene)
    if dataset in {"scannetpp", "dtu", "hiroom"}:
        return _load_native_scene(dataset, root, scene)
    raise ValueError(f"Dataset is unavailable or gated: {dataset}")


def scene_names(dataset, root=None, *, project_root=None, permission_receipt=None):
    require_dataset_access(dataset, project_root, permission_receipt)
    if dataset == "eth3d":
        return ETH3D_SCENES
    if dataset == "7scenes":
        return SEVEN_SCENES
    if dataset in {"scannetpp", "dtu"}:
        return tuple(_native_constants()[{"scannetpp": "SCANNETPP_SCENES", "dtu": "DTU_SCENES"}[dataset]])
    if dataset == "hiroom":
        if root is None:
            raise ValueError("HiRoom requires an explicit archive root containing selected_scene_list_val.txt")
        names = (Path(root) / "selected_scene_list_val.txt").read_text().splitlines()
        if not names or any(not name or Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("Invalid or empty native HiRoom scene list")
        return tuple(names)
    raise ValueError(f"Dataset is unavailable or gated: {dataset}")


def make_manifest(dataset, root, project_root, seed, budgets, *, n_views=12,
                  n_sharp=4, rho=2.0, excluded=None, dropped_scenes=(), permission_receipt=None):
    from mixres.data.sampler import farthest_points

    require_dataset_access(dataset, project_root, permission_receipt)
    if dataset in {"dtu", "hiroom"} and seed < 1000:
        raise ValueError("DTU/HiRoom are held-out only; no development seed")
    root = Path(root).resolve()
    project_root = Path(project_root).resolve()
    excluded = excluded or {}
    scenes = []
    dropped_scenes = set(dropped_scenes)
    access = {"project_root": project_root, "permission_receipt": permission_receipt}
    kept_scenes = [scene for scene in sorted(scene_names(dataset, root, **access)) if scene not in dropped_scenes]
    for scene_index, scene in enumerate(kept_scenes):
        records = load_dataset_scene(dataset, root, scene, **access)
        blocked = set(excluded.get(scene, ()))
        eligible = [index for index, record in enumerate(records) if record["frame"] not in blocked]
        if len(eligible) < n_views:
            raise ValueError(f"{dataset}/{scene} has only {len(eligible)} eligible frames")
        selected = np.random.default_rng(seed + scene_index).choice(eligible, n_views, replace=False)
        centres = np.asarray([records[index]["c2w"][:3, 3] for index in selected])
        sharp_local = set(farthest_points(centres, n_sharp))
        basis_hw = records[0].get("intrinsics_hw", records[0]["original_hw"])
        aspect = ARCHIVES[dataset].get("aspect", basis_hw[1] / basis_hw[0])
        sizes = {str(budget): budget_sizes(budget, n_sharp, n_views - n_sharp, rho, aspect)
                 for budget in budgets}
        uniform_sizes = {str(budget): budget_sizes(budget, n_views, 0, 1.0,
                                                   aspect)[0]
                         for budget in budgets}
        views = []
        for local_index, record_index in enumerate(selected):
            record = records[record_index]
            image = record["image"].resolve()
            try:
                image_path = image.relative_to(project_root)
            except ValueError as error:
                if dataset in {"eth3d", "7scenes"}:
                    raise ValueError(f"Image must live under project root: {image}") from error
                image_path = image
            with Image.open(image) as opened:
                actual_hw = [opened.height, opened.width]
            if actual_hw != record["original_hw"]:
                raise ValueError(f"Image metadata mismatch: {image}")
            sharp = local_index in sharp_local
            budget_sizes_by_name = {}
            for budget, (hi, lo) in sizes.items():
                budget_sizes_by_name[budget] = {
                    "hw": list(hi if sharp else lo), "hi_hw": list(hi),
                    "uniform_hw": list(uniform_sizes[budget]),
                }
            views.append({
                "frame": record["frame"], "image": image_path.as_posix(),
                "image_sha256": file_hash(image), "sharp": sharp,
                "rho": 1.0 if sharp else rho, "crop": [0.0, 0.0, 1.0, 1.0],
                "original_hw": actual_hw, "c2w": record["c2w"].tolist(),
                "w2c": record["w2c"].tolist(), "intrinsics": record["intrinsics"].tolist(),
                "sizes": budget_sizes_by_name,
            })
            if "source_metadata" in record:
                views[-1]["source_metadata"] = record["source_metadata"]
            if "intrinsics_hw" in record:
                views[-1]["intrinsics_hw"] = record["intrinsics_hw"]
        scenes.append({"scene": scene, "candidate_count": len(records), "views": views})
    if not scenes and dataset in {"scannetpp", "dtu", "hiroom"}:
        raise ValueError(f"No {dataset} scenes can supply the required view sets")
    manifest = {
        "study": "s23", "baseline_protocol": "fresh_v1", "dataset": dataset,
        "seed": seed, "budgets": list(budgets), "n_views": n_views,
        "n_sharp": n_sharp, "rho": rho, "scenes": scenes,
        "dropped_scenes": sorted(dropped_scenes),
        "source": {"archive_sha256": ARCHIVES[dataset]["sha256"],
                   "da3_bench_revision": ARCHIVES[dataset]["revision"]},
    }
    if dataset in {"scannetpp", "dtu", "hiroom"}:
        manifest["source"].update(native_revision=NATIVE_REVISION, root=str(root),
                                  native_loader=f"depth_anything_3.bench.datasets.{dataset}")
        if dataset == "scannetpp":
            manifest["source"]["permission_receipt_sha256"] = file_hash(permission_receipt)
    return manifest
