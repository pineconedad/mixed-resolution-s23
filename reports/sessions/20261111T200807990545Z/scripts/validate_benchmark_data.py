import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, UnidentifiedImageError

from mixres.io import atomic_json, read_json


def depth_path(dataset, view):
    image = Path(view["image"])
    if dataset == "7scenes":
        return image.with_name(image.name.replace(".color.png", ".depth.png"))
    scene_root = image.parents[2]
    relative = image.relative_to(scene_root / "images")
    return scene_root / "ground_truth_depth" / relative


def reprojection_check(view, depth_file, sample_count=1000):
    try:
        with Image.open(depth_file) as opened:
            depth = np.asarray(opened, dtype=np.float64)
    except UnidentifiedImageError:
        depth = np.fromfile(depth_file, dtype=np.float32).reshape(view["original_hw"]).astype(np.float64)
    if depth.shape != tuple(view["original_hw"]):
        raise ValueError(f"Depth/image size mismatch: {depth_file} {depth.shape}")
    valid = np.argwhere(np.isfinite(depth) & (depth > 0))
    if not len(valid):
        raise ValueError(f"No valid ground-truth depth: {depth_file}")
    selected = valid[np.linspace(0, len(valid) - 1, min(sample_count, len(valid))).astype(int)]
    y, x = selected[:, 0], selected[:, 1]
    z = depth[y, x]
    k = np.asarray(view["intrinsics"], dtype=np.float64)
    camera = np.stack([(x - k[0, 2]) / k[0, 0] * z,
                       (y - k[1, 2]) / k[1, 1] * z, z, np.ones_like(z)])
    world = np.asarray(view["c2w"], dtype=np.float64) @ camera
    recovered = np.asarray(view["w2c"], dtype=np.float64) @ world
    projected_x = recovered[0] / recovered[2] * k[0, 0] + k[0, 2]
    projected_y = recovered[1] / recovered[2] * k[1, 1] + k[1, 2]
    error = np.sqrt((projected_x - x) ** 2 + (projected_y - y) ** 2)
    return {"depth_file": str(depth_file), "valid_depth_pixels": int(len(valid)),
            "samples": int(len(selected)), "max_reprojection_error_px": float(error.max()),
            "mean_reprojection_error_px": float(error.mean())}


def validate(dataset):
    manifest = read_json(Path("runs/views/s23") / dataset / "seed42.json")
    scene = manifest["scenes"][0]
    checks = []
    for view in scene["views"][:3]:
        checks.append({"frame": view["frame"], **reprojection_check(view, depth_path(dataset, view))})
    mesh = (Path("third_party/Depth-Anything-3/workspace/benchmark_dataset/7scenes/7Scenes/meshes") /
            f"{scene['scene']}.ply" if dataset == "7scenes" else
            Path("third_party/Depth-Anything-3/workspace/benchmark_dataset/eth3d") /
            scene["scene"] / "combined_mesh.ply")
    if not mesh.exists() or mesh.stat().st_size == 0:
        raise ValueError(f"Missing ground-truth mesh: {mesh}")
    if max(check["max_reprojection_error_px"] for check in checks) > 1e-5:
        raise AssertionError("Pose/intrinsics reprojection check failed")
    return {"dataset": dataset, "scene": scene["scene"], "status": "passed",
            "mesh": str(mesh), "mesh_bytes": mesh.stat().st_size, "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["eth3d", "7scenes"],
                        default=["eth3d", "7scenes"])
    args = parser.parse_args()
    reports = [validate(dataset) for dataset in args.datasets]
    atomic_json("reports/data/geometry-checks.json", {"reports": reports})
    for report in reports:
        print(report["dataset"], report["status"], report["scene"])
