"""Download, verify and sample HiRoom for exploratory development.

HiRoom was a sealed held-out set in the S23 plan. The user unsealed it on 2026-11-13 so the
resolution findings can be checked on a second dataset with accurate ground truth. From that
point it is development data and must not be reported as held-out. DTU stays sealed.

This module reads the archive directly (same layout the pinned loader mirrors: data/<scene>/
image, pose/<frame>.npy as world-to-camera, cam_K.npy). It does not touch the frozen-run gates
in mixres.data.benchmark, which keep refusing HiRoom for the S23 held-out path.
"""
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from mixres.data.benchmark import ARCHIVES
from mixres.data.sampler import farthest_points
from mixres.io import file_hash
from mixres.rmfa.grid import budget_sizes

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data/hiroom"


def fetch():
    """Download the pinned archive, check its SHA-256, extract it. Returns the dataset root."""
    marker = EXTRACTED / ".verified"
    if not marker.exists():
        from huggingface_hub import hf_hub_download
        archive = hf_hub_download("depth-anything/DA3-BENCH", "hiroom.zip", repo_type="dataset",
                                  revision=ARCHIVES["hiroom"]["revision"], local_dir=str(ROOT / "data/benchmark_archives"))
        if file_hash(archive) != ARCHIVES["hiroom"]["sha256"]:
            raise RuntimeError("HiRoom archive checksum mismatch")
        with zipfile.ZipFile(archive) as source:
            for member in source.namelist():
                target = (EXTRACTED / member).resolve()
                if EXTRACTED.resolve() not in target.parents and target != EXTRACTED.resolve():
                    raise RuntimeError(f"Unsafe archive member: {member}")
            source.extractall(EXTRACTED)
        marker.write_text(ARCHIVES["hiroom"]["sha256"] + "\n")
    lists = sorted(EXTRACTED.rglob("selected_scene_list_val.txt"))
    if len(lists) != 1:
        raise RuntimeError(f"Expected one HiRoom scene list, found {len(lists)}")
    return lists[0].parent


def load_scene(root, scene):
    scene_root = root / "data" / scene
    intrinsics = np.load(scene_root / "cam_K.npy").astype(np.float64)
    records = []
    for image in sorted((scene_root / "image").iterdir()):
        pose = scene_root / "pose" / f"{image.name.split('.')[0]}.npy"
        if not pose.exists():
            continue
        w2c = np.eye(4)
        loaded = np.load(pose).astype(np.float64)
        w2c[:loaded.shape[0], :4] = loaded
        records.append({"frame": image.name, "image": image, "w2c": w2c, "c2w": np.linalg.inv(w2c),
                        "intrinsics": intrinsics})
    return records


def make_manifest(root, seed=42, budget=4096, n_views=12, n_sharp=4, rho=2.0):
    """Development manifest in the S23 format. Mirrors mixres.data.benchmark.make_manifest's sampling."""
    if not 42 <= seed < 1000:
        raise ValueError("Development seeds only")
    names = [name for name in (root / "selected_scene_list_val.txt").read_text().splitlines() if name.strip()]
    if not names or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("Invalid or empty HiRoom scene list")
    scenes, skipped, shapes = [], [], set()
    for scene_index, scene in enumerate(sorted(names)):
        records = load_scene(root, scene)
        if len(records) < n_views:
            skipped.append(f"{scene} ({len(records)} frames)")
            continue
        selected = np.random.default_rng(seed + scene_index).choice(len(records), n_views, replace=False)
        sharp_local = set(farthest_points(np.asarray([records[i]["c2w"][:3, 3] for i in selected]), n_sharp))
        views = []
        for local_index, record_index in enumerate(selected):
            record = records[record_index]
            with Image.open(record["image"]) as opened:
                hw = (opened.height, opened.width)
            shapes.add(hw)
            views.append({"frame": record["frame"], "image": record["image"].resolve().relative_to(ROOT).as_posix(),
                          "image_sha256": file_hash(record["image"]), "sharp": local_index in sharp_local,
                          "rho": 1.0 if local_index in sharp_local else rho, "crop": [0.0, 0.0, 1.0, 1.0],
                          "original_hw": list(hw), "c2w": record["c2w"].tolist(), "w2c": record["w2c"].tolist(),
                          "intrinsics": record["intrinsics"].tolist()})
        scenes.append({"scene": scene.replace("/", "__"), "candidate_count": len(records), "views": views})
    if not scenes:
        raise ValueError("No HiRoom scene has enough frames")
    if len(shapes) != 1:
        raise ValueError(f"HiRoom images have several sizes {sorted(shapes)}; one uniform-size plan cannot serve them")
    height, width = shapes.pop()
    aspect = width / height
    hi, lo = budget_sizes(budget, n_sharp, n_views - n_sharp, rho, aspect)
    uniform = budget_sizes(budget, n_views, 0, 1.0, aspect)[0]
    for scene in scenes:
        for view in scene["views"]:
            view["sizes"] = {str(budget): {"hw": list(hi if view["sharp"] else lo), "hi_hw": list(hi),
                                           "uniform_hw": list(uniform)}}
    return {"study": "salvation_development", "dataset": "hiroom", "seed": seed, "budgets": [budget],
            "n_views": n_views, "n_sharp": n_sharp, "rho": rho, "scenes": scenes, "skipped_scenes": skipped,
            "status": "unsealed for exploratory development on 2026-11-13; not a held-out set",
            "source": {"archive_sha256": ARCHIVES["hiroom"]["sha256"],
                       "da3_bench_revision": ARCHIVES["hiroom"]["revision"]}}, aspect
