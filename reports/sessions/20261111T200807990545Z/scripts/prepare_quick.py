"""Download only sampled 7-Scenes frames, with originals and GT kept locally."""
import argparse
import io
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from PIL import Image

from mixres.data.remote_zip import HTTPRangeReader, StoredMemberReader
from mixres.data.sampler import sample_scene
from mixres.eval.pose import validate_poses
from mixres.io import atomic_json, digest, file_hash, read_json
from mixres.rmfa.grid import patch_grid

BASE = "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"


def prepare(config):
    if config["seed"] != 42 or config["study"] != "quick":
        raise ValueError("This downloader is only for the independent seed-42 quick test")
    destination = Path(config["views"])
    root = Path(config["data_root"])
    all_scenes = []
    for scene_index, scene in enumerate(sorted(config["scenes"])):
        receipt_path = root / scene / "receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["config_hash"] != digest(config):
                raise RuntimeError("Existing scene belongs to a different configuration")
            for name, expected in receipt["files"].items():
                if file_hash(root / scene / name) != expected:
                    raise RuntimeError(f"Changed or incomplete data: {scene}/{name}")
            all_scenes.append(receipt["manifest"])
            print(f"{scene}: verified existing sampled originals", flush=True)
            continue
        url = f"{BASE}/{scene}.zip"
        remote = HTTPRangeReader(url)
        with ZipFile(remote) as outer:
            sequence = "seq-02" if scene == "stairs" else "seq-01"
            nested = next(info for info in outer.infolist() if info.filename.endswith(f"{sequence}.zip"))
            if nested.compress_type == ZIP_STORED:
                nested_file = StoredMemberReader(remote, nested)
            else:
                # Some official archives recompress seq ZIPs. Fetch just that sequence.
                nested_file = root / "_archives" / f"{scene}-{sequence}.zip"
                if not nested_file.exists():
                    nested_file.parent.mkdir(parents=True, exist_ok=True)
                    temporary = nested_file.with_suffix(".zip.tmp")
                    print(f"{scene}: downloading only {sequence}, {nested.file_size / 1e6:.1f} MB", flush=True)
                    with outer.open(nested) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    temporary.replace(nested_file)
            with ZipFile(nested_file) as inner:
                names = {Path(info.filename).name: info.filename for info in inner.infolist()}
                ids = sorted(name.removeprefix("frame-").removesuffix(".color.png")
                             for name in names if name.endswith(".color.png"))
                draw = np.random.default_rng(config["seed"] + scene_index).choice(len(ids), config["n_views"], replace=False)
                poses = np.repeat(np.eye(4)[None], len(ids), axis=0)
                files = {}
                for index in draw:
                    frame = ids[index]
                    for suffix in ("pose.txt", "color.png"):
                        name = f"frame-{frame}.{suffix}"
                        raw = inner.read(names[name])  # ZipFile verifies CRC for each extracted file.
                        path = root / scene / sequence / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix(path.suffix + ".tmp")
                        temporary.write_bytes(raw)
                        temporary.replace(path)
                        files[f"{sequence}/{name}"] = file_hash(path)
                        if suffix == "pose.txt":
                            poses[index] = np.loadtxt(io.BytesIO(raw))
                        else:
                            with Image.open(io.BytesIO(raw)) as image:
                                if image.size != (640, 480):
                                    raise ValueError("Unexpected 7-Scenes image size")
                validate_poses(poses[draw])
                views = sample_scene(ids, poses, seed=config["seed"], scene_index=scene_index,
                                     n_views=config["n_views"], n_sharp=config["n_sharp"],
                                     rho=config["rho"], budget=config["budget_tokens"])
                for view in views:
                    index = ids.index(view["frame"])
                    view["image"] = f"{scene}/{sequence}/frame-{view['frame']}.color.png"
                    view["c2w"] = poses[index].tolist()
                    view["w2c"] = np.linalg.inv(poses[index]).tolist()
                    view["intrinsics"] = [[585.0, 0, 320.0], [0, 585.0, 240.0], [0, 0, 1]]
                    view["image_sha256"] = files[f"{sequence}/frame-{view['frame']}.color.png"]
                manifest = {"scene": scene, "candidate_frames": len(ids), "views": views,
                            "actual_patch_tokens": sum(np.prod(patch_grid(v["hw"])) for v in views).item()}
                atomic_json(receipt_path, {"config_hash": digest(config), "manifest": manifest,
                            "files": files, "source": url, "source_etag": remote.etag,
                            "bytes_transferred": remote.transferred,
                            "retrieved_utc": datetime.now(timezone.utc).isoformat()})
                all_scenes.append(manifest)
                print(f"{scene}: {len(views)} frames, {remote.transferred / 1e6:.1f} MB transferred", flush=True)
        remote.session.close()
    manifest = {"schema": 1, "dataset": config["dataset"], "seed": config["seed"],
                "budget_tokens": config["budget_tokens"], "config_hash": digest(config),
                "scenes": all_scenes}
    if destination.exists() and read_json(destination) != manifest:
        raise RuntimeError("Refusing to overwrite a different saved view manifest")
    atomic_json(destination, manifest)
    print(f"Manifest: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    args = parser.parse_args()
    prepare(yaml.safe_load(Path(args.config).read_text()))
