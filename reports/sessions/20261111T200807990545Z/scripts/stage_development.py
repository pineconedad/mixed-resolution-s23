"""Download pinned archives, extracting only seed-42 image bytes for GPU inference."""
import sys
import zipfile
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import hf_hub_download
from mixres.data.benchmark import ARCHIVES, require_dataset_access
from mixres.data.remote_zip import HTTPRangeReader
from mixres.io import file_hash, read_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--discard-archives-after-staging", action="store_true")
    parser.add_argument("--range-only", action="store_true", default=True)
    parser.add_argument("--full-archive", action="store_false", dest="range_only")
    args = parser.parse_args()
    base = Path("third_party/Depth-Anything-3/workspace/benchmark_dataset")
    datasets = ["eth3d", "7scenes"]
    if Path("runs/views/s23/scannetpp/seed42.json").exists():
        require_dataset_access("scannetpp", permission_receipt="env/scannetpp-permission.json")
        datasets.append("scannetpp")
    for dataset in datasets:
        manifest = read_json(Path("runs/views/s23") / dataset / "seed42.json")
        targets = {view["image"]: view["image_sha256"]
                   for scene in manifest["scenes"] for view in scene["views"]}
        if all(Path(path).exists() and file_hash(path) == sha for path, sha in targets.items()):
            print(f"{dataset}: selected inputs already verified", flush=True)
            continue
        if args.range_only:
            from huggingface_hub import hf_hub_url
            sizes = {"eth3d": 15101568883, "7scenes": 3553806786, "scannetpp": 10830641822}
            archive = HTTPRangeReader(hf_hub_url("depth-anything/DA3-BENCH", f"{dataset}.zip",
                                      repo_type="dataset", revision=ARCHIVES[dataset]["revision"]),
                                      length=sizes[dataset])
            archive.block_size = 1024 * 1024
        else:
            archive = hf_hub_download("depth-anything/DA3-BENCH", f"{dataset}.zip", repo_type="dataset",
                                      revision=ARCHIVES[dataset]["revision"], local_dir="data/benchmark_archives")
            if file_hash(archive) != ARCHIVES[dataset]["sha256"]:
                raise RuntimeError(f"{dataset}: archive checksum mismatch")
        with zipfile.ZipFile(archive) as source:
            for path, sha in targets.items():
                target = Path(path)
                member = target.relative_to(base).as_posix()
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = source.read(member)
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(raw)
                if file_hash(temporary) != sha:
                    raise RuntimeError(f"Image checksum mismatch: {member}")
                temporary.replace(target)
        print(f"{dataset}: {len(targets)} selected images staged and verified", flush=True)
        if args.discard_archives_after_staging and not args.range_only:
            Path(archive).unlink()
