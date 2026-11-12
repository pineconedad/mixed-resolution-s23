"""Stage checksum-bound image bytes from saved manifests; no inference."""
import sys
import zipfile
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import hf_hub_download
from mixres.data.benchmark import ARCHIVES, require_dataset_access
from mixres.data.remote_zip import HTTPRangeReader
from mixres.io import file_hash, read_json


def collect_targets(dataset, seeds, permission_receipt="env/scannetpp-permission.json"):
    require_dataset_access(dataset, permission_receipt=permission_receipt)
    targets = {}
    for seed in seeds:
        if dataset in {"dtu", "hiroom"} and seed < 1000:
            raise RuntimeError("Held-out inputs require frozen seeds >=1000")
        manifest = read_json(Path("runs/views/s23") / dataset / f"seed{seed}.json")
        if manifest["dataset"] != dataset or manifest["seed"] != seed:
            raise RuntimeError("Input manifest identity mismatch")
        for scene in manifest["scenes"]:
            for view in scene["views"]:
                path, sha = view["image"], view["image_sha256"]
                if path in targets and targets[path] != sha:
                    raise RuntimeError(f"Conflicting input hashes: {path}")
                targets[path] = sha
    if not targets:
        raise RuntimeError("No selected images; refusing an empty input stage")
    return targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--discard-archives-after-staging", action="store_true")
    parser.add_argument("--range-only", action="store_true", default=True)
    parser.add_argument("--full-archive", action="store_false", dest="range_only")
    parser.add_argument("--datasets", nargs="+", choices=list(ARCHIVES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--permission-receipt", default="env/scannetpp-permission.json")
    args = parser.parse_args()
    base = Path("third_party/Depth-Anything-3/workspace/benchmark_dataset")
    datasets = args.datasets or ["eth3d", "7scenes"]
    if args.datasets is None and Path("runs/views/s23/scannetpp/seed42.json").exists():
        datasets.append("scannetpp")
    for dataset in datasets:
        targets = collect_targets(dataset, args.seeds, args.permission_receipt)
        if all(Path(path).exists() and file_hash(path) == sha for path, sha in targets.items()):
            print(f"{dataset}: selected inputs already verified", flush=True)
            continue
        if args.range_only:
            from huggingface_hub import hf_hub_url
            sizes = {"eth3d": 15101568883, "7scenes": 3553806786,
                     **{name: row["bytes"] for name, row in ARCHIVES.items() if "bytes" in row}}
            archive = HTTPRangeReader(hf_hub_url("depth-anything/DA3-BENCH", f"{dataset}.zip",
                                      repo_type="dataset", revision=ARCHIVES[dataset]["revision"]),
                                      length=sizes[dataset])
            archive.block_size = 8 * 1024 * 1024 if dataset == "eth3d" else 1024 * 1024
        else:
            archive = hf_hub_download("depth-anything/DA3-BENCH", f"{dataset}.zip", repo_type="dataset",
                                      revision=ARCHIVES[dataset]["revision"], local_dir="data/benchmark_archives")
            if file_hash(archive) != ARCHIVES[dataset]["sha256"]:
                raise RuntimeError(f"{dataset}: archive checksum mismatch")
        with zipfile.ZipFile(archive) as source:
            for path, sha in targets.items():
                target = Path(path)
                if target.exists() and file_hash(target) == sha:
                    continue
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
