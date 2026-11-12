import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixres.data.benchmark import load_dataset_scene, make_manifest, scene_names, require_dataset_access
from mixres.io import atomic_json, digest, read_json


ROOTS = {
    "eth3d": "third_party/Depth-Anything-3/workspace/benchmark_dataset/eth3d",
    "7scenes": "third_party/Depth-Anything-3/workspace/benchmark_dataset/7scenes",
    "scannetpp": "third_party/Depth-Anything-3/workspace/benchmark_dataset/scannetpp",
    "dtu": "third_party/Depth-Anything-3/workspace/benchmark_dataset/dtu",
    "hiroom": "third_party/Depth-Anything-3/workspace/benchmark_dataset/hiroom",
}


def selected_frames(manifest):
    return {scene["scene"]: [view["frame"] for view in scene["views"]]
            for scene in manifest["scenes"]}


def write_once(path, manifest):
    path = Path(path)
    if path.exists():
        previous = read_json(path)
        if digest(previous) != digest(manifest):
            raise RuntimeError(f"Refusing to overwrite a different view manifest: {path}")
        print(f"verified {path}")
        return
    atomic_json(path, manifest)
    print(f"wrote {path}")


def main(datasets, seeds, budgets, *, roots=None, permission_receipt=None):
    project_root = Path(__file__).resolve().parents[1]
    roots = roots or {}
    if 42 in seeds and seeds[0] != 42:
        raise ValueError("Seed 42 must come first")
    for dataset in datasets:
        require_dataset_access(dataset, project_root, permission_receipt)
        if dataset in {"dtu", "hiroom"}:
            if any(seed < 1000 for seed in seeds):
                raise ValueError("DTU/HiRoom require held-out seeds >=1000 only")
        elif 42 not in seeds:
            raise ValueError("Seed 42 is required to exclude development views")
    for dataset in datasets:
        # Development and frozen views must be disjoint. Drop a scene from every
        # seed when it cannot supply two complete view sets.
        root = Path(roots.get(dataset, project_root / ROOTS[dataset]))
        access = {"project_root": project_root, "permission_receipt": permission_receipt}
        required = 12 if dataset in {"dtu", "hiroom"} else (24 if any(seed != 42 for seed in seeds) else 12)
        dropped = [scene for scene in scene_names(dataset, root, **access)
                   if len(load_dataset_scene(dataset, root, scene, **access)) < required]
        development = None
        for seed in seeds:
            excluded = selected_frames(development) if development is not None and seed != 42 else None
            manifest = make_manifest(dataset, root, project_root, seed, budgets,
                                     excluded=excluded, dropped_scenes=dropped,
                                     permission_receipt=permission_receipt)
            write_once(project_root / "runs/views/s23" / dataset / f"seed{seed}.json", manifest)
            if seed == 42:
                development = manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(ROOTS), default=["7scenes", "eth3d"])
    parser.add_argument("--root", action="append", default=[], metavar="DATASET=PATH",
                        help="Override an archive root; HiRoom root contains data/ and its scene list")
    parser.add_argument("--scannetpp-permission-receipt", type=Path,
                        help="JSON with dataset=scannetpp, permission_granted=true, terms_accepted=true, "
                             "accepted_by, accepted_utc, license_url and evidence")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1000, 2000, 3000])
    parser.add_argument("--budgets", nargs="+", type=int, default=[4096, 8192])
    args = parser.parse_args()
    roots = {}
    for override in args.root:
        name, separator, path = override.partition("=")
        if not separator or name not in ROOTS or not path:
            parser.error("--root must be DATASET=PATH for a supported dataset")
        roots[name] = path
    main(args.datasets, args.seeds, args.budgets, roots=roots,
         permission_receipt=args.scannetpp_permission_receipt)
