"""Pack only selected development inputs and allowlisted source, never credentials."""
import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.io import atomic_json, file_hash, read_json


def pack(destination, source_only=False):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / "source.tar.gz"
    files = set()
    for directory in ("mixres", "scripts", "configs", "env", "tests", "tests_vggt", "runs/views/s23"):
        files.update(path for path in (root / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and not path.name.startswith((".", "actual-")))
    for name in ("pyproject.toml", ".gitignore", ".gitmodules", "env.sh", "STUDY_CHECKLIST.md"):
        files.add(root / name)
    datasets = ["eth3d", "7scenes"]
    permission = root / "artifacts/permissions/scannetpp.json"
    if (root / "runs/views/s23/scannetpp/seed42.json").exists():
        from mixres.data.benchmark import require_dataset_access
        require_dataset_access("scannetpp", root, permission)
        datasets.append("scannetpp")
    # Submodule source comes from its public pinned repository, not arbitrary local files.
    with tarfile.open(source, "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=path.relative_to(root), recursive=False)
        if "scannetpp" in datasets:
            archive.add(permission, arcname="env/scannetpp-permission.json", recursive=False)
    if source_only:
        metadata = destination / "metadata.pack"
        objects = subprocess.check_output(["git", "rev-list", "--objects", "--no-object-names",
                                            "--filter=blob:none", "HEAD"], cwd=root)
        with metadata.open("wb") as stream:
            subprocess.run(["git", "pack-objects", "--stdout"], input=objects,
                           stdout=stream, cwd=root, check=True)
        receipt = {path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
                   for path in (source, metadata)}
        atomic_json(destination / "receipt.json", receipt)
        print(receipt)
        return
    images = set()
    for dataset in datasets:
        manifest = read_json(root / "runs/views/s23" / dataset / "seed42.json")
        images.update(root / view["image"] for scene in manifest["scenes"] for view in scene["views"])
    inputs = destination / "inputs.tar"
    with tarfile.open(inputs, "w") as archive:
        for path in sorted(images):
            archive.add(path, arcname=path.relative_to(root), recursive=False)
    bundle = destination / "repository.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=root, check=True)
    receipt = {path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
               for path in (source, inputs, bundle)}
    atomic_json(destination / "receipt.json", receipt)
    print(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="artifacts/development-pack")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    pack(args.destination, args.source_only)
