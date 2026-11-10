"""Publish completed monitor logs and snapshots, refusing credential leakage."""
import argparse
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.credentials import runpod_key
from mixres.io import atomic_json, file_hash


def preserve(source, destination):
    root = Path(__file__).resolve().parents[1]
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_relative_to(root / "artifacts/control") or source == root / "artifacts/control":
        raise ValueError("Source must be one specific control session")
    if not destination.is_relative_to(root / "reports/control"):
        raise ValueError("Destination must be reports/control")
    secret = runpod_key().encode()
    files = sorted(p for p in source.rglob("*") if p.is_file())
    for path in files:
        if path.is_symlink() or secret in path.read_bytes():
            raise RuntimeError("Unsafe artifact; publication refused")
        if path.name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or secret in tar.extractfile(member).read():
                        raise RuntimeError("Unsafe archive; publication refused")
    hashes = {}
    for path in files:
        target = destination / path.relative_to(source)
        if target.exists() and file_hash(target) != file_hash(path):
            raise FileExistsError("Refusing to overwrite a different publication artifact")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(path.relative_to(source))] = file_hash(target)
    atomic_json(destination / "publication-manifest.json", {"files": hashes, "credential_scan_passed": True})
    print(f"Preserved {len(files)} monitor artifacts with credential scan and checksums")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    preserve(args.source, args.destination)
