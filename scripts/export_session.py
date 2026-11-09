"""Produce a checksummed session archive for transfer before pod termination."""
import argparse
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixres.io import atomic_json, file_hash, git_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = sorted(p for directory in ("runs", "frozen", "env") if Path(directory).exists()
                   for p in Path(directory).rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    receipt = {"created_utc": stamp, **git_state(), "files": {str(p): file_hash(p) for p in paths}}
    receipt_path = destination / f"session-{stamp}.json"
    atomic_json(receipt_path, receipt)
    archive = destination / f"session-{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=str(path))
        tar.add(receipt_path, arcname="session-manifest.json")
    print(f"{archive} sha256={file_hash(archive)}")
    print("Download and verify this archive locally before stopping or terminating any pod.")
