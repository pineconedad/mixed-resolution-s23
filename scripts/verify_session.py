"""Verify a downloaded session archive before safely restoring its files."""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixres.io import file_hash


def verify_session(archive, expected_sha256, destination):
    if file_hash(archive) != expected_sha256:
        raise ValueError("Archive SHA-256 does not match remote export")
    destination = Path(destination)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate paths")
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
        receipt = json.load(tar.extractfile("session-manifest.json"))
        if set(names) != set(receipt["files"]) | {"session-manifest.json"}:
            raise ValueError("Archive membership differs from receipt")
        for name, expected in receipt["files"].items():
            raw = tar.extractfile(name).read()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Member checksum mismatch: {name}")
        # Restore only after verifying every checksum. Never overwrite a different file.
        for member in members:
            raw = tar.extractfile(member).read()
            target = destination / member.name
            if target.exists() and target.read_bytes() != raw:
                raise FileExistsError(f"Refusing to overwrite different local data: {target}")
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(tar.extractfile(member).read())
            temporary.replace(target)
    print(f"Verified archive SHA-256 and all {len(receipt['files'])} member checksums")
    print(f"Restored session to {destination}")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    verify_session(args.archive, args.sha256, args.destination)
