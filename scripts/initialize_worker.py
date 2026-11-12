"""User-operated initialization of the allowlisted sparse source bundle."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def checked_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def store_source_blobs(paths):
    if paths:
        subprocess.run(["git", "hash-object", "-w", "--", *paths],
                       stdout=subprocess.DEVNULL, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    for path in (args.source, args.metadata):
        if checked_hash(path) != receipt[path.name]["sha256"]:
            raise SystemExit(f"Bundle checksum mismatch: {path}")
    commit = receipt["source_state"]["git_commit"]
    if not Path(".git").exists():
        subprocess.run(["git", "init", "-q"], check=True)
        with args.metadata.open("rb") as stream:
            subprocess.run(["git", "index-pack", "--stdin"], stdin=stream, check=True)
        subprocess.run(["git", "update-ref", "HEAD", commit], check=True)
        subprocess.run(["git", "read-tree", "HEAD"], check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise SystemExit("Existing worker has a different commit; use a new directory")
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).split(b"\0")
    missing = []
    present = []
    required = ("mixres/", "scripts/", "configs/", "env/", "frozen/s23/", "runs/views/s23/")
    for raw in tracked:
        if not raw:
            continue
        name = raw.decode()
        if Path(name).is_file():
            present.append(name)
        if not Path(name).exists():
            if name.startswith(required):
                raise SystemExit(f"Missing tracked execution file: {name}")
            # Exclude only absent non-execution assets such as historical reports.
            missing.append(raw)
    # Metadata packs omit blobs. Populate provided source blobs so git-show can
    # verify committed predictions/configs without fetching private GitHub data.
    store_source_blobs(present)
    if missing:
        subprocess.run(["git", "update-index", "--skip-worktree", "-z", "--stdin"],
                       input=b"\0".join(missing) + b"\0", check=True)
    print(f"Source verified at {commit}. Non-execution assets omitted: {len(missing)}")
    if receipt["source_state"]["dirty_tree"]:
        print("Bundle contains uncommitted development code. Frozen execution remains forbidden.")
