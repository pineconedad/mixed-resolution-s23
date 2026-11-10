"""Produce a checksummed session archive for transfer before pod termination."""
import argparse
import os
import hashlib
import io
import tempfile
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixres.io import atomic_json, file_hash, git_state, read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--session-id")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    paths = sorted(p for directory in ("runs", "frozen", "env") if Path(directory).exists()
                   for p in Path(directory).rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    state_path = Path("runs/control") / str(args.session_id) / "state.json"
    state = read_json(state_path) if args.session_id and state_path.exists() else {}
    receipt = {"created_utc": stamp, "pod_id": os.environ.get("RUNPOD_POD_ID"),
               "session_id": args.session_id,
               "job_quiescent": state.get("session_id") == args.session_id and state.get("quiescent") is True,
               **git_state(), "files": {}}
    receipt_path = destination / f"session-{stamp}.json"
    archive = destination / f"session-{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in paths:
            # Hash and archive the same immutable bytes, even while logs grow or
            # another process atomically replaces result.json during a live backup.
            with tempfile.TemporaryFile() as snapshot:
                digest = hashlib.sha256()
                try:
                    with path.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            snapshot.write(block)
                            digest.update(block)
                except FileNotFoundError:
                    continue
                info = tarfile.TarInfo(str(path))
                info.size = snapshot.tell()
                snapshot.seek(0)
                tar.addfile(info, snapshot)
                receipt["files"][str(path)] = digest.hexdigest()
        final_state = read_json(state_path) if args.session_id and state_path.exists() else {}
        receipt["job_quiescent"] = receipt["job_quiescent"] and final_state == state
        atomic_json(receipt_path, receipt)
        raw = receipt_path.read_bytes()
        info = tarfile.TarInfo("session-manifest.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    print(f"{archive} sha256={file_hash(archive)}")
    print("Download and verify this archive locally before stopping or terminating any pod.")
