"""Explicit project credential loading, without reading client credential stores."""
import os
import stat
from pathlib import Path


def runpod_key(root=None):
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    path = root / ".env.runpod"
    if not path.exists():
        raise RuntimeError("RUNPOD_API_KEY is not available; run python3 scripts/set_runpod_key.py locally")
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_mode & 0o077:
        raise RuntimeError(".env.runpod must be a regular, nonsymlink file with mode 0600")
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if len(lines) != 1 or not lines[0].startswith("RUNPOD_API_KEY="):
        raise RuntimeError("Invalid .env.runpod format; expected only RUNPOD_API_KEY=value")
    key = lines[0].partition("=")[2]
    if not key or any(character.isspace() for character in key):
        raise RuntimeError("Invalid API key entry")
    return key
