"""User-run hidden credential entry. Never include the key in a command or chat."""
import argparse
import getpass
import os
import subprocess
import tempfile
from pathlib import Path


def store_key(root, key, *, replace=False):
    root = Path(root).resolve()
    if not key or any(character.isspace() for character in key) or len(key) > 4096:
        raise ValueError("Expected one nonempty API key without whitespace")
    target = root / ".env.runpod"
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "--quiet", str(target)])
    if ignored.returncode != 0:
        raise RuntimeError("Credential path is not ignored by Git; refusing to write")
    tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", str(target)], capture_output=True)
    if tracked.returncode == 0:
        raise RuntimeError("Credential path is tracked by Git; refusing to write")
    if target.is_symlink() or target.exists() and not replace:
        raise FileExistsError("Credential already exists; use --replace explicitly to rotate it")
    descriptor, name = tempfile.mkstemp(prefix=".env.runpod.", dir=root)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write("RUNPOD_API_KEY=" + key + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not replace:
            # Link creates the final name atomically without replacing a concurrent write.
            os.link(name, target)
        else:
            os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    key = getpass.getpass("Runpod API key, hidden input: ")
    store_key(Path(__file__).resolve().parents[1], key.strip(), replace=args.replace)
    print("Credential stored locally with mode 0600 in Git-ignored .env.runpod.")
    print("The key was not printed. The agent can now load it without restarting OpenCode.")
