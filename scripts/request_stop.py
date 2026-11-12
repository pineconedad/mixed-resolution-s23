"""Ask this session's worker to stop between atomic scene evaluations."""
import argparse
import sys
import time
import os
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.io import read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--wait", type=float, default=120)
    args = parser.parse_args()
    if not args.session_id.isalnum():
        raise SystemExit("Invalid session id")
    root = Path("runs/control") / args.session_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "stop").touch()
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        state = read_json(root / "state.json") if (root / "state.json").exists() else {}
        if state.get("session_id") == args.session_id and state.get("quiescent") is True:
            print("Worker quiescent")
            break
        if time.monotonic() >= deadline - 30 and state.get("session_id") == args.session_id:
            pid = state.get("pid")
            if isinstance(pid, int) and pid > 1:
                try:
                    argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    if (args.session_id.encode() in argv
                            and any(arg.endswith(b"run_session.py") for arg in argv)):
                        os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, FileNotFoundError):
                    pass
        time.sleep(2)
    else:
        raise SystemExit("Worker is not confirmed quiescent; retain persistent data")
