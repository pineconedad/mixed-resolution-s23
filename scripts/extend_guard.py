"""Transfer one active session to a verified replacement deadline guard."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.io import atomic_json, read_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipt = read_json(args.receipt)
    old_ready = read_json(args.receipt.parent / "ready.json")
    if args.cap <= receipt["session_cap_usd"] or args.cap > receipt["balance_at_create"] - 3:
        raise SystemExit("Extension must increase the session cap while preserving the reserve")
    if args.destination.exists():
        raise SystemExit("Refusing to reuse a guard destination")
    args.destination.mkdir(parents=True)
    updated = {**receipt, "session_cap_usd": args.cap, "heartbeat": None}
    if args.max_seconds is not None:
        if args.max_seconds < receipt["max_seconds"]:
            raise SystemExit("Extension cannot shorten the active session deadline")
        updated["max_seconds"] = args.max_seconds
    target = args.destination.resolve() / "receipt.json"
    atomic_json(target, updated)
    command = ["/usr/bin/caffeinate", "-i", str(root / ".venv-ops/bin/python"),
               str(root / "scripts/watch_pod.py"), "--receipt", str(target),
               "--destination", str(args.destination.resolve()), "--interval", "60"]
    with (args.destination / "watchdog-console.log").open("w") as log:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 180
    while not (args.destination / "ready.json").exists():
        if process.poll() is not None or time.monotonic() > deadline:
            raise SystemExit("Replacement guard did not verify a backup; old guard remains authoritative")
        time.sleep(2)
    old_pid = old_ready["pid"]
    old_command = subprocess.run(["ps", "-p", str(old_pid), "-o", "args="], capture_output=True, text=True)
    if old_command.returncode == 0:
        if "watch_pod.py" not in old_command.stdout or str(args.receipt.resolve()) not in old_command.stdout:
            raise SystemExit("Original guard PID no longer matches; refusing to signal it")
        os.kill(old_pid, signal.SIGTERM)
    atomic_json(args.receipt.parent / "guard-handoff.json", {
        "new_receipt": str(target), "new_guard": read_json(args.destination / "ready.json"),
        "session_cap_usd": args.cap, "worker_left_running": True})
    print(f"Verified replacement guard active: {target}")
