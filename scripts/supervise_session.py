"""Local supervisor; an independently launched watchdog enforces the deadline."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.io import atomic_json, read_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipt = read_json(args.receipt)
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    heartbeat = Path(receipt["heartbeat"])
    done = Path(receipt["done_marker"])
    if done.exists() or (destination / "ready.json").exists():
        raise SystemExit("Refusing to reuse a supervisor destination")
    heartbeat.touch()
    command = [str(root / ".venv-ops/bin/python"), str(root / "scripts/watch_pod.py"),
               "--receipt", str(Path(args.receipt).resolve()), "--destination", str(destination), "--interval", "30"]
    with (destination / "watchdog-console.log").open("w") as output:
        watchdog = subprocess.Popen(["/usr/bin/caffeinate", "-i", *command], cwd=root,
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    deadline = time.monotonic() + 120
    while not (destination / "ready.json").exists():
        heartbeat.touch()
        if watchdog.poll() is not None or time.monotonic() >= deadline:
            raise SystemExit("Watchdog did not become ready; do not start paid experiments. Inspect the owned pod immediately.")
        time.sleep(2)
    from scripts.watch_pod import ssh_base
    import shlex
    remote = ("env -C " + shlex.quote(receipt["remote_root"])
              + " python scripts/run_session.py --session-id " + shlex.quote(receipt["session_id"])
              + (" --smoke-only" if args.smoke_only else " --config configs/diagnostic.yaml"))
    with (destination / "ssh-console.log").open("w") as output:
        process = subprocess.Popen(ssh_base(receipt) + [remote], stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT)
        while process.poll() is None:
            heartbeat.touch()
            if watchdog.poll() is not None:
                raise SystemExit("Watchdog exited before worker; inspect the owned pod immediately")
            time.sleep(5)
    atomic_json(done, {"ssh_exit_code": process.returncode})
    # A lost SSH connection is not proof of remote completion. The watchdog asks
    # the remote controller to quiesce before its final backup and deletion.
    while watchdog.poll() is None:
        heartbeat.touch()
        time.sleep(3)
    if not (destination / "final.json").exists():
        raise SystemExit("Watchdog exited without a final shutdown receipt; inspect the owned pod immediately")
    final = read_json(destination / "final.json")
    print(json.dumps(final, indent=2))
    if not final.get("terminated"):
        raise SystemExit("Compute stopped but persistent data remains until backup is recovered")
