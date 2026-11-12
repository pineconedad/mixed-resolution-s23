"""Run on an owned, restarted Linux pod before rebuilding an interrupted worker.

Retain the old state/logs unchanged. Refuse to reset while a project worker runs.
This script does not delete observations or claim that validation succeeded.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def recover(root, session_id):
    root = Path(root).resolve()
    if root != Path("/workspace/s23") or not session_id.isalnum() or not Path("/proc").is_dir():
        raise RuntimeError("Recovery requires the explicit workflow workspace on Linux")
    entrypoints = {"run_session.py", "01_sanity.py", "04_run.py", "02_dev_sweep.py",
                   "download_checkpoint.py", "stage_development.py", "prepare_model_sources.py"}
    for process in Path("/proc").glob("[0-9]*"):
        if process.name == str(os.getpid()):
            continue
        try:
            arguments = [p.decode(errors="replace") for p in (process / "cmdline").read_bytes().split(b"\0") if p]
            cwd = Path(os.readlink(process / "cwd"))
        except OSError:
            continue
        if any(Path(argument).name in entrypoints for argument in arguments):
            raise RuntimeError("A project worker is still running; recovery refused")
        if "pip" in arguments and cwd.is_relative_to(root):
            raise RuntimeError("Project package installation is still running; recovery refused")
    control = root / "runs/control" / session_id
    if not control.is_dir():
        raise RuntimeError("No interrupted controller directory exists")
    state = json.loads((control / "state.json").read_text())
    if state.get("session_id") != session_id:
        raise RuntimeError("Controller identity mismatch")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / "runs/_superseded/control" / (session_id + "-" + stamp)
    target.parent.mkdir(parents=True, exist_ok=True)
    control.rename(target)
    (target / "recovery.json").write_text(json.dumps({
        "session_id": session_id, "recovered_utc": stamp,
        "prior_phase": state.get("phase"), "prior_exit_code": state.get("exit_code"),
        "workers_present": False, "classification": "interrupted_setup_not_a_model_result",
        "original_state_and_log_preserved": True}, indent=2) + "\n")
    control.mkdir(parents=True)
    (control / "state.json").write_text(json.dumps({
        "session_id": session_id, "quiescent": True,
        "phase": "recovered_after_restart", "workers_present": False,
        "original_control": str(target.relative_to(root)),
        "model_validation_claimed": False}, indent=2) + "\n")
    print("Interrupted controller state and logs archived without changing observations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/s23")
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    recover(args.root, args.session_id)
