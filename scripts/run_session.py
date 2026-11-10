"""Remote worker with persistent logs and a truthful quiescence marker."""
import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.io import atomic_json


def session(session_id, config, smoke_only=False):
    if not session_id.isalnum():
        raise ValueError("Invalid session id")
    root = Path(__file__).resolve().parents[1]
    control = root / "runs/control" / session_id
    control.mkdir(parents=True, exist_ok=True)
    state_path = control / "state.json"
    if state_path.exists():
        raise FileExistsError("Refusing to reuse a session id")
    state = {"session_id": session_id, "pid": os.getpid(), "quiescent": False, "phase": "starting"}
    atomic_json(state_path, state)
    child = None
    exit_code = 1
    python = str(root / ".venv/bin/python")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0",
               XFORMERS_DISABLED="1", OMP_NUM_THREADS="4", HF_HOME=str(root / "cache/huggingface"),
               PYTHONPATH=str(root / "third_party/Pi3") + os.pathsep + str(root))

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Session interrupted")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    steps = [] if smoke_only else [
        ("venv", [sys.executable, "-m", "venv", ".venv"]),
        ("dependencies", [python, "-m", "pip", "install", "--no-cache-dir", "-q", "-r", "env/requirements.txt"]),
        ("install", [python, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"]),
        ("dependency_check", [python, "-m", "pip", "check"]),
        ("checkpoint", [python, "scripts/download_checkpoint.py", "--config", config]),
        ("tests", [python, "-m", "pytest", "-q"]),
        ("environment", [python, "scripts/record_environment.py"]),
        ("identity", [python, "scripts/01_sanity.py", "--config", config]),
        ("experiments", [python, "scripts/04_run.py", "--config", config,
                          "--stop-file", str(control / "stop"), "--max-seconds", "2400"]),
        ("aggregate", [python, "scripts/05_aggregate.py", "--config", config]),
        ("figures", [python, "scripts/06_figures.py", "--config", config]),
    ]
    try:
        with (control / "console.log").open("w") as log:
            for phase, command in steps:
                if (control / "stop").exists():
                    state["stopped_by_request"] = True
                    break
                state["phase"] = phase
                atomic_json(state_path, state)
                log.write(f"PHASE {phase}\n")
                log.flush()
                child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                result = child.wait()
                if result != 0:
                    raise RuntimeError(f"Phase {phase} exited with code {result}")
            exit_code = 0
    except BaseException as error:
        state["error"] = type(error).__name__ + ": " + str(error)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state.update(quiescent=True, exit_code=exit_code, phase="finished")
        atomic_json(state_path, state)
        signal.signal(signal.SIGTERM, previous_sigterm)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--config", default="configs/diagnostic.yaml")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(session(args.session_id, args.config, args.smoke_only))
