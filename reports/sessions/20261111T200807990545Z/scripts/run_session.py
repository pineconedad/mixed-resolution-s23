"""Remote worker with persistent logs and a truthful quiescence marker."""
import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.io import atomic_json


def session(session_id, config, smoke_only=False, model="pi3", validation_only=False, skip_unit_tests=False, env_path=".venv", task=None):
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
    environment = Path(env_path) if Path(env_path).is_absolute() else root / env_path
    python = str(environment / "bin/python")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0",
               XFORMERS_DISABLED="1", OMP_NUM_THREADS="4", HF_HOME=str(root / "cache/huggingface"),
               PYTHONPATH=os.pathsep.join(str(root / p) for p in ("third_party/Pi3", "third_party/vggt", "third_party/Depth-Anything-3/src", ".")))
    requirements = {"pi3": "env/requirements.txt", "vggt": "env/requirements-vggt.txt", "da3": "env/requirements-da3.txt"}[model]

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Session interrupted")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    steps = [] if smoke_only else [
        ("venv", [sys.executable, "-m", "venv", str(environment)]),
        ("dependencies", [python, "-m", "pip", "install", "--no-cache-dir", "-q", "-r", requirements]),
        ("install", [python, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"]),
        ("dependency_check", [python, "-m", "pip", "check"]),
        ("checkpoint", [python, "scripts/download_checkpoint.py", "--config", config]),
        ("tests", [python, "-m", "pytest", "-q", *(["tests_vggt"] if model == "vggt" else [])]),
        ("environment", [python, "scripts/record_environment.py"]),
        ("identity", [python, "scripts/01_sanity.py", "--config", config]),
    ]
    if (environment / "pyvenv.cfg").is_file():
        steps = [(phase, command) for phase, command in steps if phase != "venv"]
    if skip_unit_tests:
        steps = [(phase, command) for phase, command in steps if phase != "tests"]
        state["unit_tests_skipped_by_user_request"] = True
    if not smoke_only and not validation_only:
        steps.extend([
        ("experiments", [python, "scripts/04_run.py", "--config", config,
                          "--stop-file", str(control / "stop"), "--max-seconds", "2400"]),
        ("aggregate", [python, "scripts/05_aggregate.py", "--config", config]),
        ("figures", [python, "scripts/06_figures.py", "--config", config]),
        ])
    if task:
        steps = [
            ("model_sources", [sys.executable, "scripts/prepare_model_sources.py", "--models",
                               *(["da3", "pi3"] if task == "development" else [model])]),
            ("python", ["uv", "python", "install", "3.10"]),
            ("venv", ["uv", "venv", "--python", "3.10", "--seed", str(environment)]),
            ("dependencies", ["uv", "pip", "install", "--python", python, "-r", requirements]),
            ("install", ["uv", "pip", "install", "--python", python, "-e", ".", "--no-deps"]),
            ("dependency_check", ["uv", "pip", "check", "--python", python]),
            ("checkpoint", [python, "scripts/download_checkpoint.py", "--model", model]),
            ("stage_inputs", [python, "scripts/stage_development.py", "--discard-archives-after-staging"]),
            ("environment", [python, "scripts/record_environment.py"]),
            ("identity", [python, "scripts/01_sanity.py", "--model", model]),
        ]
        if task == "development":
            if model != "pi3":
                raise ValueError("Development tuning is Pi3-only")
            da3_environment = root / ".venv-da3"
            da3_python = str(da3_environment / "bin/python")
            steps[2:2] = [
                ("da3_venv", ["uv", "venv", "--python", "3.10", "--seed", str(da3_environment)]),
                ("da3_dependencies", ["uv", "pip", "install", "--python", da3_python, "-r", "env/requirements-da3.txt"]),
                ("da3_install", ["uv", "pip", "install", "--python", da3_python, "-e", ".", "--no-deps"]),
                ("da3_dependency_check", ["uv", "pip", "check", "--python", da3_python]),
                ("da3_checkpoint", [da3_python, "scripts/download_checkpoint.py", "--model", "da3"]),
                ("stage_inputs", [da3_python, "scripts/stage_development.py", "--discard-archives-after-staging"]),
                ("da3_environment", [da3_python, "scripts/record_environment.py", "--output", "env/da3"]),
                ("da3_identity", [da3_python, "scripts/01_sanity.py", "--model", "da3"]),
            ]
            steps.append(("sweep", [python, "scripts/02_dev_sweep.py", "--datasets", "eth3d", "scannetpp", "7scenes", "--stop-file", str(control / "stop")]))
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
    parser.add_argument("--model", choices=["pi3", "vggt", "da3"], default="pi3")
    parser.add_argument("--task", choices=["development", "identity"])
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--env-path", default=".venv")
    args = parser.parse_args()
    raise SystemExit(session(args.session_id, args.config, args.smoke_only, args.model, args.validation_only, args.skip_unit_tests, args.env_path, args.task))
