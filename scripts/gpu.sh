#!/usr/bin/env bash
# User-operated only. This script never provisions, stops or deletes cloud resources.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="$ROOT/third_party/Pi3:$ROOT/third_party/vggt:$ROOT/third_party/Depth-Anything-3/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 XFORMERS_DISABLED=1 OMP_NUM_THREADS=4
ENV_ROOT="${MIXRES_ENV_ROOT:-/opt/mixres-envs}"
export HF_HOME="${HF_HOME:-$ENV_ROOT/huggingface}"
ACTION="${1:-help}"
MODEL="${2:-}"
if [[ "$ACTION" == "help" || -z "$MODEL" ]]; then
    printf '%s\n' 'Usage: bash scripts/gpu.sh setup|inputs|sanity|dev|main|extra MODEL [arguments]'
    exit 0
fi
case "$MODEL" in pi3|da3|vggt) ;; *) exit 2 ;; esac
shift 2
PYTHON="$ENV_ROOT/$MODEL/bin/python"
if [[ "$ACTION" == "setup" ]]; then
    [[ "$(uname -s)" == Linux ]] || { printf '%s\n' 'GPU setup requires Linux.'; exit 1; }
    mkdir -p "$ENV_ROOT" /opt/mixres-checkpoints
    if ! command -v uv >/dev/null; then
        python3 -m venv "$ENV_ROOT/tools"
        "$ENV_ROOT/tools/bin/python" -m pip install uv
        export PATH="$ENV_ROOT/tools/bin:$PATH"
    fi
    uv python install 3.10
    if [[ ! -f "$ENV_ROOT/$MODEL/pyvenv.cfg" ]]; then
        uv venv --python 3.10 --seed "$ENV_ROOT/$MODEL"
    fi
    REQUIREMENTS="env/requirements.txt"
    [[ "$MODEL" != da3 ]] || REQUIREMENTS="env/requirements-da3.txt"
    [[ "$MODEL" != vggt ]] || REQUIREMENTS="env/requirements-vggt.txt"
    uv pip install --python "$PYTHON" -r "$REQUIREMENTS"
    uv pip install --python "$PYTHON" -e . --no-deps
    uv pip check --python "$PYTHON"
    "$PYTHON" scripts/prepare_model_sources.py --models "$MODEL"
    mkdir -p checkpoints
    if [[ ! -e "checkpoints/$MODEL" ]]; then
        mkdir -p "/opt/mixres-checkpoints/$MODEL"
        ln -s "/opt/mixres-checkpoints/$MODEL" "checkpoints/$MODEL"
    fi
    "$PYTHON" scripts/download_checkpoint.py --model "$MODEL"
    "$PYTHON" scripts/record_environment.py --output "env/$MODEL"
    printf '%s\n' "Setup complete for $MODEL. No tests or inference were run."
    exit 0
fi
[[ -x "$PYTHON" ]] || { printf '%s\n' "Run setup $MODEL first."; exit 1; }
case "$ACTION" in
    inputs) exec "$PYTHON" scripts/stage_development.py --range-only "$@" ;;
    sanity) exec "$PYTHON" scripts/01_sanity.py --model "$MODEL" "$@" ;;
    dev)
        [[ "$MODEL" == pi3 ]] || { printf '%s\n' 'Development tuning is Pi3-only.'; exit 2; }
        exec "$PYTHON" scripts/02_dev_sweep.py "$@" ;;
    main|extra) exec "$PYTHON" scripts/launch_study.py --phase "$ACTION" --model "$MODEL" --python "$PYTHON" --execute "$@" ;;
    *) printf '%s\n' "Unknown action: $ACTION"; exit 2 ;;
esac
