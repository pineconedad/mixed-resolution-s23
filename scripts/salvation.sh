#!/usr/bin/env bash
# Salvation kill tests. User-operated, local GPU only: never touches cloud resources.
# Usage: bash scripts/salvation.sh a|b pi3|vggt [--max-seconds N]
set -euo pipefail
TRACK="${1:-}"; MODEL="${2:-}"
case "$TRACK" in a|b) ;; *) echo "Usage: bash scripts/salvation.sh a|b pi3|vggt [--max-seconds N]"; exit 2 ;; esac
case "$MODEL" in pi3|vggt) ;; *) echo "Usage: bash scripts/salvation.sh a|b pi3|vggt [--max-seconds N]"; exit 2 ;; esac
shift 2
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD/third_party/Pi3:$PWD/third_party/vggt:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 XFORMERS_DISABLED=1
export HF_HOME="${HF_HOME:-$PWD/cache/huggingface}"
CONFIG="configs/salvation-$TRACK-$MODEL.yaml"
SANITY="runs/sanity/$MODEL-salvation_$TRACK.json"
# Checkpoint-backed fp32 identity gate. Reused only while code, config, weights and GPU are unchanged;
# 04_run.py refuses to start if it does not match.
if [[ ! -f "$SANITY" ]] || [[ -n "$(git status --porcelain)" ]] \
    || ! grep -q "\"git_commit\": \"$(git rev-parse HEAD)\"" "$SANITY"; then
    python scripts/01_sanity.py --config "$CONFIG"
fi
python scripts/04_run.py --config "$CONFIG" "$@"
python scripts/salvation_report.py --track "$TRACK" --model "$MODEL"
