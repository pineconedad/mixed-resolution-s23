#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
source env.sh
python scripts/04_run.py "$@"
python scripts/05_aggregate.py
python scripts/06_figures.py
python scripts/export_session.py
