#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root, after transferring checkpoint and sampled data.
test -f configs/quick.yaml
test -f checkpoints/pi3/model.safetensors
test -f data/7scenes/chess/receipt.json
git submodule update --init --recursive
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10"'
python -m pip install -r env/requirements.txt
python -m pip install -e . --no-deps
source env.sh
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(), torch.version.cuda)'
python -m pytest -q
python scripts/preflight.py
python scripts/record_environment.py
# Gate only. Do not start the paid experiment automatically.
python scripts/01_sanity.py
