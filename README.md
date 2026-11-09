# Mixed-resolution S23

Private research implementation of the two study plans in this folder. Start with the independent original-Pi3 quick test. All seed-42 metrics are exploratory and excluded from paper results.

## Local preparation

```bash
git submodule update --init --recursive
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
pip install -e . --no-deps
source env.sh
python -m pytest -q
python scripts/download_checkpoint.py
python scripts/prepare_quick.py
python scripts/preflight.py
python scripts/contact_sheet.py
```

The checkpoint and sampled data are already downloaded on the preparation machine. Git contains code, configs, source plans and the view manifest, not weights or dataset images. The checkpoint is 3.835 GB. Transfer only sampled scene folders and checkpoint files, not data/7scenes/_archives or a Mac virtual environment.

## First GPU session

No resource is provisioned by this repository. Choose a CUDA GPU with bf16 support, initially target at least 24 GB VRAM, then quote actual live compute and disk prices before deployment. A smaller card may fit, but fp32 T1 and full-resolution ceilings must be measured rather than assumed. A prebuilt Python 3.10/PyTorch 2.5.1 CUDA environment avoids driver installation and most setup time. The host owns the NVIDIA driver; do not install or reboot drivers inside a pod.

After transferring the repository, sampled data and checkpoint:

```bash
bash scripts/bootstrap_gpu.sh
# Only after the fp32 identity gate passes:
bash launch.sh --max-seconds 3600
```

The time limit is checked between arm evaluations. It is not a hard billing cap and does not stop or terminate a pod. Leave budget for the longest in-flight evaluation and result transfer. The default run uses 1 warm-up and 3 timed passes, and does not silently change precision/resolution if it fails.

All paid inference requires CUDA. CPU unit tests are separate. The full checkpoint has not yet passed T1 on CUDA, and no experiment results exist.

## Resume and preserve results

The runner writes each scene immediately and atomically, then refreshes arm result.json. It skips only successful scenes with identical code, config, data, weights, GPU and metric provenance. Partial and failed outputs remain visible. Changed provenance requires archiving the old run first, never overwriting it. The aggregation script uses a common set of successful paired scenes across every requested arm and reports missing/failed cells.

```bash
source env.sh
python scripts/05_aggregate.py
python scripts/06_figures.py
python scripts/export_session.py
```

Download the archive, verify its SHA-256, and verify its member checksums before terminating the pod. Keep an independent local copy. Checkpoints and sampled images already have local originals. A terminated pod loses its local disks; a stopped pod can still incur storage charges. No automatic cloud deletion is performed.

## Status

See env/model_checks.md and docs/implementation.md for exactly what is tested, what is missing, and the full-study prerequisites. In particular, T9 and the unknown S7/DA3 choices are not fabricated. Full-study and frozen runs remain disabled until those scientific prerequisites are resolved.
