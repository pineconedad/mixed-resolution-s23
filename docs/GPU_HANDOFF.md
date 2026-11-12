# GPU handoff

The assistant is build-only. You create, connect to, stop and delete GPU resources. None of the commands below provisions infrastructure or starts a cloud watchdog.

## Current evidence

The final local archive from the last automated session is restored under `reports/sessions/20261111T200807990545Z`. All 110 member checksums verified. Its watchdog receipt reports pod `hassuectufut86` terminated. This is a historical receipt, not a fresh cloud inventory read.

DA3 passed checkpoint-backed fp32 native identity on two full-resolution scenes in that archived version. The archive contains no completed development sweep. The current build changes high-head outputs and cost accounting, so run the identity gate once for this final code before new experiments. No broad unit suite is part of GPU setup.

Old pod `yvcfcbk52vr2x1` was last observed stopped with a retained 40 GB workspace. Check your console manually. Recover its files before deleting it if needed; it may still incur storage charges. The assistant will not inspect or change it.

## Hardware and storage

Use Linux and an Ampere, Ada or Hopper NVIDIA GPU. Prefer 24 GB or more VRAM for development; 48 GB or more gives headroom for ceilings. The pinned PyTorch environments are not intended for RTX 50-series/Blackwell GPUs. Do not silently change framework versions or resolution to make a run fit.

Use at least 50 GB container disk. Environments, wheel cache and model weights go under `/opt`, not the slower persistent network filesystem. Keep code, selected images and results under `/workspace/s23`. Full frozen predictions can be large; download completed cells regularly and size persistent storage for the planned batch.

Check your balance in the console and retain at least $3. Set a personal billing/time reminder before running commands. `--max-seconds` stops a batch, not the pod. Closing SSH does not reliably stop compute. Stop or terminate the pod yourself after verified download.

## Prepare locally

Run from the repository root on the Mac. Review and commit the intended source, manifests and reports before making a frozen bundle. No credentials or raw datasets should be staged.

```bash
git status --short
git add .gitignore AGENTS.md STUDY_CHECKLIST.md configs docs env mixres scripts tests runs/views reports pyproject.toml
git diff --cached --check
git commit -m "Build S23 execution pipeline and manual GPU handoff"
.venv/bin/python scripts/pack_development.py --source-only --destination artifacts/gpu-handoff
```

The bundle includes code, view manifests and the local ScanNet++ permission receipt, not your website password, download token, Runpod key, checkpoints or full datasets. A development bundle may be dirty, but frozen execution requires committed unchanged code/configs/predictions.

Set these values from your pod's SSH connection details, then transfer:

```bash
export GPU_HOST='root@YOUR_HOST'
export GPU_PORT='YOUR_PORT'
scp -P "$GPU_PORT" artifacts/gpu-handoff/source.tar.gz artifacts/gpu-handoff/metadata.pack artifacts/gpu-handoff/receipt.json "$GPU_HOST:/workspace/"
ssh -p "$GPU_PORT" "$GPU_HOST"
```

## Set up on GPU

Inside the root SSH terminal:

```bash
mkdir -p /workspace/s23
tar --no-same-owner -xzf /workspace/source.tar.gz -C /workspace/s23
cd /workspace/s23
python3 scripts/initialize_worker.py --source /workspace/source.tar.gz --metadata /workspace/metadata.pack --receipt /workspace/receipt.json
bash scripts/gpu.sh setup da3
bash scripts/gpu.sh setup pi3
bash scripts/gpu.sh inputs pi3
```

Initialization verifies both bundle hashes and sets up a sparse Git worktree. Missing historical reports are excluded from the worker checkout, but missing tracked execution files are an error. Modified/uncommitted source is not disguised as clean.

Setup is idempotent and does not run tests or inference. Selected images are fetched from pinned archives and individually checksum-verified. The 35-scene development manifests contain 420 selected views. Do not regenerate them on the worker.

If `/opt` is unavailable, use a writable container-local location through `MIXRES_ENV_ROOT`. Do not put the virtual environment on `/workspace` merely for convenience. A restarted pod may lose `/opt`; setup must then recreate its environment.

## Identity and development

Run each changed model's required native fp32 gate once. Pi3 and DA3 use separate pinned environments:

```bash
bash scripts/gpu.sh sanity da3
bash scripts/gpu.sh sanity pi3
set -o pipefail
bash scripts/gpu.sh dev pi3 --max-seconds 3600 2>&1 | tee runs/dev-console.log
```

The sweep loads Pi3 once and evaluates the 24 configurations over ETH3D, ScanNet++ and 7-Scenes. Images are cached in bounded CPU memory; timing remains one warm-up plus three synchronized model-forward passes. No held-out data is used.

Rerun the same development command to resume matching completed scenes. A partial run is not a valid freeze input. Inspect `runs/dev/progress.json`, `runs/dev/ranking.json` and scene errors. Do not delete failures or lower resolution silently. A failed scene needs diagnosis before it can be treated as resolved.

## Export and download

After the foreground command returns, export on the pod:

```bash
/opt/mixres-envs/pi3/bin/python scripts/export_session.py
```

The command prints an archive path and SHA-256. In a separate local terminal, use those exact values:

```bash
export REMOTE_ARCHIVE='session-TIMESTAMP.tar.gz'
export ARCHIVE_SHA256='PRINTED_SHA256'
mkdir -p artifacts/incoming
scp -P "$GPU_PORT" "$GPU_HOST:/workspace/s23/artifacts/$REMOTE_ARCHIVE" artifacts/incoming/
.venv/bin/python scripts/verify_session.py "artifacts/incoming/$REMOTE_ARCHIVE" --sha256 "$ARCHIVE_SHA256" --destination reports/sessions/manual-development
```

Use a new destination for a different session; verification refuses to overwrite different existing data. A direct manual run has no controller quiescence marker. Ensure all foreground commands have ended before export and shutdown. Verify the archive locally before using the console's terminate action.

## Freeze locally

Only after all 78 development cells are complete, restore `runs/dev` from the verified archive into the working study tree without overwriting different observations. Preserve the archive independently. Do not copy archived source over newer local code.

Then run locally:

```bash
.venv/bin/python scripts/03_freeze.py --ranking runs/dev/ranking.json
git add frozen/s23
git commit -m "Freeze S23 method and numeric predictions"
```

Freeze refuses incomplete sweeps, missing datasets, nonfinite rankings or existing frozen outputs. Ties use lower cost and then stable method ID. Historical T9 remains unavailable; this study uses the documented fresh baseline, not invented S7 settings. A0 is matched by nominal budget and patch rounding; report actual tokens, not an exact-token claim against A0.

## Prepare held-out data

Only after predictions are committed:

```bash
.venv/bin/python scripts/download_benchmark.py --datasets dtu hiroom
```

Extract the verified ZIPs under the native benchmark root. Do not evaluate held-out results for tuning.

```bash
unzip -q data/benchmark_archives/dtu.zip -d third_party/Depth-Anything-3/workspace/benchmark_dataset
unzip -q data/benchmark_archives/hiroom.zip -d third_party/Depth-Anything-3/workspace/benchmark_dataset
```

```bash
.venv-da3/bin/python scripts/00_make_views.py --datasets dtu hiroom --seeds 1000 2000 3000
.venv/bin/python scripts/03_freeze.py --bind-heldout-views
git add frozen/s23/heldout-views.json runs/views/s23/dtu runs/views/s23/hiroom
git commit -m "Bind held-out view manifests to frozen S23 predictions"
```

The appendix binds held-out data hashes without changing method configs. Rebuild and transfer the committed source bundle into a fresh worker directory. Stage the frozen images on the worker with:

```bash
bash scripts/gpu.sh inputs pi3 --datasets eth3d scannetpp 7scenes dtu hiroom --seeds 1000 2000 3000
```

This stages only images named by saved manifests and refuses held-out access without committed predictions. Do not remove the gate to bypass missing inputs.

## Frozen execution

On a worker containing the committed frozen files and referenced data:

```bash
bash scripts/gpu.sh setup vggt
bash scripts/gpu.sh sanity vggt
bash scripts/gpu.sh main pi3 --max-seconds 3600
bash scripts/gpu.sh main da3 --max-seconds 3600
bash scripts/gpu.sh main vggt --max-seconds 3600
```

Each model uses its own environment and is loaded once per command. Main runs use seed 1000 with probes. Dirty code, uncommitted predictions or mismatched frozen/view hashes are refused. Download partial output regularly. Ceilings that fail memory must remain explicit failures, not silently reduced workloads.

Extra seeds use the reduced 4k matrix with probes disabled:

```bash
bash scripts/gpu.sh extra pi3 --max-seconds 3600
bash scripts/gpu.sh extra da3 --max-seconds 3600
bash scripts/gpu.sh extra vggt --max-seconds 3600
```

## Reconstruction and reporting

After verified predictions are restored locally, run the native reconstruction bridge per cell in a CPU benchmark environment. For example:

```bash
.venv-da3/bin/python scripts/07_reconstruction.py runs/s23/pi3/eth3d/4096/A4/seed1000/result.json --data-root third_party/Depth-Anything-3/workspace/benchmark_dataset/eth3d
.venv-da3/bin/python scripts/07_reconstruction.py runs/s23/pi3/scannetpp/4096/A4/seed1000/result.json --data-root third_party/Depth-Anything-3/workspace/benchmark_dataset/scannetpp --permission-receipt artifacts/permissions/scannetpp.json
.venv/bin/python scripts/05_aggregate.py --study s23
.venv/bin/python scripts/06_figures.py --study s23
```

The bridge writes separate `recon.json` and `recon.log`; it never replaces original inference outputs or substitutes GT calibration for predictions. Actual-data reconstruction and full frozen/probe execution still need validation. Draft figures are not a claim that paper evidence is complete.
