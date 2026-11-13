# Mixed-resolution S23

## Start here: resolution ensembling (offline, no GPU)

Tracks A to D are done; the findings are collected in `docs/salvation_findings.md`. This test asks whether fusing poses predicted at two uniform input sizes beats one run at a larger size. It reads the poses the overnight run already stored, so it needs that PC but no GPU, and takes a few minutes.

```powershell
git pull
.venv\Scripts\Activate.ps1
python -m pytest -q tests/test_ensemble.py
python scripts/salvation_ensemble.py --models pi3 vggt --push
```

Read `reports/salvation/ensemble.txt`. The verdict is `Gate E`: the pair fixed in advance (`A0_5k+A0_7k`) must beat the best single size that costs no more compute, with the bootstrap interval above zero. All other rows are exploratory.

## Track D, the aspect-ratio check

Track C failed on both models (`reports/salvation/night-*.txt`). Those runs showed Pi3 doing best at sizes that are exactly 4:3, while the study's sizes are slightly squeezed, and by different amounts for sharp (1.304) and small (1.25) views. Track D reruns A1, A2, A3, A8 and A0 on the same frames with exactly 4:3 sizes. It needs the overnight run's downloaded draws; about 30 minutes for Pi3.

```powershell
git pull
.venv\Scripts\Activate.ps1
python scripts/salvation_aspect.py --models pi3 vggt --push
```

Read `runs/salvation_night/reports/aspect.txt` (pushed as `reports/salvation/aspect.txt`). The key line is `Mix damage (A8 - A2)`: if it is much smaller with exact 4:3 sizes, part of the mixed-resolution failure is an aspect mismatch introduced by resizing.

## The unattended overnight run (track C)

Tracks A and B both failed on Pi3 (reports under `runs/salvation_a`, `runs/salvation_b`). Track C tests the restructured method: encode each view at its own size, lift the small views' tokens to the sharp grid, run the decoder on the sharp grid (arm `A10`), and let global attention read small views through pooled keys while every token keeps its own query (`A11`).

One command runs everything and needs no attention for up to 6 hours. Setup is the same as section 1 below; if you already ran tracks A and B on this machine, nothing new needs installing.

**Windows (PowerShell):**

```powershell
cd mixed-resolution-s23
git pull
.venv\Scripts\Activate.ps1
python -m pytest -q tests/test_salvation.py
python scripts/salvation_night.py --push
```

**Linux:**

```bash
cd mixed-resolution-s23 && git pull
source .venv/bin/activate
python -m pytest -q tests/test_salvation.py
python scripts/salvation_night.py --push
```

What it does, in order. A failed step is logged and the rest continue.

| Stage | Work | Rough time |
| --- | --- | --- |
| 1 | Pi3 track C on the five seed-42 scenes | 10 min |
| 2 | Download VGGT weights (5 GB), VGGT track C on the same scenes | 20 min plus download |
| 3 | Ten fresh draws (seeds 43-52) of all seven 7-Scenes scenes: download the frames, run Pi3 and VGGT on each | most of the night; depends on download speed |
| 4 | VGGT tracks A and B on the seed-42 scenes | 20 min |

Leave the machine alone while it runs: do not commit, pull, or edit files in the repo. The runner refuses results whose code state differs from what the identity gate saw.

**When you come back**

- Read `runs/salvation_night/reports/night-pi3.txt` first, then `night-vggt.txt` and `night-summary.txt` (which step succeeded or failed). With `--push` the same files are committed to `reports/salvation/` and pushed.
- The verdict line is `Gate C on fresh draws: PASS / PARTIAL / FAIL`. PASS means `A11` stays within 1 AUC@3 of pixel upsampling, beats a uniform resize given at least as much compute (bootstrap interval above zero), and does not lose coarse accuracy. PARTIAL means only `A10` does. The rule is in `scripts/salvation_night_report.py` and was fixed before any run.
- The full console log is `runs/salvation_night/log.txt`.
- If it was interrupted (power, sleep, crash), run the same command again; finished scenes are skipped. This works only until the final push, which creates a commit.

Options: `--hours 5` changes the time limit, `--skip-vggt` runs Pi3 only, and leaving out `--push` keeps everything local.

Extra variants in track C: `A0_5k`, `A0_6k`, `A0_7k` are uniform resizes at larger token budgets (the cost-accuracy curve to beat); `A11_nobias` drops the logit bias; `A8_kv` and `A3_kv` apply the same key pooling to real and to upsampled pixels.

## The earlier `salvation` kill tests (tracks A and B)

Two short GPU experiments decide which paper we write. Run track A first. Run track B only if A fails.

| Track | Question | What it runs | Passes if |
| --- | --- | --- | --- |
| **A** | Does the mix break because small and sharp views come out of frame attention on different scales? | Zero-cost corrections on the plain mix (no extra tokens): token-statistics matching and an attention logit bias | Best correction beats mix + position alignment (A2) by at least +2 AUC@3 and in at least 4 of 5 scenes |
| **B** | Does resolution matter per view rather than across views? | Real pixels for every view, sharp frame attention, pooled global attention (new arm A9), plus the head-grid fix for RMFA | Best A9 variant beats uniform resize (A0) by at least +2 AUC@3 and in at least 4 of 5 scenes |

Everything is exploratory: 7-Scenes, 5 scenes, seed 42, 4k tokens, same views and metric as the archived A1-A8 table. These numbers pick a direction; they never go in the paper.

### 1. Set up (once, about 15 minutes, needs about 10 GB disk)

Linux, an NVIDIA GPU with bf16 support, Python 3.10.

```bash
git clone https://github.com/pineconedad/mixed-resolution-s23.git
cd mixed-resolution-s23
git checkout salvation
git submodule update --init third_party/Pi3 third_party/vggt

python3.10 -m venv .venv
source .venv/bin/activate
# RTX 50-series cards need a CUDA 12.8 build; the repo's pinned torch 2.5.1 will not run on them.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
grep -v '^torch' env/requirements.txt > /tmp/mixres-requirements.txt
pip install -r /tmp/mixres-requirements.txt einops==0.8.1
pip install -e . --no-deps
source env.sh
export PYTHONPATH="$PWD/third_party/vggt:$PYTHONPATH"

python -m pytest -q tests/test_salvation.py tests/test_engine.py   # CPU, under a minute
python scripts/download_checkpoint.py        # Pi3 weights, 3.8 GB
python scripts/prepare_quick.py              # the 60 sampled 7-Scenes frames only
```

On an older card (RTX 30/40, A40, H100) you can use `pip install -r env/requirements.txt` instead of the two torch lines.

### 2. Run track A

```bash
source .venv/bin/activate && source env.sh
bash scripts/salvation.sh a pi3
```

This runs the fp32 identity gate against the native model, then 13 variants on 5 scenes, then prints a table and a line starting `Gate A: PASS` or `Gate A: FAIL`. It is resumable: rerun the same command after an interruption. The table is also saved to `runs/salvation_a/pi3/summary/report.txt`.

### 3. Read the result

| Outcome | Next step |
| --- | --- |
| **Gate A: PASS** | Stop. Do not run B yet. Send `report.txt` back; the next build promotes the winning correction into the S23 runner and confirms it on fresh development scenes. Optionally run `bash scripts/salvation.sh a vggt` (step 4) for the second model |
| **Gate A: FAIL** | Run `bash scripts/salvation.sh b pi3` and read `Gate B` the same way |
| **Both fail** | No method paper from this repo this cycle. Keep both `report.txt` files; the measured `shift` column and the A-arm table still support the localization result |
| **INCOMPLETE** | A variant failed or did not run. The report lists which and why. Fix the cause; never drop the variant |

The pass rule is written in `scripts/salvation_report.py` and was fixed before any run. Do not change it after seeing numbers.

### 4. Second model (optional, after a pass)

```bash
python scripts/download_checkpoint.py --model vggt     # 5 GB
bash scripts/salvation.sh a vggt                       # or: b vggt
```

The VGGT identity gate runs first and stops the run if this environment does not reproduce the native model. To run the VGGT CPU tests: `python -m pytest -q tests_vggt` (one A0 test there fails on the base commit too; it is unrelated to these changes).

### What the variants are

| Name | Meaning |
| --- | --- |
| `A0` | All 12 views resized to one size at the same token budget (4032 tokens vs 4200 for the mix) |
| `A1`, `A2`, `A3`, `A4`, `A8` | Archived arms, rerun so every comparison is paired on your machine |
| `A2_measure` | A2 unchanged; only records how far small-view tokens sit from sharp-view tokens (`shift` column, in standard deviations) |
| `A2_moment` | Before each global block, rescale small-view tokens to the sharp views' per-channel mean and spread |
| `A2_moment_transient` | Same, but only global attention sees the rescaled tokens |
| `A2_bias_global`, `A2_bias_both` | Add log(area ratio) to attention logits of small-view tokens, in global attention or in global and frame attention |
| `A2_moment_bias` | Both corrections together |
| `A1_moment` | Token rescaling without position alignment |
| `A4_headhi`, `A5_headhi` | RMFA with heads reading the sharp-grid tokens (archived runs used the pooled grid) |
| `A9`, `A9_headhi`, `A9_headhi_bias` | Real pixels in every view, sharp frame attention, pooled global attention |

Code: `mixres/rmfa/calibration.py` (corrections), `mixres/rmfa/engine.py` (wiring), `mixres/rmfa/arms.py` (A9), `configs/salvation-*.yaml`, `scripts/salvation.sh`, `scripts/salvation_report.py`, `tests/test_salvation.py`. All new options default to off, so archived arms are unchanged.

---

## Background (before `salvation`)

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

All paid inference requires CUDA. CPU unit tests are separate. The first GPU session is now complete: full-checkpoint fp32 T1 passed on two scenes and all 20 A1-A4 evaluations completed. The pod was terminated after verified download. See reports/README.md for every exported artifact and docs/decisions.md for the complete decision trail.

The subsequent guarded session completed all forty A1-A8 evaluations on the same five development scenes, excluding only historical A0. Native fp32 identity also passed on the cheaper 16 GB RTX A4000. All outputs and monitor snapshots are published under reports/, and the independent watchdog terminated both its smoke-test pod and the diagnostic pod. See docs/safety.md for key handling, the reserve/deadline policy and failure behavior.

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
