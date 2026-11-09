# Mixed-Resolution Study: Setup From Scratch

Nov 9, 2026 · @wasif

## 1. What this setup produces

At the end of this guide you have three models, five datasets, frozen view lists and a test suite that all arms in the [S23 test plan](https://claude.ai/artifact/PmdzDSkxLQqP9PnCoQRtrb) run against, with nothing left to choose by hand. Build it in this order, because each step depends on the one before:

1. Copy what you need off the old machine, then install the fresh PC from zero (section 2).
2. Model checkpoints, each verified against its paper's own numbers (section 3).
3. Datasets downloaded and converted to one common format (section 4).
4. Splits and seeded view lists, written to disk once (section 5).
5. Preprocessing rules shared by every arm (section 6).
6. Fairness and reproducibility checks (sections 7 and 8).
7. Final checklist (section 9) before the first real run.

## 1b. Fresh-PC quick test (do this first)

This is a self-contained test of the new technique that needs nothing from the old machine; it decides the abstract, and the full setup below waits until the old code can be copied over. It works because it only compares arms against each other inside one new pipeline.

**Scope**

| Choice | Value |
| --- | --- |
| Model | π³ only, original checkpoint `yyfz233/Pi3` |
| Dataset | ETH3D or 7-Scenes, whichever downloads faster; 5–10 scenes |
| Views per scene | About 12: 4 sharp, 8 small at ρ = 2 |
| Seed | 42 (exploratory) |
| Metric | Pose AUC@3 against ground-truth poses |
| Arms | A1 plain mix, A2 mix + `pi_hr`, A3 upsample-all, A4 RMFA |

**Steps**

1. **Install** the driver, conda, PyTorch and the Pi3 repo (section 2, skipping everything about the old machine). Download the checkpoint once.
2. **Download** one dataset and pick 5–10 scenes. Save the scene list to a file.
3. **Write the view sampler:** for each scene, draw 12 frames with a fixed seed, choose 4 sharp views by farthest-point sampling on camera centres, and save the result as JSON. Every arm reads this file.
4. **Write the resizer** (section 6 rules): sharp views from the original frame; small views from the original at half size; upsampled small views from the small image back up to sharp size. Same crop for every view.
5. **Rewrite `pi_hr`:** first check how π³ encodes patch positions. Then scale each small view's patch coordinates by ρ so they land on the sharp views' grid.
6. **Write RMFA** (S23 plan, section 2): the two-grid loop over π³'s frame and global attention blocks, with default settings (area pooling, bilinear lift, all layers, heads on the lo grid).
7. **Run test T1:** with every view the same size, RMFA must give the same poses and points as plain π³ (max difference below 1e-4 in fp32). Do not run arms until this passes.
8. **Write pose scoring:** align predicted poses to ground truth and compute AUC@3, following DA3's benchmark protocol if its code is easy to use, otherwise the standard relative-rotation and translation-angle AUC.
9. **Run** A1, A2, A3 and A4 on every scene; save per-scene AUC.

**How to read it**

- **Sanity first:** A1 should be clearly worse than A2 and A3. If it is, the new pipeline reproduces the main finding on its own. If not, suspect the resizer or `pi_hr` before anything else.
- **Then the decision number:**

```latex
R = \frac{\mathrm{AUC}_{A4} - \mathrm{AUC}_{A2}}{\mathrm{AUC}_{A3} - \mathrm{AUC}_{A2}}
```

- R ≥ 0.5: the technique looks promising; write abstract A (method framing).
- R < 0.5, or T1 never passes: write abstract B (diagnosis framing).

**Limits.** These numbers come from a different pipeline than S1–S22, so they guide the decision only and never go in the paper. Other models, frozen seeds and held-out datasets all wait for the full setup.

## 2. Fresh machine: from zero to a working environment

This section assumes a new PC with nothing installed. Linux (Ubuntu 22.04 or 24.04) is the safe choice; on Windows, do everything inside WSL2 with Ubuntu, because all three repos are developed and tested on Linux.

**Later, once you can reach the old machine,** push or copy these, since nothing else can recreate them:

- The harness code (S1–S22) as a git repo, pushed to GitHub or a private remote.
- Every config S1–S22 used, especially the S7 frozen mix and uniform settings and the checkpoint names.
- Result summaries (CSVs or JSONs) for S1–S5 and S7, needed for test T9.
- `research_notes/` and any view lists or scene lists you already generated.
- Already-converted datasets, if you can move them by drive; re-downloading is the slowest step.

**Hardware**

- An NVIDIA GPU. Memory decides whether the ceiling arms (A3/A8) fit at 8k tokens; measure it in phase 0 and fall back to 4k for ceilings if needed.
- Disk: enough for five datasets plus `runs/` with room to spare; an SSD for the datasets makes loading much faster.
- **All latency numbers in the paper must come from this machine.** Any timing from the old machine (such as S8) is not comparable; re-time it here.

**Install steps, in order**

1. **OS packages:** `sudo apt update && sudo apt install -y build-essential git git-lfs wget curl ninja-build`. Run `git lfs install`.
2. **NVIDIA driver:** install the recommended driver (`sudo ubuntu-drivers install` on Ubuntu), reboot, and confirm the GPU appears in `nvidia-smi`. On WSL2, install the driver on Windows only, not inside Ubuntu.
3. **Miniconda:** install it, then `conda create -n mixres python=3.10 -y` and `conda activate mixres`. Use 3.10 unless a repo's README requires another version.
4. **PyTorch:** install the CUDA build from pytorch.org's selector that matches your driver. PyTorch wheels bring their own CUDA runtime, so you do not need a system CUDA toolkit for inference. Check: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`.
5. **Model repos,** each cloned to `third_party/` and installed editable at a fixed commit: VGGT, Pi3, Depth-Anything-3. For DA3 use the basic install only (`pip install -e .`); its Gaussian head needs gsplat, which this study does not use.
6. **Your harness:** clone it, `pip install -e .`, and copy the old configs and result summaries into place.
7. **Hugging Face:** set `HF_HOME` to a big disk, then download the three checkpoints once (section 3) so runs never download mid-batch.
8. **Freeze the environment:** `pip freeze > env/requirements.lock`, `conda env export > env/environment.yml`, record each repo's commit in `env/versions.md`, and commit all three.

If two repos demand conflicting library versions, make one env per model rather than forcing them together. Never mix versions between arms of the same model.

**Shared settings** in `env.sh`, sourced before every run: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, and `HF_HOME`.

**Determinism in code** (in the shared entry point, not per script): seed `torch`, `numpy` and `random`; set `torch.backends.cudnn.benchmark = False`; use `torch.use_deterministic_algorithms(True, warn_only=True)` for tests. Turn off TF32 for fp32 tests (`torch.backends.cuda.matmul.allow_tf32 = False`), otherwise T1 can fail for no real reason.

**New machine, new numbers.** A different GPU, driver or library version can shift results slightly. That is why T9 (reproduce S1–S5 within 0.5 AUC) matters even more here. If T9 fails by a small, consistent amount, rerun the S1–S5 and S7 baselines on this machine and use those numbers in the paper, so every comparison comes from one setup.

## 3. Models

Use exactly the three checkpoints that produced S1–S22; two of the repos now ship newer versions by default, so pin the old ones explicitly or every earlier number becomes incomparable.

| Model | Code | Checkpoint to use | Version trap | Weights licence |
| --- | --- | --- | --- | --- |
| VGGT | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | `facebook/VGGT-1B` | None known; pin the repo commit | Check the model card |
| π³ | [yyfz/Pi3](https://github.com/yyfz/Pi3) | `yyfz233/Pi3` (original) | The repo now recommends Pi3X (`yyfz233/Pi3X`), released Dec 2025. Do not switch mid-study | CC BY-NC 4.0 (non-commercial; fine for research) |
| DA3 | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Whatever S1–S22 used | Checkpoints with a -1.1 suffix were retrained after a training-bug fix; the originals are marked deprecated | Check the model card |

**Which DA3 checkpoint.** Check your S1–S22 configs now. If they used an original (non-1.1) checkpoint, keep it for S23 so earlier numbers stay comparable, and add one sentence in the paper saying which one. Switching to 1.1 means rerunning every DA3 result.

**Verify each model before trusting the harness.** Run each model through the harness on 2–3 scenes of a dataset its own paper reports, with uniform full-resolution input, and compare against the paper's numbers. A big gap means a preprocessing or evaluation bug, not a model property. Record the result in `env/model_checks.md`.

**Write down per model** (in `configs/models/<model>.yaml`): patch size, maximum input side, normalization (mean/std), required image-size multiples, where frame and global attention blocks are, which layers feed the heads, and what special tokens exist. π³ drops order-dependent components such as reference-frame tokens, so its special-token handling differs from VGGT's.

**Sources:** [π³ repo](https://github.com/yyfz/Pi3), [DA3 repo](https://github.com/ByteDance-Seed/Depth-Anything-3).

## 4. Datasets

Your five datasets are exactly DA3's Visual Geometry Benchmark, so use DA3's benchmark data preparation and evaluation for all three models; reviewers then cannot say your evaluation favours or hurts any model.

DA3 built its benchmark from these five datasets. It scores pose with AUC at 3° and 30°, and reconstruction with F1 (Chamfer distance on DTU). Poses are aligned with evo using RANSAC, and point clouds are fused with TSDF before scoring.

| Dataset | Type | Role in this study | Metrics | Notes |
| --- | --- | --- | --- | --- |
| ETH3D | Real, high-res, wide baseline | Development + frozen | AUC@3/30, F1 | Where mixing helped reconstruction in S7 |
| ScanNet++ | Real indoor | Development + frozen | AUC@3/30, F1 | Requires registration and accepting its terms |
| 7-Scenes | Real indoor video | Development + frozen | AUC@3/30, F1 | Where mixing hurt most (S7) |
| DTU | Real, object-centric, lab | **Held-out** | AUC@3/30, Chamfer (mm) | Use observation masks for Chamfer |
| HiRoom | Synthetic indoor | **Held-out** | AUC@3/30, F1 | 30 Blender-rendered living-room scenes; F1 threshold 0.05 m; no standalone release found outside the DA3 benchmark |

**Steps:**

1. Download each dataset from its official source, plus HiRoom through the DA3 benchmark release. Record the download date and any version tag in `env/datasets.md`.
2. Convert all five to one format with the DA3 benchmark scripts: images, intrinsics, world-to-camera poses, GT geometry, per-scene metadata.
3. Use the benchmark's scene lists and thresholds unchanged. If S1–S22 used different scenes or thresholds, keep yours for comparability and write down exactly how they differ.
4. Check each dataset: number of scenes, frames per scene, and image sizes match `env/datasets.md`; poses and GT geometry project consistently on 3 random frames.

**Sources:** [DA3 paper](https://arxiv.org/pdf/2511.10647v1), [HiRoom entry](https://paperswithcode.co/benchmark/hiroom), [benchmark summary](https://www.alphaxiv.org/benchmarks/bytedance/depth-anything-3).

## 5. Splits, seeds and view sampling

Three datasets are for development and two are held out; seeds only choose which views each scene uses, so every arm and model sees identical inputs for a given seed.

| Split | Datasets | Seeds allowed | Used for |
| --- | --- | --- | --- |
| Development | ETH3D, ScanNet++, 7-Scenes | 42 | Quick check, phase 1 tuning, any debugging |
| Frozen | ETH3D, ScanNet++, 7-Scenes | 1000, 2000, 3000 | Main results, after configs are frozen |
| Held-out | DTU, HiRoom | 1000, 2000, 3000 | Generalization; never opened before freezing |

**Leakage rule.** Development and frozen runs share datasets but not views: seed 42 picks one set of frames, seeds 1000+ pick others. That matches how S7 was done. Never tune on DTU or HiRoom, and never open any seed ≥ 1000 result before `predictions.md` is committed.

**View sampling, per scene and seed:**

1. Fix the total number of views N per dataset (use the S7 value). If a scene has fewer than N frames, drop the scene and list it in `env/dropped_scenes.md`.
2. Draw N frame ids with `numpy.random.default_rng(seed + scene_index)`, where `scene_index` is the scene's position in the sorted scene list. This makes each scene's draw independent of which other scenes exist.
3. Choose which views are sharp with farthest-point sampling on camera centres (the S7 `fps` rule), starting from the first drawn frame.
4. Write the result to `runs/views/<dataset>/seed<seed>.json` and commit it.

**Budgets and sizes are derived, not chosen.** For each budget (2k, 4k, 8k), the S7 mix gives n\_sharp, n\_small and ρ. Store the resulting pixel sizes in the view JSON (section 6), so no script recomputes them at run time.

## 6. Preprocessing

Every image is produced from the original frame by one shared function; the only thing an arm changes is the target size, so differences between arms cannot come from resizing code.

**Token budget.** With patch size 14, a view of H × W pixels costs (H/14)(W/14) tokens. For a mix with n\_sharp sharp views of T tokens each and n\_small small views shrunk by ρ:

```latex
B = n_{\text{sharp}}\,T + n_{\text{small}}\,\frac{T}{\rho^{2}}
```

Solve for T given B, n\_sharp, n\_small and ρ, then round sizes as below. Report the actual token count after rounding, not the nominal budget.

**Resizing rules:**

1. **Always resize from the original frame.** Small views: original → small size. Upsampled small views (A3, A4, A6): original → small → sharp size, so the lost detail is really lost. Real full-res views (A8): original → sharp size directly.
2. **One resampler.** Downsampling with antialiasing (the same filter S1–S22 used); upsampling with bicubic. Write both into `resize.py` and nowhere else.
3. **Same field of view for every view.** If a model needs a crop (for example a fixed width or aspect), compute the crop once in normalized coordinates and apply it identically to sharp and small views. A small view must never see a different part of the scene than it would as a sharp view.
4. **Patch-multiple rounding.** Round each side to the nearest multiple of 14, keeping the aspect ratio within one patch. Compute sizes once and store them in the view JSON.
5. **Intrinsics.** Scale GT intrinsics by the same factors as the image (and shift for crops) wherever evaluation uses them, such as TSDF fusion.
6. **Normalization.** Each model's own mean and std, applied after resizing, in the adapter.

**Check before use.** Save a contact sheet per dataset: one scene, every view, under A1, A3 and A8, side by side. Look at it once. Mismatched crops or wrong sizes are obvious there and invisible in metrics.

## 7. Fairness rules

A comparison is fair only if the two arms differ in exactly the thing being tested; everything in the table below is held identical, and the harness asserts it rather than trusting you to remember.

| Must be identical | Across | How it is enforced |
| --- | --- | --- |
| Scenes, frame ids, sharp/small assignment | All arms and models, per seed | Same view JSON, hash checked at run start |
| Field of view and crop of each view | All arms | Shared crop in normalized coordinates (section 6) |
| Global-attention token count | A1, A2, A4, A5, A7 (and A0 by budget) | Assert in the engine (test T3) |
| Checkpoint, code commit, precision | All runs of one model | Recorded in `result.json`; aggregate script refuses mixed values |
| Evaluation code, thresholds, alignment | All models | One evaluation module (DA3 benchmark), no per-model branches |
| Hyperparameters of the method | All models | Tuned once on π³ dev data, then frozen |
| GPU type and timing protocol | All latency numbers | Warm-up, CUDA sync, median of 3 |

**Fair baselines.**

- **Best uniform (A0) gets the same care as the method.** Use the S7 best uniform split per cell, chosen on development data the same way the mix was. Do not compare the method against a weak uniform setting.
- **Ceilings are labelled as ceilings.** A3, A6 and A8 use more global tokens. They appear in grey or separately, never in the same "wins" count as equal-budget arms.
- **No per-scene cherry-picking.** Report all scenes; dropped scenes are listed with a reason decided before seeing results (too few frames, missing GT).

**Fair reporting.** Report every arm, model and dataset you ran, including the ones that go against the method (DA3, pose). Count significant cells both ways (method better and method worse). State that seed 42 results are exploratory everywhere they appear.

**If your current harness already differs** from any rule here (for example a different sampler in S7), keep the S7 behaviour for comparability and write down the difference, rather than changing it halfway through the study.

## 8. Running smoothly

Most lost days in a study like this come from silent failures: a crashed batch, a stale config, an out-of-memory run that skipped scenes. The rules below make each of those loud.

**Reproducibility**

- Every `result.json` stores: git commit, dirty-tree flag, config hash, view-list hash, checkpoint name, precision, GPU name, library versions. Runs from a dirty tree are refused for frozen phases.
- Configs are files, never command-line overrides, in frozen phases. The command line only picks model, dataset, budget, arm and seed.

**Robust batches**

- **Resume, don't restart.** The launcher skips runs whose `result.json` exists with a matching config hash.
- **Atomic writes.** Write to `result.json.tmp`, then rename. A crash never leaves a half-written result that looks complete.
- **Per-scene isolation.** A scene that fails (out of memory, NaN) is logged with its error and marked failed; the run continues. A run with any failed scene is flagged in the aggregate and excluded from paired comparisons for that scene in every arm.
- **Out-of-memory policy.** Never silently lower resolution or drop views to fit. Fail the scene and report it.

**Monitoring**

- One line per scene in `log.txt`: scene, latency, memory, primary metric. Tail it while runs go.
- After each phase, `05_aggregate.py` prints: runs expected vs found, failed scenes, any arm whose token count or hash disagrees with the plan.
- **Sanity numbers to eyeball:** A1 should be worse than A2 on π³ and DA3; A3 should be close to A8. If either flips on most cells, suspect a bug before believing a finding.

**Storage and naming**

- `runs/<study>/<model>/<dataset>/<budget>/<arm>/seed<seed>/`, nothing else in the path.
- Never delete runs; superseded runs move to `runs/_superseded/` with a note in `CHANGELOG.md`.
- Back up `runs/` summaries and `frozen/` to a second location daily.

## 9. Checklist before the first real run

Tick every item before the quick check; the last group must also be done before any seed ≥ 1000 run.

**Environment**

- [ ] `mixres` env built; `requirements.lock`, `environment.yml` and repo commits committed
- [ ] Determinism settings in the shared entry point; TF32 off for fp32 tests

* [ ] Harness code, S1–S22 configs, S1–S7 result summaries and `research_notes/` copied from the old machine
* [ ] `nvidia-smi` shows the GPU; `torch.cuda.is_available()` is `True`
* [ ] All three checkpoints downloaded to `HF_HOME` before any batch starts
* [ ] Old timing results (S8) marked as not comparable; latency re-measured on this machine

**Models**

- [ ] VGGT, π³ (original, not Pi3X) and DA3 checkpoints match S1–S22
- [ ] Each model reproduces its paper's numbers on 2–3 scenes with uniform input
- [ ] `configs/models/*.yaml` filled: patch size, block layout, head layers, special tokens

**Data**

- [ ] All five datasets converted with the DA3 benchmark scripts; counts match `env/datasets.md`
- [ ] Pose and GT geometry consistency checked on 3 frames per dataset
- [ ] View lists for seeds 42, 1000, 2000, 3000 written and committed; dropped scenes listed
- [ ] Contact sheets for A1, A3, A8 inspected

**Harness**

- [ ] Tests T1–T10 from the S23 plan pass for every model
- [ ] T9 reproduces S1–S5 numbers within 0.5 AUC
- [ ] Resume, atomic writes and per-scene failure logging tested by killing a run halfway

**Before frozen runs**

- [ ] Phase 1 tuning done on π³ development data only
- [ ] Frozen configs written and committed
- [ ] `predictions.md` committed with numeric thresholds for H1–H4
