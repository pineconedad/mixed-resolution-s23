# S23 Test Plan: Resolution-Matched Frame Attention

Nov 9, 2026 · @wasif

## 1. Goal and hypotheses

S23 tests whether running the small views' frame attention at the sharp resolution, then pooling back before global attention, recovers the upsampling gain without the upsampling cost. It also re-confirms the paper's main claim (artefact, not lost information) on frozen seeds, because that claim currently rests on seed 42 only.

Two questions, one run matrix:

- **Q1 (method):** does Resolution-Matched Frame Attention (RMFA) recover most of the gain that full upsampling gives, at the mix's global-attention token count?
- **Q2 (localization and main claim):** does the damage sit in frame attention, global attention, or both? And does upsampling still recover the loss on frozen seeds?

**Hypotheses (write these into `predictions.md` before any seed-1000 run):**

- **H1:** On π³, RMFA recovers at least half of the upsampling gain over `pi_hr`. Recovery ratio R ≥ 0.5, pooled pose AUC@3, 95% interval above 0.
- **H2:** On DA3, RMFA does not hurt versus `pi_hr` (difference ≥ −1 AUC). A gain here is the headline result.
- **H3:** On VGGT, RMFA has no effect (|Δ| < 1 AUC), because VGGT benefits from real pixels, not resolution matching.
- **H4 (main claim):** On frozen seeds, upsample-all recovers ≥ 70% of the gap between as-is mix and real full-resolution views, for π³ and DA3.

**Recovery ratio**, the single number for Q1:

```latex
R = \frac{\mathrm{AUC}_{\text{RMFA}} - \mathrm{AUC}_{\text{pi\_hr}}}{\mathrm{AUC}_{\text{upsample}} - \mathrm{AUC}_{\text{pi\_hr}}}
```

R = 1 means RMFA matches full upsampling. R = 0 means it adds nothing over position alignment. Only compute R where the denominator is significant; otherwise report the raw differences.

## 2. The method under test: RMFA

Each small view keeps two token grids: a **hi grid** at the sharp views' size, used only in frame attention, and a **lo grid** at its own size, used in global attention. The hi grid is the persistent state; the lo grid is re-derived by pooling before every global block.

**One block pair, for a small view with ratio ρ:**

1. **Frame attention on the hi grid.** Tokens sit on the same grid size as a sharp view, with native positions for that grid. This is the step S14 says is broken in the plain mix.
2. **Pool hi → lo.** Area-pool the hi tokens to the small view's grid (h/ρ × w/ρ). Non-integer ρ (1.25) uses `adaptive_avg_pool2d` to the exact target size.
3. **Global attention on the lo grid**, together with the sharp views, using `pi_hr` aligned positions. Token count here equals the plain mix.
4. **Lift the update back.** Take the global block's residual update Δ = lo\_out − lo\_in, upsample Δ to the hi grid (bilinear by default), and add it to the hi tokens. Lifting only the update, not the tokens, keeps the hi-grid detail from being overwritten by pooled values.

**Two entry variants:**

- **RMFA-pixel:** upsample the small image to sharp size, then patch-embed. The hi grid starts from real patch embeddings of upsampled pixels, same as the upsample-all arm.
- **RMFA-token:** patch-embed the small image at its own size, then interpolate the tokens to the hi grid. Cheaper; tests whether pixel-level upsampling matters.

**Special tokens** (camera and register tokens in VGGT and π³) are per-view and resolution-free. Keep one copy per view; it joins both frame attention (with the hi grid) and global attention (with the lo grid). Never pool or duplicate them.

**Heads.** The DPT-style heads read intermediate-layer tokens. Parameter `head_grid`: `lo` (default) feeds pooled tokens so outputs have the same resolution as the plain mix, keeping metrics comparable; `hi` feeds hi-grid tokens and downsamples the prediction, as a secondary check. Store intermediate tokens on whichever grid the head uses.

**Model-specific mapping.** VGGT and π³ alternate frame and global attention, so the block pair maps directly. DA3 does not use the same alternating layout: before coding, list which DA3 layers attend within a view and which across views, and apply RMFA to the within-view layers only. Write this mapping into `configs/models/da3.yaml`.

**Core loop (pseudocode):**

```python
hi = embed(views, mode)              # sharp: native grid; small: hi grid
for l in range(L):
    hi = frame_block[l](hi, pos=native_hi)          # per view
    lo = pool_small_views(hi, rho)                   # sharp views pass through
    lo_out = global_block[l](lo, pos=pi_hr)
    delta = lo_out - lo
    hi = hi + lift_small_views(delta, rho)           # sharp views: hi = lo_out
    if l in head_layers:
        cache[l] = lo_out if head_grid == "lo" else hi
return heads(cache)
```

With ρ = 1 for every view, pool and lift are identities and the loop must reproduce the original model exactly (test T1).

## 3. Experiment arms

Nine arms; every arm sees the exact same views, sharp/small assignment and scene list for a given seed. A0–A3 and A8 double as the frozen re-test of the main claim (H4).

| Arm | Small views enter as | Frame attn grid | Global attn grid | Positions (global) | Global tokens vs mix | Isolates |
| --- | --- | --- | --- | --- | --- | --- |
| A0 uniform | n/a, best uniform split from S7 | native | native | native | equal budget | Reference everyone must beat |
| A1 mix as-is | small pixels | lo | lo | native | 1× | The broken baseline |
| A2 mix + `pi_hr` | small pixels | lo | lo | `pi_hr` | 1× | Current best fix |
| A3 upsample-all | upsampled pixels | hi | hi | native | ρ²× for small views | Upper bound, artefact removed everywhere |
| A4 RMFA-pixel | upsampled pixels | hi | lo | `pi_hr` | 1× | **The method** |
| A5 RMFA-token | small pixels, tokens interpolated | hi | lo | `pi_hr` | 1× | Do real upsampled patch embeddings matter? |
| A6 global-only upsample | upsampled pixels | lo (pooled) | hi | native | ρ²× | Is the benefit in global attention instead? |
| A7 RMFA-pixel, no `pi_hr` | upsampled pixels | hi | lo | native | 1× | Does RMFA need position alignment? |
| A8 real full-res | real full-resolution pixels | hi | hi | native | ρ²× | Information ceiling for H4 |

**How the arms combine into answers:**

- A3 − A2 = the total recoverable gain. A4 − A2 = what RMFA gets. Their ratio is R (section 1).
- A6 vs A4: if A6 recovers more than A4, the benefit lives in global attention and RMFA is targeting the wrong step.
- A8 − A3 = the value of real information. Small means the main claim holds.
- A3 and A6 use more global tokens than the mix. That is intended: they are diagnostic ceilings, not fair competitors. Only A0, A1, A2, A4, A5 and A7 are compared at equal global tokens.

A6 implementation: run frame attention on the lo grid (as in the mix), then upsample tokens to the hi grid for each global block and pool the update back. It mirrors A4 with the grids swapped.

## 4. What you need

Most of this already exists in your S1–S22 harness; the new work is the RMFA wrapper, three model adapters and the probes.

| Item | Detail | Status |
| --- | --- | --- |
| Models | VGGT-1B, π³, DA3: same checkpoints and preprocessing as S7 | Have |
| Datasets | DTU (held-out), ETH3D, ScanNet++, 7-Scenes, HiRoom (held-out) | Have |
| Ground truth | Camera poses for all; point clouds or meshes for reconstruction; DTU observation masks for chamfer | Have |
| View sampler | Seeded view selection plus FPS sharp-view selection, as in S7 | Have; must export view lists to JSON |
| Frozen configs | Best uniform split and mix config per model, dataset and budget from S7 | Have; copy into `frozen/s23/` |
| `pi_hr` positions | From S5 | Have |
| RMFA wrapper | Pool, lift, two-grid state, hooks into frame and global blocks | **New** |
| Model adapters | One per model: where the blocks are, where special tokens live, which layers feed heads | **New** |
| Probes | Frame-attention entropy per layer and head; token drift vs A3 | Partly have (entropy from S1–S3) |
| Stats | Paired per-scene differences, bootstrap intervals, Holm correction | Partly have |

**Hardware.** One GPU with enough memory for A3 and A8 at 8k tokens is the binding constraint, since those arms use ρ² more global tokens. If 8k A3/A8 does not fit, run them at 4k only and say so in the paper. Use fp32 for T1 (exact equivalence) and your usual precision (bf16) for real runs.

**Software.** PyTorch with `torch.nn.functional.adaptive_avg_pool2d` and `interpolate`; forward hooks or a subclassed aggregator for the block loop; a run tracker that writes the git commit and config hash into every result file. Pin library versions in `requirements.lock`.

## 5. Code structure

Keep the method separate from the models: one generic RMFA engine, thin per-model adapters, and arms defined purely in config. Names below are suggestions; map them onto your existing layout.

```
mixres/
  configs/
    s23.yaml                 # study: arms, budgets, datasets, seeds
    models/{vggt,pi3,da3}.yaml   # block layout, head layers, patch size
    datasets/*.yaml          # paths, scene lists, metric type
  mixres/
    data/
      sampler.py             # seeded view + sharp-view selection -> JSON
      resize.py              # small / upsampled / real-full-res inputs
    rmfa/
      grid.py                # pool_to(), lift_to(), grid sizes for any rho
      engine.py              # two-grid block loop (section 2 pseudocode)
      arms.py                # arm name -> (entry, frame grid, global grid, positions)
      positions.py           # native and pi_hr position builders
    adapters/
      base.py                # interface every model implements
      vggt.py  pi3.py  da3.py
    eval/
      pose.py                # AUC@3, AUC@5, AUC@30
      recon.py               # F-score, precision, recall
      dtu.py                 # chamfer mm with observation masks
    probes/
      entropy.py             # frame-attention entropy, per layer and head
      drift.py               # cosine similarity of small-view tokens vs A3
      cost.py                # FLOPs, latency, peak memory
    stats/
      paired.py              # per-scene diffs, bootstrap CI, Holm
  scripts/                   # see section 8
  tests/                     # see section 9
  frozen/s23/                # frozen configs + predictions.md (read-only)
  runs/                      # all outputs (section 12)
```

**Adapter interface** (`adapters/base.py`). Each model implements:

- `embed(images) -> tokens, special_tokens`
- `frame_blocks()` and `global_blocks()`: ordered lists of the attention modules, paired by layer index
- `head_layers()`: which layer outputs the heads read
- `run_heads(cache) -> predictions` in a common format (poses as 4×4, points as H×W×3, confidence maps)
- `patch_size` (14 for all three) and `set_positions(kind, grid_sizes)`

The engine never imports a specific model. If DA3's layer layout does not split into clean frame/global pairs, its adapter returns `None` for a missing partner and the engine skips pool/lift for that layer.

**Grid sizes.** Every grid side must be a multiple of the patch size. For a sharp view of H×W pixels and ratio ρ, the small view is round\_to\_14(H/ρ) × round\_to\_14(W/ρ), and its hi grid equals the sharp grid exactly. Compute both in `grid.py` once and assert them.

## 6. Parameters

Only four RMFA parameters are tuned, and only on development data in phase 1; everything else is fixed from S7 or by design.

| Parameter | Values | Default | Tuned in phase 1? | Notes |
| --- | --- | --- | --- | --- |
| `model` | vggt, pi3, da3 | all | No |  |
| `dataset` | dtu, eth3d, scannetpp, 7scenes, hiroom | all | No | DTU and HiRoom are held-out: never used in phase 1 |
| `budget_tokens` | 2k, 4k, 8k | 4k, 8k | No | 2k only if time allows; 16k skipped (S18 showed no mix gain) |
| `n_sharp`, `n_small`, `rho` | from S7 frozen mix per cell | frozen | No | Same mix the S7 pose and reconstruction results used |
| `sharp_select` | fps, random | fps | No |  |
| `arm` | A0–A8 | all | No | Section 3 |
| `pixel_upsample` | bilinear, bicubic | bicubic | No | Match S1–S3 upsampling control exactly |
| `token_interp` | bilinear, nearest | bilinear | No | A5 only |
| `pool_op` | area, bilinear | area | **Yes** | Hi → lo before global attention |
| `lift_op` | bilinear, nearest | bilinear | **Yes** | Lifting the global update back to hi |
| `rmfa_layers` | all, first half, first quarter | all | **Yes** | S14 per-layer entropy may show the damage is early |
| `head_grid` | lo, hi | lo | **Yes** | Lo keeps outputs comparable to the mix |
| `precision` | fp32, bf16 | bf16 | No | fp32 for tests only |
| `seed` | 42 (dev), 1000, 2000, 3000 |  | No | Seed sets views and sharp-view choice only |
| `probes` | on, off | on for seed 1000 | No | Adds memory; off for extra seeds |

**Tuning budget:** the four tuned parameters give 2 × 2 × 3 × 2 = 24 configs. Tune on π³ only, at 4k, on the development split, then apply the winner unchanged to all models. If you tune per model, say so and treat DA3 and VGGT results as exploratory.

## 7. Inputs

Every run reads the same five inputs; anything that decides which views or configs are used is written to disk once and never regenerated.

| Input | Format | Produced by | Contents |
| --- | --- | --- | --- |
| View lists | `runs/views/<dataset>/<seed>.json` | `00_make_views.py` | Per scene: frame ids, which are sharp, which are small, their target sizes per budget |
| Images | dataset folders | existing | Original full-resolution frames; all resizing happens at load time from these |
| Ground truth | dataset folders | existing | Poses (world-to-camera), intrinsics, GT point clouds or meshes, DTU observation masks |
| Frozen configs | `frozen/s23/<model>.yaml` | `02_freeze.py` | S7 mix and uniform settings plus the four tuned RMFA parameters |
| Predictions | `frozen/s23/predictions.md` | you, by hand | H1–H4 with numeric thresholds, written and committed before seed 1000 |

**View-list JSON example:**

```json
{
  "scene": "eth3d/courtyard",
  "seed": 1000,
  "budget_tokens": 4096,
  "sharp": [{"frame": 12, "hw": [518, 784]}],
  "small": [{"frame": 3, "hw": [252, 392], "rho": 2.0, "hi_hw": [518, 784]}]
}
```

**Rules:**

- Commit view lists and frozen configs to git, and store their SHA-256 hashes in every result file.
- Never look at held-out results (DTU, HiRoom) or any seed ≠ 42 before `predictions.md` is committed.
- A8 loads the real full-resolution frame for each small view, from the same frame id.

## 8. Scripts

Seven scripts run in order; each reads only files the previous ones wrote, so any step can be rerun alone.

| Script | What it does | Reads | Writes |
| --- | --- | --- | --- |
| `00_make_views.py` | Samples views and sharp/small assignment for every dataset, seed and budget | dataset index, S7 mix settings | `runs/views/*.json` |
| `01_sanity.py` | Runs the tests in section 9 on 2 scenes per model; stops everything on any failure | views, models | `runs/sanity/report.json` |
| `02_dev_sweep.py` | π³, 4k, seed 42, development datasets only: A2, A3, and A4 under all 24 RMFA configs | views, S7 configs | `runs/dev/*.json`, ranked config table |
| `03_freeze.py` | Picks the best dev config by pooled pose AUC (ties: lower cost), writes frozen configs, refuses to overwrite existing ones | dev results | `frozen/s23/*.yaml` |
| `04_run.py` | One call = one (model, dataset, budget, arm, seed). Loads frozen config, runs inference, scores, probes, saves | frozen configs, views, GT | `runs/s23/.../result.json` |
| `05_aggregate.py` | Collects all results, computes paired differences, intervals, R, Holm-adjusted significance | `runs/s23/**` | `runs/s23/summary/*.csv` |
| `06_figures.py` | Draws the paper figures and tables from the summary CSVs only | summary CSVs | `runs/s23/figures/*` |

**`04_run.py` command line:**

```bash
python scripts/04_run.py --model pi3 --dataset eth3d --budget 4096 \
  --arm A4 --seed 1000 --frozen frozen/s23/pi3.yaml --probes on
```

**Batch launcher.** `launch.sh` loops over the run matrix (section 11) and calls `04_run.py`. It skips any run whose `result.json` exists with a matching config hash, so a crash mid-batch resumes cleanly.

**`04_run.py` steps, in order:**

1. Load and hash the frozen config and view list; abort if hashes do not match the frozen record.
2. Build inputs for the arm (small, upsampled or real full-res pixels).
3. Build the engine with the arm's grids and positions; assert global-attention token count against the arm table.
4. Run inference per scene; time it with CUDA sync; record peak memory.
5. Score pose and reconstruction; run probes if enabled.
6. Write `result.json` and, if `--save_preds`, `preds.npz`.

## 9. Sanity and unit tests

No dev or frozen run starts until T1–T10 pass for all three models; T1 and T9 catch the bugs that would silently invalidate the whole study.

| Test | What it checks | Pass condition |
| --- | --- | --- |
| T1 identity | All views at one resolution (ρ = 1), engine vs the original model forward | Max abs difference < 1e-4 on poses and points, fp32 |
| T2 pool/lift round trip | lift(pool(x)) for a smooth random field at ρ = 2 and 3; pool(lift(x)) = x for area pooling at integer ρ | Round-trip error within interpolation tolerance; pool(lift(x)) exact to 1e-6 |
| T3 global token count | Tokens entering each global block, per arm | Equals the arm table: 1× for A1, A2, A4, A5, A7 |
| T4 grid sizes | Every grid side for ρ ∈ {1.25, 2, 3} and every budget | Multiple of 14; hi grid equals the sharp grid |
| T5 special tokens | Camera and register tokens per view | Same count and shape in both grids; one copy per view |
| T6 positions | Positions used in frame vs global blocks | Native-hi in frame attention; `pi_hr` in global for A2, A4, A5; native for A7 |
| T7 sharp views untouched | Sharp-view tokens under A2 vs A4 after layer 1 global block | Identical inputs give identical sharp-view frame-attention outputs |
| T8 determinism | Same run twice | Bit-identical metrics in fp32; within 1e-3 AUC in bf16 |
| T9 reproduce old numbers | A1, A2, A3 on 2 seed-42 cells from S1–S3 and S5 | Within 0.5 AUC of the recorded values |
| T10 cost logging | Latency and memory recorded for a dummy run | Non-zero, CUDA-synchronized, warm-up excluded |

**If T9 fails,** stop and find the harness drift before anything else. Every comparison in the paper assumes the S1–S22 numbers and the new runs come from the same pipeline.

**If T1 fails** only in bf16, rerun in fp32. If it fails in fp32, the block pairing or special-token handling is wrong; check T5 and the adapter's block order first.

## 10. Metrics, probes and statistics

The primary metric is pooled pose AUC@3, because the upsampling control that motivates RMFA was measured on pose; reconstruction is secondary.

**Task metrics**

| Metric | Role | Where |
| --- | --- | --- |
| Pose AUC@3 | **Primary** for H1–H4 | All datasets |
| Pose AUC@5, AUC@30 | Secondary, robustness of the conclusion | All datasets |
| F-score, with precision and recall | Secondary; recall shows coverage gains as in S7 | ETH3D, ScanNet++, 7-Scenes, HiRoom |
| Chamfer (mm), accuracy and completeness | Secondary | DTU |

**Mechanism probes** (seed 1000 only, to save memory)

- **Frame-attention entropy** per layer and head, for small views, normalized as in S1–S3. Prediction: RMFA's small-view entropy matches the uniform input, A1/A2 stay over-sharpened.
- **Token drift**: cosine similarity between each arm's small-view tokens (pooled to lo grid) and A3's, per layer. Shows where in depth each arm diverges from the artefact-free reference.
- **Cost**: latency per scene (median of 3 timed passes after 1 warm-up), peak GPU memory, and analytic FLOPs split into frame and global attention.

**Statistics**

- **Unit of analysis:** the scene. Every comparison is a paired per-scene difference between two arms on identical views.
- **Interval:** 95% bootstrap over scenes (10,000 resamples), reported beside mean ± 2 SE for consistency with S1–S22. "Significant" means the interval excludes zero.
- **Pooling:** pooled AUC = mean over scenes across datasets, each dataset weighted equally (not by scene count).
- **Multiple comparisons:** the primary family is A4 vs A2 across 3 models × 5 datasets × 2 budgets = 30 cells. Apply Holm correction within it, and also report the raw count of significant cells as before.
- **Seeds:** report seed 1000 as the main result; seeds 2000 and 3000 as a consistency check (same sign in at least 2 of 3 seeds per pooled comparison).
- **R's interval:** bootstrap R directly by resampling scenes and recomputing the ratio; skip R when the A3 − A2 interval includes zero.

## 11. Run protocol

About 500 runs in four phases (one run = one model, dataset, budget, arm and seed over all scenes); the order guarantees the main-claim result exists even if the method runs out of time.

1. **Phase 0, sanity.** `00_make_views.py`, then `01_sanity.py`. Gate: T1–T10 pass for all models.
2. **Phase 1, development (seed 42).** π³ only, 4k, ETH3D + ScanNet++ + 7-Scenes. A2, A3 and A4 × 24 configs = 78 runs. Gate: run `03_freeze.py`, then write and commit `predictions.md`.
3. **Phase 2, frozen main run (seed 1000).** Every model, dataset, arm, at 4k and 8k, probes on. Gate: none; this is the result.
4. **Phase 3, extra seeds (2000, 3000).** Reduced matrix for the consistency check, probes off.

**Quick check before the abstract (first 24 hours).** The abstract is due before phase 1 can finish, so a small test decides its framing. It runs in parallel with phase 0 and uses only development data, so it does not touch held-out sets or frozen seeds.

| When (from now) | Task | Scope | Output |
| --- | --- | --- | --- |
| Hours 0–4 | Build the engine and π³ adapter; pass T1 (identity) and T3 (token count) | π³ only | Engine trusted for π³ |
| Hours 4–8 | Run A2, A3, A4 with default RMFA settings; add A6 if time allows | π³, 4k, seed 42, one dev dataset (7-Scenes or ETH3D) | Pose AUC@3 per arm |
| Hour \~10 | Compute R = (A4 − A2) / (A3 − A2) | Same | One number |
| Hours 10–20 | Write the abstract: R ≥ 0.5 → method framing (A); otherwise diagnosis framing (B) |  | Draft abstract |
| Hours 20–24 | Final read, register abstract |  | Submitted |

**Rules for the quick check:**

- Treat its result as exploratory. It only picks the abstract's framing; no number from it goes in the paper.
- No tuning: use the defaults in section 6. Tuning happens in phase 1.
- If T1 fails by hour 6, stop debugging and register abstract B. Abstracts are often editable until the paper deadline; check CVPR's rules and switch later if the frozen runs support RMFA.

| Phase | Models | Datasets | Budgets | Arms | Seeds | Runs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 dev | π³ | 3 dev | 4k | A2, A3, A4 × 24 configs | 42 | 78 |
| 2 main | 3 | 5 | 4k, 8k | A0–A8 | 1000 | 270 |
| 3 seeds | 3 | 5 | 4k | A1, A2, A3, A4, A8 | 2000, 3000 | 150 |

**Launch order inside phase 2,** so partial results are still usable:

1. A1, A2, A3, A8 at 4k for π³ and DA3. This alone answers H4, the main claim.
2. A4 at 4k for π³ and DA3. This answers H1 and H2.
3. A5, A6, A7 at 4k, then VGGT at 4k.
4. Everything at 8k, then A0.

**Time estimate.** Fill in after phase 0: time one A1 run and one A3 run per model, then multiply. A3, A6 and A8 cost roughly ρ² more in global attention. If the total exceeds 3 GPU-days, cut 8k from A5–A7 first, then cut phase 3 to seed 2000 only.

**Rule during phase 2:** do not change code or configs. A bug found mid-run means fixing it, bumping the config hash, and rerunning every affected run, noted in `runs/s23/CHANGELOG.md`.

## 12. Outputs

Every run writes one self-describing `result.json`; all tables and figures are generated from those files only, never edited by hand.

**Folder layout:** `runs/s23/<model>/<dataset>/<budget>/<arm>/seed<seed>/`

**`result.json` fields:**

```json
{
  "study": "S23", "model": "pi3", "dataset": "eth3d",
  "budget": 4096, "arm": "A4", "seed": 1000,
  "config_hash": "...", "views_hash": "...", "git_commit": "...",
  "per_scene": [
    {"scene": "courtyard", "auc3": 0.0, "auc5": 0.0, "auc30": 0.0,
     "f": 0.0, "precision": 0.0, "recall": 0.0, "chamfer_mm": null,
     "latency_s": 0.0, "peak_mem_gb": 0.0}
  ],
  "global_tokens": 4096, "frame_tokens_total": 0,
  "flops": {"frame": 0, "global": 0, "heads": 0},
  "probes": {"entropy_file": "entropy.npz", "drift_file": "drift.npz"}
}
```

**Other per-run files:** `entropy.npz` (layers × heads × views), `drift.npz` (layers × small views), `preds.npz` (optional), `log.txt`.

**Aggregates** (`runs/s23/summary/`):

| File | Contents |
| --- | --- |
| `cells.csv` | Mean per model × dataset × budget × arm × seed, all metrics |
| `paired.csv` | Every planned pair (A4−A2, A3−A2, A8−A3, A6−A4, A5−A4, A7−A4, best−A0), mean diff, bootstrap CI, raw and Holm p |
| `recovery.csv` | R per model and budget with CI |
| `cost.csv` | Latency, memory, FLOPs per arm |
| `predictions_check.md` | Each of H1–H4 marked held or failed, with the number |

**Figures** (`runs/s23/figures/`), each meant for the paper:

- **Recovery chart:** per model, pose AUC for A1, A2, A4, A3, A8 at 4k, with intervals. The single figure that shows the main claim and the method together.
- **Localization chart:** A4 vs A6 vs A3, showing whether the gain lives in frame or global attention.
- **Per-layer entropy and drift:** small-view curves per arm against the uniform reference, one panel per model.
- **Cost vs accuracy:** latency against pose AUC, every arm, so reviewers see RMFA keeps the mix's cost.
- **Main table:** A0, A2, A4 (and A3 as a grey ceiling) across datasets, pose and reconstruction.

## 13. Decision rules

Every likely outcome maps to a paper you can still submit; decide the framing from this table, not after looking for the most flattering cut.

| Outcome (seed 1000, pooled pose AUC@3, 4k) | What it means | Paper framing |
| --- | --- | --- |
| R ≥ 0.5 on π³ and DA3 improves over A2 | Artefact sits in frame attention and RMFA removes it cheaply | **Method paper:** diagnosis → localization → RMFA, with DA3 as the headline |
| R ≥ 0.5 on π³ only; DA3 flat or worse | Frame attention is the site for π³; DA3 breaks elsewhere | Diagnosis paper with RMFA as a fix for one architecture family; DA3 as an open problem |
| R < 0.5 and A6 recovers more than A4 | Damage is mainly in global attention | Diagnosis paper; localization result replaces the method; `pi_hr` stays the simple fix |
| R < 0.5 and A6 also low; only A3 recovers | Damage needs matching in both steps | Diagnosis paper; state that the fix must act in both attention types |
| A5 ≈ A4 | Interpolated tokens are enough | Prefer A5 as the method: cheaper, no pixel upsampling |
| A7 ≈ A4 | RMFA does not need `pi_hr` | Report RMFA alone as simpler |
| H4 fails: A8 − A3 large on frozen seeds | Real information matters more than seed 42 suggested | **Stop and rethink the main claim** before writing; report it honestly |
| Seeds 2000/3000 disagree in sign with 1000 | Result is not stable | Weaken claims to the cells that agree; report the disagreement |

**VGGT** is expected to show no effect under any arm except A8 (H3). If RMFA helps VGGT, report it, but it does not change the framing.

**Reconstruction** results are reported whatever happens. If RMFA helps pose but hurts reconstruction (or the reverse), say so in the main table rather than splitting it into separate claims.

## 14. Risks and 7-day schedule

The biggest risk is spending the week debugging RMFA on DA3 and losing the frozen main-claim result; the schedule protects H4 first.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| DA3 has no clean frame/global split | RMFA undefined or wrong on DA3 | Map layers on day 1; if unclear, run DA3 for A1–A3 and A8 only and state RMFA applies to alternating-attention models |
| A3 uses more global tokens than the mix | Upsampling gain partly from extra tokens, not removing the artefact | A4 and A6 separate the two; say it explicitly in the paper |
| Lifting Δ with bilinear blurs updates | RMFA underperforms for a fixable reason | `lift_op` is tuned in phase 1; also log drift to see where it diverges |
| Dev and held-out leakage | Reviewers distrust the results | Phase 1 on 3 dev datasets, seed 42 only; predictions committed before seed 1000 |
| A3/A8 at 8k does not fit in memory | Missing ceilings at 8k | Run ceilings at 4k only and say so |
| G-ray (arXiv 2609.15018) overlaps with position alignment | Novelty challenge on `pi_hr` | Read it day 1; position RMFA and the diagnosis, not `pi_hr`, as the contribution |
| Code change mid phase 2 | Mixed versions in one table | Config hash in every result; changelog; rerun affected runs |

**Schedule** (day 1 = today):

1. **Day 1:** RMFA engine, `grid.py`, π³ adapter; T1–T10 on π³. Run the quick check (section 11) and register the abstract. Map DA3 layers. Read G-ray.
2. **Day 2:** VGGT and DA3 adapters; tests pass on all three; view lists generated; time one run per arm for the estimate.
3. **Day 3:** phase 1 dev sweep; freeze; write and commit `predictions.md`; start phase 2 in launch order.
4. **Day 4:** phase 2 continues; aggregate partial results as they land; draft method and setup sections.
5. **Day 5:** phase 2 finishes; start phase 3; draft figures; pick the framing from section 13.
6. **Day 6:** phase 3 finishes; final aggregates and figures; write results and limitations.
7. **Day 7:** full read-through, check every number against `summary/` CSVs, supplementary material, submit.

**Go/no-go checkpoint, end of day 3:** if T1 still fails for a model, drop RMFA for that model and run only A0–A3 and A8 for it. The main claim must be finished by day 5 whatever happens to the method.
