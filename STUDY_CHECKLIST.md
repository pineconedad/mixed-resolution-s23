# S23 execution checklist

Read this file first after interruption. Update it before provisioning, after a completed batch, and before ending a session. The two root Markdown plans remain the specification. Never mark historical reproduction or missing datasets as passed.

## Current checkpoint

- TRACK D RESULT (reports/salvation/aspect.txt, 75 paired scene-draws per model): aspect lead is DEAD as an explanation of the mix failure. Mix damage (A8 - A2) is unchanged with exact 4:3 sizes: Pi3 7.34 -> 7.55, VGGT 5.41 -> 7.71. All arms gain 1-3 AUC@3 with the 4:3 sizes, but those sizes also carry more tokens, so the gain cannot be attributed to aspect.
- STATE AFTER TRACKS A-D: no in-model method beats uniform resize at matched compute on Pi3 or VGGT. The only supported paper-1 claim is the negative result plus the four-cause decomposition. No further track is built or planned.
- TRACK C RESULT (commit 613a4ea, 70 fresh scene-draws per model, all steps ok): Gate C FAILED on Pi3 and VGGT. A10 trails A3 by 7.1 AUC@3 on Pi3; kv pooling costs about 2. Uniform resize beats every in-model arm at matched compute on both models. Reports: reports/salvation/night-*.txt.
- OPEN LEAD: Pi3 uniform curve is non-monotonic and peaks at exactly-4:3 sizes (A0_7k 14.36 > A8 12.64). Study sizes are anisotropic (sharp 1.304, small 1.25). Hypothesis only; VGGT curve is monotonic.
- TRACK D BUILT, NOT RUN: `sizes` override in scripts/04_run.py and scripts/salvation_aspect.py. NEXT COMMAND (user, local GPU): `python scripts/salvation_aspect.py --models pi3 vggt --push`, then read reports/salvation/aspect.txt.
- TRACK C BUILT, NOT RUN (restructured method after gates A and B failed): arm A10 (encode each view at its own size, lift tokens, sharp-grid decoder), option `kv_pool=mix` (dense queries, pooled keys/values for small views; `asymmetric_block` in mixres/rmfa/calibration.py), A0 at other budgets via `uniform_budget`, development draws seeds 43-52 via scripts/prepare_quick.py (`study: salvation_draw`, frames bound to seed 42 and frozen seeds excluded). The exploratory runner now accepts any `salvation_*` study with seed 42-999; seeds from 1000 up remain frozen-only.
- NEXT COMMAND (user, local GPU, unattended): `python scripts/salvation_night.py --push`. Reads: runs/salvation_night/reports/night-pi3.txt. Gate C rule is pre-stated in scripts/salvation_night_report.py; do not edit it after results exist.
- Not verified here: the 7-Scenes range downloads for new draws and all GPU paths (CPU tiny-model tests only). kv_pool has no checkpoint-backed identity gate of its own; it falls back to the gated path when no view is pooled.
- SALVATION BRANCH (`salvation`, from e1876b0). The paper direction is being re-decided by two exploratory kill tests on a user-operated local GPU. Run instructions are at the top of README.md. No cloud action is involved or authorized.
- Built, not yet run on any GPU: track A zero-cost corrections (`calibrate`, `logit_bias` in `run_engine`, code in mixres/rmfa/calibration.py), track B arm A9 (real pixels, sharp frame attention, pooled global attention), A0 and named variants in scripts/04_run.py, configs/salvation-{a,b}-{pi3,vggt}.yaml, scripts/salvation.sh, scripts/salvation_report.py, tests/test_salvation.py.
- New options default to off. They are wired into the exploratory seed-42 runner only. They are NOT in StudyRunner METHOD_DEFAULTS, the 24-config sweep, freeze, or the 270/150 matrix; add them there only after a gate passes.
- Adding A9 and calibration.py changes `sanity_source_hash`, so every S23 identity gate must be rerun before any S23 cell.
- Pass rule is pre-stated in scripts/salvation_report.py (+2 AUC@3 and 4 of 5 scene wins against A2 for track A, against A0 for track B). Do not edit it after results exist.
- RESULT (commit 5bf1164, RTX 5090, torch 2.8.0+cu128, clean tree 1518afa, Pi3, 7-Scenes 5 scenes, seed 42, exploratory): BOTH GATES FAILED. Reports: runs/salvation_{a,b}/pi3/summary/report.txt. Archived A1/A2/A4/A3/A8 means reproduced.
  - Gate A FAIL: best zero-cost variant A2_bias_both +1.20 AUC@3 over A2 (4/5 scenes), below the +2 rule and below A0. Moment calibration hurts (-2.1); measured group shift is only about 0.2 SD.
  - Gate B FAIL: A9 8.69 vs A0 7.47 AUC@3 (+1.21, 2/5 scenes) at about twice the frame tokens (8340 vs 4092). A9 is about equal to A4; head_grid=hi changes nothing. Every pooled-global arm collapses on office (AUC@30 about 17 vs 71 for A0).
  - A0 first measurement: 7.47 AUC@3, 80.0 AUC@30. A9 vs A8 (13.17) isolates pooled global attention as costing about 4.5 AUC@3, so "damage is not in global attention" is not supported.
  - Per the pre-stated rule: no RMFA method paper from this repo this cycle. VGGT salvation configs were not run. Do not promote the new options into StudyRunner.
- SUPERSEDED: NEXT COMMAND (user, local GPU): `bash scripts/salvation.sh a pi3`. Run `bash scripts/salvation.sh b pi3` only if Gate A fails. Then return runs/salvation_*/pi3/summary/report.txt.
- Known limits: logit bias supports Pi3 and VGGT blocks only and raises on DA3; VGGT corrections pass tiny-model CPU tests only (tests_vggt has one pre-existing A0 failure); track B cost is not matched to A0 (the report prints the latency ratio).
- The entries below predate this branch and describe the cloud sessions.


- ACTIVE NEW POD hassuectufut86, created 2026-11-11T18:42:45.022Z, US-MO-1 H100 SXM $3.99/hour, 30 GB container + 40 GB workspace. Marker devfull20261111a. $2 session cap/$3 reserve. Launch independent watchdog before setup. Source pack artifacts/development-pack-next/source.tar.gz, metadata.pack. NEW PACK includes all three development manifests and local approval receipt. Retained old yvcfcbk52vr2x1 remains stopped separately; do not confuse IDs.
- UPDATE: verified guard handoff to artifacts/control/devfull20261111a-extended/receipt.json. Session ceiling extended to $4 at unchanged rate, reserve $3. Original supervisor exits after handoff but remote worker continues; replacement watcher detects remote quiescent archives and automatically stops/deletes after backup. Follow the EXTENDED final.json, not the original supervisor exit.
- Active worker passed dependency checks, verified DA3 weights and is staging selected inputs. Range staging avoids full 30 GB ZIP downloads; all selected image hashes are checked. An actual selected 7-Scenes member was verified over HTTP ranges. CPU input cache added before any sweep output to avoid repeated JPEG decode. Runtime overlays: mixres/data/remote_zip.py, scripts/stage_development.py, mixres/study_runner.py uploaded before gates/inference.
- Development manifest count is 35 scenes: ETH3D 8 + ScanNet++ 20 + 7-Scenes 7. No CUDA identity/sweep completion is claimed yet. Read remote runs/control/devfull20261111a/state.json and console.log for exact phase.

- Local branch is main, last published commit 85680d5. Substantial new source changes are uncommitted. No new benchmark inference completed during the interrupted setup.
- Resume read confirmed zero pods/volumes. The interrupted e9y82zq7kxi4ss is gone.
- Pod v3ff0vv5jb5lm2 was stopped by watchdog before setup, then deleted and confirmed 404. Cause was Python 3.10 rejecting two-digit fractional seconds in an eagerly evaluated timestamp fallback. Fixed scripts/watch_pod.py.
- H100 r7jn02w51p7eun stopped on initial backup failure before setup, then deleted and confirmed 404. Corrected OpenSSH known-hosts filename quoting for workspace paths with spaces. No model batch ran.
- RETAINED STOPPED pod yvcfcbk52vr2x1, created 2026-11-11T12:50:00.16Z, US-MO-1 H100 SXM. No GPU compute billing. Its 40 GB workspace remains billed because the latest quiescent backup is unverified. Do not delete until recovery/backup. Restart was refused for host capacity. No separate network volume exists.
- Initial backup verified: artifacts/control/devresume20261111i/session-20261111T125650973708Z.tar.gz, SHA256 d21b409f77e79b6d30efe388910bcee1bdc3db7067532405131c504e276bcba1, 18 members. Independent watchdog ready, deadline epoch 1794403605.4817927. Worker starts DA3 CUDA identity before Pi3 identity/sweep. Read state.json or latest archived console for progress.
- Latest verified balance: $11.6413142792 at 2026-11-11T13:11:51.963512+00:00. Preserve the existing $3 reserve.
- Worker stopped during model source staging, before any recorded CUDA identity or development result. Periodic archives are locally verified but not final/quiescent. Last verified archive: reports/control/devresume20261111i/session-20261111T125914441371Z.tar.gz, SHA256 9620abffb7603a7882a6332db8c2ab0f54641288150d72a8315fcc0c70db4cb0, 20 members.
- Exact failure: export_session called root git-status while submodule vggt was mid-clone, causing fatal missing .git/modules/third_party/vggt. Fixed exporter to preserve data even if Git inspection fails; added pinned sparse source preparation and bounded controller termination on stop. No additional smoke tests were run.
- Previous remote setup reached Python 3.10 / torch 2.5.1 installation, Pi3 weights and ETH3D/7-Scenes downloads/extraction. Its final `python -m pip check` failed because uv-created environments do not include pip. Use `uv pip check --python .venv/bin/python`, not reinstalling the environment.
- Do not repeat archived Pi3/VGGT exploratory matrices or watchdog smoke tests.

## Completed locally

- [x] Pi3/VGGT adapters and checkpoint-backed fp32 identity, prior exploratory A1-A8 controls. Archives under reports/sessions; immutable historical results.
- [x] ETH3D and 7-Scenes official DA3-BENCH archives verified and extracted under third_party/Depth-Anything-3/workspace/benchmark_dataset.
- [x] Fresh baseline choices documented in docs/implementation.md and configs/s23.yaml, not historical S7 reproduction.
- [x] Development/frozen metadata manifests for both available datasets, four seeds, two budgets under runs/views/s23. Frozen frames exclude seed-42 frames.
- [x] Matched native ETH3D filename-suffix exclusions before inference. Eight ETH3D and seven 7-Scenes scenes retained. pipes=14, relief=19 and relief_2=20 cannot supply disjoint 12-view sets; dropped consistently. Old metadata retained in runs/views/_superseded/eth3d-before-native-filters.
- [x] Development contact sheets generated and visually inspected, reports/data/*contact.png.
- [x] Algebraic depth backprojection/reprojection checks on three frames per available dataset, reports/data/geometry-checks.json. This is a convention round-trip check, not independent mesh registration validation.
- [x] DA3-LARGE-1.1 weights pinned/downloaded locally: revision 0e109ae307c5982f319a67cf6f9f99ccdc0ec97c, SHA256 739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64. Local isolated .venv-da3 loads network structurally.
- [x] Draft DA3 adapter, S23 cell runner, 24-config sweep, freeze, aggregation and pose/reconstruction helpers exist. These are NOT all scientifically validated or complete.
- [x] Native reconstruction bridge built using DA3 fuse3d/eval3d, separate recon.json/log, 15 focused bridge tests. Pi3 predicted intrinsics are still missing, so Pi3 reconstruction is explicitly unsupported rather than substituting GT.
- [x] Entropy/drift observers implemented and integrated as untimed seed-1000 passes; no extra-seed probes. Main/extra matrix enumerates exactly 270/150 cells. A0 drift is unavailable because its grid differs.
- [x] Stratified equal-dataset bootstrap, paired sign-flip p-values, Holm family 30, missing-cell audits and truthful predictions_check.md implemented; 24 focused tests passed.
- [x] Fixed head_grid=hi to read persistent high-grid detail, not lifted pooled tokens, before the first development sweep. Default-lo archived results are unaffected.

## Next exact actions

1. Recover retained yvcfcbk52vr2x1 first. Try `pod-action start` only after reading its state and current balance; if host capacity still refuses, keep stopped rather than creating more setup-only rentals. Once reachable, upload corrected scripts/export_session.py and scripts/recover_interrupted_session.py, run recovery for devresume20261111i, export/download/verify a quiescent archive, then delete the pod. No current background worker/watchdog is running.
2. Rebuild the allowlisted source pack using `python scripts/pack_development.py`. Next guarded worker command is `python3 scripts/run_session.py --session-id NEW_ALPHANUMERIC_ID --task development --model pi3 --skip-unit-tests`. Activate its independent watchdog before invoking this command. It stages sparse DA3/Pi3 sources, runs DA3 CUDA identity, then Pi3 identity and the two-dataset sweep.
3. Run DA3 native fp32 identity once on a guarded GPU. If it fails, fix the evidenced issue; no DA3 metrics until it passes.
4. Run Pi3 24-config development sweep on available ETH3D/7-Scenes with scene-level resume and verified periodic backups. Label it incomplete three-dataset tuning until ScanNet++ permission is supplied.
5. Finish reconstruction/probes, full matrix launcher and inference provenance. Freeze only after the required development datasets are complete, or obtain an explicit reduced-study amendment. Commit predictions/configs before seed>=1000 results. User authorizes autonomous compute/cleanup, not fabricated gates.
6. Download/verify outputs locally, terminate each owned pod and any created volumes, and update this checklist with exact artifacts and next command.

## Known build gaps and risks

- [ ] DA3 CUDA study identity remains unverified. Small-grid CPU check PASS: exactly one normalized native vs actual run_engine A1 comparison, rho=1, three synthetic 56x56 views, fp32 in .venv-da3. All seven output fields have max absolute error 0.0; report reports/data/da3-local-identity.json includes checkpoint/source hashes and per-view errors. No mixres changes or GPU runs for this check. Normalization, head fp32/autocast and pose-convention fixes are present. Next action remains step 3, the guarded CUDA identity gate; do not repeat this CPU check as a study gate.
- [ ] DA3 pi_hr A2 is explicitly a no-op because native global RoPE is spatially nondifferentiating. Do not pretend this reproduces missing S5 DA3 implementation.
- [ ] Current fixed-12-view A0 uses a rounded nominal budget, not necessarily exactly the same actual spatial tokens as mixed arms. Label counts honestly and resolve fairness before frozen results.
- [x] StudyRunner sanity receipts bind adapter/engine source, model metadata and checkpoint. Frozen execution checks committed predictions/configs and hashes.
- [x] Freeze prevalidates all artifacts and publishes a staged directory rather than partially writing files.
- [ ] DA3 analytic costs omit local prefix; Pi3 decoder dimension assumption needs source confirmation. Head FLOPs are null. Do not call partial analytic counts total model FLOPs.
- [ ] Native reconstruction bridge needs actual-data validation; Pi3 needs a defensible predicted-intrinsics path. Probe integration needs first real seed-1000 execution only after freeze. Figure drafts still need interval and entropy/drift panels from real CSV/NPZ outputs.
- [x] Pi3 reconstruction export now derives depth/K from predicted local_points using native recover_intrinsic_from_rays_d, with explicit provenance and no GT K. Focused synthetic export tests passed. Actual-data reconstruction remains pending.
- [x] ScanNet++ user permission received through local handoff; archive verified and extracted. Permission receipt remains local and is included explicitly in authorized worker packs, never credentials.
- [x] Update on resume: user confirmed approval/terms through local handoff form at 2026-11-11T15:22:09.223769+00:00. Receipt artifacts/permissions/scannetpp.json. Archive checksum verified and extracted automatically. All 20 scenes now have four-seed/two-budget manifests. Native undistort/ROI/resize and calibrated intrinsics basis implemented; synthetic check passed. Sweep/staging now includes all three development datasets.
- [ ] Historical S7 settings and T9 targets absent permanently. Report T9 unavailable, not passed.
- [ ] DTU/HiRoom remain sealed. No seed>=1000 model results exist.
- [x] Remaining dataset metadata loaders, pinned archive receipts, ScanNet++ permission gate and committed-predictions held-out gate implemented without downloading restricted/held-out data.
- [x] Held-out hash appendix workflow implemented: after primary freeze is committed, generate held-out metadata, run `python scripts/03_freeze.py --bind-heldout-views`, then commit appendix before inference. Method configs remain unchanged.

## Deployment rules

- Use existing runpod_control/watch_pod safety rather than another smoke test. Watchdog must start before paid setup and have an independent credential/deadline. No API key in a transfer bundle or pod.
- Source transfer must use an explicit allowlist. Do not tar the whole workspace: it can include ignored secrets, .venvs and macOS metadata. Use COPYFILE_DISABLE=1 and extraction --no-same-owner.
- Transfer sampled inputs only when possible; full extracted ETH3D/7-Scenes is ~53 GB. Do not redownload on every short rental if a sampled input pack suffices.
- Latest successful GPU shape: secure A40 in EU-SE-1, $0.59/hour, official image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404. Always re-read stock/pricing.
- Slow SSH transfer workaround: generate a metadata-only Git pack with `git rev-list --objects --no-object-names --filter=blob:none HEAD | git pack-objects --stdout`. Initialize remote repo, index-pack, update-ref to the recorded HEAD, read-tree, then extract allowlisted source. Model submodules remain full clean public pinned clones. Do not upload the 35 MB old report bundle over a 12 KB/s route.
- Never leave paid compute running at handoff without an active independently enforced shutdown deadline. Download and checksum artifacts before deletion.
