# Study status at report checkpoint

The rebuilt Pi3 and VGGT development/control pipelines work. The full S23 study is unfinished. Latest evidence is exploratory seed 42 on five 7-Scenes scenes at nominal 4k, not frozen paper evidence.

## Completed

- Original Pi3 and VGGT-1B weights pinned and checksum-verified, separate model environments, adapters and shared two-grid inference.
- Exact full-checkpoint native fp32 identity on both models and both checked scenes.
- Complete eight-arm matrices on both models, under the preserved global-only protocol and separately under the full-scope nominal-rho protocol. No failed or excluded scenes in those matrices.
- Saved camera matrices/errors, cost logs, complete console/environment records, summaries, figures, every periodic/final archive, outer/member checksum verification and local metric rescoring.
- Independent balance/deadline/backup monitor, live smoke test, automatic shutdown, no API key in Git or pod environment.
- Official DA3-BENCH ETH3D and 7-Scenes archives downloaded and verified locally, 18.66 GB total. No paid GPU used for these downloads.

## Correction that matters

The initial position fix was restricted to global attention and effective grid ratios. Inspection of native Pi3 showed that it reuses positions across frame/global blocks and head decoders. The explicit full-scope nominal-rho interpretation gives a stronger position-only baseline. The appropriate Pi3 default recovery headline is now 37.6%, not the earlier restricted-baseline 53.8%. Both protocols and the reason for the change are retained.

See development/fullscope/README.md for the latest table and architecture-controls.png. Default RMFA has a Pi3 gain but does not yet recover half the upsampling gain; no reliable VGGT gain is established. Provisional framing is diagnosis, not a demonstrated general method.

## Not completed

- DA3 adapter is built and has exact small-grid CPU checkpoint identity; full-resolution CUDA identity remains pending.
- Full five-dataset common conversion/evaluation, including ground-truth consistency checks.
- Historical S7 best-uniform/mix reconstruction and T9 reproduction; those inputs were not supplied.
- A0 baseline selection, 24-configuration Pi3 development sweep, committed frozen configurations and predictions.
- Seeds 1000/2000/3000, all-model/all-dataset/budget frozen matrix, reconstruction, entropy/drift/FLOP probes, final paper tables.

## Restart entry point

Read ../STUDY_CHECKLIST.md first. It records exact current resources, verified archives and next commands. AGENTS.md directs future sessions there instead of rediscovering the project.

Fresh execution tooling now includes the available development manifests, native benchmark pose scoring, a 24-config sweep, immutable freeze and committed-artifact checks, entropy/drift probes, native reconstruction bridge, stratified bootstrap/sign-flip inference and the 270+150 frozen/extra matrix. These paths are built but the full study is not executed. ScanNet++ permission, historical T9 and actual-data validation remain explicit gates. No frozen model result has been opened.

## Cloud and credit

Latest verified balance: $11.6413142792 at 2026-11-11T13:11:51.963512+00:00. No GPU compute is running. The user's earlier pod u64lksix8j610q and volume oq84uxp88q were deleted with explicit authorization. Setup pods v3ff0vv5jb5lm2 and r7jn02w51p7eun are also deleted and confirmed 404.

Workflow pod yvcfcbk52vr2x1 is EXITED with a retained 40 GB workspace. The watchdog stopped it when Git source staging temporarily broke the backup export. Periodic archives are verified locally, but the final quiescent archive is not. Restart is currently refused for host GPU capacity. Storage billing continues until recovery and deletion; this is not a zero-resource handoff. No separate network volume exists. See the checklist for the exact recovery command order.
