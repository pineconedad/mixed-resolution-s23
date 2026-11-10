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

- DA3 adapter and validated study checkpoint.
- Full five-dataset common conversion/evaluation, including ground-truth consistency checks.
- Historical S7 best-uniform/mix reconstruction and T9 reproduction; those inputs were not supplied.
- A0 baseline selection, 24-configuration Pi3 development sweep, committed frozen configurations and predictions.
- Seeds 1000/2000/3000, all-model/all-dataset/budget frozen matrix, reconstruction, entropy/drift/FLOP probes, final paper tables.

The next research step is common development-data preparation followed by the specified development sweep and DA3 integration, not declaring the current controls to be the full main study.

## Cloud and credit

Latest verified balance: $13.8291713412 at 2026-11-10T20:10:54.144810+00:00. Credits are not exhausted. No inference job or GPU is running. All workflow-created completed temporary pods are deleted. User pod u64lksix8j610q is stopped with a 20 GB workspace. User volume oq84uxp88q is 10 GB, unattached, and independent. Those retained resources cost approximately $4.70/month in storage and need explicit deletion authorization for zero residual storage billing.
