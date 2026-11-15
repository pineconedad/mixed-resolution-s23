# Paper 2 (*Views or Pixels?*): where it stands and what to do before the deadline

15 Nov 2026. Based on `STUDY_REPORT.md` (paper-2 repo, commit `c1f6fb7`) and on the paper-1 runs in this repo (`reports/salvation/`, `docs/salvation_findings.md`). I have not seen the paper-2 repo or its draft; commands for it are written as specifications for whoever works there.

## 1. Bottom line

Paper 2 is submittable and is the stronger of the two papers. Its pose evidence is complete on a reporting seed that no choice ever saw, across 3 models and 5 datasets, with failed predictions reported. **Submit paper 2; do not spend the remaining time on a standalone paper 1.** Three things can still hurt it: the VGGT conclusions, the reconstruction claims, and its own verification rule.

## 2. What this repo's runs add (independent code, metric and seed)

| Paper-2 finding | Paper-1 evidence | Effect on paper 2 |
| --- | --- | --- |
| F1 mixing collapses the models | Reproduced on 7-Scenes, ETH3D, HiRoom | Supports |
| F2 representation, not information (Pi3) | Upsampled matches real pixels on ETH3D and HiRoom | Supports |
| F4 in-model fixes help little | Seven more fixes fail: RMFA, real-pixel frame attention, lifted tokens, pooled keys, logit bias, token-statistics matching, pose ensembling | Supports; usable as extra negative controls |
| F5 "VGGT differs" | **VGGT peaks at 504 px width (about 1,300 tokens per view) and is near zero below it.** HiRoom AUC@3 by side 364/392/504/546/588/630: 6.0, 10.7, 45.5, 41.5, 31.3, 19.6. ETH3D shows the same shape | **Threat.** Paper 2 runs VGGT at about 470 and about 1,024 tokens per view, both below the peak |
| (not in paper 2) | For Pi3 the mix is worse than every view at the small size: HiRoom 10.0 vs 27.9, 19/19 scenes; ETH3D 5.8 vs 13.3 | A stronger headline than "worse than dropping"; paper 2 can check it from existing jobs |
| (not in paper 2) | 7-Scenes cannot resolve rotations below about 1.6 to 2 degrees: SIFT two-view, Pi3 and VGGT share that median error on well-matched pairs | Caveat for any claim resting on 7-Scenes AUC@3 alone |

These numbers come from a different pipeline (seed 42, a different pose AUC). They must not be put in paper 2's tables. They can be cited as an independent check or placed in the supplementary, clearly labelled.

## 3. The three risks, in order

1. **VGGT is evaluated off its training size.** The report already notes VGGT scores 2 to 6 AUC@3 with all views sharp at the low token level. Findings about VGGT ("hurt by uneven detail", "needs tokens per view", "drop the small views") may be properties of the operating point, not of the model. As written, F5 and the VGGT row of the recipe are not safe.
2. **Reconstruction is mostly exploratory.** Only S20 passed its gate. S26, S27, S29 and S30 have no usable reconstruction readout.
3. **The repo's own rule is unmet.** No study from S1 to S31 is verified-2, and tables are typed by hand.

## 4. What to do, in priority order

### A. No GPU, do first (about half a day)

1. **Reword every VGGT claim** to "at the token levels tested, which are below VGGT's training resolution". Move "VGGT's cause is narrowed" to "VGGT is confounded by its operating range". Keep the numbers; change the interpretation.
2. **Add one contrast from existing S23 jobs: down vs aligned** (all 12 views small vs the aligned mix), W/L per model, all-pair and common-frame. No new jobs, only a readout. If Pi3 and DA3 win most cells, the abstract can say the mix is worse than shrinking everything.
3. **Make the paper pose-first.** Reconstruction claims only from S20 (complete) and from S23, S25, S28 with the unfused counts stated beside each table. Remove reconstruction statements that depend on S26, S27, S29, S30.
4. **Verify the headline numbers.** One fresh independent check of the S23 and S25 pose readouts against raw job outputs, then diff every hand-typed table cell in the `.tex` against the readout files. This is the minimum that honours the verified-2 rule for the claims in the abstract.
5. **Add the 7-Scenes caveat** to limitations, and check that no headline claim depends on 7-Scenes alone.
6. **State the limitations the report lists**, plus the VGGT operating range.

### B. One new study, only if a GPU is free for a few hours (the hail mary)

Pose only, reporting seed 1000, same frames and scoring as S23. Commit the protocol and predictions before any job runs, like every earlier study.

**S32: VGGT at its training resolution.**

- **Part 1, size curve.** 12 views at one size, sharp side 364, 448, 518, 588 (the long side for non-square data), all three models, HiRoom (19 scenes) and ETH3D.
  - Prediction 1: VGGT's AUC@3 peaks at 518 and is lower at 448 and at 588 in both datasets.
  - Prediction 2: Pi3 changes by less than 5 AUC@3 between 448 and 588.
- **Part 2, the mix at 518.** 4 sharp at 518 + 8 small at 259 (ratio 2), strategies as-is, aligned, up, down, true, drop, VGGT only, same two datasets.
  - Prediction 3: true minus up stays significant for VGGT (it uses real detail).
  - Prediction 4: aligned stays below drop on common frames.
- **Why it is worth it:** it turns "VGGT is unexplained" into a tested statement either way. If the VGGT pattern persists at 518, F5 stands on firm ground. If it changes, the paper reports that VGGT's behaviour at low budgets was an operating-range effect, which is an honest and useful result.
- **Cost:** 12 views at 518 is about 16k tokens, which fit on the 5090 in this repo's runs. Part 1 is 4 sizes x 3 models x 2 datasets; part 2 is 6 strategies x 2 datasets. Pose jobs only, no fusion.
- **If there is no time to finish it:** do not start it. A half-run study on the reporting seed is worse than a stated limitation.

### C. Reconstruction fusion, in the background only

Resume `bash paper/scripts/chain_fuse_priority.sh 6 10` for S23 first (72 unfused), then S25. Do not wait for it: DTU fusion takes about 12 hours per study. Whatever is not fused by writing time stays labelled exploratory.

## 5. Paper 1

- Do not submit it standalone. Its method failed every pre-stated gate, and its surviving analysis rests on one development seed.
- Offer its results to paper 2 as supplementary material: the independent replication, the seven failed fixes, and the VGGT size curve. Check first whether CVPR still allows the author list to change; if not, cite nothing from it and keep it for a later venue with frozen runs.
- The registered paper-1 abstract claims a method and a mixing benefit that both papers now contradict. If paper 1 is not submitted, withdraw it so the two do not conflict.

## 6. Order for the remaining time

| When | Do |
| --- | --- |
| Now | A1, A2, A3; start C in the background |
| Next | Decide on S32 within the hour; if yes, commit predictions and launch |
| While jobs run | A4, A5, A6; write |
| Last half day | Freeze numbers, regenerate or diff tables, check every co-author's OpenReview profile, submit early |
