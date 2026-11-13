# Paper 1: all findings in one place, and the best angle

Nov 13, 2026. Sources: `reports/salvation/night-pi3.txt`, `night-vggt.txt`, `aspect.txt`, `track-*-vggt.txt`, `runs/salvation_a`, `runs/salvation_b`. All numbers are exploratory development data on 7-Scenes (seed 42 plus draws 43-52, 12 views, 4 sharp + 8 small). Nothing here is a frozen result. Pose AUC@3 unless stated.

## 1. Findings

Strength: **solid** = 70+ paired scene-draws with an interval excluding zero; **weak** = 5 scenes or a confounded comparison.

| # | Finding | Evidence | Strength |
| --- | --- | --- | --- |
| F1 | Uniform resize beats every in-model fix at matched compute, on both models | Pi3: A0_5k 12.66 at 10.3k GFLOPs vs RMFA 7.42 at 13.0k. VGGT: A0_6k 3.71 at 15.9k vs A9 3.28 at 15.5k | Solid |
| F2 | On Pi3, a smaller uniform size beats the "full resolution" ceiling | A0_7k (294x392) 14.36 vs A8 (322x420) 12.64, with 15% fewer tokens | Solid, cause unknown |
| F3 | Pi3 accuracy is not monotonic in resolution | Uniform 4k/5k/6k/7k/8.3k tokens: 9.36, 12.66, 11.70, 14.36, 12.64. VGGT is monotonic: 1.10, 2.42, 3.71, 5.61, 7.13 | Solid |
| F4 | The mix damage is not an aspect-ratio artifact | A8 - A2 with original vs exact 4:3 sizes: Pi3 7.34 vs 7.55, VGGT 5.41 vs 7.71 | Solid |
| F5 | Position alignment is the largest single fix for the mix | A1 -> A2: Pi3 1.43 -> 5.32, VGGT 0.51 -> 1.68; AUC@30 +23 and +32 | Solid |
| F6 | Global attention needs dense queries; pooling the stream is harmful | A9 vs A8: Pi3 7.54 vs 12.64. Pooling only keys still costs about 2 (A8_kv - A8: Pi3 -2.26, VGGT -1.95) | Solid |
| F7 | Lifted tokens are not upsampled pixels | A10 - A3: Pi3 -7.06, VGGT -1.45. They only look equal when pooled straight back (A5 vs A4) | Solid |
| F8 | Pi3 does not use the extra pixels; VGGT does | A8 - A3: Pi3 +0.4, VGGT +3.1. A9 vs A4 (real vs upsampled in frame attention): Pi3 +0.1, VGGT +1.9 | Solid |
| F9 | The two models rank the mix differently against uniform | At equal 4k tokens VGGT prefers the mix (A2 1.68 vs A0 1.10); Pi3 prefers uniform (5.32 vs 9.36) | Solid |
| F10 | Token statistics do not drift between small and sharp views | Measured shift about 0.2 SD; rescaling hurts by 2 | Weak (5 scenes) |
| F11 | Logit bias for coarse tokens helps VGGT slightly, not Pi3 | A2_bias_both - A2: VGGT +0.78 [0.51, 1.06]; Pi3 +0.02 | Solid, small |
| F12 | RMFA collapses coarse accuracy | A4 AUC@30: Pi3 72.4 vs A0 79.4; VGGT 49.3 vs 62.5 | Solid |
| F13 | Exact 4:3 sizes score 1-3 points higher on every arm | Track D | Weak: token counts also rose |
| F14 | Fusing predictions from several input sizes never beats the best member | 20 ensembles per model; best Pi3 pair 13.74 vs 14.36 alone; errors across sizes are correlated | Solid |
| F15 | Classical two-view refinement lowers Pi3 accuracy and helps VGGT only at small sizes | Pi3 A0 9.36 -> 7.88; VGGT A0 1.10 -> 2.39; 28% of pairs matched | Solid |
| F16 | 7-Scenes cannot resolve rotations below about 1.6-2 degrees | SIFT, Pi3 and VGGT all have 1.6-1.9 deg median error on the best-matched pairs, at any focal length | Solid for the floor; cause unverified |
| F17 | On ETH3D, Pi3 is resolution-invariant under uniform resize | AUC@3 28.5, 27.4, 28.2, 28.2, 29.3 from 4k to 16k tokens (6x compute) | 8 scenes; consistent across 5 sizes |
| F18 | On ETH3D, the same Pi3 collapses when sizes are mixed | A2 5.83 vs A0 28.51 at equal budget, 8/8 scenes; upsampling the small views restores 29.76 | 8 scenes; very large effect |
| F19 | VGGT's accuracy peaks near its training width and falls on both sides | 2.6, 6.0, 10.6, 21.0, 9.2 at widths 322, 392, 448, 546, 630; last step -11.8, 0/8 wins | 8 scenes; large effect |
| F20 | F2 and F3 (Pi3 non-monotonic on 7-Scenes) did not reproduce on ETH3D | Pi3 curve is flat there | Retract F2/F3 as general claims |
| F21 | Pi3 has a resolution floor, then is flat | ETH3D uniform: 13.33 at 154x224; 26-29 at every size from 210x322 to 420x630 | 8 scenes |
| F22 | For Pi3, mixing is worse than making every view small | A2 5.83 vs all views at 154x224 13.33: +7.50 [2.82, 12.72], 7/8 scenes | 8 scenes; interval excludes zero |
| F23 | VGGT's peak is at its training width | ETH3D AUC@3 by width 224..630: 0.24, 2.60, 5.66, 10.55, 23.12 (504), 21.00, 14.06, 9.18 | 8 scenes; large effect |
| F24 | VGGT's peak at its training width replicates on HiRoom | AUC@3 by side 182..630: 0.09, 0.22, 6.0, 10.7, 45.5 (504), 41.5, 31.3, 19.6; 19 scenes | Two datasets, 27 scenes |
| F25 | Pi3 mix worse than all-small replicates on HiRoom | A2 10.0 vs 27.9: +17.9 [13.1, 23.0], 19/19 | Two datasets |
| F26 | Pi3 is not flat on HiRoom: it rises, then plateaus near 504 px | 27.9, 49.7, 57.1, 59.8, 62.9, 63.4, 63.0, 61.2 | Corrects F17/F21: plateau point is dataset-dependent |
| F27 | At each model's best size the Pi3-VGGT gap is far smaller than at the study's sizes | HiRoom: 63 vs 45 at best size; 57 vs 6 at the mix's sharp size | Two datasets (ETH3D: 29 vs 23; 28 vs 11) |

**Warning added after F16:** F2, F3, F11, F13 and F14 are AUC@3 effects measured only on 7-Scenes. Treat them as unconfirmed until replicated on ETH3D or ScanNet++.

Dead hypotheses, for the record: damage lives only in frame attention; feature-scale drift; real pixels rescue RMFA on Pi3; keys can be pooled for free; aspect mismatch.

## 2. Candidate angles

| Angle | Claim | Rests on | Problem |
| --- | --- | --- | --- |
| A. Negative result | In-model fixes for mixed resolution do not beat uniform resize | F1, F6, F7, F12 | Close to paper 2's conclusion; reads as a failed method |
| B. Decomposition | Four separable causes of the mix failure | F5, F6, F10, F11 | Shares overlap; no payoff, because the fixes still lose to uniform |
| **C. Resolution is not a monotone knob** | These models have model-specific resolution responses: Pi3 peaks below its nominal full size and ignores extra pixels, VGGT keeps gaining and uses them. Consequences: "full resolution" is the wrong ceiling, and token-saving methods judged against it are mis-ranked | F2, F3, F8, F9, with F1, F6, F7 as the consequences | F2/F3 need the cause pinned down and a second dataset |
| D. Queries versus keys | Dense queries are required; even key-only pooling costs accuracy at 12 views | F6 | Contradicts AVGGT's claim only in the few-view regime; needs a many-view run to be fair |

**Recommendation: angle C.** It is the only surprising, checkable result in the table, it does not overlap paper 2 (which fixes the model's response and studies allocation), and every failed method becomes supporting evidence instead of a failure. Working title: *More Pixels, Worse Poses: Resolution Response in Feed-Forward 3D Reconstruction*.

What angle C still needs before it can be written:

1. A fine uniform-size sweep on Pi3 (every 4:3 grid from 12x16 to 30x40) to show the curve's shape and whether peaks sit at particular grids, at trained sizes, or at 4:3-exact sizes. Inference only.
2. The same sweep on VGGT and DA3, and on ETH3D, to show it is model-specific and not a 7-Scenes effect.
3. One mechanism test: the same pixels at a peak size and an off-peak size with positions rescaled to the peak grid. If accuracy follows the positions, the response is a positional effect.

## 3. Resolution ensembling (tested, failed: see F14)

F3 says Pi3's errors change with input size in a non-smooth way. If errors at different sizes are partly independent, running the same views at two cheap sizes and averaging the relative poses should beat one run at a large size. Two runs at 4k and 5k tokens cost 17.9k GFLOPs, less than one full-resolution run (18.7k).

- **Free first test, no GPU:** every result file already stores `predicted_c2w` for A0, A0_5k, A0_6k, A0_7k and A8 on all 70 draws. Average relative rotations and translation directions across two or three of them offline and rescore.
- **Pass rule (as built in `scripts/salvation_ensemble.py`):** one pair fixed in advance, A0_5k + A0_7k, must beat the best single size that costs no more compute, with a bootstrap interval above zero on the fresh draws. Other ensembles are reported as exploratory.
- **If it passes:** training-free method with a cost-accuracy win over the strongest baseline in this repo, motivated directly by angle C. If it fails, the offline hour cost nothing and angle C stands alone.
- **Risk:** errors across sizes may be strongly correlated, in which case averaging gains little. This is untested.
