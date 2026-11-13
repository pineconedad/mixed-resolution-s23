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

Dead hypotheses, for the record: damage lives only in frame attention; feature-scale drift; real pixels rescue RMFA on Pi3; keys can be pooled for free; aspect mismatch.

| F14 | Fusing predictions from several input sizes never beats the best member | 20 ensembles per model; best Pi3 pair 13.74 vs 14.36 alone; errors across sizes are correlated | Solid |

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
