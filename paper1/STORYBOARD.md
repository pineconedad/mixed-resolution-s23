# Storyboard: paper 1 (diagnostic and empirical analysis)

Working title: *Where Do Feed-Forward 3D Models Use Resolution? Probing VGGT and π³*.

## Positioning against paper 2

| | Paper 1 (this) | Paper 2 (*Views or Pixels?*) |
| --- | --- | --- |
| Question | How does accuracy respond to resolution, and which stages of the network need it? | Why do models fail on mixed-resolution input, and how should a token budget be spent? |
| Unit of analysis | Stages of the network (input pixels, encoder, frame attention, global queries and keys) and the uniform-size response curve | Whole inputs (as-is, aligned, upsampled, one size, views vs pixels) |
| Owns | Response curves per model; stage decomposition; queries vs keys; correlated errors across sizes; 7-Scenes precision floor | Mixed-input collapse; representation vs information; DA3 control; in-model fixes vs input fixes; allocation and view count |
| Must not claim | Anything about allocation, view count, mixed input being worse than uniform, the DA3 control, or "representation not information" | — |

Paper 1 uses reduced-resolution views only as a probing tool: lower the resolution of a stage for 8 of 12 views and measure the cost. Its claims are about stages, not about mixed input as a use case.

## Structure (modelled on El Banani et al., CVPR 2024)

1. Introduction: question, approach (frozen models, controlled probes), findings as a short list, teaser figure.
2. Related work.
3. What we mean by using resolution: two properties we posit (response, stage use).
4. Experimental setup: questions 1–4 stated explicitly, models, probes table, data, metric, statistics.
5. Results, one subsection per question.
6. Discussion: what it implies, limitations, conclusion.

## Evidence (all from `reports/salvation/` in this repository)

- Q1 response curves: `accurate.txt` (ETH3D 8 scenes, HiRoom 19 scenes, seed 42).
- Q2 stage probes: `night-pi3.txt`, `night-vggt.txt` (7-Scenes, 70 fresh scene-draws), `accurate.txt` (ETH3D, HiRoom).
- Q3 correlated errors: `ensemble.txt`.
- Q4 benchmark precision: `gtcheck.txt`, `geometry.txt`, and the 7-Scenes uniform curve in `night-*.txt`.

## Caveats carried into the text

- Exploratory development data; ETH3D and HiRoom use one frame sample per scene; no frozen seed.
- Two models only; DA3 not run.
- The repository's own relative-pose AUC (exact empirical CDF), not the DA3 benchmark histogram AUC.
- Intervals are reported only where they were computed; other differences are means.
- Correction found while drafting: one of the 20 resolution ensembles on π³ (A0_5k + A8, 12.94) does exceed both members (12.66, 12.64). The claim is therefore "no ensemble beats the best single size at no more compute", not "no ensemble beats its members".

## Narrative (added for the submission version)

The results are told as a chain of questions, each raised by the previous answer:
1. Response: does accuracy rise and saturate with input size? No: VGGT peaks and falls, so size itself matters to the network.
2. Stages: which stages depend on it? The encoder and global attention both do; π³ does not use fine pixel detail.
3. Queries vs keys: global attention is the costly stage that needs resolution; keys tolerate lowering better than queries.
4. Correlated errors: can errors at several cheap sizes be averaged away? No, they are shared.
5. Benchmark precision: why was 7-Scenes irregular throughout? Its ground truth has a ~2° floor.

Figures: Fig. 1 response curves, Fig. 2 stage probes (both in the main paper), Fig. S1 predicted camera layouts (supplement). All drawn from data in this repository by `scripts/`.
