# Implementation decisions

The two root Markdown plans are the experiment specification. Upstream source is consulted only to implement those instructions against real models. The source plans are not edited.

## Current scope

The fresh-PC quick test in setup section 1b comes first. This repository implements its Pi3 data preparation, model adapter, two-grid inference, pose evaluation, per-scene resume, summaries, figure and session archive. A1-A8 are implemented for Pi3, but the default quick launch runs only A1-A4. No tuning or frozen/held-out evaluation starts here.

S23's complete matrix is recorded in configs/s23.yaml, not claimed executable or complete. Additional adapters, full reconstruction, mechanism probes, the development sweep and frozen-run workflow follow validation of the first model. The quick test is the first point that requires renting a GPU.

## Model mapping

Original Pi3 has a DINOv2 encoder and 36 decoder blocks alternating 18 frame/global pairs. Five register tokens per view are kept separate from spatial pooling. Its three head decoders consume the concatenated outputs of decoder layers 34 and 35. Its point/confidence heads are linear, not DPT. The adapter preserves this layout.

Pi3 frame attention batches views by equal grid shape. Global attention concatenates ragged per-view sequences with no padding. Every view retains one set of five special tokens. Output heads batch views by equal grid shape. Grouping is shared across arms.

For pi_hr, zero-based y/x patch coordinates scale by the actual high/low grid ratio per axis, then receive Pi3's +1 patch offset. Register coordinates stay zero. Rounding never quantizes fractional coordinates. Frame and output-head positions stay native to their current grid; only global positions use pi_hr. The exact S5 head-position behavior is unknown, so this is an explicitly fresh pipeline.

When a grid transfer is an identity the engine assigns the global output directly. It does not compute x + (out - x), which can introduce fp32 rounding drift. Special tokens likewise take the global output directly.

For A4/A5/A7, high-grid state persists. Pooling creates the global input, and only the global residual update is lifted. A6 maintains low-grid state, lifts for global attention, then pools only the update back. Limited-layer RMFA switches its persistent state to low at the cutoff. Hi-grid head output maps are downsampled to low; camera matrices are never spatially interpolated.

## Tests and evaluation

T2's exact reverse property holds for integer nearest lifting followed by area pooling, not arbitrary bilinear lifting. The tests verify the exact pair and separately bound smooth-field error for bilinear lifting. This does not change the specified default bilinear method.

T7 compares the first sharp-view frame-attention outputs, as its written pass condition requests. Sharp-view global outputs can differ because small-view inputs differ.

The quick test uses the setup guide's permitted standard relative rotation/translation-angle AUC fallback. It uses all nondegenerate camera pairs, exact empirical-CDF integration, translation sign ambiguity, and AUC in percent. Relative poses already cancel the global similarity gauge; there is no additional GT-fitted pose correction. This is not labelled as DA3 histogram AUC and cannot be mixed with it. Frozen evaluation will require one finalized shared benchmark protocol.

For five scenes, intervals and recovery ratios are exploratory. R is suppressed when the upsampling gain is not significantly positive or the bootstrap denominator crosses zero. No conclusion is forced when A1 is not worse than A2/A3.

The actual rounded mixed patch count is recorded, not relabelled as the nominal 4096. Special tokens are counted separately.

## Missing historical inputs

The user confirmed only the two plans will be provided. Therefore S7 configurations, historical numbers for T9, the DA3 checkpoint choice and original preprocessing cannot be recovered. A0 and historical-reproduction claims are blocked rather than invented. Completing a new full study will require a documented fresh-baseline amendment and development selection of replacement settings. No such settings have been silently selected.

## DA3 source inspection

At the pinned source, DA3 does have conditional alternating local/global layers after alt_start. For da3-large: 24 layers, alt_start=rope_start=8, global layers 9,11,13,15,17,19,21,23; other layers local. DA3-base/small start at 4; DA3-giant starts at 13. Global RoPE uses position-nondifferentiating coordinates, unlike local RoPE. Reference selection can reorder views before alternation. The unknown checkpoint prevents choosing one mapping as the study's mapping. The engine must not treat DA3 as a uniform sequence of Pi3 block pairs.

VGGT has 24 frame/global pairs, a camera token plus four register tokens, first-view-specific special tokens, and cached DPT features at pairs 4,11,17,23. Its adapter is implemented and locally tested in a separate torch-2.3.1 environment. The joint camera head receives all views together; only dense heads are grouped by output grid. First-view token identity follows the original view index rather than restarting in each resolution group. All 19 local VGGT tests pass, including native-forward identity for A1-A8 at rho=1. Full-checkpoint CUDA validation remains a distinct gate.

The inference package's torch minimum is now 2.3.1 because the pinned VGGT source requests that version. Pi3 still uses its separately pinned torch-2.5.1 environment. No experiment mixes model environments between arms. The worker supports validation-only sessions so a new model can pass native fp32 identity without pretending to complete its dataset or historical-reproduction checks.
