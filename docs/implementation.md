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

The user confirmed only the two plans will be provided. Therefore S7 configurations, historical numbers for T9, the DA3 checkpoint choice and original preprocessing cannot be recovered. Historical-reproduction claims remain unavailable rather than being inferred.

## Fresh-baseline amendment

The new study uses `fresh_v1`, fixed before its development sweep. It keeps the setup guide's independent 12-view design: four FPS-selected sharp views, eight small views and rho=2. A0 uses the same 12 frame IDs at one equal-budget uniform resolution. This keeps scene and frame identity fixed across arms rather than inventing an unavailable S7 view-count search. Sizes are derived separately at nominal 4k and 8k, and every result reports the rounded token count.

S23's written positional protocol is authoritative for the frozen study: frame attention and heads use native positions, while A2/A4/A5 use aligned positions in global attention. Alignment uses the effective per-axis high/low grid ratio after patch rounding. The preserved full-scope/nominal-rho results are architecture controls, not frozen S23 cells.

Because this is a new baseline rather than an S1-S22 reproduction, DA3 uses the corrected `depth-anything/DA3-LARGE-1.1` release at the pinned source commit. The deprecated original is not selected merely to mimic an unknown historical checkpoint. DA3 results must identify this choice and cannot be compared as reproductions of old DA3 numbers. DA3's global RoPE deliberately uses nondifferentiating coordinates, so it has no direct `pi_hr` global-position operation. A2 is therefore the explicit no-op position control for DA3 instead of an invented architecture change.

ScanNet++ remains blocked until the user has accepted its terms. This amendment does not treat a public processed mirror as permission. Development code may run on ETH3D and 7-Scenes while that gate is unresolved, but the three-dataset phase-1 sweep cannot be called complete without ScanNet++.

## DA3 source inspection

At the pinned source, DA3 does have conditional alternating local/global layers after alt_start. For da3-large: 24 layers, alt_start=rope_start=8, global layers 9,11,13,15,17,19,21,23; other layers local. DA3-base/small start at 4; DA3-giant starts at 13. Global RoPE uses position-nondifferentiating coordinates, unlike local RoPE. Reference selection can reorder views before alternation. The unknown checkpoint prevents choosing one mapping as the study's mapping. The engine must not treat DA3 as a uniform sequence of Pi3 block pairs.

VGGT has 24 frame/global pairs, a camera token plus four register tokens, first-view-specific special tokens, and cached DPT features at pairs 4,11,17,23. Its adapter is implemented and locally tested in a separate torch-2.3.1 environment. The joint camera head receives all views together; only dense heads are grouped by output grid. First-view token identity follows the original view index rather than restarting in each resolution group. All 19 local VGGT tests pass, including native-forward identity for A1-A8 at rho=1. Full-checkpoint CUDA validation remains a distinct gate.

The inference package's torch minimum is now 2.3.1 because the pinned VGGT source requests that version. Pi3 still uses its separately pinned torch-2.5.1 environment. No experiment mixes model environments between arms. The worker supports validation-only sessions so a new model can pass native fp32 identity without pretending to complete its dataset or historical-reproduction checks.

## Final build and manual operation

The assistant no longer operates cloud resources. User commands are in GPU_HANDOFF.md. Setup and identity are separate operations; setup does not launch tests, experiments or watchdogs.

`rmfa_layers` selects a fraction of the alternating aggregation pairs. Pi3/VGGT image encoders keep their entry-resolution behavior. DA3's local feature prefix likewise keeps its entry grid; the fraction applies to its eight subsequent global pairs. This mapping is fixed in the model config, not tuned per model.

High-grid head predictions are reduced to the declared low grid for points, confidences and depth. Predicted intrinsics scale with that reduction; camera poses and pose encodings do not. Transformer cost uses two FLOPs per multiply-add, includes DA3's real local-prefix blocks and excludes its identity placeholder. The analytic count is explicitly partial: encoder and head FLOPs are not manufactured. Archived MAC counts remain unchanged and cannot be pooled as new FLOP counts.

The frozen launcher keeps one model loaded per selected model/environment. Its preprocessing cache is CPU-only and bounded to 2 GiB; input hashes remain checked. Reconstruction summaries join separate native recon.json records by cell and scene, with missing reconstruction kept explicit rather than reported as zero.
