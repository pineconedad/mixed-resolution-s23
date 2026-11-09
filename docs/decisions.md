# Decision and execution record

This file records decisions, failures and stops. The two root study plans remain unchanged. All times in exported session filenames are UTC.

## First implementation and GPU session

1. Start with setup section 1b, the independent Pi3 quick test. No old harness or results were supplied. Do not invent S7 settings, an old DA3 checkpoint, or a pass for historical reproduction T9.
2. Select five development scenes before collecting metrics: chess, fire, heads, office and pumpkin, seq-01. This is the minimum five-scene scope permitted by the quick-test plan. Sample 12 views, select four sharp camera centres by FPS, use seed 42 and nominal rho=2.
3. Pin original Pi3 code and weights. Do not use Pi3X. The inspected original has five decoder register tokens and transformer/linear heads, so implement its actual heads rather than the plan's generic DPT description.
4. Use the quick-test guide's permitted standard relative-pose AUC fallback. Store `relative_c2w_exact_auc_v1` and percent units in outputs. Do not call these results the DA3 benchmark or mix protocols. No reconstruction metrics are fabricated.
5. Round image sides as instructed and record actual budgets. The nominal 4096 mix is 4200 spatial patch tokens plus 60 register tokens. Full upsampling is 8280 spatial tokens plus 60 registers. Pi_hr scales by actual high/low grid ratios, including fractional coordinates after rounding; the +1 patch offset and special-token zero coordinates are preserved.
6. Preserve high-grid state and lift only global residual updates for RMFA. Exact identity transfers use assignment rather than x+(out-x), avoiding avoidable rounding differences.
7. Correct two inconsistent written tests transparently: exact T2 inversion applies to nearest lift plus integer area pool, while bilinear gets a smooth-field tolerance check. T7 checks the first sharp-view frame output, not unchanged global outputs after different small-view inputs.
8. Build locally, download/checksum inputs and push a private repository before deploying. Commit `938cc8f` is the initial implementation. Weights and raw dataset archives are downloaded inputs, not generated observations, and are excluded from Git. Their source URLs, immutable revisions, sampler and checksums are recorded.
9. Inspect live stock. A5000 and 3090 had no available stock. Quote RTX 4090 Community compute at $0.34/hour plus 30 GB container storage, approximately $0.344/hour. The create request returned a capacity refusal and created no resource. Never start multiple speculative pods.
10. A fresh capacity read showed A40 Secure stock. Quote $0.59/hour compute plus approximately $0.0042/hour for 30 GB container disk. Create only pod `y9fmrpnrpscv0b`, which landed in CA-MTL-1. No persistent disk, network volume or public notebook was requested. Use only SSH.
11. Use the catalog's official `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` image because the catalog had no original-Pi3 torch-2.5.1/Python-3.10 image. Install pinned torch 2.5.1 in an isolated Python-3.10 virtual environment. CUDA runtime is 12.4; host driver 595.91.07. Do not install a host driver inside the pod.
12. Initial base-environment installation exposed unrelated image-package dependency conflicts. Do not ignore them. Switch to an isolated virtual environment; `pip check` passes there. The actual environment is exported in the session archive.
13. Clean-checkout validation exposed a packaging error: the ignore pattern `data/` also hid `mixres/data`. No inference had started. Anchor input-storage ignores to the repository root, add the missing source files, add a tracked-source regression test and isolate the bootstrap environment. Push fix `65af845` before running it on the pod. All 63 tests pass on the pod. This is the first point the initial implementation went wrong and the exact corrective commit.
14. Full-checkpoint fp32 T1 passes on chess and fire, using all 12 views at the uniform sharp size. Maximum absolute error is exactly 0 for poses, points, local points and confidence on both scenes. All recorded decoder global layers receive 8340 tokens for this uniform check.
15. Run A1-A4 on all five scenes, one warm-up and three timed passes per evaluation. Every one of the 20 scene/arm evaluations completes. Keep failed-scene handling and per-scene writes enabled. Do not silently alter resolution or precision.
16. Export `session-20261109T223406Z.tar.gz`, download it, verify SHA-256 `a947f97c055c980d95e829858ecdd4139ae65e24aa2419aa7d902b9145bc7868` and verify all 41 member checksums before deleting anything remote. Restore an independent local copy and a publication copy under `reports/sessions/20261109T223406Z/`.
17. Terminate pod `y9fmrpnrpscv0b` after verified download. A fresh get returns 404 and list-pods returns no pods. No separately billed volume was created. Compute and this pod's disk are no longer billing. Do not retain an idle pod overnight.
18. Billing detail had no settled records at the first two post-session reads. This is not evidence that the run was free. Do not report an exact charge or a verified remaining balance from an empty billing response. Reserve $1 conservatively against the user's stated initial $15 for planning until billing settles.

## Result interpretation

- All five scenes contribute to every arm; none were dropped after metrics.
- Mean AUC@3 in percent: A1 1.476632, A2 1.652770, A3 12.873600, A4 7.664844.
- A4-A2 is +6.012074 AUC points, bootstrap 95% interval [3.374724, 8.615540].
- R is 0.535796, bootstrap 95% interval [0.496563, 0.674601]. Its point estimate crosses the guide's 0.5 framing threshold, but the lower bound does not establish recovery above one half.
- A2-A1 is only +0.176138, interval [-0.470708, 1.094077]. The guide's expected position-only sanity gain is not established. Investigate the baseline/position behavior before stronger claims. Do not cherry-pick the one scene where A2 helps.
- RMFA's mean recorded latency is close to full upsampling here, despite fewer global tokens. Equal global tokens are not proof of equal runtime. Do not claim a demonstrated speedup.
- These seed-42 results are exploratory and cannot be put into the paper as frozen evidence. No held-out or seed>=1000 evaluation has occurred.

## Overnight safety decision

The user authorized autonomous continuation, preservation of every generated artifact, low-balance handling and complete shutdown when idle.

- The current session is already terminated and all its generated files are local. Publish the complete archive and its extracted outputs, not only the flattering summary.
- OAuth authenticates the interactive MCP tools, but it does not supply a standalone watchdog credential. `RUNPOD_API_KEY` is not set in the local environment. Do not read OpenCode's credential store, export OAuth tokens, or put credentials into Git or a pod.
- Runpod REST v2/MCP billing reads expose historical spend, not live prepaid credit balance. The official balance concept directs users to `runpodctl user` or GraphQL `myself`, which need separate API-key access. The user's stated $15 is not an independently verified live balance.
- Therefore do not advertise a live-balance monitor or unattended termination watchdog as working. Do not start a new overnight paid deployment without those safeguards. Continue free local verification and preserve the completed result instead. This is a deliberate fail-closed stop, not a claim that the full study is complete.
- A future session may start only after current balance is verified, a conservative reserve and deadline are recorded, and an independently authenticated cleanup path is tested. Never wait for a zero balance: the official billing guide warns that pods without network storage can be terminated and lose their data at zero credits.

## Full-study boundary

The full S23 matrix still needs additional adapters, benchmark reconstruction/probes, development tuning and frozen execution. Its exact historical S7 configurations and DA3 checkpoint are not specified anywhere in the two supplied files. No additional historical material will be provided. A genuinely new full study must explicitly select and document fresh baselines; it cannot honestly pass historical T9 or claim continuity with missing runs. Do not guess an old checkpoint and label it reproduced. The independent quick test is the valid completed result so far.

## Next experimental order

After a safe funding/shutdown path is available, the next short session should stay on Pi3 seed 42 and the same five-scene view list. First investigate the unestablished A2 position-only gain without tuning on the reported scenes' frozen seeds. Then add A8 to separate real information from upsampling, and A5/A6/A7 for entry, localization and position-alignment controls. Preserve the original A1-A4 session unchanged. Any implementation correction requires rerunning affected arms under a new code/config identity, not silently combining versions.

The full 24-configuration development sweep follows the model's sanity gates, not before them. Seeds 1000/2000/3000 and DTU/HiRoom remain unopened until the protocol and predictions are committed. No venue registration or manuscript claim is submitted automatically on the strength of this exploratory five-scene run.
