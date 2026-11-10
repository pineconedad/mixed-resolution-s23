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

## Credential and safety continuation

1. The environment-variable handoff did not reach the agent process. Add a user-run hidden-input helper and explicit Git-ignored mode-0600 credential loading. Do not extract client credentials or ask for the key in chat.
2. The locally entered key authenticates. The official runpodctl v3.1.0 user read reported clientBalance=14.7874481444 USD, and the independent MCP SDK connection lists no pods. The earlier lack-of-key restriction is historical, not the current state.
3. An attempted read-only GraphQL GET returned HTTP 400. Replace it with the official CLI user command, which handles the supported account query and keeps the key out of request URLs. Verify the downloaded binary against its official release checksum; do not treat failed reads as zero balance.
4. Keep a $3 reserve and $1 cap per short session. Test a no-inference backup/stop/delete cycle first. Future guarded pods need persistent workspace storage so an interrupted final download does not force deletion of the only output copy. Quote that storage before deployment and delete it after verified recovery.
5. Prepare configs/diagnostic.yaml for A1-A8 except the unavailable historical A0. Reuse the exact original seed-42 manifest by its source-config and file hashes, without regenerating views or modifying the original config. Rerun A1-A4 under the same new code/GPU provenance as A5-A8 rather than mixing old and new runs.
6. The next matrix is still exploratory. Preserve the original unexpected A2 result; there is no evidence justifying a position implementation change merely to make the expected gain appear. Add the specified localization/information controls before interpreting it.

## Guarded diagnostic execution

1. All 86 local tests pass before deployment. Push execution code `61e9acc` and transfer only its Git bundle, never the credential file.
2. Quote a no-inference safety test on the available RTX A4000 at $0.25/hour, 10 GB container disk and 20 GB persistent workspace. The combined running rate is approximately $0.254/hour. This is pod-local persistent storage, not a new network volume.
3. Create smoke pod `la8oe50jah4x29` in EUR-IS-1 with session marker guardsmoke20261110a. Its independent watchdog verifies periodic archives, requests worker quiescence, verifies the final eleven-member archive, stops compute and deletes the pod. Fresh MCP reads return 404 and an empty pod list. Preserve all monitor logs and all snapshots in reports/control/guardsmoke20261110a.
4. Fresh stock advertised an A5000 at $0.27/hour. Its create request was refused for capacity and created no pod. Do not hide the refusal or create multiple speculative instances. The next read still showed the A4000 at $0.25/hour. Use it, with the unchanged fp32 identity gate deciding whether its 16 GB VRAM is sufficient. Never lower resolution to force a pass.
5. Create diagnostic pod `8l7h3gvjlk4npx`, same quoted disk/rate, marker pi3diag20261110a. Activate the independent monitor before setup/inference. All setup, testing, scoring and figure logs are captured in the persistent controller log.
6. The full checkpoint passes fp32 identity on chess and fire with exactly zero maximum difference on poses, points, local points and confidence. The 16 GB A4000 is therefore sufficient for this tested 4k diagnostic shape. Do not extrapolate that fit to 8k or other models.
7. All forty A1-A8 scene/arm evaluations complete, with no failed or excluded scenes. The independent watcher downloads the final quiescent archive, verifies its SHA-256 and all 73 members, stops compute and terminates the pod. A fresh MCP get returns 404 and list-pods is empty. No workspace remains billing.
8. The official CLI balance read at 2026-11-10T09:10:10.088906+00:00 is 14.6942937462 USD. This is a live balance observation, not a fabricated settled per-pod invoice.
9. Recompute all forty pose metrics from saved poses locally. The diagnostic recovery ratio is 0.537690, interval [0.499718, 0.675294]. Upsampling recovers 0.970382 of the A1-to-A8 gap, interval [0.901119, 1.023339]. These are exploratory, not a pass for frozen H1/H4.
10. A6 global-only upsampling scores 1.633443 mean AUC@3 versus A4's 7.674891. A7 without pi_hr scores 3.937313. A5 token entry scores 6.278705, with A5-A4 interval including zero; do not claim equivalence or choose a winner from this small sample. A2's isolated gain remains unestablished.
11. A4 mean latency on this A4000 is 0.734694 seconds versus A3's 0.810284, about 9.3% lower. Keep this separate from the earlier A40 timings, where A4 did not show a speedup. The office scene still hurts AUC@30 despite improving AUC@3; preserve and report it.

## Next model preparation

1. Implement VGGT next, as the day-2 adapter order requires, while no pod is running. Its camera head jointly attends across views and cannot be invoked separately for each image-size group. Keep that computation joint and group only its dense point/depth heads.
2. Preserve first-reference camera/register tokens by original view index. Do not accidentally assign a first-reference token to the first view in every resolution group.
3. Use a separate environment with the pinned VGGT repository's torch 2.3.1 and numpy 1.26.1 instead of forcing it into Pi3's torch-2.5.1 environment. All 19 CPU tests pass against the actual upstream forward and scaled-down native components. The 86 existing Pi3/safety tests still pass separately.
4. Pin and download the explicitly requested facebook/VGGT-1B weights locally. Record the public revision, non-commercial license and file checksum in model_checks.md. Never substitute a newer model or infer the unspecified old DA3 checkpoint.
5. Prepare a validation-only worker: download/verify weights, run model-specific tests and compare full-checkpoint native poses/points/confidence/depth in fp32 on the same two development scenes. No VGGT task metrics, frozen seeds or held-out data are implied by this gate.

## User-supplied L40S continuation

1. An earlier VGGT setup was stopped by the monitor when API SSH metadata changed. Its periodic snapshots are retained locally; it had not reached validation. Treat this as interrupted setup, not a model result. Correct the monitor to use successful, strictly host-key-pinned backup verification instead of treating a missing runtime SSH mapping alone as a reason to stop.
2. The user terminated that VM and explicitly supplied pod u64lksix8j610q for the work. Its fresh read shows NVIDIA L40S, compute $1.09/hour, 50 GB container disk, Python 3.12 and uv installed. Record user attribution rather than claiming the workflow rented it.
3. The user also created STANDARD network volume oq84uxp88q, 10 GB in EU-SE-1, at the documented $0.07/GB/month rate, $0.70/month. The pod has no matching data-center ID and the attach request is refused. Do not retry shape variants or delete the user's volume without an explicit request.
4. Quote and add a 20 GB host-local persistent workspace for safe interrupted-download recovery, about $0.003/hour while running. A fresh SSH df confirms /workspace is backed by a 20 GB device, not the 50 GB container overlay. The update restarts the container and reassigns SSH from port 19033 to 11476; reread and use the fresh mapping.
5. Use uv to prepare the already specified Python 3.10 environment. Keep VGGT's pinned torch 2.3.1, rather than silently adopting the image's torch 2.8. Do not rerun unit suites at the user's request. The mandatory full-checkpoint fp32 identity gate remains before the eight-arm development matrix.
6. Run configs/diagnostic-vggt.yaml on exactly the original five development scenes and view manifest, defaults unchanged. All outputs are exploratory seed 42, not the frozen main-paper matrix. Stop the user-named pod after a locally verified final backup; do not infer permission to delete it or the independent volume.

## Positional-scope clarification before broader runs

The original setup says to scale each small view's patch coordinates by rho. The first implementation restricted this to global attention and used effective grid ratios after rounding. That restriction was an implementation decision, not a recovered historical S5 implementation. It leaves the prescribed position-only sanity improvement unestablished.

The inspected native Pi3 decode computes one position tensor, reuses it in frame/global blocks, and returns it to all three head decoders. Before any frozen result, explicitly run the full-scope interpretation: nominal rho scaling in every applicable positional operation. VGGT has RoPE in its aggregator but not in its camera/DPT heads, so its normalized dense-head UV embedding is unchanged.

Configs/alignment-{pi3,vggt}.yaml fix this interpretation before seeing its outcomes. They use the identical original development views and default RMFA parameters. The existing global-only/effective-ratio results remain unchanged and must be labelled as that protocol, not silently substituted as S5 reproduction. No variant is chosen merely because it wins. All new comparisons are exploratory seed 42.

The user allowed continued autonomous work. Download only ETH3D and 7-Scenes development archives from the pinned official DA3-BENCH release on the local PC, not a paid GPU. Do not download or inspect held-out data yet. ScanNet++ original license/registration requirements remain a separate permission gate even though a public processed mirror exists.
