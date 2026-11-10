# Model checks

The first full-checkpoint CUDA identity test passed on an NVIDIA A40. On chess and fire, all 12 views at uniform sharp size match original Pi3.forward exactly in fp32: maximum absolute error 0 for camera poses, global points, local points and confidence. See reports/sessions/20261109T223406Z/runs/sanity/pi3-quick.json. The checkpoint and code hashes are recorded there.

GPU gate: scripts/01_sanity.py compares the actual original Pi3 checkpoint forward against the wrapper in fp32 on two 12-view scenes at uniform resolution. Poses, global points, local points and confidence must all have maximum absolute difference below 1e-4. The run script refuses to start without a passing gate with identical provenance.

All 20 quick-test scene/arm evaluations completed. Their raw outputs and exploratory summary are under reports/sessions/20261109T223406Z/. This is not paper-number reproduction. T9 remains unverified. Neither historical result tables nor the exact old pipeline were supplied. No synthetic or freshly produced numbers can truthfully stand in for historical reproduction.

The guarded diagnostic run also passes full-checkpoint fp32 T1 on an NVIDIA RTX A4000 with 16 GB VRAM, at the unchanged 12-view 4k shape, with maximum difference 0 on both scenes. All forty A1-A8 exploratory evaluations complete. See reports/sessions/20261110T090917939514Z/. The two GPU sessions remain separate provenance groups; no old/new metrics or timings are silently pooled.
