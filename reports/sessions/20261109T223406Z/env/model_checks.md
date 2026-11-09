# Model checks

No checkpoint-backed accuracy or CUDA equivalence results exist yet. Local tests use scaled-down upstream Pi3 components and its unmodified forward/decode code. They are implementation tests, not a validation of the full checkpoint.

GPU gate: scripts/01_sanity.py compares the actual original Pi3 checkpoint forward against the wrapper in fp32 on two 12-view scenes at uniform resolution. Poses, global points, local points and confidence must all have maximum absolute difference below 1e-4. The run script refuses to start without a passing gate with identical provenance.

Paper-number reproduction and T9 remain unverified. Neither historical result tables nor the exact old pipeline were supplied. No synthetic or freshly produced numbers can truthfully stand in for historical reproduction.
