# Local verification

2026-11-10, Python 3.10.20, PyTorch 2.5.1, CPU, macOS arm64.

| Check | Observed result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | 68 passed after packaging and archive-verification fixes |
| `.venv/bin/python -m pip check` | No broken requirements found |
| `bash -n launch.sh scripts/bootstrap_gpu.sh` | Exit 0 |
| `git diff --check` | Exit 0 |
| `scripts/preflight.py` | 5 scenes, 60 original images/poses verified, checkpoint SHA-256 matches |
| Actual mixed patch budget | 4200 per scene, plus 60 decoder register tokens |
| Sampled image + checkpoint transfer size | 3.860 GB, excluding local sequence archives |
| Re-running data preparation | All 5 receipts verified; identical view manifest preserved |
| Contact sheet | A1/A3/A8 for all 12 chess views inspected; crops match |

The tests cover scaled-down upstream Pi3.forward/decode identity for A1-A8 at rho=1, mixed-grid shapes/token counts, special-token preservation, continuous rotary coordinates, pooling/lifting, first sharp frame output, head-grid/layer variants, determinism, FPS sampling, disjoint exclusions, intrinsics, controlled resizing, pose metrics, ratio guards, Holm adjustment, atomic writes, nested ZIP reading, scene resume, interruption after a durable result, provenance refusal, failed-scene retry, session limits and aggregation pairing.

Synthetic test inputs/results live only in pytest temporary directories. They are never research observations.

The clean GPU checkout passed 63 tests before collecting metrics. Five archive-restoration tests were added locally afterward, for 68 total local tests. The earlier local test run did not detect the ignored source package; docs/decisions.md records the clean-checkout failure and corrective commit.

Full-checkpoint CUDA T1 and the first quick-test metrics/timing/memory are now recorded under reports/sessions/20261109T223406Z/. Paper-number reproduction, T9, other model adapters and full-study evaluation remain pending. Local success is not reported as a pass for these gates.
