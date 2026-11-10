# Preserved research outputs

## VGGT development matrix

All forty VGGT scene/arm evaluations completed on the user-supplied L40S. The full-checkpoint fp32 identity check matched native pose encoding, camera poses, world points, point confidence, depth and depth confidence with exactly zero maximum error on both checked scenes. Previously completed unit suites were skipped at the user's request.

- [Complete restored session](sessions/20261110T182520898917Z/).
- [VGGT summary](sessions/20261110T182520898917Z/runs/diagnostic/summary/snapshot.json).
- [Independent audit](audits/20261110T182520898917Z.md).
- [All monitor logs and periodic/final archives](control/userl40svggt20261110a/).
- [Two-model development comparison and limitations](development/README.md).

The final 74-member archive passed outer and member checksum verification before the GPU was stopped. The pod is user-owned and was not deleted without explicit authorization. Its workspace and the user's unattached network volume are recorded in operations.json, including their standing storage charges.

## Guarded diagnostic session

The complete eight-arm Pi3 run is under [sessions/20261110T090917939514Z](sessions/20261110T090917939514Z/). All forty evaluations completed on the original seed-42 views. The full native fp32 identity test passed on an RTX A4000. The session is exploratory, not frozen evidence.

- [Diagnostic summary](sessions/20261110T090917939514Z/runs/diagnostic/summary/snapshot.json).
- [Forty-scene/arm CSV](sessions/20261110T090917939514Z/runs/diagnostic/summary/cells.csv).
- [Independent metric audit](audits/20261110T090917939514Z.md).
- [Persistent worker/setup log](sessions/20261110T090917939514Z/runs/control/pi3diag20261110a/console.log).
- [Every diagnostic monitor snapshot and local control log](control/pi3diag20261110a/).
- [Live shutdown smoke-test artifacts](control/guardsmoke20261110a/).

Both pods were independently backed up, stopped and deleted. Their final archives and every periodic archive are preserved in the control directories, with outer and member checksum verification. There is no active pod or retained cloud workspace from these sessions.

## Session 20261109T223406Z

- Complete downloaded archive: [session-20261109T223406Z.tar.gz](../artifacts/session-20261109T223406Z.tar.gz).
- Archive SHA-256: `a947f97c055c980d95e829858ecdd4139ae65e24aa2419aa7d902b9145bc7868`.
- Every one of its 41 member checksums is verified against [the export receipt](sessions/20261109T223406Z/session-manifest.json).
- Full exported files are under [sessions/20261109T223406Z](sessions/20261109T223406Z/), including per-scene poses/errors, raw arm result JSONs, run logs, summaries, the figure, actual environment and fp32 identity report.
- [Summary JSON](sessions/20261109T223406Z/runs/quick/summary/snapshot.json).
- [Per-scene CSV](sessions/20261109T223406Z/runs/quick/summary/cells.csv).
- [Recovery figure](sessions/20261109T223406Z/runs/quick/summary/figures/quick-recovery.png).
- [CUDA identity report](sessions/20261109T223406Z/runs/sanity/pi3-quick.json).
- Execution code commit: `65af845d721ad556b66c09075deaa3bd733cb2c6`.
- [Post-download audit](audits/20261109T223406Z.md) recomputes every AUC from saved camera poses and checks token budgets and provenance without a GPU.
- [Preprocessing contact sheet](../artifacts/7scenes-chess-contact.png).
- [Exact sampled input transfer pack](../artifacts/quick-inputs.tar.gz), containing the 60 original sampled images, poses and download receipts. The pretrained checkpoint remains a pinned external download.
- Original source-transfer bundles are retained in `artifacts/source.bundle` and `artifacts/source-fixed.bundle`, including the pre-fix state for tracing the packaging failure.

All metrics are exploratory seed-42 outputs. See [the decision record](../docs/decisions.md) for failures, deviations, resource cleanup and limitations. The archived environment/docs are immutable copies from execution time, not current status statements.

The preparation machine's CPU environment is captured separately under `local-preparation/`; it is not the CUDA execution environment. Runtime captures in the root `env/actual-*` remain ignored so capturing a new machine does not make the experiment's source tree dirty.

Recheck and restore independently:

```bash
python scripts/verify_session.py artifacts/session-20261109T223406Z.tar.gz \
  --sha256 a947f97c055c980d95e829858ecdd4139ae65e24aa2419aa7d902b9145bc7868 \
  --destination artifacts/rechecked-session
```

The archive contains all files exported by the runner. Large downloaded inputs such as pretrained weights and original full-sequence dataset ZIPs are not new research outputs. They remain local and are reproducible through pinned download scripts and checksums; they are not silently added to GitHub as multi-GB binaries. Identical redundant local restore directories are represented by this complete archive and the extracted publication copy rather than repeated in Git.
