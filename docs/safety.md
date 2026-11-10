# Budget, backup and shutdown controls

The key is entered with `python3 scripts/set_runpod_key.py`. It is loaded from the environment or the explicit project-local `.env.runpod`, never from OpenCode's credential store. The helper checks Git exclusion and creates the file with mode 0600. The key is never passed on a command line, included in a source bundle or transferred to a pod.

## Independent access

- Operations dependencies use `.venv-ops` and `env/requirements-ops.txt`, separate from inference.
- Balance reads use the official `runpodctl user` command, not historical billing buckets. Version 3.1.0, Darwin ARM64 SHA-256 `2581913eacac577a3fd5426219962529d37172f294a80e807e7d920a65475e2c`, was downloaded from the official release and verified. Only clientBalance is exported; email/account details are discarded.
- MCP mutations use the official MCP Python SDK over the hosted server, with the API key in an authorization header. Tool names come from the live list. No direct REST mutation fallback exists.
- `scripts/check_runpod_access.py` is read-only. A verified key, balance and MCP connection are necessary but not sufficient: an actual backup/stop/delete smoke test must pass before unattended inference.

Install the operations dependencies with Python 3.10, then download the pinned CLI binary and its release checksum using `gh release download v3.1.0 --repo runpod/runpodctl`. Keep the CLI in ignored `artifacts/runpodctl-darwin-arm64`; do not commit credentials. The client rechecks its binary hash before every balance read.

## Per-session limits

An explicit create receipt binds the monitor to one pod ID, its creation timestamp and a create-time S23_SESSION_ID marker. Listing a familiar name is not authority to change a resource. No other resource is stopped or deleted.

Each session keeps at least a $3 reserve, has a $1 spend cap and a one-hour hard duration. The monitor starts shutdown ten minutes before the smaller cost/time deadline. It uses both live balance and conservative wall-clock accounting. An unverified balance, stale supervisor heartbeat, failed backup or completed worker triggers shutdown rather than permission to keep billing.

The local supervisor launches the watchdog in an independent process under macOS caffeinate. Normal idle sleep is inhibited while it runs. A missing heartbeat after the supervisor exits or crashes requests shutdown. No software can guarantee cloud control during a total local power/network failure; a watchdog transport failure reconnects and retries stopping the owned pod, and otherwise explicitly reports that shutdown is not confirmed.

## Artifact protection

Future unattended pods use a persistent workspace for both results and the environment. This is host-local pod storage, not a separately provisioned network volume. Its exact running and stopped rates must be quoted before creation. A stopped workspace still bills storage; termination removes it.

The watchdog periodically exports and downloads snapshots. Export hashes and archives the same spooled bytes even while an atomic result file changes. Every downloaded archive must pass its outer SHA-256, all member checksums and its pod/session binding.

A periodic snapshot is not a final backup. Before deletion the monitor requests the worker to stop between evaluations, waits for the remote controller to confirm its child processes exited, and downloads a final quiescent snapshot. Only that proof permits termination. If the last backup is unverified, the monitor stops compute and retains the workspace, reporting the continuing storage charge instead of deleting unbacked data.

`scripts/run_session.py` preserves setup, test, inference and aggregation output in its persistent console log and records phase changes and exit codes. Failed or interrupted runs remain labelled as such. Completed observations are never replaced to manufacture a clean result.
