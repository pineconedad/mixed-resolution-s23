"""Independent local monitor for one explicitly receipted workflow-owned pod.

Requires a live-tested API key and a results-only persistent mount. On backup
failure it stops compute but retains that disk; it never destroys unverified data.
Run under caffeinate on macOS so normal idle sleep does not suspend the monitor.
"""
import argparse
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixres.credentials import runpod_key
from mixres.io import atomic_json, read_json
from mixres.runpod_control import AccessError, account_balance, control_plane
from mixres.safety import Budget, verified_backup


def ssh_base(receipt):
    trust = ["-o", "UserKnownHostsFile=" + receipt["ssh_known_hosts"]] if receipt.get("ssh_known_hosts") else []
    return ["ssh", "-i", receipt["ssh_key"], "-p", str(receipt["ssh_port"]), *trust,
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
            "root@" + receipt["ssh_host"]]


def quiesce(receipt):
    command = ("env -C " + shlex.quote(receipt["remote_root"])
               + " python scripts/request_stop.py --session-id " + shlex.quote(receipt["session_id"]))
    subprocess.run(ssh_base(receipt) + [command], capture_output=True, timeout=140, check=True)


def sync_backup(receipt, destination):
    root = receipt["remote_root"]
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", root) or ".." in Path(root).parts:
        raise ValueError("Unsafe remote root")
    command = ("env -C " + shlex.quote(root) + " RUNPOD_POD_ID=" + shlex.quote(receipt["pod_id"])
               + " python scripts/export_session.py --session-id "
               + shlex.quote(receipt["session_id"]))
    result = subprocess.run(ssh_base(receipt) + [command], capture_output=True, text=True, timeout=120, check=True)
    matches = re.findall(r"^(artifacts/session-[0-9TZ]+\.tar\.gz) sha256=([a-f0-9]{64})$", result.stdout, flags=re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError("Remote export did not identify one archive")
    remote_path, expected_hash = matches[0]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / Path(remote_path).name
    transfer = ["scp", "-i", receipt["ssh_key"], "-P", str(receipt["ssh_port"]),
                *(["-o", "UserKnownHostsFile=" + receipt["ssh_known_hosts"]] if receipt.get("ssh_known_hosts") else []),
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
                "root@" + receipt["ssh_host"] + ":" + root + "/" + remote_path, str(target)]
    subprocess.run(transfer, capture_output=True, timeout=120, check=True)
    proof = verified_backup(target, expected_hash, receipt["pod_id"])
    proof["local_archive"] = str(target)
    return proof


def verify_storage_and_ssh(pod, receipt):
    persistent = pod.get("mounts", {}).get("persistent")
    if not persistent or persistent.get("path") != receipt.get("results_mount"):
        raise AccessError("Unattended work requires the receipted results-only persistent disk")
    if not Path(receipt["remote_root"]).is_relative_to(Path(persistent["path"])):
        raise AccessError("Remote project is outside persistent storage")
    direct = pod.get("ssh", {}).get("direct") or {}
    # Runtime SSH metadata may temporarily disappear while the established route
    # still works. Do not follow a changed route; validate the pinned host key and
    # session-bound archive instead. Failed transfer still stops compute below.
    return direct.get("host") == receipt["ssh_host"] and direct.get("port") == receipt["ssh_port"]


async def watch(receipt_path, destination, interval=60):
    if not 10 <= interval <= 120:
        raise ValueError("Monitor interval must be 10-120 seconds")
    receipt = read_json(receipt_path)
    if receipt.get("created_by_workflow") != "mixres-s23":
        raise ValueError("Receipt is not from this workflow")
    key = runpod_key()
    created = receipt.get("budget_start_epoch", datetime.fromisoformat(receipt["created_at"].replace("Z", "+00:00")).timestamp())
    initial_balance = await account_balance(key)
    # Balance is current; add elapsed charges back only for the conservative clock
    # accounting. Both the live balance and the original create-time cap still apply.
    elapsed = max(0, time.time() - created)
    budget = Budget(balance_at_start=min(receipt["balance_at_create"], initial_balance + receipt["hourly_total"] * elapsed / 3600),
                    hourly_total=receipt["hourly_total"], reserve_usd=receipt.get("reserve_usd", 3),
                    session_cap_usd=receipt.get("session_cap_usd", 1), max_seconds=receipt.get("max_seconds", 3600))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    events_path = destination / "watchdog.jsonl"
    proof = None
    last_backup_failed = False

    def event(kind, **fields):
        record = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind,
                  "pod_id": receipt["pod_id"], **fields}
        with events_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    async with control_plane(key) as control:
        while True:
            pod = await control.owned_pod(receipt)
            if pod is None:
                event("pod_absent")
                return
            if pod["status"] in {"EXITED", "ERROR"}:
                event("compute_not_running", status=pod["status"], unverified_data_retained=True)
                atomic_json(destination / "final.json", {"pod_id": receipt["pod_id"],
                            "reason": "already_stopped", "compute_stopped": True,
                            "terminated": False, "persistent_data_retained": True})
                return
            try:
                metadata_matches = verify_storage_and_ssh(pod, receipt)
            except AccessError as error:
                await control.stop_owned(receipt)
                event("unsafe_storage_compute_stopped", reason=str(error))
                atomic_json(destination / "final.json", {"pod_id": receipt["pod_id"],
                            "reason": "unsafe_storage", "compute_stopped": True,
                            "terminated": False, "persistent_data_retained": True})
                return
            try:
                proof = await asyncio.to_thread(sync_backup, receipt, destination)
                event("backup_verified", **{k: v for k, v in proof.items() if k != "pod_id"})
                if not metadata_matches:
                    event("ssh_metadata_changed_pinned_backup_verified")
                if not (destination / "ready.json").exists():
                    atomic_json(destination / "ready.json", {"pid": os.getpid(),
                                "pod_id": receipt["pod_id"], "balance_usd": initial_balance,
                                "hard_deadline_epoch": created + budget.hard_seconds})
                last_backup_failed = False
            except Exception:
                last_backup_failed = True
                event("backup_failed")
            try:
                live_balance = await account_balance(key)
                reason = budget.stop_reason(max(0, time.time() - created), live_balance)
                event("balance", balance_usd=live_balance)
            except Exception:
                reason = "balance_unverified"
            done = Path(receipt["done_marker"]).exists()
            if receipt.get("heartbeat"):
                heartbeat = Path(receipt["heartbeat"])
                if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 180:
                    reason = reason or "supervisor_heartbeat_stale"
            reason = reason or ("work_complete" if done else None)
            if last_backup_failed:
                reason = reason or "backup_failed"
            if reason:
                final_proof = None
                try:
                    await asyncio.to_thread(quiesce, receipt)
                    final_proof = await asyncio.to_thread(sync_backup, receipt, destination)
                except Exception:
                    event("final_backup_or_quiesce_failed")
                # A live periodic snapshot is not a final backup. Terminate only
                # after the worker is quiescent and its last output is local.
                await control.stop_owned(receipt)
                event("compute_stopped", reason=reason)
                can_terminate = final_proof is not None and final_proof.get("job_quiescent") is True
                if can_terminate and receipt.get("cleanup_action", "terminate") == "terminate":
                    await control.terminate_owned(receipt, final_proof)
                    event("terminated_after_verified_backup")
                elif can_terminate:
                    event("user_named_pod_stopped_after_verified_backup", storage_billing_continues=True)
                else:
                    event("persistent_disk_retained", reason="latest_backup_unverified",
                          storage_billing_continues=True)
                atomic_json(destination / "final.json", {"pod_id": receipt["pod_id"], "reason": reason,
                            "compute_stopped": True, "terminated": can_terminate and receipt.get("cleanup_action", "terminate") == "terminate",
                            "backup": final_proof or proof})
                return
            await asyncio.sleep(interval)


async def guarded_watch(receipt_path, destination, interval=60):
    key = runpod_key()
    receipt = read_json(receipt_path)
    try:
        await watch(receipt_path, destination, interval)
    except Exception:
        # A transport/session exception must not silently disable the cost guard.
        # Reconnect independently and stop compute, retaining unverified disk data.
        for attempt in range(5):
            try:
                async with control_plane(key) as control:
                    await control.stop_owned(receipt)
                atomic_json(Path(destination) / "final.json", {
                    "pod_id": receipt["pod_id"], "reason": "watchdog_exception",
                    "compute_stopped": True, "terminated": False,
                    "persistent_data_retained": True})
                return
            except Exception:
                await asyncio.sleep(min(2**attempt, 16))
        raise AccessError("Emergency compute stop is not confirmed; intervene through MCP immediately") from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    try:
        asyncio.run(guarded_watch(args.receipt, args.destination, args.interval))
    except Exception as error:
        print(f"Watchdog failed: {type(error).__name__}. Shutdown is NOT confirmed; use the control-plane tools immediately.")
        raise SystemExit(1) from None
