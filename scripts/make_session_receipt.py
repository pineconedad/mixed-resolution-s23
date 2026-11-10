"""Bind a local watchdog to an explicitly created pod, not a name search."""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.credentials import runpod_key
from mixres.io import atomic_json
from mixres.runpod_control import account_balance, control_plane


async def make(args):
    key = runpod_key()
    async with control_plane(key) as control:
        pod = await control.call("get-pod", {"id": args.pod_id})
    if pod.get("id") != args.pod_id or pod.get("createdAt") != args.created_at:
        raise RuntimeError("Fresh pod read does not match the explicit create receipt")
    if not args.user_named and pod.get("env", {}).get("S23_SESSION_ID") != args.session_id:
        raise RuntimeError("Pod does not carry this workflow's create-time session marker")
    direct = pod.get("ssh", {}).get("direct")
    if not direct:
        raise RuntimeError("Direct SSH is not ready")
    persistent = pod.get("mounts", {}).get("persistent")
    if not persistent or persistent.get("path") != "/workspace":
        raise RuntimeError("Persistent workspace was not provisioned")
    balance = await account_balance(key)
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {"created_by_workflow": "mixres-s23", "pod_id": args.pod_id,
               "resource_origin": "user_named" if args.user_named else "conversation_created",
               "cleanup_action": "stop" if args.user_named else "terminate",
               "created_at": args.created_at, "session_id": args.session_id,
               "budget_start_epoch": time.time(),
               "ssh_host": direct["host"], "ssh_port": direct["port"],
               "ssh_key": str(Path.home() / ".ssh/id_ed25519"), "remote_root": "/workspace/s23",
               "results_mount": "/workspace", "balance_at_create": balance,
               "model": args.model, "validation_only": args.validation_only,
               "worker_python": args.worker_python, "skip_unit_tests": args.skip_unit_tests,
               "config": args.config, "env_path": args.env_path,
               "hourly_total": pod["cost"] + (pod["disk"] + persistent["size"]) * 0.10 / 730,
               "reserve_usd": 3, "session_cap_usd": 1, "max_seconds": 3600,
               "heartbeat": str(destination / "heartbeat"), "done_marker": str(destination / "done.json")}
    if args.known_hosts:
        trust = Path(args.known_hosts).resolve()
        if not trust.is_file():
            raise RuntimeError("Recovery host-key trust file does not exist")
        receipt["ssh_known_hosts"] = str(trust)
    atomic_json(destination / "receipt.json", receipt)
    print(str(destination / "receipt.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--model", choices=["pi3", "vggt"], default="pi3")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--known-hosts")
    parser.add_argument("--user-named", action="store_true")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--worker-python", default="python")
    parser.add_argument("--config")
    parser.add_argument("--env-path", default=".venv")
    args = parser.parse_args()
    try:
        asyncio.run(make(args))
    except Exception as error:
        print(f"Receipt verification failed: {type(error).__name__}; do not start unattended work")
        raise SystemExit(1) from None
