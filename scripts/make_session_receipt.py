"""Bind a local watchdog to an explicitly created pod, not a name search."""
import argparse
import asyncio
import sys
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
    if pod.get("env", {}).get("S23_SESSION_ID") != args.session_id:
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
               "created_at": args.created_at, "session_id": args.session_id,
               "ssh_host": direct["host"], "ssh_port": direct["port"],
               "ssh_key": str(Path.home() / ".ssh/id_ed25519"), "remote_root": "/workspace/s23",
               "results_mount": "/workspace", "balance_at_create": balance,
               "hourly_total": pod["cost"] + (pod["disk"] + persistent["size"]) * 0.10 / 730,
               "reserve_usd": 3, "session_cap_usd": 1, "max_seconds": 3600,
               "heartbeat": str(destination / "heartbeat"), "done_marker": str(destination / "done.json")}
    atomic_json(destination / "receipt.json", receipt)
    print(str(destination / "receipt.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        asyncio.run(make(args))
    except Exception as error:
        print(f"Receipt verification failed: {type(error).__name__}; do not start unattended work")
        raise SystemExit(1) from None
