"""Read-only verification. Never create a pod to test credentials."""
import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixres.credentials import runpod_key
from mixres.io import atomic_json
from mixres.runpod_control import AccessError, account_balance, control_plane


async def check(output):
    key = runpod_key()
    balance = await account_balance(key)
    async with control_plane(key) as control:
        pods = await control.call("list-pods", {})
        # Do not export raw pod objects; they can include configuration/secret fields.
        result = {"api_key_authenticated": True, "live_balance_usd": balance,
                  "observed_utc": datetime.now(timezone.utc).isoformat(),
                  "independent_mcp_authenticated": True,
                  "pods": [{"id": p["id"], "status": p["status"]} for p in pods.get("pods", [])],
                  "watchdog_live_tested": None}
    atomic_json(output, result)
    print(f"Live balance: ${balance:.4f}; independent MCP access verified")
    print("No resource was created. This read-only check does not assess watchdog execution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/control/access.json")
    args = parser.parse_args()
    try:
        asyncio.run(check(args.output))
    except Exception as error:
        # Never print HTTP exception objects or credential-bearing request URLs.
        reason = str(error) if isinstance(error, AccessError) else type(error).__name__
        print(f"Access verification blocked: {reason}. No paid action taken.")
        raise SystemExit(1) from None
