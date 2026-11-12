"""Small control-plane client. Mutations use MCP, never direct REST writes."""
import asyncio
import json
import logging
import math
import os
from pathlib import Path
from contextlib import asynccontextmanager


class AccessError(RuntimeError):
    pass


def tool_payload(result):
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if not isinstance(structured, dict):
            raise AccessError("Runpod MCP result has an unexpected shape")
        if result.isError and structured.get("status") != 404:
            raise AccessError("Runpod MCP tool returned an error")
        return structured
    for item in result.content:
        if getattr(item, "type", None) == "text":
            try:
                value = json.loads(item.text)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                if result.isError and value.get("status") != 404:
                    raise AccessError("Runpod MCP tool returned an error")
                return value
    raise AccessError("Runpod MCP tool did not return a JSON object")


async def account_balance(key):
    """Read only clientBalance through the official, pinned CLI user command.

    The key is passed in the child environment, never an argument or request URL.
    REST v2 has no prepaid-balance route. Mutations stay on MCP.
    """
    try:
        from mixres.io import file_hash
        executable = Path(__file__).resolve().parents[1] / "artifacts/runpodctl-darwin-arm64"
        if file_hash(executable) != "2581913eacac577a3fd5426219962529d37172f294a80e807e7d920a65475e2c":
            raise AccessError("Official runpodctl v3.1.0 binary checksum mismatch")
        process = await asyncio.create_subprocess_exec(
            str(executable), "user", env=dict(os.environ, RUNPOD_API_KEY=key),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode != 0:
            raise AccessError("Official CLI live balance read failed")
        data = json.loads(stdout)
        value = float(data["clientBalance"])
        if not math.isfinite(value) or value < 0:
            raise AccessError("Invalid live balance")
        return value
    except AccessError:
        raise
    except Exception:
        # Never expose child output, account details, credentials or raw exceptions.
        raise AccessError("Could not verify live prepaid balance; paid work is blocked") from None


class RunpodControl:
    def __init__(self, session, tool_names):
        self.session = session
        self.tool_names = tool_names

    async def call(self, name, arguments):
        if name not in self.tool_names:
            raise AccessError("Required tool is missing from the live MCP schema")
        try:
            result = await self.session.call_tool(name, arguments)
            return tool_payload(result)
        except AccessError:
            raise
        except Exception:
            raise AccessError("Runpod control-plane request failed") from None

    async def owned_pod(self, receipt):
        if receipt.get("created_by_workflow") != "mixres-s23" or not receipt.get("created_at"):
            raise AccessError("Missing explicit workflow ownership receipt")
        pod = await self.call("get-pod", {"id": receipt["pod_id"]})
        if pod.get("status") == 404:
            return None
        if pod.get("id") != receipt["pod_id"] or pod.get("createdAt") != receipt["created_at"]:
            raise AccessError("Pod ownership receipt does not match the fresh read")
        if pod.get("status") == 403:
            raise AccessError("Pod is outside the key's permitted scope")
        return pod

    async def stop_owned(self, receipt):
        pod = await self.owned_pod(receipt)
        if pod is None or pod["status"] in {"EXITED", "ERROR"}:
            return pod
        if "stop" not in pod.get("actions", []):
            raise AccessError("Pod cannot currently be stopped")
        await self.call("pod-action", {"id": receipt["pod_id"], "body": {"action": "stop"}})
        for _ in range(10):
            after = await self.owned_pod(receipt)
            if after is None or after["status"] in {"EXITED", "ERROR"}:
                return after
            await asyncio.sleep(2)
        raise AccessError("Stop was requested but compute shutdown is not confirmed")

    async def terminate_owned(self, receipt, verified_backup):
        if receipt.get("cleanup_action") == "stop":
            raise AccessError("User-named pod is authorized for stopping, not deletion")
        # The caller must supply a receipt from the local archive verifier.
        if (verified_backup.get("pod_id") != receipt["pod_id"]
                or verified_backup.get("session_id") != receipt.get("session_id")
                or not verified_backup.get("all_members_verified")
                or not verified_backup.get("job_quiescent")):
            raise AccessError("Termination requires a locally verified backup for this pod")
        pod = await self.owned_pod(receipt)
        if pod is None:
            return
        await self.call("delete-pod", {"id": receipt["pod_id"]})
        if await self.owned_pod(receipt) is not None:
            raise AccessError("Pod termination is not confirmed")


@asynccontextmanager
async def control_plane(key):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        async with httpx.AsyncClient(headers={"Authorization": "Bearer " + key}, timeout=60) as client:
            async with streamable_http_client("https://mcp.runpod.io/", http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    names = {tool.name for tool in tools.tools}
                    required = {"get-pod", "list-pods", "pod-action", "delete-pod"}
                    if not required <= names:
                        raise AccessError("Live MCP tool schema is missing required control tools")
                    yield RunpodControl(session, names)
    except AccessError:
        raise
    except Exception:
        raise AccessError("Independent MCP authentication/control failed; do not deploy") from None
