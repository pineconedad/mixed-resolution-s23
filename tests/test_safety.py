import asyncio
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from mixres.credentials import runpod_key
from mixres.io import file_hash, read_json
from mixres.runpod_control import AccessError, RunpodControl, tool_payload
from mixres.safety import Budget, verified_backup


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_has_finite_deadline_and_reserves_backup_time():
    budget = Budget(14, 0.6)
    assert budget.hard_seconds == 3600
    assert budget.stop_reason(2999, 13) is None
    assert budget.stop_reason(3000, 13) == "session_limit"
    assert budget.stop_reason(1, 3.01) == "credit_reserve"
    assert budget.stop_reason(1, None) == "balance_unverified"
    assert budget.stop_reason(1, float("nan")) == "balance_unverified"
    assert budget.stop_reason(-1, 14) == "invalid_clock"
    assert Budget(3.01, 0.6).stop_reason(0, 3.01) == "credit_reserve"


@pytest.mark.parametrize("kwargs", [{"hourly_total": 0}, {"hourly_total": float("nan")},
                                   {"reserve_usd": 0}, {"session_cap_usd": 0}, {"max_seconds": 10}])
def test_invalid_safety_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Budget(**{**dict(balance_at_start=14, hourly_total=0.6), **kwargs})


def test_credential_entry_is_private_ignored_and_never_echoed(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".env.*\n")
    setter = script("set_runpod_key")
    key = "unit-test-secret-not-a-real-api-key"
    path = setter.store_key(tmp_path, key)
    assert path.stat().st_mode & 0o777 == 0o600
    assert runpod_key(tmp_path) == key
    assert key not in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        setter.store_key(tmp_path, "another-test-value")
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        runpod_key(tmp_path)


def test_credential_entry_refuses_tracked_or_unignored_file(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(RuntimeError, match="not ignored"):
        script("set_runpod_key").store_key(tmp_path, "test-only")


def test_tool_errors_never_leak_server_text():
    raw = SimpleNamespace(structuredContent=None, isError=True,
                          content=[SimpleNamespace(type="text", text='{"error":"unit-test-secret"}')])
    with pytest.raises(AccessError) as error:
        tool_payload(raw)
    assert "unit-test-secret" not in str(error.value)
    missing = SimpleNamespace(structuredContent={"status": 404}, isError=True, content=[])
    assert tool_payload(missing)["status"] == 404


def test_only_receipted_pod_can_be_mutated_and_deletion_requires_final_backup():
    calls = []
    state = {"id": "unit-test-pod", "createdAt": "unit-test-time", "status": "RUNNING",
             "actions": ["stop", "terminate"]}

    class Session:
        async def call_tool(self, name, args):
            calls.append(name)
            if name == "pod-action":
                state["status"] = "EXITED"
                payload = {"status": "EXITED"}
            elif name == "delete-pod":
                state["status"] = 404
                payload = {"status": 204}
            else:
                payload = dict(state)
            return SimpleNamespace(structuredContent=payload, isError=False, content=[])

    receipt = {"created_by_workflow": "mixres-s23", "created_at": "unit-test-time",
               "pod_id": "unit-test-pod", "session_id": "testsession"}
    control = RunpodControl(Session(), {"get-pod", "pod-action", "delete-pod"})
    with pytest.raises(AccessError):
        asyncio.run(control.owned_pod({**receipt, "created_at": "wrong-creation"}))
    assert "pod-action" not in calls
    with pytest.raises(AccessError):
        asyncio.run(control.terminate_owned(receipt, {"pod_id": "unit-test-pod", "all_members_verified": True}))
    assert "delete-pod" not in calls
    asyncio.run(control.stop_owned(receipt))
    assert state["status"] == "EXITED"
    proof = {"pod_id": "unit-test-pod", "session_id": "testsession",
             "all_members_verified": True, "job_quiescent": True}
    asyncio.run(control.terminate_owned(receipt, proof))
    assert state["status"] == 404


def test_backup_proof_is_bound_to_pod_session_and_immutable_bytes(tmp_path):
    archive = tmp_path / "session.tar.gz"
    raw = b"result data"
    manifest = {"pod_id": "test-pod", "session_id": "testsession", "job_quiescent": True,
                "files": {"runs/result.json": hashlib.sha256(raw).hexdigest()}}
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in (("runs/result.json", raw), ("session-manifest.json", json.dumps(manifest).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    proof = verified_backup(archive, file_hash(archive), "test-pod")
    assert proof["job_quiescent"] and proof["session_id"] == "testsession"
    with pytest.raises(ValueError, match="bound"):
        verified_backup(archive, file_hash(archive), "different-pod")


def test_remote_worker_smoke_writes_truthful_quiescence(tmp_path, monkeypatch):
    module = script("run_session")
    # Redirect the module's repository root into a temporary fixture.
    fake_script = tmp_path / "scripts/run_session.py"
    fake_script.parent.mkdir()
    monkeypatch.setattr(module, "__file__", str(fake_script))
    assert module.session("testsession", "unused.yaml", smoke_only=True) == 0
    state = read_json(tmp_path / "runs/control/testsession/state.json")
    assert state["quiescent"] is True and state["exit_code"] == 0
    with pytest.raises(FileExistsError):
        module.session("testsession", "unused.yaml", smoke_only=True)


def test_watchdog_refuses_container_only_results():
    module = script("watch_pod")
    receipt = {"results_mount": "/workspace", "remote_root": "/workspace/s23", "ssh_host": "testhost", "ssh_port": 22}
    with pytest.raises(AccessError, match="persistent"):
        module.verify_storage_and_ssh({"mounts": {}}, receipt)
    pod = {"mounts": {"persistent": {"path": "/workspace", "size": 20}},
           "ssh": {"direct": {"host": "testhost", "port": 22}}}
    module.verify_storage_and_ssh(pod, receipt)
    with pytest.raises(AccessError, match="outside"):
        module.verify_storage_and_ssh(pod, {**receipt, "remote_root": "/tmp/s23"})


@pytest.mark.parametrize("final_backup_ok", [True, False])
def test_watchdog_never_terminates_from_a_periodic_live_snapshot(tmp_path, monkeypatch, final_backup_ok):
    module = script("watch_pod")
    from mixres.io import atomic_json
    receipt = {"created_by_workflow": "mixres-s23", "pod_id": "test-pod",
               "created_at": datetime.now(timezone.utc).isoformat(), "session_id": "testsession",
               "balance_at_create": 14, "hourly_total": 0.6, "results_mount": "/workspace",
               "remote_root": "/workspace/s23", "ssh_host": "testhost", "ssh_port": 22,
               "done_marker": str(tmp_path / "done")}
    atomic_json(tmp_path / "receipt.json", receipt)
    (tmp_path / "done").touch()
    actions = []
    backups = []

    class Control:
        async def owned_pod(self, _):
            return {"status": "RUNNING", "mounts": {"persistent": {"path": "/workspace"}},
                    "ssh": {"direct": {"host": "testhost", "port": 22}}}
        async def stop_owned(self, _):
            actions.append("stop")
        async def terminate_owned(self, _, proof):
            assert proof["job_quiescent"] is True
            actions.append("terminate")

    @asynccontextmanager
    async def connection(_):
        yield Control()

    async def balance(_):
        return 14

    def backup(*args):
        backups.append(True)
        if len(backups) == 2 and not final_backup_ok:
            raise OSError("Synthetic transfer failure")
        return {"pod_id": "test-pod", "session_id": "testsession", "all_members_verified": True,
                "job_quiescent": len(backups) == 2, "local_archive": "unit-test-archive"}

    monkeypatch.setattr(module, "runpod_key", lambda: "unit-test-only")
    monkeypatch.setattr(module, "account_balance", balance)
    monkeypatch.setattr(module, "control_plane", connection)
    monkeypatch.setattr(module, "sync_backup", backup)
    monkeypatch.setattr(module, "quiesce", lambda _: actions.append("quiesce"))
    asyncio.run(module.watch(tmp_path / "receipt.json", tmp_path / "received", interval=10))
    assert actions == (["quiesce", "stop", "terminate"] if final_backup_ok else ["quiesce", "stop"])
    final = read_json(tmp_path / "received/final.json")
    assert final["terminated"] is final_backup_ok
