import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("verify_session", Path(__file__).resolve().parents[1] / "scripts/verify_session.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_archive(path, *, payload=b"result", checksum=None, member_name="runs/test/result.json"):
    receipt = {"files": {member_name: checksum or hashlib.sha256(payload).hexdigest()}}
    with tarfile.open(path, "w:gz") as tar:
        for name, raw in ((member_name, payload), ("session-manifest.json", json.dumps(receipt).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restore_verifies_every_member_and_is_idempotent(tmp_path):
    path = tmp_path / "session.tar.gz"
    sha = make_archive(path)
    target = tmp_path / "restored"
    module.verify_session(path, sha, target)
    module.verify_session(path, sha, target)
    assert (target / "runs/test/result.json").read_bytes() == b"result"


def test_corrupt_archive_hash_writes_nothing(tmp_path):
    path = tmp_path / "session.tar.gz"
    make_archive(path)
    with pytest.raises(ValueError, match="Archive SHA"):
        module.verify_session(path, "0" * 64, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_corrupt_member_writes_nothing(tmp_path):
    path = tmp_path / "session.tar.gz"
    sha = make_archive(path, checksum="0" * 64)
    with pytest.raises(ValueError, match="Member checksum"):
        module.verify_session(path, sha, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_archive_path_traversal_is_rejected(tmp_path):
    path = tmp_path / "session.tar.gz"
    sha = make_archive(path, member_name="../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        module.verify_session(path, sha, tmp_path / "restored")
    assert not (tmp_path / "outside").exists()


def test_restore_cannot_overwrite_different_local_data(tmp_path):
    path = tmp_path / "session.tar.gz"
    sha = make_archive(path)
    target = tmp_path / "restored"
    file = target / "runs/test/result.json"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"previous valuable result")
    with pytest.raises(FileExistsError):
        module.verify_session(path, sha, target)
    assert file.read_bytes() == b"previous valuable result"


def test_export_preserves_results_when_submodule_git_status_fails(tmp_path, monkeypatch):
    import runpy
    import subprocess
    import sys
    import mixres.io
    from mixres.safety import verified_backup

    root = tmp_path / "runs/control/recovery"
    root.mkdir(parents=True)
    (root / "state.json").write_text(json.dumps({"session_id": "recovery", "quiescent": True}))
    (root / "console.log").write_text("Source clone interrupted\n")

    def unavailable():
        raise subprocess.CalledProcessError(128, ["git", "status"])

    monkeypatch.setattr(mixres.io, "git_state", unavailable)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNPOD_POD_ID", "owned-test-pod")
    monkeypatch.setattr(sys, "argv", ["export_session.py", "--session-id", "recovery"])
    script = Path(__file__).resolve().parents[1] / "scripts/export_session.py"
    runpy.run_path(str(script), run_name="__main__")
    archive = next((tmp_path / "artifacts").glob("*.tar.gz"))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    proof = verified_backup(archive, sha, "owned-test-pod")
    assert proof["job_quiescent"] and proof["all_members_verified"]
