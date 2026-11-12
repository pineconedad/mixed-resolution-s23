import importlib.util
import subprocess
from pathlib import Path


def test_sparse_worker_stores_provided_blob_without_creating_commit(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/initialize_worker.py"
    spec = importlib.util.spec_from_file_location("initialize_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    source = tmp_path / "predictions.md"
    source.write_text("Frozen predictions\n")
    module.store_source_blobs(["predictions.md"])
    oid = subprocess.check_output(["git", "hash-object", "predictions.md"], text=True).strip()
    assert subprocess.check_output(["git", "cat-file", "-p", oid]) == source.read_bytes()
    assert subprocess.run(["git", "rev-parse", "--verify", "HEAD"], capture_output=True).returncode != 0
