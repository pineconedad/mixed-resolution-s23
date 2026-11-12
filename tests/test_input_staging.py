import importlib.util
from pathlib import Path

import pytest


def test_staging_binds_manifest_identity_hashes_and_permission(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/stage_development.py"
    spec = importlib.util.spec_from_file_location("stage_inputs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []
    monkeypatch.setattr(module, "require_dataset_access", lambda dataset, **kw: checks.append(dataset))
    manifest = {"dataset": "dtu", "seed": 1000,
                "scenes": [{"views": [{"image": "dtu/image.png", "image_sha256": "a" * 64}]}]}
    monkeypatch.setattr(module, "read_json", lambda path: manifest)
    assert module.collect_targets("dtu", [1000]) == {"dtu/image.png": "a" * 64}
    assert checks == ["dtu"]
    with pytest.raises(RuntimeError, match="frozen seeds"):
        module.collect_targets("dtu", [42])
    with pytest.raises(RuntimeError, match="identity mismatch"):
        module.collect_targets("hiroom", [1000])
