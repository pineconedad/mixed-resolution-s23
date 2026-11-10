import json
from pathlib import Path

import pytest
import yaml

from mixres.io import atomic_json, file_hash
from mixres.runtime import validate_views


def test_diagnostic_uses_exact_original_views_without_rewriting_them():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/diagnostic.yaml").read_text())
    config["views"] = str(root / config["views"])
    manifest = json.loads(Path(config["views"]).read_text())
    before = file_hash(config["views"])
    validate_views(config, manifest)
    assert file_hash(config["views"]) == before
    assert config["arms"] == ["A1", "A2", "A3", "A8", "A4", "A5", "A6", "A7"]
    assert config["seed"] == 42


def test_reused_views_require_both_source_hash_and_exact_bytes(tmp_path):
    path = tmp_path / "views.json"
    config = {"source_views_config_hash": "original", "views": str(path), "seed": 42,
              "dataset": "test", "budget_tokens": 4096, "scenes": ["scene"]}
    manifest = {"config_hash": "original", "seed": 42, "dataset": "test", "budget_tokens": 4096,
                "scenes": [{"scene": "scene"}]}
    atomic_json(path, manifest)
    config["views_sha256"] = file_hash(path)
    validate_views(config, manifest)
    with pytest.raises(RuntimeError):
        validate_views({**config, "seed": 1000}, manifest)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="checksum"):
        validate_views(config, manifest)
