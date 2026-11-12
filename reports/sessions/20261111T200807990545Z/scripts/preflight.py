"""No GPU needed: verify all quick-test assets before renting compute."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from PIL import Image

from mixres.eval.pose import validate_poses
from mixres.io import digest, file_hash, read_json
from mixres.rmfa.grid import patch_grid

config = yaml.safe_load(Path("configs/quick.yaml").read_text())
manifest = read_json(config["views"])
assert manifest["config_hash"] == digest(config)
assert sorted(s["scene"] for s in manifest["scenes"]) == sorted(config["scenes"])
files = []
for scene in manifest["scenes"]:
    views = scene["views"]
    assert len(views) == config["n_views"]
    assert len(set(v["frame"] for v in views)) == config["n_views"]
    assert sum(v["sharp"] for v in views) == config["n_sharp"]
    validate_poses([v["c2w"] for v in views])
    for view in views:
        path = Path(config["data_root"]) / view["image"]
        assert file_hash(path) == view["image_sha256"]
        files.append(path)
        with Image.open(path) as image:
            image.verify()
        assert np.allclose(np.asarray(view["c2w"]) @ np.asarray(view["w2c"]), np.eye(4), atol=1e-6)
        assert patch_grid(view["hi_hw"])
        assert patch_grid(view["hw"])
    actual = sum(np.prod(patch_grid(v["hw"])) for v in views)
    assert actual == scene["actual_patch_tokens"]
    print(f"{scene['scene']}: {len(views)} verified originals/poses; actual mix patch tokens={actual}")
receipt = read_json("checkpoints/pi3/receipt.json")
checkpoint = Path("checkpoints/pi3/model.safetensors")
assert receipt["checkpoint"] == config["checkpoint"]
assert receipt["revision"] == config["checkpoint_revision"]
assert receipt["sha256"] == file_hash(checkpoint)
print(f"Checkpoint SHA-256 verified; transfer payload={(sum(p.stat().st_size for p in files) + checkpoint.stat().st_size) / 1e9:.3f} GB")
print("CPU asset preflight passed. Checkpoint-backed CUDA sanity remains pending.")
