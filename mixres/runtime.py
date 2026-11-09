import importlib.metadata
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from mixres.io import digest, file_hash, git_state, read_json


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def provenance(config, views_path, checkpoint):
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "-C", str(root / "third_party/Pi3"), "rev-parse", "HEAD"], text=True).strip()
    model_config = root / "configs/models/pi3.yaml"
    import yaml
    pinned = yaml.safe_load(model_config.read_text())
    if revision != pinned["code_commit"]:
        raise RuntimeError("Pi3 upstream commit differs from inspected adapter")
    dirty_upstream = subprocess.check_output(["git", "-C", str(root / "third_party/Pi3"), "status", "--porcelain"], text=True).strip()
    if dirty_upstream:
        raise RuntimeError("Refusing modified Pi3 source")
    receipt = read_json(Path(checkpoint).with_name("receipt.json"))
    if receipt["checkpoint"] != config["checkpoint"] or receipt["revision"] != config["checkpoint_revision"]:
        raise RuntimeError("Checkpoint revision mismatch")
    weights_hash = file_hash(checkpoint)
    if weights_hash != receipt["sha256"]:
        raise RuntimeError("Checkpoint checksum mismatch")
    return {**git_state(), "config_hash": digest(config), "views_hash": file_hash(views_path),
            "model_config_hash": file_hash(model_config), "upstream_commit": revision,
            "checkpoint": config["checkpoint"], "checkpoint_revision": config["checkpoint_revision"],
            "checkpoint_sha256": weights_hash, "precision": config["precision"],
            "gpu_name": torch.cuda.get_device_name(), "cuda_version": torch.version.cuda,
            "driver_version": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchvision", "numpy", "Pillow", "safetensors", "huggingface-hub")}}
