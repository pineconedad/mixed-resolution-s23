import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from huggingface_hub import hf_hub_download

from mixres.io import atomic_json, file_hash

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    path = hf_hub_download(config["checkpoint"], "model.safetensors",
                           revision=config["checkpoint_revision"], local_dir="checkpoints/pi3")
    atomic_json("checkpoints/pi3/receipt.json", {
        "checkpoint": config["checkpoint"], "revision": config["checkpoint_revision"],
        "sha256": file_hash(path), "bytes": Path(path).stat().st_size})
    print(f"Verified checkpoint: {path}")
