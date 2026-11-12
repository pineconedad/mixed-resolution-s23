import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from mixres.io import atomic_json, file_hash

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--model", choices=["pi3", "vggt", "da3"])
    args = parser.parse_args()
    config_path = Path(f"configs/models/{args.model}.yaml") if args.model else Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    model = args.model or config["model"]
    directory = Path("checkpoints") / model
    if model == "da3":
        snapshot_download(config["checkpoint"], revision=config["checkpoint_revision"],
                          allow_patterns=["config.json", config["checkpoint_file"]],
                          local_dir=directory)
        path = directory / config["checkpoint_file"]
    else:
        path = Path(hf_hub_download(config["checkpoint"], config.get("checkpoint_file", "model.safetensors"),
                                    revision=config["checkpoint_revision"], local_dir=directory))
    atomic_json(directory / "receipt.json", {
        "checkpoint": config["checkpoint"], "revision": config["checkpoint_revision"],
        "sha256": file_hash(path), "bytes": Path(path).stat().st_size})
    print(f"Verified checkpoint: {path}")
