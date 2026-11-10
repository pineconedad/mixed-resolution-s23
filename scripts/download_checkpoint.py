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
    parser.add_argument("--model", choices=["pi3", "vggt"])
    args = parser.parse_args()
    config_path = Path(f"configs/models/{args.model}.yaml") if args.model else Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    model = args.model or config["model"]
    directory = Path("checkpoints") / model
    path = hf_hub_download(config["checkpoint"], config.get("checkpoint_file", "model.safetensors"),
                           revision=config["checkpoint_revision"], local_dir=directory)
    atomic_json(directory / "receipt.json", {
        "checkpoint": config["checkpoint"], "revision": config["checkpoint_revision"],
        "sha256": file_hash(path), "bytes": Path(path).stat().st_size})
    print(f"Verified checkpoint: {path}")
