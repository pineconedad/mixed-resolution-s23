"""Fetch pinned executable source without cloning demo/data assets."""
import argparse
import subprocess
from pathlib import Path


MODELS = {
    "pi3": ("Pi3", "https://github.com/yyfz/Pi3.git", "pi3",
            "9fa3ddb3f8d53041f8b2738df404f62223bbaa7b"),
    "da3": ("Depth-Anything-3", "https://github.com/ByteDance-Seed/Depth-Anything-3.git",
            "src/depth_anything_3", "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"),
    "vggt": ("vggt", "https://github.com/facebookresearch/vggt.git", "vggt",
             "a288dd0f14786c93483e45524328726ab7b1b4ce"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["da3", "pi3"])
    args = parser.parse_args()
    for model in args.models:
        directory, url, source, revision = MODELS[model]
        target = Path("third_party") / directory
        if not (target / ".git").exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", url, str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "sparse-checkout", "set", source], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision or subprocess.check_output(
                ["git", "-C", str(target), "status", "--porcelain"], text=True).strip():
            raise RuntimeError(f"Model source does not match clean pin: {model}")
        print(f"{model}: clean pinned sparse source ready", flush=True)
