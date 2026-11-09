"""GPU gate for the independent Pi3 quick test. Never marks T9 as passed."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/Pi3"))

import torch
import yaml

from mixres.adapters.pi3 import load_pi3
from mixres.data.resize import load_view
from mixres.io import atomic_json, digest, read_json
from mixres.rmfa.engine import run_engine
from mixres.runtime import provenance, seed_everything


def sanity(config, checkpoint):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for checkpoint-backed T1; local tests are separate")
    seed_everything(config["seed"])
    meta = provenance(config, config["views"], checkpoint)
    manifest = read_json(config["views"])
    if manifest["config_hash"] != digest(config):
        raise RuntimeError("Saved view manifest belongs to a different configuration")
    adapter = load_pi3(checkpoint, "cuda")
    comparisons = []
    with torch.inference_mode():
        for scene in manifest["scenes"][:2]:
            # Use the planned sharp size and all 12 views for the uniform identity gate.
            views = [{**v, "hw": v["hi_hw"], "rho": 1.0, "sharp": True} for v in scene["views"]]
            images = [load_view(Path(config["data_root"]) / v["image"], v, "real").cuda() for v in views]
            native = adapter.model(torch.stack(images)[None])
            actual, trace = run_engine(adapter, images, views, "A4")
            differences = {}
            for key in ("camera_poses", "points", "local_points", "conf"):
                reference = native[key][0]
                candidate = torch.stack([v[key] for v in actual])
                error = (reference - candidate).abs().max().item()
                differences[key] = error
                if not torch.isfinite(candidate).all() or error >= 1e-4:
                    atomic_json("runs/sanity/pi3-quick-failed.json", {"scene": scene["scene"], "errors": differences, "provenance": meta})
                    raise AssertionError(f"T1 failed: {scene['scene']} {key} error={error}")
            comparisons.append({"scene": scene["scene"], "max_abs_error": differences, "global_trace": trace})
            del native, actual, images
    atomic_json("runs/sanity/pi3-quick.json", {"status": "passed", "scope": "quick-pi3-fp32-T1", "provenance": meta,
                "comparisons": comparisons, "T9": "unavailable: no historical results supplied"})
    print("Pi3 checkpoint-backed fp32 identity gate passed on two scenes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/pi3/model.safetensors")
    args = parser.parse_args()
    sanity(yaml.safe_load(Path(args.config).read_text()), args.checkpoint)
