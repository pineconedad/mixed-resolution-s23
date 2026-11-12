import argparse
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from mixres.io import digest, file_hash, read_json
from mixres.study_runner import METHOD_DEFAULTS, committed_hash, model_identity


REQUIRED_DATASETS = {"eth3d", "scannetpp", "7scenes"}


def write_new(path, text):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def freeze(ranking_path):
    destination = Path("frozen/s23")
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite frozen artifact: {destination}")
    ranking = read_json(ranking_path)
    if (ranking.get("status") != "complete" or set(ranking["datasets"]) != REQUIRED_DATASETS
            or ranking.get("model") != "pi3" or ranking.get("seed") != 42
            or ranking.get("budget") != 4096):
        raise RuntimeError("Freeze requires the complete three-dataset Pi3 development sweep")
    winner = ranking["ranking"][0]
    if (set(winner["method"]) != set(METHOD_DEFAULTS)
            or winner["method_id"] != digest(winner["method"])[:12]):
        raise RuntimeError("Invalid selected method identity")
    allowed = {"pool_op": {"area", "bilinear"}, "lift_op": {"bilinear", "nearest"},
               "token_interp": {"bilinear"}, "rmfa_layers": {"all", "first_half", "first_quarter"},
               "head_grid": {"lo", "hi"}, "alignment_scope": {"global"}, "position_scale": {"actual"}}
    if any(value not in allowed[key] for key, value in winner["method"].items()):
        raise RuntimeError("Invalid selected method parameters")
    artifacts = {}
    view_hashes = {}
    for dataset in sorted(REQUIRED_DATASETS):
        for seed in (42, 1000, 2000, 3000):
            path = Path("runs/views/s23") / dataset / f"seed{seed}.json"
            if not path.exists():
                raise RuntimeError(f"Missing view manifest: {path}")
            manifest = read_json(path)
            if (manifest["dataset"] != dataset or manifest["seed"] != seed
                    or not {4096, 8192}.issubset(manifest["budgets"]) or not manifest["scenes"]):
                raise RuntimeError(f"Invalid view manifest: {path}")
            view_hashes[f"{dataset}/seed{seed}"] = file_hash(path)
    for model in ("pi3", "vggt", "da3"):
        model_config = yaml.safe_load(Path(f"configs/models/{model}.yaml").read_text())
        receipt = read_json(Path("checkpoints") / model / "receipt.json")
        if (receipt["checkpoint"] != model_config["checkpoint"]
                or receipt["revision"] != model_config["checkpoint_revision"]
                or len(receipt["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in receipt["sha256"])):
            raise RuntimeError(f"Invalid checkpoint receipt: {model}")
        identity = model_identity(model, dict(upstream_commit=model_config["code_commit"],
                                  checkpoint=model_config["checkpoint"],
                                  checkpoint_revision=model_config["checkpoint_revision"],
                                  checkpoint_sha256=receipt["sha256"],
                                  model_config_hash=file_hash(f"configs/models/{model}.yaml")))
        frozen = {"study": "s23", "baseline_protocol": "fresh_v1", "model": model,
                  "model_identity": identity, "study_config_hash": file_hash("configs/s23.yaml"),
                  "checkpoint": model_config["checkpoint"],
                  "checkpoint_revision": model_config["checkpoint_revision"],
                  "precision": "bf16", "n_views": 12, "n_sharp": 4, "rho": 2.0,
                  "uniform": "equal_budget_all_12_views", "view_hashes": view_hashes,
                  "selected_on": "pi3_equal_dataset_weighted_dev_auc3",
                  "selected_method_id": winner["method_id"], **winner["method"]}
        artifacts[f"{model}.yaml"] = frozen
    predictions = """# S23 predictions

These thresholds were specified before opening any seed-1000 or held-out result.

- H1: On Pi3, A4 recovers at least half of A3's gain over A2. R >= 0.5 and the 95% interval for A4-A2 excludes zero.
- H2: On DA3-Large-1.1, A4 does not hurt the architecture-specific A2 control by more than 1 AUC point. A4-A2 >= -1.
- H3: On VGGT-1B, RMFA has no material effect. Absolute A4-A2 is below 1 AUC point.
- H4: On frozen seeds, A3 recovers at least 70% of the A1-to-A8 gap for Pi3 and DA3.

Historical T9 remains unavailable because no S1-S5 targets or exact S7 configs were supplied.
"""
    # Serialize and validate everything before creating any frozen output.
    import hashlib
    predictions_hash = hashlib.sha256(predictions.encode()).hexdigest()
    texts = {name: yaml.safe_dump({**config, "predictions_hash": predictions_hash}, sort_keys=True)
             for name, config in artifacts.items()}
    texts["predictions.md"] = predictions
    message = f"Frozen method {winner['method_id']} at pooled dev AUC@3={winner['pooled_auc3']}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".s23-", dir=destination.parent))
    try:
        for name, text in texts.items():
            write_new(staging / name, text)
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite frozen artifact: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(message)


def bind_heldout_views():
    """Append data hashes after freeze without changing any selected method."""
    predictions_hash = committed_hash("frozen/s23/predictions.md")
    hashes = {model: committed_hash(f"frozen/s23/{model}.yaml")
              for model in ("pi3", "vggt", "da3")}
    views = {}
    for dataset in ("dtu", "hiroom"):
        for seed in (1000, 2000, 3000):
            path = Path("runs/views/s23") / dataset / f"seed{seed}.json"
            manifest = read_json(path)
            if manifest["dataset"] != dataset or manifest["seed"] != seed or not manifest["scenes"]:
                raise RuntimeError(f"Invalid held-out manifest: {path}")
            views[f"{dataset}/seed{seed}"] = file_hash(path)
    import json
    write_new("frozen/s23/heldout-views.json", json.dumps({
        "predictions_hash": predictions_hash, "frozen_hashes": hashes,
        "view_hashes": views}, indent=2, sort_keys=True) + "\n")
    print("Held-out metadata hashes bound. Commit this appendix before inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", default="runs/dev/ranking.json")
    parser.add_argument("--bind-heldout-views", action="store_true")
    args = parser.parse_args()
    if args.bind_heldout_views:
        bind_heldout_views()
    else:
        freeze(args.ranking)
