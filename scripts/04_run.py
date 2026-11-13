import argparse
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/Pi3"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/vggt"))

import numpy as np
import torch
import yaml

from mixres.adapters import load_adapter
from mixres.data.resize import load_view
from mixres.eval.pose import PROTOCOL, score_poses
from mixres.io import atomic_json, digest, file_hash, read_json
from mixres.probes.cost import timed_inference
from mixres.rmfa.arms import ARMS
from mixres.rmfa.engine import run_engine
from mixres.rmfa.grid import budget_sizes
from mixres.runtime import provenance, seed_everything, validate_views, sanity_path


METHOD_KEYS = ("pool_op", "lift_op", "token_interp", "head_grid", "rmfa_layers")
OPTIONAL_KEYS = ("alignment_scope", "position_scale", "calibrate", "logit_bias", "kv_pool")
RUNNER_KEYS = ("uniform_budget",)  # A0 only: uniform size at another token budget


def parse_variant(entry):
    """An arm entry is either "A4" or {name: ..., arm: A4, <method overrides>}."""
    if isinstance(entry, str):
        return entry, entry, {}
    overrides = dict(entry)
    name, arm = overrides.pop("name"), overrides.pop("arm")
    allowed = set(METHOD_KEYS) | set(OPTIONAL_KEYS) | (set(RUNNER_KEYS) if arm == "A0" else set())
    if set(overrides) - allowed or arm not in ARMS or name in ARMS:
        raise ValueError(f"Invalid variant: {entry}")
    return name, arm, overrides


def scene_views(views, arm, config, uniform_budget=None):
    """A0 resizes every view to one size at the mix's nominal token budget (or uniform_budget)."""
    sizes = config.get("sizes")
    if sizes:
        # Track D: replace every view size, e.g. with exactly 4:3 sizes, keeping the same frames.
        if arm == "A0":
            return [{**view, "hw": list(sizes["uniform"]), "hi_hw": list(sizes["uniform"]),
                     "sharp": True, "rho": 1.0} for view in views]
        return [{**view, "hw": list(sizes["hi"] if view["sharp"] else sizes["lo"]),
                 "hi_hw": list(sizes["hi"])} for view in views]
    if arm != "A0":
        return views
    uniform = budget_sizes(uniform_budget or config["budget_tokens"], len(views), 0, 1.0,
                           config.get("aspect", 4 / 3))[0]
    return [{**view, "hw": list(uniform), "hi_hw": list(uniform), "sharp": True, "rho": 1.0}
            for view in views]


def run(config, checkpoint, max_seconds=None, stop_file=None):
    variants = [parse_variant(entry) for entry in config["arms"]]
    if len({name for name, _, _ in variants}) != len(variants):
        raise ValueError("Duplicate arm/variant names")
    salvation = config["study"].startswith("salvation_")
    # Salvation studies may use extra development draws; seeds from 1000 up belong to frozen runs.
    seed_allowed = 42 <= config["seed"] < 1000 if salvation else config["seed"] == 42
    if (not salvation and config["study"] not in {"quick", "diagnostic", "alignment"}) or not seed_allowed or config["model"] not in {"pi3", "vggt"}:
        raise RuntimeError("Only exploratory development is enabled; full-study prerequisites are not satisfied")
    if not torch.cuda.is_available():
        raise RuntimeError("Run on a CUDA GPU; no CPU or MPS substitution for research runs")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU lacks bf16 support; never silently change precision")
    seed_everything(config["seed"])
    meta = provenance(config, config["views"], checkpoint)
    manifest = read_json(config["views"])
    validate_views(config, manifest)
    sanity = read_json(sanity_path(config))
    if sanity["status"] != "passed" or sanity["scope"] != f"{config['study']}-{config['model']}-fp32-T1" or sanity["provenance"] != meta:
        raise RuntimeError("Run 01_sanity.py with this exact code/config/checkpoint/GPU first")
    adapter = load_adapter(config["model"], checkpoint, "cuda")
    started = time.monotonic()

    def identity_of(scene, name, arm_name, overrides):
        fields = {"provenance": meta, "scene": scene, "arm": arm_name, "metric": PROTOCOL}
        if name != arm_name or overrides:
            fields.update(variant=name, overrides=overrides)
        return digest(fields)

    def method_of(overrides):
        method = {k: config[k] for k in METHOD_KEYS}
        method.update({k: config[k] for k in OPTIONAL_KEYS if k in config})
        return {**method, **{k: v for k, v in overrides.items() if k not in RUNNER_KEYS}}

    def publish_result(out, name, arm_name, overrides):
        records = []
        for s in manifest["scenes"]:
            path = out / "scenes" / f"{s['scene']}.json"
            if path.exists():
                record = read_json(path)
                expected = identity_of(s["scene"], name, arm_name, overrides)
                if record["identity"] != expected:
                    raise RuntimeError(f"Existing result has different provenance: {path}; archive it before rerunning")
                records.append(record)
        atomic_json(out / "result.json", {"study": config["study"], "exploratory": True, "model": config["model"], "dataset": config["dataset"],
                    "budget": config["budget_tokens"], "arm": arm_name, "variant": name, "seed": config["seed"],
                    "metric_protocol": PROTOCOL, "auc_units": "percent", "provenance": meta,
                    "method_parameters": {**{"alignment_scope": "global", "position_scale": "actual",
                                             "calibrate": "none", "logit_bias": "none", "kv_pool": "none"},
                                          **method_of(overrides), **{k: overrides[k] for k in RUNNER_KEYS if k in overrides}},
                    "scenes_expected": [s["scene"] for s in manifest["scenes"]],
                    "status": "completed" if len(records) == len(manifest["scenes"]) and all(r["status"] == "completed" for r in records) else "partial_or_failed",
                    "per_scene": records})

    # Scene-major order preserves complete paired snapshots if a session ends early.
    for scene in manifest["scenes"]:
        for name, arm_name, overrides in variants:
            if stop_file is not None and Path(stop_file).exists():
                print("Stop requested between evaluations; completed scenes are durable", flush=True)
                return
            out = Path("runs") / config["study"] / config["model"] / config["dataset"] / str(config["budget_tokens"]) / name / f"seed{config['seed']}"
            scene_path = out / "scenes" / f"{scene['scene']}.json"
            identity = identity_of(scene["scene"], name, arm_name, overrides)
            if scene_path.exists():
                previous = read_json(scene_path)
                if previous["identity"] != identity:
                    raise RuntimeError(f"Existing result has different provenance: {scene_path}; archive it before rerunning")
                if previous["status"] == "completed":
                    publish_result(out, name, arm_name, overrides)
                    print(f"skip {scene['scene']} {name}: verified complete", flush=True)
                    continue
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                print("Session time limit reached between scenes/arms; outputs saved. Pod is NOT stopped automatically.", flush=True)
                return
            record = {"scene": scene["scene"], "arm": arm_name, "variant": name, "identity": identity, "status": "failed"}
            try:
                images = []
                views = scene_views(scene["views"], arm_name, config, overrides.get("uniform_budget"))
                for view in views:
                    path = Path(config["data_root"]) / view["image"]
                    if file_hash(path) != view["image_sha256"]:
                        raise RuntimeError(f"Image checksum mismatch: {path}")
                    images.append(load_view(path, view, ARMS[arm_name].entry).cuda())
                def forward():
                    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if config["precision"] == "bf16" else nullcontext()
                    with torch.inference_mode(), autocast:
                        return run_engine(adapter, images, views, arm_name, **method_of(overrides))
                (predictions, trace), costs = timed_inference(forward, config["warmup"], config["timed_passes"])
                for prediction in predictions:
                    if any(not torch.isfinite(value).all() for value in prediction.values()):
                        raise ValueError("Nonfinite model output")
                c2w = torch.stack([p["camera_poses"] for p in predictions]).cpu().numpy()
                metrics = score_poses(c2w, np.asarray([v["c2w"] for v in views]))
                record.update(status="completed", **metrics, **costs, global_trace=trace, predicted_c2w=c2w.tolist())
                del predictions, images
            except Exception as error:
                record.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                torch.cuda.empty_cache()
            atomic_json(scene_path, record)
            out.mkdir(parents=True, exist_ok=True)
            with (out / "log.txt").open("a") as log:
                log.write(f"{scene['scene']} {record['status']} latency={record.get('latency_s')} memory={record.get('peak_mem_gb')} auc3={record.get('auc3')}\n")
            publish_result(out, name, arm_name, overrides)
            print(f"{scene['scene']} {name}: {record['status']} AUC@3={record.get('auc3')}"
                  + (f" ERROR {record['error']}" if "error" in record else ""), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--stop-file")
    parser.add_argument("--model", choices=["pi3", "vggt", "da3"])
    parser.add_argument("--dataset", choices=["eth3d", "scannetpp", "7scenes", "dtu", "hiroom"])
    parser.add_argument("--budget", type=int, choices=[4096, 8192])
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--frozen")
    parser.add_argument("--phase", default="main")
    parser.add_argument("--probes", choices=["on", "off"], default="off")
    parser.add_argument("--save-preds", action="store_true")
    args = parser.parse_args()
    if args.probes == "on" and (not args.model or args.seed != 1000):
        parser.error("--probes on requires an S23 model and seed 1000")
    if args.model:
        if None in (args.dataset, args.budget, args.arm, args.seed):
            parser.error("S23 runs require --model, --dataset, --budget, --arm and --seed")
        from mixres.study_runner import StudyRunner
        frozen = yaml.safe_load(Path(args.frozen).read_text()) if args.frozen else {}
        if args.seed >= 1000 or args.phase != "dev":
            if not args.frozen:
                parser.error("Frozen execution requires --frozen")
        method = {key: frozen.get(key, default) for key, default in
                  (("pool_op", "area"), ("lift_op", "bilinear"), ("token_interp", "bilinear"),
                   ("rmfa_layers", "all"), ("head_grid", "lo"),
                   ("alignment_scope", "global"), ("position_scale", "actual"))}
        runner = StudyRunner(args.model, args.checkpoint or f"checkpoints/{args.model}",
                             precision=frozen.get("precision", "bf16"))
        runner.run_cell(args.dataset, args.budget, args.arm, args.seed, method,
                        phase=args.phase, save_preds=args.save_preds,
                        max_seconds=args.max_seconds, stop_file=args.stop_file,
                         frozen_path=args.frozen, probes=args.probes == "on")
    else:
        config = yaml.safe_load(Path(args.config).read_text())
        run(config, args.checkpoint or f"checkpoints/{config['model']}/model.safetensors", args.max_seconds, args.stop_file)
