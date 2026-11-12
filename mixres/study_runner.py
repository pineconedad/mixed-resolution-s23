import time
import traceback
import subprocess
import signal
import threading
from functools import wraps
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from mixres.adapters import load_adapter
from mixres.data.resize import load_view
from mixres.eval.pose import DA3_PROTOCOL, score_poses_da3
from mixres.io import atomic_json, digest, file_hash, git_state, read_json
from mixres.probes.cost import timed_inference, transformer_flops
from mixres.rmfa.arms import ARMS
from mixres.rmfa.engine import run_engine
from mixres.runtime import provenance, seed_everything


METHOD_DEFAULTS = dict(pool_op="area", lift_op="bilinear", token_interp="bilinear",
                       rmfa_layers="all", head_grid="lo", alignment_scope="global",
                       position_scale="actual")


def cell_deadline(function):
    @wraps(function)
    def bounded(*args, **kwargs):
        seconds = kwargs.get("max_seconds")
        if seconds is None:
            return function(*args, **kwargs)
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
            raise RuntimeError("Whole-cell deadlines require a Unix main thread")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError("Refusing to replace an existing deadline timer")

        def expire(signum, frame):
            raise TimeoutError("Whole-cell deadline reached")

        previous = signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return function(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    return bounded


def sanity_source_hash(model):
    root = Path(__file__).resolve().parents[1]
    paths = [root / "mixres/adapters/__init__.py", root / f"mixres/adapters/{model}.py",
             root / "mixres/data/resize.py", *sorted((root / "mixres/rmfa").glob("*.py"))]
    return digest({str(path.relative_to(root)): file_hash(path) for path in paths})


def model_identity(model, meta):
    return {**{key: meta[key] for key in
               ("upstream_commit", "checkpoint", "checkpoint_revision", "checkpoint_sha256",
                "model_config_hash")}, "sanity_source_hash": sanity_source_hash(model)}


def committed_hash(path):
    path = Path(path)
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    relative = path.resolve().relative_to(root).as_posix()
    try:
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Artifact must be committed: {path}") from error
    if path.read_bytes() != committed:
        raise RuntimeError(f"Artifact differs from committed content: {path}")
    return file_hash(path)


def expected_global_tokens(views, arm, adapter):
    size = "hi_hw" if ARMS[arm].global_grid == "hi" else "hw"
    patch = adapter.patch_size
    if any(any(side <= 0 or side % patch for side in view[size]) for view in views):
        raise ValueError("View sizes must be positive patch multiples")
    return sum((view[size][0] // patch) * (view[size][1] // patch)
               + adapter.special_count for view in views)


def materialize_views(scene, budget, arm):
    output = []
    for source in scene["views"]:
        sizes = source["sizes"][str(budget)]
        view = {key: value for key, value in source.items() if key != "sizes"}
        if arm == "A0":
            view.update(hw=sizes["uniform_hw"], hi_hw=sizes["uniform_hw"], sharp=True, rho=1.0)
        else:
            view.update(hw=sizes["hw"], hi_hw=sizes["hi_hw"])
        output.append(view)
    return output


def atomic_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    arrays = {}
    for index, prediction in enumerate(predictions):
        pi3_geometry = ("local_points" in prediction
                        and ("depth" not in prediction or "intrinsics" not in prediction))
        if pi3_geometry:
            prediction = dict(prediction)
            local_points = prediction["local_points"].detach().float()
            if "depth" not in prediction:
                prediction["depth"] = local_points[..., 2]
            if "intrinsics" not in prediction:
                from pi3.utils.geometry import recover_intrinsic_from_rays_d

                prediction["intrinsics"] = recover_intrinsic_from_rays_d(
                    local_points.unsqueeze(0), ndc_coords=False,
                    force_center_principal_point=False)[0]
                arrays[f"intrinsics_source_{index}"] = np.array(
                    "Pi3 recover_intrinsic_from_rays_d from predicted local_points; "
                    "ndc_coords=False; force_center_principal_point=False; no GT K")
        if "intrinsics" in prediction and f"intrinsics_source_{index}" not in arrays:
            arrays[f"intrinsics_source_{index}"] = np.array("model-predicted intrinsics; no GT K")
        for key in ("camera_poses", "depth", "depth_conf", "intrinsics", "points", "conf"):
            if pi3_geometry and key == "points":
                continue
            if key in prediction:
                arrays[f"{key}_{index}"] = prediction[key].detach().float().cpu().numpy()
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


class StudyRunner:
    def __init__(self, model, checkpoint, *, precision="bf16", warmup=1, timed_passes=3,
                 require_sanity=True):
        if not torch.cuda.is_available():
            raise RuntimeError("S23 inference requires CUDA")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU lacks bf16; refusing a silent precision change")
        self.model = model
        checkpoint = Path(checkpoint)
        weights = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
        self.checkpoint = checkpoint if model == "da3" and checkpoint.is_dir() else weights
        self.precision = precision
        self.warmup = warmup
        self.timed_passes = timed_passes
        self.image_cache = {}
        model_config = yaml.safe_load(Path(f"configs/models/{model}.yaml").read_text())
        self.config = model_config
        synthetic = {"model": model, "checkpoint": model_config["checkpoint"],
                     "checkpoint_revision": model_config["checkpoint_revision"],
                     "precision": precision}
        self.base_meta = provenance(synthetic, "configs/s23.yaml", weights)
        self.base_meta["model_identity"] = model_identity(model, self.base_meta)
        root = Path(__file__).resolve().parents[1]
        self.base_meta["source_hash"] = digest({str(path.relative_to(root)): file_hash(path)
                                              for path in sorted((root / "mixres").rglob("*.py"))})
        if require_sanity:
            sanity_path = Path("runs/sanity") / f"{model}-s23.json"
            if not sanity_path.exists():
                raise RuntimeError(f"Missing checkpoint-backed sanity gate: {sanity_path}")
            sanity = read_json(sanity_path)
            expected = self.base_meta["model_identity"]
            if (sanity.get("status") != "passed" or sanity.get("model_identity") != expected
                    or sanity.get("scope") != f"s23-{model}-fp32-T1"
                    or sanity.get("test_precision") != "fp32"):
                raise RuntimeError("S23 sanity gate does not match this source and checkpoint")
        self.adapter = load_adapter(model, self.checkpoint, "cuda")

    def cached_view(self, path, view, entry):
        key = digest({"view": view, "entry": "real" if view["sharp"] else entry})
        cache = self.__dict__.setdefault("image_cache", {})
        if key in cache:
            return cache[key]
        pixels = load_view(path, view, entry)
        size = pixels.numel() * pixels.element_size()
        limit = self.__dict__.get("image_cache_limit_bytes", 2 * 1024**3)
        used = self.__dict__.get("image_cache_bytes", 0)
        while cache and used + size > limit:
            previous = cache.pop(next(iter(cache)))
            used -= previous.numel() * previous.element_size()
        if size <= limit:
            cache[key] = pixels
            used += size
        self.image_cache_bytes = used
        return pixels

    @cell_deadline
    def run_cell(self, dataset, budget, arm, seed, method, *, phase, save_preds=False,
                 max_seconds=None, stop_file=None, frozen_path=None, probes=False):
        started = time.monotonic()
        if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
            raise ValueError("max_seconds must be finite and positive")

        def check_deadline():
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                raise TimeoutError("Whole-cell deadline reached")
            if stop_file and Path(stop_file).exists():
                raise TimeoutError("Stop requested")

        if arm not in ARMS:
            raise ValueError(arm)
        if probes and seed != 1000:
            raise ValueError("Mechanism probes are restricted to seed 1000")
        if set(method) - set(METHOD_DEFAULTS):
            raise ValueError("Unknown method parameters")
        method = {**METHOD_DEFAULTS, **method}
        frozen_run = seed >= 1000 or phase != "dev"
        if frozen_run:
            if seed < 1000 or phase == "dev" or frozen_path is None:
                raise RuntimeError("Frozen execution requires seed >= 1000, non-dev phase and a frozen config")
            state = git_state()
            if state["dirty_tree"]:
                raise RuntimeError("Frozen runs require a clean committed tree")
            frozen_hash = committed_hash(frozen_path)
            frozen = yaml.safe_load(Path(frozen_path).read_text())
            predictions_hash = committed_hash("frozen/s23/predictions.md")
            if (frozen.get("study") != "s23" or frozen.get("model") != self.model
                    or frozen.get("precision") != self.precision
                    or frozen.get("model_identity") != self.base_meta["model_identity"]
                    or frozen.get("study_config_hash") != file_hash("configs/s23.yaml")
                    or frozen.get("predictions_hash") != predictions_hash
                    or any(frozen.get(key) != value for key, value in method.items())):
                raise RuntimeError("Frozen configuration/model/predictions identity mismatch")
        views_path = Path("runs/views/s23") / dataset / f"seed{seed}.json"
        manifest = read_json(views_path)
        if manifest["dataset"] != dataset or manifest["seed"] != seed or budget not in manifest["budgets"]:
            raise RuntimeError("View manifest does not match the requested cell")
        if frozen_run:
            bound_views = frozen.get("view_hashes", {})
            if dataset in {"dtu", "hiroom"}:
                appendix_path = "frozen/s23/heldout-views.json"
                committed_hash(appendix_path)
                appendix = read_json(appendix_path)
                if (appendix.get("predictions_hash") != predictions_hash
                        or appendix.get("frozen_hashes", {}).get(self.model) != frozen_hash):
                    raise RuntimeError("Held-out appendix belongs to different frozen predictions/config")
                bound_views = appendix.get("view_hashes", {})
            if not isinstance(bound_views, dict) or bound_views.get(f"{dataset}/seed{seed}") != file_hash(views_path):
                raise RuntimeError("Frozen view manifest hash mismatch")
        seed_everything(seed)
        cell = {"study": "s23", "baseline_protocol": "fresh_v1", "model": self.model,
                "dataset": dataset, "budget": budget, "arm": arm, "seed": seed,
                "phase": phase, "precision": self.precision, **method}
        meta = {**self.base_meta, "cell_hash": digest(cell), "views_hash": file_hash(views_path)}
        if frozen_run:
            meta.update(frozen_hash=frozen_hash, predictions_hash=predictions_hash)
        # Git state and hardware remain provenance, not scientific resume keys.
        scientific = {key: value for key, value in meta.items() if key not in
                      {"git_commit", "dirty_tree", "gpu_name", "driver_version", "cpu_threads"}}
        scientific.update(warmup=self.warmup, timed_passes=self.timed_passes,
                          metric_protocol=DA3_PROTOCOL, save_preds=save_preds, probes=probes)
        timing = {key: meta.get(key) for key in
                  ("gpu_name", "driver_version", "cpu_threads", "cuda_version", "versions")}
        if phase == "dev":
            variant = "baseline" if arm in {"A2", "A3"} else digest(method)[:12]
            output = Path("runs/dev") / self.model / dataset / str(budget) / arm / variant / f"seed{seed}"
        else:
            output = Path("runs/s23") / self.model / dataset / str(budget) / arm / f"seed{seed}"
        records = []
        deadline_reached = False
        for scene in manifest["scenes"]:
            if stop_file and Path(stop_file).exists():
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                break
            scene_path = output / "scenes" / f"{scene['scene']}.json"
            identity = digest({"scientific": scientific, "scene": scene["scene"]})
            if scene_path.exists():
                previous = read_json(scene_path)
                if previous["identity"] != identity:
                    raise RuntimeError(f"Refusing changed provenance at {scene_path}")
                if previous.get("timing_provenance") != timing:
                    print(f"Timing runtime changed at {scene_path}; timings remain in separate groups", flush=True)
                records.append(previous)
                continue
            record = {"scene": scene["scene"], "identity": identity, "status": "failed",
                      "provenance": meta, "timing_provenance": timing}
            try:
                check_deadline()
                # Resume must not depend on how many earlier scenes ran this session.
                seed_everything(int(digest({"seed": seed, "scene": scene["scene"]})[:8], 16))
                views = materialize_views(scene, budget, arm)
                images = []
                for view in views:
                    check_deadline()
                    image_path = Path(view["image"])
                    if file_hash(image_path) != view["image_sha256"]:
                        raise RuntimeError(f"Image checksum mismatch: {image_path}")
                    images.append(self.cached_view(image_path, view, ARMS[arm].entry).cuda())

                def forward():
                    check_deadline()
                    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                                if self.precision == "bf16" else nullcontext())
                    with torch.inference_mode(), autocast:
                        result = run_engine(self.adapter, images, views, arm, **method)
                    check_deadline()
                    return result

                (predictions, trace), costs = timed_inference(forward, self.warmup, self.timed_passes)
                check_deadline()
                for prediction in predictions:
                    if any(not torch.isfinite(value).all() for value in prediction.values()
                           if torch.is_tensor(value)):
                        raise ValueError("Nonfinite model output")
                predicted_c2w = torch.stack([item["camera_poses"] for item in predictions]).float().cpu().numpy()
                metrics = score_poses_da3(predicted_c2w, np.asarray([view["c2w"] for view in views]))
                expected = expected_global_tokens(views, arm, self.adapter)
                if not trace or any(layer["total_tokens"] != expected for layer in trace):
                    raise AssertionError("Global token count does not match view sizes")
                flops = transformer_flops(trace, self.adapter.embed_dim, self.adapter.mlp_ratio)
                prediction_path = output / "predictions" / f"{scene['scene']}.npz"
                if save_preds:
                    check_deadline()
                    atomic_predictions(prediction_path, predictions)
                probe_files = None
                if probes:
                    from mixres.probes.study import observe_scene
                    probe_files = observe_scene(self.adapter, images, views, arm, method,
                                                self.precision, output / "probes" / scene["scene"])
                check_deadline()
                record.update(status="completed", **metrics, **costs, global_tokens=expected,
                              global_trace=trace,
                              frame_tokens_total=trace[0]["frame_total_tokens"], flops=flops,
                               predictions_file=str(prediction_path) if save_preds else None,
                               predicted_c2w=predicted_c2w.tolist(), reconstruction=None,
                               views=views, probes=probe_files)
                del predictions, images
            except Exception as error:
                deadline_reached = isinstance(error, TimeoutError)
                record.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                torch.cuda.empty_cache()
            atomic_json(scene_path, record)
            output.mkdir(parents=True, exist_ok=True)
            with (output / "log.txt").open("a") as log:
                log.write(f"{scene['scene']} {record['status']} latency={record.get('latency_s')} "
                          f"memory={record.get('peak_mem_gb')} auc3={record.get('auc3')}\n")
            records.append(record)
            atomic_json(output / "result.json", self._result(cell, meta, manifest, records))
            print(f"{self.model} {dataset} {budget} {arm} {scene['scene']}: {record['status']}", flush=True)
            if deadline_reached:
                break
        atomic_json(output / "result.json", self._result(cell, meta, manifest, records))
        return read_json(output / "result.json")

    @staticmethod
    def _result(cell, meta, manifest, records):
        expected = [scene["scene"] for scene in manifest["scenes"]]
        completed = {record["scene"] for record in records if record["status"] == "completed"}
        timing_groups = {}
        for record in records:
            key = digest(record["timing_provenance"])
            timing_groups.setdefault(key, {"provenance": record["timing_provenance"], "scenes": []})["scenes"].append(record["scene"])
        return {**cell, "metric_protocol": DA3_PROTOCOL, "auc_units": "percent",
                "timing_comparable": len(timing_groups) <= 1, "timing_groups": timing_groups,
                "timing_warning": "Do not pool timing across runtime groups" if len(timing_groups) > 1 else None,
                "provenance": meta, "scenes_expected": expected,
                "status": "completed" if completed == set(expected) else "partial_or_failed",
                "per_scene": records}
