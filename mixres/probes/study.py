"""Run untimed mechanism probes without contaminating latency measurements."""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from mixres.data.resize import load_view
from mixres.probes.drift import DriftObserver
from mixres.probes.entropy import EntropyObserver
from mixres.rmfa.engine import run_engine


class CombinedObserver:
    def __init__(self, *observers):
        self.observers = observers

    def frame(self, *args):
        for observer in self.observers:
            observer.frame(*args)

    def global_input(self, *args):
        for observer in self.observers:
            observer.global_input(*args)

    def tokens(self, *args):
        for observer in self.observers:
            observer.tokens(*args)


def save_arrays(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def observe_scene(adapter, images, views, arm, method, precision, output):
    entropy = EntropyObserver()
    drift = None

    def context():
        return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()

    # A0 has a different uniform grid. Comparing it with mixed-grid A3 tokens
    # would not be the prescribed per-small-view drift comparison.
    if arm != "A0":
        reference = DriftObserver()
        reference_images = [load_view(view["image"], view, "pixel").to(images[0].device) for view in views]
        with torch.inference_mode(), context():
            reference_predictions, _ = run_engine(adapter, reference_images, views, "A3",
                                                    observer=reference, **method)
        del reference_predictions, reference_images
        drift = DriftObserver(reference=reference.saved_tokens)
    combined = CombinedObserver(entropy, *([drift] if drift is not None else []))
    with torch.inference_mode(), context():
        probe_predictions, _ = run_engine(adapter, images, views, arm, observer=combined, **method)
    del probe_predictions
    entropy_arrays = {f"{row['scope']}_layer{row['layer']}_view{row['view']}": row["entropy"]
                      for row in entropy.records}
    entropy_path = Path(output) / "entropy.npz"
    save_arrays(entropy_path, entropy_arrays)
    drift_path = None
    if drift is not None:
        drift_path = Path(output) / "drift.npz"
        save_arrays(drift_path, {f"layer{row['layer']}_view{row['view']}": np.asarray(row["mean_cosine"])
                                for row in drift.records})
    return {"entropy_file": str(entropy_path), "drift_file": str(drift_path) if drift_path else None,
            "drift_unavailable_reason": "uniform A0 grid differs from mixed A3 grid" if drift is None else None,
            "entropy_definition": "per_head_mean_query_entropy_divided_by_log_all_tokens",
            "drift_definition": "mean_per_patch_cosine_on_lo_grid_against_A3",
            "untimed_extra_passes": 1 if drift is None else 2}
