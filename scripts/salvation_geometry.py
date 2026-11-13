"""Coarse tokens, fine geometry: refine stored feed-forward poses with two-view geometry. CPU only.

For every scene-draw of the overnight run, match the original full-resolution
images pairwise (SIFT, essential matrix), then refine the poses each uniform
input size produced and rescore.

    python scripts/salvation_geometry.py --models pi3 vggt --push

Calibrated setting: the two-view step uses the dataset's known intrinsics, which
the models do not receive. A pass here earns the fair version (model-predicted
focal length); it is not yet the claim.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from mixres.eval import refine as geometry
from mixres.eval.pose import score_poses
from mixres.io import atomic_json, read_json
from salvation_ensemble import SIZES, load
from salvation_night_report import bootstrap

# Fixed in advance: the cheapest forward pass plus geometry must beat the best single forward pass.
CANDIDATE = "A0"
CACHE = ROOT / "runs/salvation_night/geometry"


def scene_views(seed, scene):
    manifest, data = ((ROOT / "runs/views/7scenes/seed42.json", ROOT / "data/7scenes") if seed == 42 else
                      (ROOT / f"runs/salvation_night/views/seed{seed}.json", ROOT / f"data/7scenes_draws/seed{seed}"))
    views = next(s for s in read_json(manifest)["scenes"] if s["scene"] == scene)["views"]
    return [(data / view["image"], np.array(view["intrinsics"], dtype=np.float64)) for view in views]


def pair_estimates(seed, scene):
    """Two-view estimates for all pairs of one scene-draw, cached. Independent of any model."""
    import cv2
    path = CACHE / f"seed{seed}-{scene}.json"
    if path.exists():
        cached = read_json(path)
        if cached["settings"] == settings():
            return ({tuple(map(int, key.split("-"))): (np.array(v["R"]), np.array(v["t"]), v["inliers"])
                     for key, v in cached["pairs"].items()}, cached["seconds"])
    started = time.perf_counter()
    views = scene_views(seed, scene)
    features = []
    for image_path, _ in views:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(image_path)
        features.append(geometry.detect(gray))
    pairs = {}
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            estimate = geometry.pair_from_matches(*geometry.match(features[i], features[j]),
                                                  views[i][1], views[j][1])
            if estimate is not None:
                pairs[(i, j)] = estimate
    seconds = time.perf_counter() - started
    atomic_json(path, {"settings": settings(), "seconds": seconds, "pairs": {
        f"{i}-{j}": {"R": r.tolist(), "t": t.tolist(), "inliers": n} for (i, j), (r, t, n) in pairs.items()}})
    return pairs, seconds


def settings():
    return {name: getattr(geometry, name) for name in
            ("SIFT_FEATURES", "RATIO", "RANSAC_PIXELS", "MIN_INLIERS", "MAX_ROTATION_DISAGREEMENT_DEG",
             "REFINED_WEIGHT", "COARSE_WEIGHT")}


def evaluate(model):
    draws = load(model)
    lines = [f"Coarse tokens, fine geometry: {model} (offline, exploratory, calibrated two-view step)"]
    for label, keys in (("Fresh draws (seeds 43+)", sorted(k for k in draws if k[0] != 42)),
                        ("Seed 42", sorted(k for k in draws if k[0] == 42))):
        if not keys:
            lines += ["", f"{label}: no scene-draws with all five uniform sizes."]
            continue
        scores = {arm: {"forward3": [], "refined3": [], "forward30": [], "refined30": [], "accepted": []}
                  for arm in SIZES}
        seconds, missing = [], []
        for seed, scene in keys:
            try:
                pairs, elapsed = pair_estimates(seed, scene)
            except FileNotFoundError as error:
                missing.append(f"seed{seed} {scene}: image not found ({error})")
                continue
            seconds.append(elapsed)
            entry = draws[(seed, scene)]
            for arm in SIZES:
                coarse = entry[arm][0]
                refined, accepted = geometry.refine(coarse, pairs)
                before, after = score_poses(coarse, entry["truth"]), score_poses(refined, entry["truth"])
                row = scores[arm]
                row["forward3"].append(before["auc3"]); row["refined3"].append(after["auc3"])
                row["forward30"].append(before["auc30"]); row["refined30"].append(after["auc30"])
                row["accepted"].append(accepted)
        count = len(seconds)
        lines += ["", f"{label}: {count} scene-draws"]
        lines += [f"MISSING {item}" for item in missing]
        if not count:
            continue
        pair_total = len(next(iter(draws.values()))["truth"]) * (len(next(iter(draws.values()))["truth"]) - 1) // 2
        lines.append(f"{'size':<8}{'GFLOPs':>8}{'fwd AUC@3':>11}{'+geo AUC@3':>12}{'gain [95% interval]':>26}"
                     f"{'fwd AUC@30':>12}{'+geo AUC@30':>13}{'pairs used':>12}")
        table = {}
        for arm in SIZES:
            row = {key: np.array(value) for key, value in scores[arm].items()}
            table[arm] = row
            mean, low, high = bootstrap(row["refined3"] - row["forward3"])
            gflops = float(np.mean([draws[key][arm][1] for key in keys]))
            lines.append(f"{arm:<8}{gflops:>8.0f}{row['forward3'].mean():>11.2f}{row['refined3'].mean():>12.2f}"
                         f"{f'{mean:+.2f} [{low:+.2f}, {high:+.2f}]':>26}{row['forward30'].mean():>12.2f}"
                         f"{row['refined30'].mean():>13.2f}{row['accepted'].mean() / pair_total:>11.0%}")
        lines.append(f"Geometry step: {np.mean(seconds):.1f} s CPU per scene-draw (12 images, {pair_total} pairs), "
                     "no transformer compute.")
        if label.startswith("Fresh"):
            best = max(SIZES, key=lambda arm: table[arm]["forward3"].mean())
            mean, low, high = bootstrap(table[CANDIDATE]["refined3"] - table[best]["forward3"])
            wins = int((table[CANDIDATE]["refined3"] > table[best]["forward3"]).sum())
            coarse = bootstrap(table[CANDIDATE]["refined30"] - table[CANDIDATE]["forward30"])
            verdict = "PASS" if low > 0 and coarse[0] >= -1.0 else "FAIL"
            lines += ["", f"Gate G ({CANDIDATE} + geometry vs best forward pass {best}, fixed in advance): {verdict}. "
                      f"{mean:+.2f} AUC@3 [{low:+.2f}, {high:+.2f}], wins {wins}/{count} "
                      "(rule: interval above 0, and AUC@30 not reduced by more than 1).",
                      f"  AUC@30 change from geometry on {CANDIDATE}: {coarse[0]:+.2f} [{coarse[1]:+.2f}, {coarse[2]:+.2f}]"]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["pi3"], choices=["pi3", "vggt"])
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    text = "\n\n".join(evaluate(model) for model in args.models)
    text += "\n\nSettings (fixed before any real image was processed): " + str(settings())
    print(text)
    local = ROOT / "runs/salvation_night/reports"
    local.mkdir(parents=True, exist_ok=True)
    (local / "geometry.txt").write_text(text + "\n")
    if args.push:
        destination = ROOT / "reports/salvation"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "geometry.txt").write_text(text + "\n")
        for command in (["git", "add", "reports/salvation/geometry.txt"],
                        ["git", "commit", "-m", "Add coarse-tokens fine-geometry report"], ["git", "push"]):
            subprocess.run(command, cwd=ROOT)
