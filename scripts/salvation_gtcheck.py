"""Diagnostic after Gate G failed: is 7-Scenes precision limited by its intrinsics or ground truth?

For several assumed focal lengths, compares classical two-view rotations on the
best-matched pairs with the dataset's ground truth, next to the models' own
rotations on the same pairs. Exploratory by construction: the focal values were
chosen after seeing Gate G fail, so nothing here is a gate.

    python scripts/salvation_gtcheck.py --push
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
import numpy as np

from mixres.eval import refine as geometry
from mixres.eval.pose import score_poses
from salvation_ensemble import load
from salvation_geometry import scene_views

FOCALS = [500.0, 525.0, 550.0, 585.0, 620.0]  # 585 is the dataset's nominal value
STRONG = 100  # inliers for a pair to count as well matched


def main(push):
    draws = {model: load(model) for model in ("pi3", "vggt")}
    keys = sorted(k for k in draws["pi3"] if k[0] != 42 and k in draws["vggt"])
    stats = {f: {"sift": [], "pi3": [], "vggt": [], "refined": {"pi3": [], "vggt": []}} for f in FOCALS}
    forward = {m: [] for m in draws}
    for seed, scene in keys:
        views = scene_views(seed, scene)
        features = [geometry.detect(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)) for path, _ in views]
        matches = {(i, j): geometry.match(features[i], features[j])
                   for i in range(len(views)) for j in range(i + 1, len(views))}
        truth = draws["pi3"][(seed, scene)]["truth"]
        for model in draws:
            forward[model].append(score_poses(draws[model][(seed, scene)]["A0"][0], truth)["auc3"])
        for focal in FOCALS:
            k = np.array([[focal, 0, 320.0], [0, focal, 240.0], [0, 0, 1]])
            pairs = {key: estimate for key, (a, b) in matches.items()
                     if (estimate := geometry.pair_from_matches(a, b, k, k)) is not None}
            for (i, j), (rotation, _, inliers) in pairs.items():
                if inliers < STRONG:
                    continue
                relative = lambda poses: poses[j][:3, :3].T @ poses[i][:3, :3]
                stats[focal]["sift"].append(geometry.rotation_angle(rotation, relative(truth)))
                for model in draws:
                    stats[focal][model].append(geometry.rotation_angle(
                        relative(draws[model][(seed, scene)]["A0_7k"][0]), relative(truth)))
            for model in draws:
                refined, _ = geometry.refine(draws[model][(seed, scene)]["A0"][0], pairs)
                stats[focal]["refined"][model].append(score_poses(refined, truth)["auc3"])
    lines = [f"7-Scenes precision diagnostic ({len(keys)} fresh scene-draws, exploratory, not a gate)", "",
             f"Rotation error against ground truth on well-matched pairs (at least {STRONG} inliers), degrees:",
             f"{'focal':>6}{'pairs':>7}{'SIFT median':>13}{'SIFT <1deg':>12}{'Pi3 median':>12}{'VGGT median':>13}"
             f"{'Pi3 A0+geo':>12}{'VGGT A0+geo':>13}"]
    for focal in FOCALS:
        row = stats[focal]
        sift = np.array(row["sift"])
        lines.append(f"{focal:>6.0f}{len(sift):>7}{np.median(sift):>13.2f}{(sift < 1).mean():>12.0%}"
                     f"{np.median(row['pi3']):>12.2f}{np.median(row['vggt']):>13.2f}"
                     f"{np.mean(row['refined']['pi3']):>12.2f}{np.mean(row['refined']['vggt']):>13.2f}")
    lines += ["", f"Forward pass alone, A0 AUC@3: Pi3 {np.mean(forward['pi3']):.2f}, VGGT {np.mean(forward['vggt']):.2f}",
              "Pi3/VGGT medians use the A0_7k poses on the same pairs. 'A0+geo' is AUC@3 after refinement.",
              "Reading: if SIFT's median error on its best pairs stays near or above 1 degree at every focal,",
              "the ground truth or calibration, not the methods, limits AUC@3 on this dataset."]
    text = "\n".join(lines)
    print(text)
    local = ROOT / "runs/salvation_night/reports"
    local.mkdir(parents=True, exist_ok=True)
    (local / "gtcheck.txt").write_text(text + "\n")
    if push:
        (ROOT / "reports/salvation/gtcheck.txt").write_text(text + "\n")
        for command in (["git", "add", "reports/salvation/gtcheck.txt"],
                        ["git", "commit", "-m", "Add 7-Scenes precision diagnostic"], ["git", "push"]):
            subprocess.run(command, cwd=ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    main(parser.parse_args().push)
