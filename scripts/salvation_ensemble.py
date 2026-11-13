"""Resolution ensembling, scored offline from poses stored by the overnight run. No GPU.

Fuses the same views predicted at several uniform input sizes and compares each
ensemble with the best single size that costs no more compute.

    python scripts/salvation_ensemble.py --models pi3 vggt --push
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from mixres.eval.ensemble import fuse
from mixres.eval.pose import score_poses
from mixres.io import read_json
from salvation_night_report import bootstrap, cost_gflops

SIZES = ["A0", "A0_5k", "A0_6k", "A0_7k", "A8"]
# Pre-stated before any ensemble was scored. Do not edit after seeing results.
PRIMARY = ("A0_5k", "A0_7k")


def load(model):
    """{(seed, scene): {"truth": c2w, arm: (c2w, gflops)}} for scene-draws with every size."""
    draws = {}
    for study in sorted((ROOT / "runs").glob("salvation_[cd]*")):
        seed = 42 if study.name == "salvation_c" else int(study.name.removeprefix("salvation_d"))
        manifest = (ROOT / "runs/views/7scenes/seed42.json" if seed == 42
                    else ROOT / f"runs/salvation_night/views/seed{seed}.json")
        if not manifest.exists():
            continue
        truth = {s["scene"]: np.array([v["c2w"] for v in s["views"]]) for s in read_json(manifest)["scenes"]}
        for arm in SIZES:
            path = study / model / "7scenes" / "4096" / arm / f"seed{seed}" / "result.json"
            if not path.exists():
                continue
            for record in read_json(path)["per_scene"]:
                if record["status"] == "completed":
                    entry = draws.setdefault((seed, record["scene"]), {"truth": truth[record["scene"]]})
                    entry[arm] = (np.array(record["predicted_c2w"]), cost_gflops(record))
    return {key: value for key, value in draws.items() if all(arm in value for arm in SIZES)}


def evaluate(model):
    draws = load(model)
    lines = [f"Resolution ensembling, {model} (offline, exploratory)"]
    for label, keys in (("Fresh draws (seeds 43+)", sorted(k for k in draws if k[0] != 42)),
                        ("Seed 42", sorted(k for k in draws if k[0] == 42))):
        if not keys:
            lines += ["", f"{label}: no scene-draws with all five uniform sizes."]
            continue
        rows = {}
        for count in (1, 2, 3):
            for members in itertools.combinations(SIZES, count):
                scores = {"auc3": [], "auc30": []}
                for key in keys:
                    runs = [draws[key][arm][0] for arm in members]
                    fused = runs[0] if count == 1 else fuse(runs)[0]
                    metrics = score_poses(fused, draws[key]["truth"])
                    scores["auc3"].append(metrics["auc3"])
                    scores["auc30"].append(metrics["auc30"])
                rows[members] = {"auc3": np.array(scores["auc3"]), "auc30": np.array(scores["auc30"]),
                                 "gflops": float(np.mean([sum(draws[key][arm][1] for arm in members)
                                                          for key in keys]))}
        singles = {members: row for members, row in rows.items() if len(members) == 1}

        def baseline(row):
            """Best single size that costs no more than this ensemble."""
            options = [m for m, single in singles.items() if single["gflops"] <= row["gflops"]]
            return max(options, key=lambda m: singles[m]["auc3"].mean())

        lines += ["", f"{label}: {len(keys)} scene-draws",
                  f"{'sizes':<22}{'AUC@3':>7}{'AUC@30':>8}{'GFLOPs':>8}  vs best single at no more compute"]
        for members, row in sorted(rows.items(), key=lambda item: item[1]["gflops"]):
            text = ""
            if len(members) > 1:
                best = baseline(row)
                mean, low, high = bootstrap(row["auc3"] - singles[best]["auc3"])
                wins = int((row["auc3"] > singles[best]["auc3"]).sum())
                text = f"  {best[0]}: {mean:+.2f} [{low:+.2f}, {high:+.2f}], wins {wins}/{len(keys)}"
            lines.append(f"{'+'.join(members):<22}{row['auc3'].mean():>7.2f}{row['auc30'].mean():>8.2f}"
                         f"{row['gflops']:>8.0f}{text}")
        if label.startswith("Fresh"):
            row, best = rows[PRIMARY], baseline(rows[PRIMARY])
            mean, low, high = bootstrap(row["auc3"] - singles[best]["auc3"])
            verdict = "PASS" if low > 0 else "FAIL"
            lines += ["", f"Gate E ({'+'.join(PRIMARY)}, fixed in advance): {verdict}. {mean:+.2f} AUC@3 "
                      f"[{low:+.2f}, {high:+.2f}] against {best[0]} (rule: interval above 0).",
                      "Every other row is exploratory: with 20 ensembles, some will look good by chance."]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["pi3"], choices=["pi3", "vggt"])
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    text = "\n\n".join(evaluate(model) for model in args.models)
    print(text)
    local = ROOT / "runs/salvation_night/reports"
    local.mkdir(parents=True, exist_ok=True)
    (local / "ensemble.txt").write_text(text + "\n")
    if args.push:
        destination = ROOT / "reports/salvation"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "ensemble.txt").write_text(text + "\n")
        for command in (["git", "add", "reports/salvation/ensemble.txt"],
                        ["git", "commit", "-m", "Add resolution-ensemble report"], ["git", "push"]):
            subprocess.run(command, cwd=ROOT)
