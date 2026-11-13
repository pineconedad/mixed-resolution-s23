"""Replicate the key 7-Scenes results on ETH3D, whose ground truth is laser-scanned.

Stages only the 96 seed-42 development images (range requests, roughly 1 GB), then runs the
mix arms and a uniform-size curve on Pi3 and VGGT. Eight scenes, exploratory.

    python scripts/salvation_eth3d.py --push          # Pi3 then VGGT, stops starting work after 2 h
    python scripts/salvation_eth3d.py --models pi3 --hours 1
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import yaml

from mixres.io import atomic_json, digest, file_hash, read_json
from salvation_night import WORK, Night
from salvation_night_report import bootstrap, cost_gflops

BUDGET = 4096
ARMS = ["A1", "A2", "A3", "A8", "A0",
        {"name": "A0_6k", "arm": "A0", "uniform_budget": 6144},
        {"name": "A0_12k", "arm": "A0", "uniform_budget": 12288},
        {"name": "A0_16k", "arm": "A0", "uniform_budget": 16384}, "A4", "A9"]
ORDER = ["A1", "A2", "A4", "A9", "A3", "A0", "A0_6k", "A8", "A0_12k", "A0_16k"]


def build_views():
    """Quick-runner manifest from the committed S23 ETH3D seed-42 manifest. Same frames and sizes."""
    source = read_json(ROOT / "runs/views/s23/eth3d/seed42.json")
    identity = {"study": "salvation_eth_views", "dataset": "eth3d", "seed": 42, "budget_tokens": BUDGET,
                "source_sha256": file_hash(ROOT / "runs/views/s23/eth3d/seed42.json")}
    scenes = []
    for scene in source["scenes"]:
        views = []
        for view in scene["views"]:
            sizes = view["sizes"][str(BUDGET)]
            views.append({**{k: v for k, v in view.items() if k != "sizes"},
                          "hw": sizes["hw"], "hi_hw": sizes["hi_hw"]})
        scenes.append({"scene": scene["scene"], "views": views})
    path = WORK / "views" / "eth3d-seed42.json"
    atomic_json(path, {"schema": 1, "dataset": "eth3d", "seed": 42, "budget_tokens": BUDGET,
                       "config_hash": digest(identity), "scenes": scenes})
    return path, digest(identity), [scene["scene"] for scene in scenes]


def build_config(model, views, views_hash, scenes):
    config = yaml.safe_load((ROOT / f"configs/salvation-c-{model}.yaml").read_text())
    config.update(study="salvation_eth", dataset="eth3d", scenes=scenes, arms=ARMS, aspect=1.5,
                  views=views.as_posix(), data_root=".", source_views_config_hash=views_hash,
                  views_sha256=file_hash(views))
    path = WORK / "configs" / f"eth3d-{model}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def report(models, scenes):
    lines = ["ETH3D replication (seed 42, exploratory; laser-scanned ground truth)"]
    for model in models:
        rows = {}
        for arm in ORDER:
            path = ROOT / "runs/salvation_eth" / model / "eth3d" / str(BUDGET) / arm / "seed42" / "result.json"
            if path.exists():
                done = {r["scene"]: r for r in read_json(path)["per_scene"] if r["status"] == "completed"}
                if all(scene in done for scene in scenes):
                    rows[arm] = [done[scene] for scene in scenes]
        lines += ["", f"{model}: {len(scenes)} scenes, {len(rows)}/{len(ORDER)} arms complete",
                  f"{'arm':<9}{'AUC@3':>7}{'AUC@5':>7}{'AUC@30':>8}{'GFLOPs':>8}{'tokens':>8}{'size (sharp view)':>19}"]
        for arm, records in rows.items():
            first = records[0]["global_trace"][0]
            # 04_run does not store sizes; recover the first (sharp) view's token count from the trace.
            lines.append(f"{arm:<9}{np.mean([r['auc3'] for r in records]):>7.2f}"
                         f"{np.mean([r['auc5'] for r in records]):>7.2f}{np.mean([r['auc30'] for r in records]):>8.2f}"
                         f"{np.mean([cost_gflops(r) for r in records]):>8.0f}{first['total_tokens']:>8}"
                         f"{'mix' if arm in ('A1', 'A2', 'A4', 'A9') else 'uniform' if arm != 'A3' else 'upsampled':>19}")
        lines.append("Paired differences, AUC@3, mean [95% bootstrap interval over scenes]:")
        for left, right, note in (("A8", "A2", "mix damage"), ("A2", "A1", "position alignment"),
                                  ("A8", "A3", "real vs upsampled pixels"), ("A0", "A2", "uniform vs mix, same budget"),
                                  ("A4", "A0", "RMFA vs uniform"), ("A8", "A0_6k", "8k vs 6k uniform"),
                                  ("A0_12k", "A8", "12k vs 8k uniform"), ("A0_16k", "A0_12k", "16k vs 12k uniform")):
            if left in rows and right in rows:
                difference = np.array([a["auc3"] - b["auc3"] for a, b in zip(rows[left], rows[right])])
                mean, low, high = bootstrap(difference)
                lines.append(f"  {left} - {right} ({note}): {mean:+.2f} [{low:+.2f}, {high:+.2f}], "
                             f"wins {int((difference > 0).sum())}/{len(difference)}")
    lines += ["", "Eight scenes only: intervals are wide. Compare directions and sizes of effects with",
              "reports/salvation/night-*.txt (7-Scenes), not small differences."]
    text = "\n".join(lines)
    print(text)
    (WORK / "reports").mkdir(parents=True, exist_ok=True)
    (WORK / "reports" / "eth3d.txt").write_text(text + "\n")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["pi3", "vggt"], choices=["pi3", "vggt"])
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    night = Night(args.hours)
    if not night.step("eth3d stage 96 images", "scripts/stage_development.py", "--datasets", "eth3d",
                      "--seeds", "42", timeout=3600):
        night.say("Could not stage ETH3D images; nothing was run.")
        return
    views, views_hash, scenes = build_views()
    for model in args.models:
        night.study(f"{model} eth3d", model, build_config(model, views, views_hash, scenes))
        report(args.models, scenes)
    text = report(args.models, scenes)
    if args.push:
        (ROOT / "reports/salvation").mkdir(parents=True, exist_ok=True)
        (ROOT / "reports/salvation/eth3d.txt").write_text(text + "\n")
        for command in (["git", "add", "reports/salvation/eth3d.txt"],
                        ["git", "commit", "-m", "Add ETH3D replication report"], ["git", "push"]):
            subprocess.run(command, cwd=ROOT)


if __name__ == "__main__":
    main()
