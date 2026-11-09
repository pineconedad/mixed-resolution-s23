import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from mixres.io import atomic_json, read_json
from mixres.stats.paired import paired_bootstrap, recovery


def aggregate(config):
    root = Path("runs") / config["study"] / config["model"] / config["dataset"] / str(config["budget_tokens"])
    results = {}
    common_provenance = None
    rows = []
    expected_scenes = sorted(config["scenes"])
    for arm in config["arms"]:
        path = root / arm / f"seed{config['seed']}" / "result.json"
        if not path.exists():
            continue
        run = read_json(path)
        signature = {"provenance": run["provenance"], "metric": run["metric_protocol"], "scenes_expected": run["scenes_expected"]}
        if common_provenance is not None and signature != common_provenance:
            raise ValueError("Refusing mixed code, config, view lists, checkpoints, precision, hardware or metrics")
        common_provenance = signature
        results[arm] = {r["scene"]: r for r in run["per_scene"] if r["status"] == "completed"}
        for scene in run["per_scene"]:
            rows.append({"arm": arm, **{key: scene.get(key) for key in ("scene", "status", "auc3", "auc5", "auc30", "latency_s", "peak_mem_gb", "error")}})
    common = sorted(set.intersection(*(set(results.get(arm, {})) for arm in config["arms"])))
    summary = {"exploratory": config["seed"] == 42, "scenes_expected": expected_scenes, "scenes_paired": common,
               "arms_expected": config["arms"], "arms_found": sorted(results),
               "missing_or_failed": {arm: sorted(set(expected_scenes) - set(results.get(arm, {}))) for arm in config["arms"]},
               "means_auc3": {arm: float(np.mean([results[arm][s]["auc3"] for s in common])) for arm in results} if common else {},
               "paired": {}, "recovery": None}
    if len(common) >= 2:
        for left, right in (("A4", "A2"), ("A3", "A2"), ("A2", "A1")):
            if left in results and right in results:
                summary["paired"][f"{left}-{right}"] = paired_bootstrap([results[left][s]["auc3"] - results[right][s]["auc3"] for s in common], config["bootstrap_resamples"])
        if all(arm in results for arm in ("A2", "A3", "A4")):
            summary["recovery"] = recovery(*[[results[a][s]["auc3"] for s in common] for a in ("A2", "A3", "A4")], resamples=config["bootstrap_resamples"])
    destination = Path("runs") / config["study"] / "summary"
    atomic_json(destination / "snapshot.json", summary)
    with (destination / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "scene", "status", "auc3", "auc5", "auc30", "latency_s", "peak_mem_gb", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Paired scenes: {len(common)}/{len(expected_scenes)}; arms: {len(results)}/{len(config['arms'])}")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    args = parser.parse_args()
    aggregate(yaml.safe_load(Path(args.config).read_text()))
