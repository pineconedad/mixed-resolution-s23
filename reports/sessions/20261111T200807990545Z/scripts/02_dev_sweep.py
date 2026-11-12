import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mixres.io import atomic_json, digest
from mixres.study_runner import StudyRunner


def configurations():
    for pool_op, lift_op, rmfa_layers, head_grid in itertools.product(
            ("area", "bilinear"), ("bilinear", "nearest"),
            ("all", "first_half", "first_quarter"), ("lo", "hi")):
        yield {"pool_op": pool_op, "lift_op": lift_op, "rmfa_layers": rmfa_layers,
               "head_grid": head_grid, "token_interp": "bilinear",
               "alignment_scope": "global", "position_scale": "actual"}


def completed_auc(result):
    if result.get("timing_comparable") is False:
        raise RuntimeError("Development cost tie-breaking requires matching timing runtimes")
    rows = [row for row in result["per_scene"] if row["status"] == "completed"]
    if len(rows) != len(result["scenes_expected"]):
        raise RuntimeError(f"Incomplete development result: {result['dataset']} {result['arm']}")
    return float(np.mean([row["auc3"] for row in rows])), float(np.mean([row["latency_s"] for row in rows]))


def sweep(datasets, checkpoint, max_seconds=None, stop_file=None):
    runner = StudyRunner("pi3", checkpoint)
    default = next(configurations())
    baselines = {}
    for dataset in datasets:
        for arm in ("A2", "A3"):
            result = runner.run_cell(dataset, 4096, arm, 42, default, phase="dev",
                                     max_seconds=max_seconds, stop_file=stop_file)
            baselines[f"{dataset}/{arm}"] = completed_auc(result)[0]
    ranking = []
    for method in configurations():
        dataset_scores, latencies = {}, []
        for dataset in datasets:
            result = runner.run_cell(dataset, 4096, "A4", 42, method, phase="dev",
                                     max_seconds=max_seconds, stop_file=stop_file)
            score, latency = completed_auc(result)
            dataset_scores[dataset] = score
            latencies.append(latency)
        ranking.append({"method_id": digest(method)[:12], "method": method,
                        "dataset_auc3": dataset_scores,
                        "pooled_auc3": float(np.mean(list(dataset_scores.values()))),
                        "mean_latency_s": float(np.mean(latencies))})
    ranking.sort(key=lambda row: (-row["pooled_auc3"], row["mean_latency_s"], row["method_id"]))
    output = Path("runs/dev")
    atomic_json(output / "ranking.json", {"status": "complete", "model": "pi3",
                "budget": 4096, "seed": 42, "datasets": datasets,
                "baseline_auc3": baselines, "ranking": ranking})
    with (output / "ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", "method_id", "pooled_auc3", "mean_latency_s",
                                                        "pool_op", "lift_op", "rmfa_layers", "head_grid"])
        writer.writeheader()
        for rank, row in enumerate(ranking, 1):
            writer.writerow({"rank": rank, "method_id": row["method_id"],
                             "pooled_auc3": row["pooled_auc3"], "mean_latency_s": row["mean_latency_s"],
                             **{key: row["method"][key] for key in
                                ("pool_op", "lift_op", "rmfa_layers", "head_grid")}})
    print(f"Development sweep complete: {len(ranking)} configurations over {len(datasets)} datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["eth3d", "scannetpp", "7scenes"],
                        default=["eth3d", "scannetpp", "7scenes"])
    parser.add_argument("--checkpoint", default="checkpoints/pi3/model.safetensors")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--stop-file")
    args = parser.parse_args()
    sweep(args.datasets, args.checkpoint, args.max_seconds, args.stop_file)
