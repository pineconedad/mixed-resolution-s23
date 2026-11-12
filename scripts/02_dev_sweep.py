import argparse
import csv
import itertools
import math
import sys
import time
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
    rows = result["per_scene"]
    expected = result["scenes_expected"]
    scenes = [row["scene"] for row in rows]
    if (not expected or len(set(expected)) != len(expected) or len(scenes) != len(set(scenes))
            or set(scenes) != set(expected) or any(row["status"] != "completed" for row in rows)
            or result.get("status") != "completed"):
        raise RuntimeError(f"Incomplete development result: {result['dataset']} {result['arm']}")
    if any(isinstance(row[key], bool) or not isinstance(row[key], (int, float))
           or not math.isfinite(row[key]) for row in rows for key in ("auc3", "latency_s")):
        raise RuntimeError("Development scores and costs must be finite numbers")
    if any(row["latency_s"] < 0 for row in rows):
        raise RuntimeError("Development costs must be nonnegative")
    return float(np.mean([row["auc3"] for row in rows])), float(np.mean([row["latency_s"] for row in rows]))


def sweep(datasets, checkpoint, max_seconds=None, stop_file=None):
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Development datasets must be nonempty and distinct")
    deadline = time.monotonic() + max_seconds if max_seconds is not None else None
    runner = None
    default = next(configurations())
    baselines = {}
    ranking = []
    progress = []
    output = Path("runs/dev")
    stopped = None

    def publish():
        ranking.sort(key=lambda row: (-row["pooled_auc3"], row["mean_latency_s"], row["method_id"]))
        status = ("complete" if len(ranking) == 24 and len(baselines) == 2 * len(datasets)
                  and all(cell["status"] == "completed" for cell in progress) else "partial_or_failed")
        payload = {"status": status, "model": "pi3", "budget": 4096, "seed": 42,
                   "datasets": datasets, "baseline_auc3": baselines, "ranking": ranking,
                   "cells": progress, "stop_reason": stopped}
        atomic_json(output / "ranking.json", payload)
        with (output / "ranking.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["rank", "method_id", "pooled_auc3", "mean_latency_s",
                                                       "pool_op", "lift_op", "rmfa_layers", "head_grid"])
            writer.writeheader()
            for rank, row in enumerate(ranking, 1):
                writer.writerow({"rank": rank, **{key: row[key] for key in
                                 ("method_id", "pooled_auc3", "mean_latency_s")},
                                 **{key: row["method"][key] for key in
                                    ("pool_op", "lift_op", "rmfa_layers", "head_grid")}})
        return payload

    publish()
    jobs = [(default, arm, True) for arm in ("A2", "A3")]
    jobs.extend((method, "A4", False) for method in configurations())
    for method, arm, baseline in jobs:
        dataset_scores, latencies = {}, []
        for dataset in datasets:
            if stop_file and Path(stop_file).exists():
                stopped = "stop_file"
            elif deadline is not None and time.monotonic() >= deadline:
                stopped = "deadline"
            if stopped:
                break
            if runner is None:
                runner = StudyRunner("pi3", checkpoint)
            remaining = None if deadline is None else deadline - time.monotonic()
            if (remaining is not None and remaining <= 0) or (stop_file and Path(stop_file).exists()):
                stopped = "stop_file" if stop_file and Path(stop_file).exists() else "deadline"
                break
            cell = {"dataset": dataset, "arm": arm, "method_id": digest(method)[:12],
                    "status": "partial_or_failed"}
            progress.append(cell)
            publish()
            remaining = None if deadline is None else deadline - time.monotonic()
            if (remaining is not None and remaining <= 0) or (stop_file and Path(stop_file).exists()):
                stopped = "stop_file" if stop_file and Path(stop_file).exists() else "deadline"
                break
            try:
                result = runner.run_cell(dataset, 4096, arm, 42, method, phase="dev",
                                         max_seconds=remaining, stop_file=stop_file)
            except TimeoutError as error:
                cell["error"] = str(error)
                stopped = "stop_file" if stop_file and Path(stop_file).exists() else "deadline"
                publish()
                break
            try:
                score, latency = completed_auc(result)
            except RuntimeError as error:
                cell["error"] = str(error)
                print(f"{dataset} {arm}: {error}", flush=True)
            else:
                cell.update(status="completed", auc3=score, latency_s=latency)
                if baseline:
                    baselines[f"{dataset}/{arm}"] = score
                else:
                    dataset_scores[dataset] = score
                    latencies.append(latency)
            if not baseline and len(dataset_scores) == len(datasets):
                ranking.append({"status": "complete", "method_id": digest(method)[:12], "method": method,
                                "dataset_auc3": dataset_scores,
                                "pooled_auc3": float(np.mean(list(dataset_scores.values()))),
                                "mean_latency_s": float(np.mean(latencies))})
            publish()
        if stopped:
            break
    payload = publish()
    print(f"Development sweep {payload['status']}: {len(ranking)} configurations over {len(datasets)} datasets")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["eth3d", "scannetpp", "7scenes"],
                        default=["eth3d", "scannetpp", "7scenes"])
    parser.add_argument("--checkpoint", default="checkpoints/pi3/model.safetensors")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--stop-file")
    args = parser.parse_args()
    sweep(args.datasets, args.checkpoint, args.max_seconds, args.stop_file)
