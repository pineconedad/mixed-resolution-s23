import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from mixres.io import atomic_json, file_hash, read_json
from mixres.stats.paired import holm, paired_bootstrap, paired_sign_flip, recovery, stratified_recovery


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
               "paired": {}, "recovery": None, "information_recovery": None}
    if len(common) >= 2:
        for left, right in (("A4", "A2"), ("A3", "A2"), ("A2", "A1"), ("A8", "A3"),
                            ("A6", "A4"), ("A5", "A4"), ("A7", "A4")):
            if left in results and right in results:
                summary["paired"][f"{left}-{right}"] = paired_bootstrap([results[left][s]["auc3"] - results[right][s]["auc3"] for s in common], config["bootstrap_resamples"])
        if all(arm in results for arm in ("A2", "A3", "A4")):
            summary["recovery"] = recovery(*[[results[a][s]["auc3"] for s in common] for a in ("A2", "A3", "A4")], resamples=config["bootstrap_resamples"])
        if all(arm in results for arm in ("A1", "A8", "A3")):
            summary["information_recovery"] = recovery(*[[results[a][s]["auc3"] for s in common] for a in ("A1", "A8", "A3")], resamples=config["bootstrap_resamples"])
    destination = Path("runs") / config["study"] / "summary"
    atomic_json(destination / "snapshot.json", summary)
    with (destination / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "scene", "status", "auc3", "auc5", "auc30", "latency_s", "peak_mem_gb", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Paired scenes: {len(common)}/{len(expected_scenes)}; arms: {len(results)}/{len(config['arms'])}")
    print(summary)


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _merge_reconstruction(result, path):
    result = {**result, "per_scene": [dict(scene) for scene in result["per_scene"]]}
    fields = ("fscore", "precision", "recall", "accuracy_mm", "completeness_mm", "chamfer_mm")
    for scene in result["per_scene"]:
        for field in fields:
            scene.pop(field, None)
    audit = {"status": "missing", "completed_scenes": [], "unverified_predictions": [],
             "missing_prediction_hashes": [], "source_hash_verified": False}
    recon_path = path.with_name("recon.json")
    if recon_path.exists():
        report = read_json(recon_path)
        if (report.get("protocol") != "da3_native_recon_unposed"
                or report.get("dataset") != result["dataset"]
                or result["dataset"] not in ("eth3d", "scannetpp", "7scenes", "hiroom", "dtu")):
            raise ValueError(f"Reconstruction protocol or dataset mismatch: {recon_path}")
        for name in ("model", "budget", "arm", "seed"):
            if name in report and report[name] != result[name]:
                raise ValueError(f"Reconstruction cell mismatch: {recon_path}/{name}")
        if "source_sha256" in report:
            if report["source_sha256"] != file_hash(path):
                raise ValueError(f"Reconstruction source hash mismatch: {recon_path}")
            audit["source_hash_verified"] = True
        records = report.get("per_scene", [])
        names = [scene["scene"] for scene in records]
        if len(names) != len(set(names)) or set(names) - set(result["scenes_expected"]):
            raise ValueError(f"Duplicate or unexpected reconstruction scenes: {recon_path}")
        by_scene = {scene["scene"]: scene for scene in records}
        audit["status"] = report.get("status", "missing")
        for scene in result["per_scene"]:
            recon = by_scene.get(scene["scene"], {})
            if (scene["status"] != "completed" or recon.get("status") != "completed"
                    or report.get("status") not in ("completed", "partial_or_failed")
                    or report.get("error")):
                continue
            prediction_hash = recon.get("predictions_sha256")
            recorded_hash = scene.get("predictions_sha256")
            if recorded_hash is not None and prediction_hash != recorded_hash:
                raise ValueError(f"Reconstruction prediction hash mismatch: {recon_path}/{scene['scene']}")
            if prediction_hash is not None:
                prediction = Path(scene["predictions_file"]) if scene.get("predictions_file") else None
                if prediction is not None and prediction.is_file():
                    if file_hash(prediction) != prediction_hash:
                        raise ValueError(f"Reconstruction prediction hash mismatch: {recon_path}/{scene['scene']}")
                elif recorded_hash is None:
                    audit["unverified_predictions"].append(scene["scene"])
                    continue
            else:
                audit["missing_prediction_hashes"].append(scene["scene"])
            metrics = recon.get("metrics", {})
            normalized = {}
            for target, aliases in (("fscore", ("fscore", "F1")),
                                    ("precision", ("precision",)), ("recall", ("recall",))):
                values = [metrics[name] for name in aliases if name in metrics]
                if values:
                    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                            or not np.isfinite(value) or not 0 <= value <= 1 for value in values)
                            or len(set(values)) != 1):
                        raise ValueError(f"Invalid reconstruction {target}: {recon_path}/{scene['scene']}")
                    normalized[target] = values[0]
            # Native DTU eval3d reverses the directional labels, but not overall.
            dtu = result["dataset"] == "dtu"
            scale = 1 if dtu else 1000
            for target, source in (("accuracy_mm", "comp" if dtu else "acc"),
                                   ("completeness_mm", "acc" if dtu else "comp"),
                                   ("chamfer_mm", "overall")):
                if source in metrics:
                    value = metrics[source]
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not np.isfinite(value) or value < 0):
                        raise ValueError(f"Invalid reconstruction {source}: {recon_path}/{scene['scene']}")
                    normalized[target] = value * scale
            if normalized:
                scene.update(normalized)
                audit["completed_scenes"].append(scene["scene"])
    audit["missing_or_failed"] = sorted(set(result["scenes_expected"]) - set(audit["completed_scenes"]))
    audit["complete"] = bool(result["scenes_expected"]) and not audit["missing_or_failed"]
    return result, audit


def aggregate_s23(root=Path("runs/s23"), output=Path("runs/s23/summary")):
    root, output = Path(root), Path(output)
    models = ("pi3", "da3", "vggt")
    datasets = ("eth3d", "scannetpp", "7scenes", "dtu", "hiroom")
    expected_keys = {(model, dataset, budget, f"A{arm}", 1000)
                     for model in models for dataset in datasets
                     for budget in (4096, 8192) for arm in range(9)}
    expected_keys.update((model, dataset, 4096, arm, seed)
                         for model in models for dataset in datasets
                         for arm in ("A1", "A2", "A3", "A4", "A8") for seed in (2000, 3000))
    runs, reconstruction_audit = [], []
    signatures, view_signatures = {}, {}
    # Config/cell hashes encode the experimental treatment. Hardware only affects timing.
    varying = {"config_hash", "cell_hash", "views_hash", "frozen_hash",
               "gpu_name", "driver_version", "cpu_threads", "dirty_tree"}
    required = {"source_hash", "upstream_commit", "checkpoint", "checkpoint_revision",
                "checkpoint_sha256", "model_config_hash", "precision", "versions", "cuda_version"}
    for path in sorted(root.glob("*/*/*/*/seed*/result.json")):
        result = read_json(path)
        if result.get("study") != "s23":
            continue
        meta = result.get("provenance", {})
        if (any(meta.get(name) in (None, "", {}) for name in required)
                or not meta.get("views_hash") or not result.get("metric_protocol")
                or not result.get("auc_units") or not result.get("baseline_protocol")):
            raise ValueError(f"Missing scientific provenance: {path}")
        signature = {name: value for name, value in meta.items() if name not in varying}
        signature.update(metric_protocol=result["metric_protocol"], auc_units=result["auc_units"],
                         baseline_protocol=result["baseline_protocol"])
        model = result["model"]
        if model in signatures and signatures[model] != signature:
            raise ValueError(f"Mixed code, checkpoint, model environment or metric provenance for {model}")
        signatures[model] = signature
        view_key = (result["dataset"], result["seed"])
        view_signature = (meta["views_hash"], sorted(result["scenes_expected"]))
        if view_key in view_signatures and view_signatures[view_key] != view_signature:
            raise ValueError(f"Mixed view lists or expected scenes for {view_key}")
        view_signatures[view_key] = view_signature
        for scene in result["per_scene"]:
            if "provenance" in scene:
                scene_meta = scene["provenance"]
                if ({k: v for k, v in scene_meta.items() if k not in varying}
                        != {k: v for k, v in meta.items() if k not in varying}
                        or scene_meta.get("views_hash") != meta["views_hash"]):
                    raise ValueError(f"Mixed scene provenance: {path}")
        result, recon_audit = _merge_reconstruction(result, path)
        reconstruction_audit.append({"cell": [result[name] for name in
                                              ("model", "dataset", "budget", "arm", "seed")],
                                     **recon_audit})
        runs.append(result)
    cells, costs = [], []
    indexed, complete_keys, incomplete = {}, set(), []
    for result in runs:
        key = tuple(result[name] for name in ("model", "dataset", "budget", "arm", "seed"))
        if key in indexed:
            raise ValueError(f"Duplicate study cell: {key}")
        expected_scenes = result["scenes_expected"]
        scene_names = [row["scene"] for row in result["per_scene"]]
        if (len(scene_names) != len(set(scene_names))
                or len(expected_scenes) != len(set(expected_scenes))
                or set(scene_names) - set(expected_scenes)):
            raise ValueError(f"Duplicate or unexpected scenes: {key}")
        completed = {row["scene"]: row for row in result["per_scene"] if row["status"] == "completed"}
        for scene in completed.values():
            if not isinstance(scene.get("auc3"), (int, float)) or not np.isfinite(scene["auc3"]):
                raise ValueError(f"Nonfinite or missing auc3: {key}/{scene['scene']}")
        if (expected_scenes and set(completed) == set(expected_scenes)
                and result["status"] == "completed"):
            complete_keys.add(key)
        else:
            incomplete.append({"cell": key, "missing_or_failed": sorted(set(expected_scenes) - set(completed)),
                               "status": result["status"]})
        indexed[key] = completed
        row = {name: result[name] for name in ("model", "dataset", "budget", "arm", "seed")}
        row.update(status=result["status"], n=len(completed))
        row.update(recon_n=sum(any(scene.get(metric) is not None for metric in
                                  ("fscore", "precision", "recall", "chamfer_mm"))
                               for scene in completed.values()),
                   fscore_units="fraction", distance_units="mm")
        for metric in ("auc3", "auc5", "auc15", "auc30", "fscore", "precision", "recall",
                       "accuracy_mm", "completeness_mm", "chamfer_mm"):
            values = [scene[metric] for scene in completed.values() if scene.get(metric) is not None]
            row[metric] = float(np.mean(values)) if values else None
        cells.append(row)
        if completed and result.get("timing_comparable", True):
            costs.append({**{name: row[name] for name in ("model", "dataset", "budget", "arm", "seed")},
                          "latency_s": float(np.mean([scene["latency_s"] for scene in completed.values()]))
                          if all(scene.get("latency_s") is not None for scene in completed.values()) else None,
                          "peak_mem_gb": float(np.max([scene["peak_mem_gb"] for scene in completed.values()]))
                          if all(scene.get("peak_mem_gb") is not None for scene in completed.values()) else None,
                          "frame_flops": completed[next(iter(completed))].get("flops", {}).get("frame"),
                          "global_flops": completed[next(iter(completed))].get("flops", {}).get("global")})
    pairs = (("A4", "A2"), ("A3", "A2"), ("A8", "A3"), ("A6", "A4"),
             ("A5", "A4"), ("A7", "A4"), ("A4", "A0"))
    paired_rows = []
    for model, dataset, budget, seed in sorted({(key[0], key[1], key[2], key[4]) for key in indexed}):
        for left, right in pairs:
            left_rows = indexed.get((model, dataset, budget, left, seed), {})
            right_rows = indexed.get((model, dataset, budget, right, seed), {})
            common = sorted(set(left_rows) & set(right_rows))
            if not common:
                continue
            stats = paired_bootstrap([left_rows[scene]["auc3"] - right_rows[scene]["auc3"]
                                      for scene in common])
            inference = paired_sign_flip([left_rows[scene]["auc3"] - right_rows[scene]["auc3"]
                                          for scene in common])
            paired_rows.append({"model": model, "dataset": dataset, "budget": budget,
                                "seed": seed, "pair": f"{left}-{right}", "n": len(common),
                                "mean": stats["mean"], "ci_low": stats["ci95"][0],
                                "ci_high": stats["ci95"][1], "p_raw": inference["p_two_sided"],
                                "p_method": inference["p_method"], "two_se": stats["two_se"],
                                "complete": all((model, dataset, budget, arm, seed) in complete_keys
                                                for arm in (left, right)), "p_holm": None})
    primary = [index for index, row in enumerate(paired_rows)
               if row["pair"] == "A4-A2" and row["seed"] == 1000
               and (row["model"], row["dataset"], row["budget"], "A4", 1000) in expected_keys]
    if primary:
        adjusted = holm([paired_rows[index]["p_raw"] for index in primary], family_size=30)
        for index, value in zip(primary, adjusted):
            paired_rows[index]["p_holm"] = float(value)
    recovery_rows = []
    combinations = sorted({(key[0], key[2], key[4]) for key in indexed})
    for model, budget, seed in combinations:
        for kind, arms in (("rmfa", ("A2", "A3", "A4")),
                           ("information", ("A1", "A8", "A3"))):
            strata, used_datasets = [], []
            for dataset in datasets:
                rows = [indexed.get((model, dataset, budget, arm, seed), {}) for arm in arms]
                common = sorted(set.intersection(*(set(row) for row in rows)))
                if common:
                    used_datasets.append(dataset)
                    strata.append([[row[scene]["auc3"] for row in rows] for scene in common])
            if strata:
                stats = stratified_recovery(strata)
                recovery_rows.append({"model": model, "budget": budget, "seed": seed, "kind": kind,
                                      "datasets": ",".join(used_datasets), "R": stats["R"],
                                      "ci_low": stats["ci95"][0] if stats["ci95"] else None,
                                      "ci_high": stats["ci95"][1] if stats["ci95"] else None,
                                      "gain": stats["gain"]["mean"],
                                      "gain_ci_low": stats["gain"]["ci95"][0],
                                      "gain_ci_high": stats["gain"]["ci95"][1],
                                      "complete": all((model, dataset, budget, arm, seed) in complete_keys
                                                      for dataset in datasets for arm in arms),
                                      "reason": stats.get("reason")})
    cell_fields = ["model", "dataset", "budget", "arm", "seed", "status", "n",
                   "auc3", "auc5", "auc15", "auc30", "recon_n", "fscore", "precision", "recall",
                   "fscore_units", "accuracy_mm", "completeness_mm", "chamfer_mm", "distance_units"]
    _write_csv(output / "cells.csv", cells, cell_fields)
    _write_csv(output / "paired.csv", paired_rows,
               ["model", "dataset", "budget", "seed", "pair", "n", "mean", "ci_low", "ci_high",
                "two_se", "p_raw", "p_method", "p_holm", "complete"])
    _write_csv(output / "recovery.csv", recovery_rows,
               ["model", "budget", "seed", "kind", "datasets", "R", "ci_low", "ci_high",
                "gain", "gain_ci_low", "gain_ci_high", "complete", "reason"])
    _write_csv(output / "cost.csv", costs,
               ["model", "dataset", "budget", "arm", "seed", "latency_s", "peak_mem_gb", "frame_flops", "global_flops"])
    audit = {"runs_expected": len(expected_keys), "runs_found": len(runs),
             "complete": len(complete_keys & expected_keys),
             "matrix_complete": expected_keys <= complete_keys,
             "missing": sorted(expected_keys - set(indexed)),
             "unexpected": sorted(set(indexed) - expected_keys),
              "partial_or_failed": incomplete,
              "reconstruction_incomplete_count": sum(not item["complete"] for item in reconstruction_audit),
              "reconstruction": reconstruction_audit,
              "reconstruction_metric_units": {"fscore": "fraction", "precision": "fraction",
                                               "recall": "fraction", "accuracy_mm": "mm",
                                               "completeness_mm": "mm", "chamfer_mm": "mm"},
             "primary_family_size": 30, "primary_tests_available": len(primary)}
    atomic_json(output / "audit.json", audit)
    checks = ["# Predictions check", "",
              "Thresholds follow the study plan; this does not verify preregistration or historical T9.",
              "Each required budget/seed is checked separately with equal dataset weights.", ""]
    for hypothesis, selected_models, kind in (("H1", ("pi3",), "rmfa"),
                                              ("H2", ("da3",), "rmfa"),
                                              ("H3", ("vggt",), "rmfa"),
                                              ("H4", ("pi3", "da3"), "information")):
        required_groups = {(model, budget, seed) for model in selected_models
                           for budget, seed in ((4096, 1000), (8192, 1000))}
        if hypothesis == "H4":
            required_groups.update((model, 4096, seed) for model in selected_models
                                   for seed in (2000, 3000))
        rows = [row for row in recovery_rows if row["kind"] == kind
                and (row["model"], row["budget"], row["seed"]) in required_groups]
        available = {(row["model"], row["budget"], row["seed"]) for row in rows if row["complete"]}
        status = "insufficient"
        if available == required_groups:
            if hypothesis in ("H1", "H4"):
                if all(row["R"] is not None for row in rows):
                    threshold = 0.5 if hypothesis == "H1" else 0.7
                    status = "held" if all(row["R"] >= threshold and row["ci_low"] > 0
                                           for row in rows) else "failed"
            elif hypothesis == "H2":
                status = "held" if all(row["gain"] >= -1 for row in rows) else "failed"
            else:
                status = "held" if all(abs(row["gain"]) < 1 for row in rows) else "failed"
        checks.append(f"- {hypothesis}: {status}. Complete groups {len(available)}/{len(required_groups)}.")
        for row in rows:
            checks.append(f"  {row['model']} budget={row['budget']} seed={row['seed']}: "
                          f"gain={row['gain']:.6g}, R={row['R']}, CI={row['ci_low']}, {row['ci_high']}; "
                          f"complete={row['complete']}; reason={row['reason']}.")
    checks.extend(["", "H2/H3 are descriptive threshold checks, not noninferiority/equivalence tests.",
                   "Partial-data estimates are not confirmation. Missing or unstable ratios are insufficient."])
    (output / "predictions_check.md").write_text("\n".join(checks) + "\n")
    print(f"S23 runs found: {len(runs)}/{len(expected_keys)}; matrix complete: {audit['matrix_complete']}")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--study", choices=["exploratory", "s23"], default="exploratory")
    args = parser.parse_args()
    if args.study == "s23":
        aggregate_s23()
    else:
        aggregate(yaml.safe_load(Path(args.config).read_text()))
