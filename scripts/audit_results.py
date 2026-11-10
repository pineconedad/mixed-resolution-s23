"""Audit preserved quick-test outputs without renting another GPU."""
import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mixres.eval.pose import score_poses
from mixres.io import atomic_json, file_hash, read_json


def audit(session, destination, study="quick", model="pi3"):
    session, destination = Path(session), Path(destination)
    receipt = read_json(session / "session-manifest.json")
    for name, expected in receipt["files"].items():
        if file_hash(session / name) != expected:
            raise ValueError(f"Preserved member checksum changed: {name}")
    manifest = read_json(session / "runs/views/7scenes/seed42.json")
    ground_truth = {s["scene"]: s for s in manifest["scenes"]}
    expected_scenes = set(ground_truth)
    snapshot = read_json(session / f"runs/{study}/summary/snapshot.json")
    arms = snapshot["arms_expected"]
    summary = {"git_commit": receipt["git_commit"], "exploratory": True,
               "checksummed_members": len(receipt["files"]), "arms": {}, "scene_warnings": []}
    provenance = None
    evaluations = 0
    for arm in arms:
        result = read_json(session / f"runs/{study}/{model}/7scenes/4096/{arm}/seed42/result.json")
        if result["status"] != "completed" or {s["scene"] for s in result["per_scene"]} != expected_scenes:
            raise ValueError(f"Incomplete paired scene set for {arm}")
        if provenance is not None and result["provenance"] != provenance:
            raise ValueError("Mixed run provenance")
        provenance = result["provenance"]
        if provenance["dirty_tree"] or provenance["git_commit"] != receipt["git_commit"]:
            raise ValueError("Export and execution provenance differ")
        for scene in result["per_scene"]:
            views = ground_truth[scene["scene"]]["views"]
            recomputed = score_poses(np.asarray(scene["predicted_c2w"]), np.asarray([v["c2w"] for v in views]))
            for key in ("auc3", "auc5", "auc30"):
                if not math.isclose(recomputed[key], scene[key], rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Pose rescore mismatch: {scene['scene']} {arm} {key}")
            traces = scene["global_trace"]
            if len(traces) != {"pi3": 18, "vggt": 24}[model]:
                raise ValueError("Missing global-layer traces")
            expected_patches = 8280 if arm in {"A3", "A6", "A8"} else 4200
            if any(t["patch_tokens"] != expected_patches or t["special_tokens"] != 60
                   or t["total_tokens"] != expected_patches + 60 for t in traces):
                raise ValueError("Global-token fairness failure")
            if len(scene["timed_passes_s"]) != 3 or scene["latency_s"] <= 0 or scene["peak_mem_gb"] <= 0:
                raise ValueError("Invalid cost record")
            evaluations += 1
        rows = result["per_scene"]
        summary["arms"][arm] = {
            **{key: statistics.mean(s[key] for s in rows) for key in ("auc3", "auc5", "auc30", "latency_s")},
            "peak_mem_gb_max": max(s["peak_mem_gb"] for s in rows), "scenes": len(rows)}
    csv_path = session / f"runs/{study}/summary/cells.csv"
    with csv_path.open() as stream:
        records = list(csv.DictReader(stream))
    by_scene = {}
    for row in records:
        by_scene.setdefault(row["scene"], {})[row["arm"]] = row
    for scene, rows in by_scene.items():
        primary_gain = float(rows["A4"]["auc3"]) - float(rows["A2"]["auc3"])
        secondary_gain = float(rows["A4"]["auc30"]) - float(rows["A2"]["auc30"])
        if primary_gain > 0 and secondary_gain < 0:
            summary["scene_warnings"].append({"scene": scene, "auc3_gain_A4_A2": primary_gain,
                                              "auc30_gain_A4_A2": secondary_gain})
    identity = read_json(session / f"runs/sanity/{model}-{study}.json")
    summary["fp32_T1"] = {s["scene"]: s["max_abs_error"] for s in identity["comparisons"]}
    summary["evaluations_rescored"] = evaluations
    summary["latency_ratio_A4_A3"] = summary["arms"]["A4"]["latency_s"] / summary["arms"]["A3"]["latency_s"]
    summary["limitations"] = ["Five-scene exploratory seed 42 only; not frozen paper evidence",
                               "No reconstruction, extra seeds or other models evaluated"]
    alignment_gain = snapshot.get("paired", {}).get("A2-A1", {}).get("ci95")
    if alignment_gain is None or alignment_gain[0] <= 0:
        summary["limitations"].append("A2-A1 position-only sanity gain is not established")
    if summary["latency_ratio_A4_A3"] >= 0.95:
        summary["limitations"].append("No demonstrated runtime speedup of A4 over A3 in this session")
    else:
        summary["limitations"].append("Measured runtime improvement is specific to this GPU and input size")
    if "A8" not in arms:
        summary["limitations"].append("A8 information ceiling was not evaluated")
    atomic_json(destination.with_suffix(".json"), summary)
    lines = ["# Preserved result audit", "", f"Execution commit: `{receipt['git_commit']}`.", "",
             f"Verified {len(receipt['files'])} member checksums and rescored all {evaluations} evaluations from saved camera poses.", "",
             "| Arm | AUC@3 [%] | AUC@5 [%] | AUC@30 [%] | Mean latency [s] | Max allocated GPU memory [GiB] |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, row in summary["arms"].items():
        lines.append(f"| {arm} | {row['auc3']:.6f} | {row['auc5']:.6f} | {row['auc30']:.6f} | {row['latency_s']:.6f} | {row['peak_mem_gb_max']:.6f} |")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {message}." for message in summary["limitations"])
    for warning in summary["scene_warnings"]:
        lines.append(f"- {warning['scene']}: A4-A2 improves AUC@3 by {warning['auc3_gain_A4_A2']:.6f} but changes AUC@30 by {warning['auc30_gain_A4_A2']:.6f} points.")
    destination.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"Audited {evaluations} evaluations; wrote {destination.with_suffix('.json')} and .md")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="reports/sessions/20261109T223406Z")
    parser.add_argument("--output", default="reports/audits/20261109T223406Z")
    parser.add_argument("--study", choices=["quick", "diagnostic", "alignment"], default="quick")
    parser.add_argument("--model", choices=["pi3", "vggt"], default="pi3")
    args = parser.parse_args()
    audit(args.session, args.output, args.study, args.model)
