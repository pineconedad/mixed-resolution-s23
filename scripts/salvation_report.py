"""Print the salvation kill-test table and its pre-stated gate verdict.

Exploratory seed-42 numbers on five scenes: a go/no-go signal, never a paper result.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from mixres.io import atomic_json, read_json

# Pre-stated before any run. Do not edit after seeing results.
MIN_GAIN_AUC3 = 2.0
MIN_SCENE_WINS = 4
CANDIDATES = {
    "a": ["A2_moment", "A2_moment_transient", "A2_bias_global", "A2_bias_both",
          "A2_moment_bias", "A1_moment"],
    "b": ["A9", "A9_headhi", "A9_headhi_bias"],
}
BASELINE = {"a": "A2", "b": "A0"}


def load(track, model):
    config = yaml.safe_load(Path(f"configs/salvation-{track}-{model}.yaml").read_text())
    root = (Path("runs") / config["study"] / model / config["dataset"]
            / str(config["budget_tokens"]))
    names = [entry if isinstance(entry, str) else entry["name"] for entry in config["arms"]]
    results, problems = {}, []
    for name in names:
        path = root / name / f"seed{config['seed']}" / "result.json"
        if not path.exists():
            problems.append(f"{name}: not run")
            continue
        scenes = {record["scene"]: record for record in read_json(path)["per_scene"]}
        failed = [scene for scene in config["scenes"]
                  if scenes.get(scene, {}).get("status") != "completed"]
        if failed:
            errors = {scenes[s].get("error", "missing") if s in scenes else "missing" for s in failed}
            problems.append(f"{name}: failed or missing {failed}: {sorted(errors)}")
            continue
        results[name] = [scenes[scene] for scene in config["scenes"]]
    return config, names, results, problems


def column(records, key):
    return np.array([record[key] for record in records], dtype=float)


def report(track, model):
    config, names, results, problems = load(track, model)
    baseline = BASELINE[track]
    lines = [f"Salvation track {track.upper()}, {model}, {config['dataset']} "
             f"{len(config['scenes'])} scenes, seed {config['seed']} (exploratory)", "",
             f"{'variant':<22}{'AUC@3':>7}{'AUC@5':>7}{'AUC@30':>8}{'d vs ' + baseline:>10}"
             f"{'wins':>6}{'lat s':>8}{'mem GB':>8}{'glob tok':>10}{'shift':>7}"]
    rows = {}
    for name in names:
        if name not in results:
            continue
        records = results[name]
        auc3 = column(records, "auc3")
        row = {"auc3": auc3.mean(), "auc5": column(records, "auc5").mean(),
               "auc30": column(records, "auc30").mean(),
               "latency_s": float(np.median(column(records, "latency_s"))),
               "peak_mem_gb": column(records, "peak_mem_gb").max(),
               "global_tokens": records[0]["global_trace"][0]["total_tokens"]}
        shifts = [layer.get("group_shift") for record in records for layer in record["global_trace"]]
        shifts = [value for value in shifts if value is not None]
        row["group_shift"] = float(np.mean(shifts)) if shifts else None
        if baseline in results:
            difference = auc3 - column(results[baseline], "auc3")
            row.update(delta=difference.mean(), wins=int((difference > 0).sum()))
        rows[name] = row
        shift = "" if row["group_shift"] is None else f"{row['group_shift']:.2f}"
        delta = f"{row['delta']:+.2f}" if "delta" in row else ""
        lines.append(f"{name:<22}{row['auc3']:>7.2f}{row['auc5']:>7.2f}{row['auc30']:>8.2f}"
                     f"{delta:>10}{str(row.get('wins', '')):>6}{row['latency_s']:>8.3f}"
                     f"{row['peak_mem_gb']:>8.2f}{row['global_tokens']:>10}{shift:>7}")
    lines.append("")
    lines.extend(f"INCOMPLETE {problem}" for problem in problems)
    candidates = [name for name in CANDIDATES[track] if name in rows and "delta" in rows[name]]
    verdict = "INCOMPLETE"
    if candidates and baseline in rows and not problems:
        best = max(candidates, key=lambda name: rows[name]["delta"])
        row = rows[best]
        passed = row["delta"] >= MIN_GAIN_AUC3 and row["wins"] >= MIN_SCENE_WINS
        verdict = "PASS" if passed else "FAIL"
        lines.append(f"Gate {track.upper()}: {verdict}. Best candidate {best}: {row['delta']:+.2f} AUC@3 "
                     f"vs {baseline}, better in {row['wins']}/{len(config['scenes'])} scenes "
                     f"(rule: at least +{MIN_GAIN_AUC3} and {MIN_SCENE_WINS} scene wins).")
        if track == "a" and "A4" in rows:
            lines.append(f"  RMFA (A4) is {rows['A4']['auc3']:.2f}; the zero-cost candidate is "
                         f"{'at or above' if row['auc3'] >= rows['A4']['auc3'] else 'below'} it.")
        if track == "b":
            ratio = row["latency_s"] / rows[baseline]["latency_s"]
            lines.append(f"  Cost is NOT matched: {best} takes {ratio:.2f}x the latency of {baseline}. "
                         "A pass here only earns the cost-matched comparison, it is not the claim.")
        lines.append(f"  The best of {len(candidates)} candidates on five scenes is optimistic; "
                     "confirm the single chosen variant on fresh development scenes before freezing.")
    elif not problems:
        lines.append(f"Gate {track.upper()}: INCOMPLETE (no candidate or baseline results).")
    text = "\n".join(lines)
    print(text)
    destination = Path("runs") / config["study"] / model / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.txt").write_text(text + "\n")
    atomic_json(destination / "report.json", {"track": track, "model": model, "verdict": verdict,
                                              "rule": {"min_gain_auc3": MIN_GAIN_AUC3,
                                                       "min_scene_wins": MIN_SCENE_WINS},
                                              "rows": rows, "problems": problems})
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["a", "b"], required=True)
    parser.add_argument("--model", choices=["pi3", "vggt"], required=True)
    args = parser.parse_args()
    report(args.track, args.model)
