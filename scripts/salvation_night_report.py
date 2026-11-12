"""Pooled report for salvation track C (lift tokens, dense queries, pooled keys).

Reads the seed-42 run (runs/salvation_c) and the fresh development draws
(runs/salvation_d<seed>). Exploratory development numbers, never paper results.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mixres.io import atomic_json, read_json
from mixres.probes.cost import transformer_flops

# Pre-stated before any run. Do not edit after seeing results.
MAX_LOSS_VS_UPSAMPLE = 1.0      # candidate may trail A3 by at most this much AUC@3
MAX_AUC30_LOSS_VS_UNIFORM = 2.0  # candidate may trail A0 by at most this much AUC@30
UNIFORM = ["A0", "A0_5k", "A0_6k", "A0_7k", "A8"]
ORDER = ["A1", "A2", "A2_bias_both", "A4", "A9", "A0", "A0_5k", "A0_6k", "A0_7k",
         "A10", "A11", "A11_nobias", "A3_kv", "A8_kv", "A3", "A8"]
ENCODER_LAYERS, DIM, MLP_RATIO = 24, 1024, 4


def cost_gflops(record):
    """Encoder (ViT-L, per view) plus decoder transformer FLOPs. Heads excluded. A proxy."""
    trace = record["global_trace"]
    decoder = transformer_flops(trace, DIM, MLP_RATIO)
    first = trace[0]
    encoder = 2 * ENCODER_LAYERS * ((4 + 2 * MLP_RATIO) * first["encoder_patch_tokens"] * DIM**2
                                    + 2 * first["encoder_attention_quadratic"] * DIM)
    return (encoder + decoder["frame"] + decoder["global"]) / 1e9


def collect(model):
    """{arm: {(seed, scene): record}} for completed scenes, plus a list of problems."""
    data, problems = {}, []
    for study in sorted(Path("runs").glob("salvation_[cd]*")):
        for path in sorted((study / model).glob("*/*/*/seed*/result.json")):
            result = read_json(path)
            name = result.get("variant", result["arm"])
            for record in result["per_scene"]:
                if record["status"] == "completed":
                    data.setdefault(name, {})[(result["seed"], record["scene"])] = record
                else:
                    problems.append(f"{study.name} {name} {record['scene']}: {record.get('error', 'failed')}")
    return data, problems


def bootstrap(difference, resamples=10000):
    rng = np.random.default_rng(0)
    means = difference[rng.integers(0, len(difference), (resamples, len(difference)))].mean(1)
    return float(difference.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def table(data, keys, title):
    arms = [arm for arm in ORDER if arm in data] + sorted(set(data) - set(ORDER))
    arms = [arm for arm in arms if all(key in data[arm] for key in keys)]
    scenes = len(keys)
    lines = [f"{title}: {scenes} scene-draws paired across {len(arms)} arms", "",
             f"{'arm':<14}{'AUC@3':>7}{'AUC@5':>7}{'AUC@30':>8}{'GFLOPs':>9}{'lat s':>8}"
             f"{'enc tok':>9}{'glob Q':>8}{'glob KV':>9}"]
    rows = {}
    for arm in arms:
        records = [data[arm][key] for key in keys]
        first = records[0]["global_trace"][0]
        rows[arm] = {"auc3": np.array([r["auc3"] for r in records]),
                     "auc30": np.array([r["auc30"] for r in records]),
                     "auc5_mean": float(np.mean([r["auc5"] for r in records])),
                     "gflops": float(np.mean([cost_gflops(r) for r in records])),
                     "latency_s": float(np.median([r["latency_s"] for r in records]))}
        row = rows[arm]
        lines.append(f"{arm:<14}{row['auc3'].mean():>7.2f}{row['auc5_mean']:>7.2f}{row['auc30'].mean():>8.2f}"
                     f"{row['gflops']:>9.0f}{row['latency_s']:>8.3f}{first['encoder_patch_tokens']:>9}"
                     f"{first['total_tokens']:>8}{first['global_kv_tokens']:>9}")
    return lines, rows


def compare(rows, left, right, metric="auc3"):
    if left not in rows or right not in rows:
        return None
    mean, low, high = bootstrap(rows[left][metric] - rows[right][metric])
    wins = int((rows[left][metric] > rows[right][metric]).sum())
    return {"left": left, "right": right, "metric": metric, "mean": mean, "low": low, "high": high,
            "wins": wins, "n": len(rows[left][metric])}


def matched_uniform(rows, arm):
    """Cheapest uniform-resize arm that costs at least as much as `arm`."""
    options = [u for u in UNIFORM if u in rows and rows[u]["gflops"] >= rows[arm]["gflops"]]
    return min(options, key=lambda u: rows[u]["gflops"]) if options else None


def judge(rows, arm):
    """Apply the pre-stated rule to one candidate. Returns (passed, lines, comparisons)."""
    uniform = matched_uniform(rows, arm)
    checks = [compare(rows, arm, "A3"), compare(rows, arm, uniform) if uniform else None,
              compare(rows, arm, "A0", "auc30")]
    if any(check is None for check in checks):
        return None, [f"  {arm}: cannot be judged (missing A3, A0 or a costlier uniform arm)"], checks
    vs_upsample, vs_uniform, coarse = checks
    passed = (vs_upsample["mean"] >= -MAX_LOSS_VS_UPSAMPLE and vs_uniform["low"] > 0
              and coarse["mean"] >= -MAX_AUC30_LOSS_VS_UNIFORM)
    lines = [f"  {arm}: {'PASS' if passed else 'FAIL'}",
             f"    vs A3 (pixel upsampling): {vs_upsample['mean']:+.2f} AUC@3 "
             f"[{vs_upsample['low']:+.2f}, {vs_upsample['high']:+.2f}]  (rule: mean >= -{MAX_LOSS_VS_UPSAMPLE})",
             f"    vs {uniform} (uniform resize with at least as much compute, "
             f"{rows[uniform]['gflops']:.0f} vs {rows[arm]['gflops']:.0f} GFLOPs): {vs_uniform['mean']:+.2f} AUC@3 "
             f"[{vs_uniform['low']:+.2f}, {vs_uniform['high']:+.2f}], wins {vs_uniform['wins']}/{vs_uniform['n']}"
             "  (rule: interval above 0)",
             f"    vs A0 on AUC@30 (coarse robustness): {coarse['mean']:+.2f} "
             f"[{coarse['low']:+.2f}, {coarse['high']:+.2f}]  (rule: mean >= -{MAX_AUC30_LOSS_VS_UNIFORM})"]
    return passed, lines, checks


def report(model, destination=Path("runs/salvation_night/reports")):
    data, problems = collect(model)
    if not data:
        print(f"No track C results for {model} yet.")
        return "INCOMPLETE"
    lines, summary = [f"Salvation track C, {model} (exploratory development data)", ""], {}
    verdict = "INCOMPLETE"
    core = [arm for arm in ("A0", "A3", "A8", "A10", "A11") if arm in data]
    for label, keep in (("Fresh draws (seeds 43+)", lambda key: key[0] != 42),
                        ("Seed 42 (already seen, shown for continuity)", lambda key: key[0] == 42)):
        keys = sorted(set.intersection(*[set(data[arm]) for arm in core]) if core else set())
        keys = [key for key in keys if keep(key)]
        if not keys:
            lines.extend([f"{label}: no complete scene-draws yet.", ""])
            continue
        block, rows = table(data, keys, label)
        lines.extend(block + [""])
        section = {"scene_draws": len(keys), "means": {arm: {"auc3": float(row["auc3"].mean()),
                   "auc30": float(row["auc30"].mean()), "auc5": row["auc5_mean"], "gflops": row["gflops"],
                   "latency_s": row["latency_s"]} for arm, row in rows.items()}, "comparisons": []}
        lines.append("Paired differences, mean [95% bootstrap interval]:")
        for left, right, metric in (("A10", "A3", "auc3"), ("A11", "A10", "auc3"), ("A11_nobias", "A11", "auc3"),
                                    ("A8_kv", "A8", "auc3"), ("A3_kv", "A3", "auc3"), ("A11", "A4", "auc30"),
                                    ("A2_bias_both", "A2", "auc3"), ("A10", "A0", "auc3")):
            result = compare(rows, left, right, metric)
            if result:
                section["comparisons"].append(result)
                lines.append(f"  {left} - {right} ({metric}): {result['mean']:+.2f} "
                             f"[{result['low']:+.2f}, {result['high']:+.2f}], wins {result['wins']}/{result['n']}")
        lines.append("")
        lines.append("Gate C candidates:")
        outcomes = {}
        for arm in ("A11", "A10"):
            if arm in rows:
                outcomes[arm], text, checks = judge(rows, arm)
                lines.extend(text)
                section["comparisons"].extend(check for check in checks if check)
        if label.startswith("Fresh"):
            if outcomes.get("A11"):
                verdict = "PASS"
            elif outcomes.get("A10"):
                verdict = "PARTIAL"
            elif None not in outcomes.values() and outcomes:
                verdict = "FAIL"
            lines.append(f"Gate C on fresh draws: {verdict} "
                         "(PASS = A11 passes; PARTIAL = only A10 passes; FAIL = neither).")
        lines.append("")
        summary[label] = section
    if verdict == "INCOMPLETE":
        lines.append("Gate C: INCOMPLETE. The verdict is taken on fresh draws only.")
    lines.extend(f"FAILED CELL {problem}" for problem in problems)
    text = "\n".join(lines)
    print(text)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"night-{model}.txt").write_text(text + "\n")
    atomic_json(destination / f"night-{model}.json", {"model": model, "verdict": verdict, "sections": summary,
                "problems": problems, "rule": {"max_loss_vs_upsample_auc3": MAX_LOSS_VS_UPSAMPLE,
                                               "max_auc30_loss_vs_uniform": MAX_AUC30_LOSS_VS_UNIFORM,
                                               "uniform_interval_above_zero": True}})
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pi3", "vggt"], required=True)
    parser.add_argument("--output", default="runs/salvation_night/reports",
                        help="kept under the ignored runs/ folder so the git tree stays clean mid-run")
    args = parser.parse_args()
    report(args.model, Path(args.output))
