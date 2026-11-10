"""Cross-model pose summary, keeping execution/timing provenance separate."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mixres.io import atomic_json, read_json

SESSIONS = {
    "Pi3": "reports/sessions/20261110T090917939514Z",
    "VGGT": "reports/sessions/20261110T182520898917Z",
}


def compare():
    summaries = {}
    view_hash = None
    rows = []
    for name, path in SESSIONS.items():
        root = Path(path)
        snapshot = read_json(root / "runs/diagnostic/summary/snapshot.json")
        model = name.lower()
        result = read_json(root / f"runs/diagnostic/{model}/7scenes/4096/A1/seed42/result.json")
        provenance = result["provenance"]
        if view_hash is not None and provenance["views_hash"] != view_hash:
            raise ValueError("Models did not use identical committed views")
        view_hash = provenance["views_hash"]
        if any(snapshot["missing_or_failed"].values()):
            raise ValueError("Comparison requires complete paired scenes")
        summaries[name] = {"summary": snapshot, "provenance": provenance,
                           "limitations": "Exploratory seed 42; hardware/timing provenance is not pooled"}
        rows.extend({"model": name, "arm": arm, "auc3": value, "gpu": provenance["gpu_name"],
                     "git_commit": provenance["git_commit"]} for arm, value in snapshot["means_auc3"].items())
    destination = Path("reports/development")
    atomic_json(destination / "comparison.json", {"views_hash": view_hash, "models": summaries,
                "alignment_protocol": "global_only_effective_grid_ratio"})
    with (destination / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "arm", "auc3", "gpu", "git_commit"])
        writer.writeheader()
        writer.writerows(rows)
    order = ["A1", "A2", "A4", "A5", "A6", "A7", "A3", "A8"]
    labels = ["Mix", "+ pi_hr", "RMFA", "Token", "Global", "No pi_hr", "Upsample", "Real pixels"]
    colors = ["#7a858f", "#596f85", "#175c53", "#39866d", "#bb7b42", "#b4a276", "#8c5550", "#422f55"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (name, record) in zip(axes, summaries.items()):
        values = record["summary"]["means_auc3"]
        ax.bar(range(len(order)), [values[arm] for arm in order], color=colors, width=0.72)
        ax.set_xticks(range(len(order)), labels, rotation=38, ha="right")
        ax.set_title(name)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Mean pose AUC@3 [%]")
    fig.suptitle("Global-only alignment control: 7-Scenes, five scenes, seed 42", fontsize=11)
    fig.tight_layout()
    fig.savefig(destination / "architecture-controls.png", dpi=200)
    print("Published development comparison; no GPU timings pooled")


if __name__ == "__main__":
    compare()
