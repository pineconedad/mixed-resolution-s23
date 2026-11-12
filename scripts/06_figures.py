import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def study_figures(summary):
    import numpy as np

    summary = Path(summary)
    rows = list(csv.DictReader((summary / "cells.csv").open()))
    rows = [row for row in rows if row["seed"] == "1000" and row["budget"] == "4096"
            and row["status"] == "completed" and row.get("auc3")]
    if not rows:
        raise SystemExit("No completed seed-1000 4k cells; no paper figure generated")
    destination = summary.parent / "figures"
    destination.mkdir(exist_ok=True)
    models = sorted({row["model"] for row in rows})
    for filename, arms in (("recovery", ("A1", "A2", "A4", "A3", "A8")),
                           ("localization", ("A4", "A6", "A3"))):
        figure, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
        for axis, model in zip(axes[0], models):
            means, labels = [], []
            for arm in arms:
                values = [float(row["auc3"]) for row in rows if row["model"] == model and row["arm"] == arm]
                if values:
                    means.append(np.mean(values))
                    labels.append(arm)
            axis.bar(labels, means, color=["#b7b7b7" if arm in {"A3", "A6", "A8"} else "#28718e"
                                          for arm in labels])
            axis.set_title(model)
            axis.set_ylabel("Mean dataset pose AUC@3 [%]")
        figure.suptitle("Seed 1000, nominal 4k. Missing cells are not imputed.", fontsize=10)
        figure.tight_layout()
        figure.savefig(destination / f"{filename}.png", dpi=180)
        plt.close(figure)
    costs = list(csv.DictReader((summary / "cost.csv").open()))
    figure, axis = plt.subplots(figsize=(7, 4))
    for row in rows:
        cost = next((cost for cost in costs if all(cost[key] == row[key]
                     for key in ("model", "dataset", "budget", "arm", "seed"))), None)
        if cost and cost.get("latency_s"):
            axis.scatter(float(cost["latency_s"]), float(row["auc3"]))
            axis.annotate(f"{row['model']}/{row['dataset']}/{row['arm']}",
                          (float(cost["latency_s"]), float(row["auc3"])), fontsize=6)
    axis.set_xlabel("Median timed scene latency [s]")
    axis.set_ylabel("Pose AUC@3 [%]")
    figure.tight_layout()
    figure.savefig(destination / "cost-accuracy.png", dpi=180)
    plt.close(figure)
    print(destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="runs/quick/summary/cells.csv")
    parser.add_argument("--config")
    parser.add_argument("--study", choices=["exploratory", "s23"], default="exploratory")
    args = parser.parse_args()
    if args.study == "s23":
        study_figures("runs/s23/summary")
        raise SystemExit(0)
    if args.config:
        import yaml
        config = yaml.safe_load(Path(args.config).read_text())
        args.summary = f"runs/{config['study']}/summary/cells.csv"
    rows = list(csv.DictReader(Path(args.summary).open()))
    arms = sorted(set(r["arm"] for r in rows))
    groups = {a: {r["scene"]: float(r["auc3"]) for r in rows if r["arm"] == a and r["status"] == "completed"} for a in arms}
    common = sorted(set.intersection(*(set(g) for g in groups.values()))) if groups else []
    if not common:
        raise SystemExit("No identically paired scenes; no figure generated")
    fig, ax = plt.subplots(figsize=(7, 4))
    for scene in common:
        ax.plot(arms, [groups[a][scene] for a in arms], marker="o", alpha=0.6, label=scene)
    ax.set_ylabel("Pose AUC@3 [%]")
    ax.set_title("Exploratory seed-42 quick test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    destination = Path(args.summary).parent / "figures"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / "quick-recovery.png", dpi=180)
