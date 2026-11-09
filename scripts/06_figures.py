import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="runs/quick/summary/cells.csv")
    args = parser.parse_args()
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
