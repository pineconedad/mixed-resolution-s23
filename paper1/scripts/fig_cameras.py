"""Predicted camera layouts for one development scene (Fig. 3).

Reads the predicted camera-to-world poses stored in runs/salvation_a/*/7scenes/4096/<probe>/seed42/scenes/*.json
and the ground truth in runs/views/7scenes/seed42.json. Predictions are aligned to the ground-truth camera
centres with a similarity transform (Umeyama) before plotting.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = sys.argv[1] if len(sys.argv) > 1 else "office"
PROBES = [("A8", "Full resolution"), ("A3", "Detail removed"), ("A2", "All lowered"), ("A1", "All lowered,\npositions not mapped")]
COLORS = {"pi3": "#2a78d6", "vggt": "#eb6834"}
NAMES = {"pi3": r"$\pi^3$", "vggt": "VGGT"}


def umeyama(source, target):
    mu_s, mu_t = source.mean(0), target.mean(0)
    s, t = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(t.T @ s / len(source))
    sign = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / s.var(0).sum()
    return lambda x: scale * x @ rotation.T + (mu_t - scale * rotation @ mu_s)


scene = next(s for s in json.load(open(ROOT / "runs/views/7scenes/seed42.json"))["scenes"] if s["scene"] == SCENE)
truth = np.array([v["c2w"] for v in scene["views"]])[:, :3, 3]
sharp = np.array([v["sharp"] for v in scene["views"]])
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6})
fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.4))
scores = {}
for r, model in enumerate(("pi3", "vggt")):
    for c, (arm, title) in enumerate(PROBES):
        record = json.load(open(ROOT / f"runs/salvation_a/{model}/7scenes/4096/{arm}/seed42/scenes/{SCENE}.json"))
        predicted = umeyama(np.array(record["predicted_c2w"])[:, :3, 3], truth)(np.array(record["predicted_c2w"])[:, :3, 3])
        ax = axes[r, c]
        for p, g in zip(predicted, truth):
            ax.plot([g[0], p[0]], [g[2], p[2]], color="#b8b7b1", lw=0.7, zorder=1)
        ax.scatter(truth[:, 0], truth[:, 2], s=26, facecolors="white", edgecolors="#555550", linewidths=0.8, zorder=2)
        ax.scatter(predicted[sharp, 0], predicted[sharp, 2], s=22, marker="s", color=COLORS[model], edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter(predicted[~sharp, 0], predicted[~sharp, 2], s=20, marker="o", color=COLORS[model], alpha=0.55, edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\n{NAMES[model]}  AUC@3 {record['auc3'] + 1e-9:.1f}", fontsize=7.5)
        scores[(model, arm)] = record["auc3"]
fig.tight_layout(w_pad=0.6, h_pad=0.6)
fig.savefig(ROOT / "paper1/figures/fig_cameras.pdf")
fig.savefig(ROOT / "paper1/figures/fig_cameras.png", dpi=200)
print(SCENE, {k: round(v, 2) for k, v in scores.items()})
