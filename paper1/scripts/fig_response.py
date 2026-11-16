"""Uniform-resolution response curves (Fig. 1). Values copied from reports/salvation/accurate.txt."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = {  # long side in px -> pose AUC@3, 12 views at one size, seed-42 frames
    "ETH3D (8 scenes)": {"sides": [224, 322, 392, 448, 504, 546, 588, 630],
                         "Pi3": [13.33, 28.51, 26.26, 28.21, 28.23, 28.15, 29.13, 29.33],
                         "VGGT": [0.24, 2.60, 5.66, 10.55, 23.12, 21.00, 14.06, 9.18]},
    "HiRoom (19 scenes)": {"sides": [182, 252, 364, 392, 504, 546, 588, 630],
                           "Pi3": [27.89, 49.71, 57.14, 59.76, 62.85, 63.43, 63.02, 61.16],
                           "VGGT": [0.09, 0.22, 6.00, 10.71, 45.49, 41.51, 31.28, 19.56]},
}
COLORS = {"Pi3": "#2a78d6", "VGGT": "#eb6834"}
LABELS = {"Pi3": r"$\pi^3$", "VGGT": "VGGT"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.2), sharey=False)
for ax, (title, d) in zip(axes, DATA.items()):
    ax.axvline(518, color="#9a9a96", lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.text(522, 2, "VGGT default\n(518 px)", color="#6b6b66", fontsize=6.5, va="bottom")
    for model in ("Pi3", "VGGT"):
        ax.plot(d["sides"], d[model], color=COLORS[model], lw=2, marker="o", ms=4,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(d["sides"][-1] + 8, d[model][-1], LABELS[model], color="#222222", fontsize=7.5, va="center")
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("Long side of every view (px)")
    ax.set_xlim(170, 680)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Pose AUC@3")
fig.tight_layout(w_pad=2.5)
fig.savefig("paper1/figures/fig_response.pdf")
fig.savefig("paper1/figures/fig_response.png", dpi=200)
