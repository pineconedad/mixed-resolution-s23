"""Stage probes across datasets (Fig. 2). Values copied from reports/salvation/night-*.txt and accurate.txt."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROBES = ["Full resolution", "Detail removed", "Keys lowered", "Encoder lowered", "Global lowered", "All lowered"]
DATA = {  # model -> dataset -> AUC@3 per probe (None = not run)
    "Pi3": {"7-Scenes (70 draws)": [12.64, 12.25, 10.38, 5.18, 7.54, 5.32],
            "ETH3D (8 scenes)": [28.21, 29.76, None, None, 6.38, 5.83],
            "HiRoom (19 scenes)": [57.14, 56.03, None, None, 15.99, 10.00]},
    "VGGT": {"7-Scenes (70 draws)": [7.13, 4.01, 5.18, 2.56, 3.28, 1.68],
             "ETH3D (8 scenes)": [10.55, 4.53, None, None, 3.92, 2.63],
             "HiRoom (19 scenes)": [6.00, 3.53, None, None, 2.35, 1.59]},
}
COLORS = {"Pi3": "#2a78d6", "VGGT": "#eb6834"}
NAMES = {"Pi3": r"$\pi^3$", "VGGT": "VGGT"}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6})
fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.3), sharey=True)
for r, model in enumerate(DATA):
    for c, (dataset, values) in enumerate(DATA[model].items()):
        ax = axes[r, c]
        full = values[0]
        for i, v in enumerate(values):
            if v is None:
                ax.text(2, i, "not run", va="center", ha="left", fontsize=6.5, color="#9a9a96")
                continue
            ax.barh(i, v / full * 100, height=0.62, color=COLORS[model], edgecolor="white", linewidth=0.8)
            ax.text(v / full * 100 + 2, i, f"{v + 1e-9:.1f}", va="center", fontsize=6.5, color="#333333")
        ax.axvline(100, color="#9a9a96", lw=0.7, ls=(0, (2, 2)))
        ax.set_xlim(0, 135)
        ax.set_yticks(range(len(PROBES)), PROBES)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e4e3df", lw=0.5)
        ax.set_title(f"{NAMES[model]}, {dataset}", fontsize=8)
        if r == 1:
            ax.set_xlabel("% of full-resolution AUC@3")
axes[0, 0].invert_yaxis()
fig.tight_layout(w_pad=1.0, h_pad=0.8)
fig.savefig("paper1/figures/fig_stages.pdf")
fig.savefig("paper1/figures/fig_stages.png", dpi=200)
