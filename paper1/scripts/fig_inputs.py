"""Input views of one ETH3D scene (Fig. 2). Cuts the full-resolution column out of the contact sheet
reports/data/eth3d-courtyard-contact.png, which shows the exact seed-42 inputs used in our ETH3D runs, and
outlines the 4 anchor views listed as sharp in runs/views/s23/eth3d/seed42.json."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
im = np.asarray(Image.open(ROOT / "reports/data/eth3d-courtyard-contact.png").convert("RGB")).astype(int)
H, W, _ = im.shape
ch = H / 12
views = next(s for s in json.load(open(ROOT / "runs/views/s23/eth3d/seed42.json"))["scenes"] if s["scene"] == "courtyard")["views"]
cells = []
for r in range(12):
    block = im[int(r * ch):int((r + 1) * ch), W * 2 // 3:]
    nw = (block < 235).any(2)
    ys, xs = np.where(nw.mean(1) > 0.5)[0], np.where(nw.mean(0) > 0.5)[0]
    cells.append(Image.fromarray(block[ys.min() + 1:ys.max(), xs.min() + 1:xs.max()].astype("uint8")).resize((196, 129)))
w, h, pad = 196, 129, 12
sheet = Image.new("RGB", (6 * w + 7 * pad, 2 * h + 3 * pad), "white")
draw = ImageDraw.Draw(sheet)
for i, (cell, view) in enumerate(zip(cells, views)):
    x, y = pad + (i % 6) * (w + pad), pad + (i // 6) * (h + pad)
    sheet.paste(cell, (x, y))
    if view["sharp"]:
        draw.rectangle([x - 5, y - 5, x + w + 4, y + h + 4], outline=(42, 120, 214), width=4)
sheet.save(ROOT / "paper1/figures/fig_inputs.png")
sheet.save(ROOT / "paper1/figures/fig_inputs.pdf", resolution=150)
