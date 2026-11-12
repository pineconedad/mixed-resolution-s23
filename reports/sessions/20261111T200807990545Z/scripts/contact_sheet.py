import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from PIL import Image, ImageDraw

from mixres.data.resize import load_view
from mixres.io import read_json
from mixres.study_runner import materialize_views

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--dataset", choices=["eth3d", "7scenes", "scannetpp"])
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.dataset:
        config = {"dataset": args.dataset}
        scene = read_json(Path("runs/views/s23") / args.dataset / "seed42.json")["scenes"][0]
        source_views = scene["views"]
    else:
        config = yaml.safe_load(Path(args.config).read_text())
        scene = read_json(config["views"])["scenes"][0]
        source_views = scene["views"]
    sheet = Image.new("RGB", (3 * 200, len(scene["views"]) * 172), "white")
    draw = ImageDraw.Draw(sheet)
    for row, source_view in enumerate(source_views):
        for column, (arm, entry) in enumerate((("A1", "small"), ("A3", "pixel"), ("A8", "real"))):
            view = (materialize_views({"views": [source_view]}, args.budget, arm)[0]
                    if args.dataset else source_view)
            path = Path(view["image"]) if args.dataset else Path(config["data_root"]) / view["image"]
            tensor = load_view(path, view, entry)
            image = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            image.thumbnail((196, 144))
            x, y = column * 200, row * 172
            sheet.paste(image, (x, y + 22))
            draw.text((x + 4, y + 3), f"{arm} {Path(view['frame']).name} {'sharp' if view['sharp'] else 'small'}", fill="black")
    destination = (Path(args.output) if args.output else
                   Path("artifacts") / f"{config['dataset']}-{scene['scene']}-contact.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    print(destination)
