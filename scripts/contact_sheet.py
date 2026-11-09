import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from PIL import Image, ImageDraw

from mixres.data.resize import load_view
from mixres.io import read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    scene = read_json(config["views"])["scenes"][0]
    sheet = Image.new("RGB", (3 * 200, len(scene["views"]) * 172), "white")
    draw = ImageDraw.Draw(sheet)
    for row, view in enumerate(scene["views"]):
        for column, (arm, entry) in enumerate((("A1", "small"), ("A3", "pixel"), ("A8", "real"))):
            tensor = load_view(Path(config["data_root"]) / view["image"], view, entry)
            image = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            image.thumbnail((196, 144))
            x, y = column * 200, row * 172
            sheet.paste(image, (x, y + 22))
            draw.text((x + 4, y + 3), f"{arm} {view['frame']} {'sharp' if view['sharp'] else 'small'}", fill="black")
    destination = Path("artifacts") / f"{config['dataset']}-{scene['scene']}-contact.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    print(destination)
