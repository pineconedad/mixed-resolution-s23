"""Track D: is part of the mixed-resolution damage an aspect-ratio mismatch?

Reruns A1, A2, A3, A8 and A0 on the same frames with sizes that are exactly 4:3
(sharp 336x448, small 168x224, uniform 252x336) and prints them next to the
original sizes (sharp 322x420 = 1.304, small 168x210 = 1.25, uniform 224x294).

    python scripts/salvation_aspect.py            # Pi3, seed 42 plus every downloaded draw
    python scripts/salvation_aspect.py --models pi3 vggt --push
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import yaml

from mixres.io import read_json
from salvation_night import WORK, Night

ARMS = ["A1", "A2", "A3", "A8", "A0"]
SIZES = {"hi": [336, 448], "lo": [168, 224], "uniform": [252, 336]}


def means(study, model, seed):
    out = {}
    for arm in ARMS:
        path = ROOT / "runs" / study / model / "7scenes" / "4096" / arm / f"seed{seed}" / "result.json"
        if path.exists():
            out[arm] = {r["scene"]: (r["auc3"], r["auc30"]) for r in read_json(path)["per_scene"]
                        if r["status"] == "completed"}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["pi3"], choices=["pi3", "vggt"])
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    night = Night(args.hours)
    folder = WORK / "configs"
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["Track D: exact 4:3 sizes vs original sizes (exploratory)", ""]
    for model in args.models:
        sources = [(42, ROOT / f"configs/salvation-c-{model}.yaml", "salvation_c")]
        sources += [(int(p.stem.split("seed")[1]), p, f"salvation_d{p.stem.split('seed')[1]}")
                    for p in sorted(folder.glob(f"run-{model}-seed*.yaml"))]
        pooled = {kind: {arm: [] for arm in ARMS} for kind in ("original", "exact")}
        for seed, source, original_study in sources:
            config = yaml.safe_load(source.read_text())
            config.update(study=f"salvation_x{seed}", arms=ARMS, sizes=SIZES)
            path = folder / f"aspect-{model}-seed{seed}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            night.study(f"{model} aspect seed{seed}", model, path)
            exact, original = means(f"salvation_x{seed}", model, seed), means(original_study, model, seed)
            scenes = set.intersection(*[set(d[a]) for d in (exact, original) for a in ARMS if a in d]) \
                if all(a in exact and a in original for a in ARMS) else set()
            for scene in sorted(scenes):
                for arm in ARMS:
                    pooled["original"][arm].append(original[arm][scene])
                    pooled["exact"][arm].append(exact[arm][scene])
        count = len(pooled["exact"]["A1"])
        lines += [f"{model}: {count} paired scene-draws", f"{'arm':<6}{'orig AUC@3':>11}{'4:3 AUC@3':>11}"
                  f"{'change':>8}{'orig AUC@30':>13}{'4:3 AUC@30':>12}"]
        if count:
            for arm in ARMS:
                o, e = np.array(pooled["original"][arm]), np.array(pooled["exact"][arm])
                lines.append(f"{arm:<6}{o[:, 0].mean():>11.2f}{e[:, 0].mean():>11.2f}"
                             f"{(e[:, 0] - o[:, 0]).mean():>+8.2f}{o[:, 1].mean():>13.2f}{e[:, 1].mean():>12.2f}")
            e = {arm: np.array(pooled["exact"][arm])[:, 0].mean() for arm in ARMS}
            o = {arm: np.array(pooled["original"][arm])[:, 0].mean() for arm in ARMS}
            lines += ["", f"Mix damage (A8 - A2), AUC@3: original sizes {o['A8'] - o['A2']:.2f}, "
                      f"exact 4:3 {e['A8'] - e['A2']:.2f}",
                      "Token counts differ between the two size sets (mix 4200 vs 4608, uniform 4032 vs 5184),",
                      "so compare gaps between arms within a column, not one arm across columns.", ""]
    text = "\n".join(lines)
    print(text)
    (WORK / "reports").mkdir(parents=True, exist_ok=True)
    (WORK / "reports" / "aspect.txt").write_text(text + "\n")
    if args.push:
        destination = ROOT / "reports/salvation"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "aspect.txt").write_text(text + "\n")
        for command in (["git", "add", "reports/salvation"], ["git", "commit", "-m", "Add track D aspect report"],
                        ["git", "push"]):
            subprocess.run(command, cwd=ROOT)


if __name__ == "__main__":
    main()
