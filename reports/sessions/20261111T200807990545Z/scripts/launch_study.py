"""Enumerate the prescribed matrix; execute only committed, frozen cells."""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.io import atomic_json


def matrix(phase):
    models = ("pi3", "da3", "vggt")
    datasets = ("eth3d", "scannetpp", "7scenes", "dtu", "hiroom")
    if phase == "main":
        # Main-claim controls first, then RMFA, architecture controls, 8k and A0.
        cells = []
        for budget in (4096, 8192):
            for arms in (("A1", "A2", "A3", "A8"), ("A4",), ("A5", "A6", "A7"), ("A0",)):
                for model in models:
                    for dataset in datasets:
                        cells.extend((model, dataset, budget, arm, 1000) for arm in arms)
        return cells
    return [(model, dataset, 4096, arm, seed)
            for seed in (2000, 3000) for model in models for dataset in datasets
            for arm in ("A1", "A2", "A3", "A4", "A8")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("main", "extra"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", choices=("pi3", "da3", "vggt"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stop-file")
    args = parser.parse_args()
    cells = [cell for cell in matrix(args.phase) if args.model is None or cell[0] == args.model]
    atomic_json(Path("artifacts") / f"matrix-{args.phase}.json",
                {"phase": args.phase, "count": len(cells),
                 "cells": [dict(zip(("model", "dataset", "budget", "arm", "seed"), cell)) for cell in cells]})
    print(f"{args.phase}: {len(cells)} planned cells")
    if args.execute:
        if not Path("frozen/s23/predictions.md").is_file():
            raise SystemExit("Freeze the complete development sweep and commit predictions before execution")
        for model, dataset, budget, arm, seed in cells:
            if args.stop_file and Path(args.stop_file).exists():
                break
            command = [args.python, "scripts/04_run.py", "--model", model, "--dataset", dataset,
                       "--budget", str(budget), "--arm", arm, "--seed", str(seed),
                       "--frozen", f"frozen/s23/{model}.yaml", "--phase", args.phase,
                       "--probes", "on" if seed == 1000 else "off", "--save-preds"]
            if args.stop_file:
                command.extend(["--stop-file", args.stop_file])
            subprocess.run(command, check=True)
