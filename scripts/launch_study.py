"""Enumerate the prescribed matrix; execute only committed, frozen cells."""
import argparse
import math
import sys
import time
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("main", "extra"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", choices=("pi3", "da3", "vggt"))
    parser.add_argument("--python", default=sys.executable,
                        help="Must be the current interpreter; execution stays in this process")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--stop-file")
    args = parser.parse_args(argv)
    if args.execute and args.model is None:
        parser.error("--execute requires an explicit --model")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("--max-seconds must be finite and positive")
    if args.execute and Path(args.python).absolute() != Path(sys.executable).absolute():
        parser.error("--python must match the current interpreter; launch with that interpreter instead")
    cells = [cell for cell in matrix(args.phase) if args.model is None or cell[0] == args.model]
    atomic_json(Path("artifacts") / f"matrix-{args.phase}.json",
                {"phase": args.phase, "count": len(cells),
                 "cells": [dict(zip(("model", "dataset", "budget", "arm", "seed"), cell)) for cell in cells]})
    print(f"{args.phase}: {len(cells)} planned cells")
    if args.execute:
        deadline = time.monotonic() + args.max_seconds if args.max_seconds is not None else None
        if not Path("frozen/s23/predictions.md").is_file():
            raise SystemExit("Freeze the complete development sweep and commit predictions before execution")
        import yaml
        from mixres.study_runner import METHOD_DEFAULTS, StudyRunner

        frozen_path = Path(f"frozen/s23/{args.model}.yaml")
        frozen = yaml.safe_load(frozen_path.read_text())
        method = {key: frozen.get(key, default) for key, default in METHOD_DEFAULTS.items()}
        runner = None
        for model, dataset, budget, arm, seed in cells:
            if ((args.stop_file and Path(args.stop_file).exists())
                    or (deadline is not None and time.monotonic() >= deadline)):
                print("Batch stopped between cells; completed scene outputs remain durable")
                break
            if runner is None:
                runner = StudyRunner(model, f"checkpoints/{model}",
                                     precision=frozen.get("precision", "bf16"))
            remaining = None if deadline is None else deadline - time.monotonic()
            if (remaining is not None and remaining <= 0) or (args.stop_file and Path(args.stop_file).exists()):
                print("Batch stopped before next cell")
                break
            try:
                runner.run_cell(dataset, budget, arm, seed, method, phase=args.phase,
                                frozen_path=str(frozen_path), probes=seed == 1000, save_preds=True,
                                max_seconds=remaining, stop_file=args.stop_file)
            except TimeoutError:
                print("Batch stopped during cell deadline; prior scene outputs remain durable")
                break


if __name__ == "__main__":
    main()
