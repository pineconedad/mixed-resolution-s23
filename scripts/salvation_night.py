"""Unattended salvation run for a local GPU. Start it and leave it.

    python scripts/salvation_night.py            # run everything, about 6 hours at most
    python scripts/salvation_night.py --push     # also commit and push the reports at the end

Order (most important first; a failed step is logged and the rest continue):
  1. Pi3 track C on the five seed-42 scenes
  2. VGGT weights, then VGGT track C on the same scenes
  3. Fresh development draws (seeds 43-52, seven 7-Scenes scenes): download frames, run Pi3, then VGGT
  4. VGGT tracks A and B on the seed-42 scenes
The pooled report (runs/salvation_night/reports/night-<model>.txt) is rewritten after every stage,
so a partial night still leaves a readable result.

User-operated and local only: never touches cloud resources. If it is interrupted, run the same
command again: finished scenes are skipped. Do not commit or edit tracked files between runs; the
runner refuses results whose code state differs. Stage 4 and --push come last for that reason.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from mixres.io import digest, file_hash

WORK = Path("runs/salvation_night")
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
DRAW_SEEDS = list(range(43, 53))
# Frames already bound to seed 42 or to frozen seeds are never drawn again.
BOUND = ["runs/views/7scenes/seed42.json", "runs/views/s23/7scenes/seed42.json",
         "runs/views/s23/7scenes/seed1000.json", "runs/views/s23/7scenes/seed2000.json",
         "runs/views/s23/7scenes/seed3000.json"]
EXTRA_ARMS = [{"name": "A2_bias_both", "arm": "A2", "logit_bias": "both"}, "A9"]


class Night:
    def __init__(self, hours):
        self.deadline = time.monotonic() + hours * 3600
        self.results = []
        WORK.mkdir(parents=True, exist_ok=True)
        self.log = (WORK / "log.txt").open("a", encoding="utf-8")
        self.env = dict(os.environ)
        paths = [str(ROOT / "third_party/Pi3"), str(ROOT / "third_party/vggt"),
                 str(ROOT / "third_party/Depth-Anything-3/src"), str(ROOT)]
        self.env["PYTHONPATH"] = os.pathsep.join(paths + [p for p in [os.environ.get("PYTHONPATH")] if p])
        self.env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0", XFORMERS_DISABLED="1",
                        PYTHONIOENCODING="utf-8")
        self.env.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))

    def say(self, text):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def step(self, name, *arguments, timeout=5400):
        """Run one python script. Returns True on success; never raises."""
        if time.monotonic() >= self.deadline:
            self.results.append((name, "SKIPPED (time limit)"))
            self.say(f"SKIP {name}: time limit reached")
            return False
        self.say(f"START {name}")
        started = time.monotonic()
        try:
            process = subprocess.run([sys.executable, *arguments], env=self.env, cwd=ROOT, timeout=timeout,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace")
            output, code = process.stdout, process.returncode
        except subprocess.TimeoutExpired as error:
            output, code = (error.stdout or "") if isinstance(error.stdout, str) else "", "timeout"
        self.log.write(output + "\n")
        self.log.flush()
        status = "ok" if code == 0 else f"FAILED ({code})"
        self.results.append((name, status))
        tail = "" if code == 0 else " | " + " / ".join(output.strip().splitlines()[-3:])
        self.say(f"END   {name}: {status} in {time.monotonic() - started:.0f}s{tail}")
        return code == 0

    def study(self, label, model, config):
        """Identity gate, then the run. The run is attempted only if the gate passes."""
        if not self.step(f"{label} identity gate", "scripts/01_sanity.py", "--config", str(config)):
            return False
        return self.step(f"{label} run", "scripts/04_run.py", "--config", str(config))


def draw_configs(seed, models):
    """Write the download config and one run config per model for a development draw."""
    folder = WORK / "configs"
    folder.mkdir(parents=True, exist_ok=True)
    views = WORK / "views" / f"seed{seed}.json"
    prepare = {"study": "salvation_draw", "dataset": "7scenes", "seed": seed, "budget_tokens": 4096,
               "n_views": 12, "n_sharp": 4, "rho": 2.0, "scenes": SCENES, "views": views.as_posix(),
               "data_root": f"data/7scenes_draws/seed{seed}", "archive_root": "data/7scenes_draws/_archives",
               "exclude_manifests": [path for path in BOUND if (ROOT / path).exists()]}
    prepare_path = folder / f"draw-seed{seed}.yaml"
    prepare_path.write_text(yaml.safe_dump(prepare, sort_keys=False))

    def run_config(model):
        base = yaml.safe_load((ROOT / f"configs/salvation-c-{model}.yaml").read_text())
        base.update(study=f"salvation_d{seed}", seed=seed, scenes=SCENES, views=prepare["views"],
                    data_root=prepare["data_root"], arms=base["arms"] + EXTRA_ARMS,
                    source_views_config_hash=digest(yaml.safe_load(prepare_path.read_text())),
                    views_sha256=file_hash(ROOT / views))
        path = folder / f"run-{model}-seed{seed}.yaml"
        path.write_text(yaml.safe_dump(base, sort_keys=False))
        return path

    return prepare_path, views, run_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=6.0, help="stop starting new steps after this long")
    parser.add_argument("--push", action="store_true", help="commit and push reports/salvation at the end")
    parser.add_argument("--skip-vggt", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    night = Night(args.hours)
    night.say(f"Salvation night started; limit {args.hours} h; log {WORK / 'log.txt'}")

    def reports(models):
        for model in models:
            night.step(f"{model} pooled report", "scripts/salvation_night_report.py", "--model", model, timeout=600)

    # 1. Pi3 track C on the seed-42 scenes.
    night.study("pi3 track C seed42", "pi3", "configs/salvation-c-pi3.yaml")
    reports(["pi3"])

    # 2. VGGT: weights, then tracks C, A, B. One failed identity gate disables VGGT for the night.
    vggt = not args.skip_vggt
    if vggt and not (ROOT / "checkpoints/vggt/receipt.json").exists():
        vggt = night.step("vggt weights download", "scripts/download_checkpoint.py", "--model", "vggt", timeout=7200)
    if vggt:
        vggt = night.study("vggt track C seed42", "vggt", "configs/salvation-c-vggt.yaml")
        if not vggt:
            night.say("VGGT disabled for the rest of the night (identity gate or run failed).")
    if vggt:
        reports(["vggt"])
    models = ["pi3"] + (["vggt"] if vggt else [])

    # 3. Fresh development draws.
    for seed in DRAW_SEEDS:
        prepare_path, views, run_config = draw_configs(seed, models)
        if not night.step(f"draw seed{seed} download", "scripts/prepare_quick.py", "--config", str(prepare_path),
                          timeout=7200):
            continue
        for model in models:
            night.study(f"{model} draw seed{seed}", model, run_config(model))
        reports(models)

    # 4. VGGT tracks A and B. Last, because their outputs are tracked paths and dirty the git tree.
    if vggt:
        for track in ("a", "b"):
            if night.study(f"vggt track {track.upper()} seed42", "vggt", f"configs/salvation-{track}-vggt.yaml"):
                night.step(f"vggt track {track.upper()} report", "scripts/salvation_report.py",
                           "--track", track, "--model", "vggt", timeout=600)

    summary = ["Salvation night summary", ""] + [f"{status:<24}{name}" for name, status in night.results]
    local = WORK / "reports"
    local.mkdir(parents=True, exist_ok=True)
    (local / "night-summary.txt").write_text("\n".join(summary) + "\n")
    night.say("\n".join(summary))
    for track in ("a", "b"):
        source = ROOT / f"runs/salvation_{track}/vggt/summary/report.txt"
        if source.exists():
            (local / f"track-{track}-vggt.txt").write_text(source.read_text())
    if args.push:
        destination = ROOT / "reports/salvation"
        destination.mkdir(parents=True, exist_ok=True)
        for source in local.iterdir():
            (destination / source.name).write_bytes(source.read_bytes())
        # Done last: a new commit changes the provenance that resumed runs are checked against.
        for command in (["git", "add", "reports/salvation"],
                        ["git", "commit", "-m", "Add salvation night reports"],
                        ["git", "push"]):
            outcome = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            night.say(f"{' '.join(command[:2])}: {'ok' if outcome.returncode == 0 else outcome.stderr.strip()[-300:]}")
    night.say(f"Night finished. Read {local / 'night-pi3.txt'} first.")


if __name__ == "__main__":
    main()
