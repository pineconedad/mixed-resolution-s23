"""Resolution response and mix damage on datasets with accurate ground truth.

Runs, for each dataset and model, the mix arms plus a uniform-size curve that includes the
mix's own small size and several sizes around 518 px width. Seed-42 development views only.

    python scripts/salvation_accurate.py --push                      # ETH3D + ScanNet++, Pi3 + VGGT
    python scripts/salvation_accurate.py --models pi3 vggt da3 --push  # also attempt DA3 (see README)

ScanNet++ runs only if the licence receipt env/scannetpp-permission.json is on this machine.
A failed step is logged and the rest continue.
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

from mixres.io import atomic_json, digest, file_hash, read_json
from mixres.rmfa.grid import round_to_patch
from salvation_night import WORK, Night
from salvation_night_report import bootstrap, cost_gflops

BUDGET = 4096
ASPECT = {"eth3d": 3 / 2, "scannetpp": 4 / 3}
WIDTHS = [392, 504, 546, 588, 630]  # brackets the 518 px training width of VGGT
MIX = ["A1", "A2", "A3", "A4", "A8", "A9"]  # neighbours share decoded images
RECEIPT = "env/scannetpp-permission.json"


def build(dataset, source_path=None):
    """Quick-runner manifest and arm list from an S23-format seed-42 manifest. Same frames and sizes."""
    source_path = source_path or ROOT / "runs/views/s23" / dataset / "seed42.json"
    source = read_json(source_path)
    identity = {"study": "salvation_accurate_views", "dataset": dataset, "seed": 42, "budget_tokens": BUDGET,
                "source_sha256": file_hash(source_path)}
    scenes, small, sharp, uniform = [], set(), set(), set()
    for scene in source["scenes"]:
        views = []
        for view in scene["views"]:
            sizes = view["sizes"][str(BUDGET)]
            views.append({**{k: v for k, v in view.items() if k != "sizes"}, "hw": sizes["hw"], "hi_hw": sizes["hi_hw"]})
            (sharp if view["sharp"] else small).add(tuple(sizes["hw"]))
            uniform.add(tuple(sizes["uniform_hw"]))
        scenes.append({"scene": scene["scene"], "views": views})
    if len(small) != 1 or len(sharp) != 1 or len(uniform) != 1:
        raise RuntimeError(f"{dataset}: expected one small, one sharp and one uniform size")
    path = WORK / "views" / f"{dataset}-seed42.json"
    atomic_json(path, {"schema": 1, "dataset": dataset, "seed": 42, "budget_tokens": BUDGET,
                       "config_hash": digest(identity), "scenes": scenes})
    curve = {"A0_small": list(small.pop()), "A0": list(uniform.pop())}
    curve.update({f"A0_w{w}": [round_to_patch(w / ASPECT[dataset]), w] for w in WIDTHS})
    arms = MIX + [{"name": name, "arm": "A0", "uniform_hw": hw} if name != "A0" else "A0" for name, hw in curve.items()]
    curve["A8"] = list(sharp.pop())
    return path, digest(identity), [s["scene"] for s in scenes], arms, curve


def config_for(dataset, model, views, views_hash, scenes, arms):
    base = yaml.safe_load((ROOT / "configs/salvation-c-pi3.yaml").read_text())
    pinned = yaml.safe_load((ROOT / f"configs/models/{model}.yaml").read_text())
    base.update(study=f"salvation_acc_{dataset}", model=model, dataset=dataset, scenes=scenes, arms=arms,
                aspect=ASPECT[dataset], views=views.as_posix(), data_root=".",
                checkpoint=pinned["checkpoint"], checkpoint_revision=pinned["checkpoint_revision"],
                source_views_config_hash=views_hash, views_sha256=file_hash(views))
    path = WORK / "configs" / f"acc-{dataset}-{model}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return path


def table(dataset, model, scenes, curve):
    rows = {}
    for arm in MIX + [name for name in curve if name not in MIX]:
        path = ROOT / "runs" / f"salvation_acc_{dataset}" / model / dataset / str(BUDGET) / arm / "seed42" / "result.json"
        if path.exists():
            done = {r["scene"]: r for r in read_json(path)["per_scene"] if r["status"] == "completed"}
            if all(scene in done for scene in scenes):
                rows[arm] = [done[scene] for scene in scenes]
    total = len(MIX) + len(curve) - 1
    lines = ["", f"{dataset} / {model}: {len(scenes)} scenes, {len(rows)}/{total} arms complete"]
    if not rows:
        return lines + ["  no complete arm (see runs/salvation_night/log.txt)"]
    mean = lambda arm, key: float(np.mean([r[key] for r in rows[arm]]))
    lines.append(f"  {'mix arm':<10}{'AUC@3':>7}{'AUC@30':>8}{'GFLOPs':>8}")
    lines += [f"  {arm:<10}{mean(arm, 'auc3'):>7.2f}{mean(arm, 'auc30'):>8.2f}"
              f"{np.mean([cost_gflops(r) for r in rows[arm]]):>8.0f}" for arm in MIX if arm in rows]
    lines.append(f"  {'uniform size':<14}{'tokens':>7}{'AUC@3':>7}{'AUC@30':>8}{'GFLOPs':>8}")
    for arm, hw in sorted(curve.items(), key=lambda item: item[1][0] * item[1][1]):
        if arm in rows:
            note = "  <- the mix's small size" if arm == "A0_small" else "  <- the mix's sharp size" if arm == "A8" else ""
            lines.append(f"  {f'{hw[0]}x{hw[1]}':<14}{12 * (hw[0] // 14) * (hw[1] // 14):>7}{mean(arm, 'auc3'):>7.2f}"
                         f"{mean(arm, 'auc30'):>8.2f}{np.mean([cost_gflops(r) for r in rows[arm]]):>8.0f}{note}")
    lines.append("  Paired differences, AUC@3, mean [95% bootstrap interval over scenes]:")
    for left, right, note in (("A8", "A2", "mix damage"), ("A0_small", "A2", "all views at the small size vs the mix"),
                              ("A8", "A0_small", "sharp size vs small size, uniform"),
                              ("A0", "A2", "uniform vs mix at equal budget"), ("A4", "A0", "RMFA vs uniform"),
                              ("A8", "A3", "real vs upsampled pixels"), ("A0_w546", "A8", "width 546 vs sharp size"),
                              ("A0_w630", "A0_w546", "width 630 vs 546")):
        if left in rows and right in rows:
            difference = np.array([a["auc3"] - b["auc3"] for a, b in zip(rows[left], rows[right])])
            m, low, high = bootstrap(difference)
            lines.append(f"    {left} - {right} ({note}): {m:+.2f} [{low:+.2f}, {high:+.2f}], "
                         f"wins {int((difference > 0).sum())}/{len(difference)}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["eth3d", "scannetpp"],
                        choices=["eth3d", "scannetpp", "hiroom"])
    parser.add_argument("--unseal-hiroom", action="store_true",
                        help="required for hiroom: confirms it is now development data, not a held-out set")
    parser.add_argument("--models", nargs="+", default=["pi3", "vggt"], choices=["pi3", "vggt", "da3"])
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    night = Night(args.hours)
    plans, notes = [], []
    for dataset in args.datasets:
        if dataset == "hiroom":
            if not args.unseal_hiroom:
                parser.error("hiroom was a sealed held-out set; pass --unseal-hiroom to use it as development data")
            try:
                import salvation_hiroom_data as hiroom
                night.say("hiroom: downloading and verifying the 0.7 GB archive")
                manifest, ASPECT["hiroom"] = hiroom.make_manifest(hiroom.fetch())
                source = WORK / "views" / "hiroom-s23-seed42.json"
                atomic_json(source, manifest)
                notes.extend(f"hiroom: scene skipped, too few frames: {item}" for item in manifest["skipped_scenes"])
                plans.append((dataset, *build(dataset, source)))
                night.say(f"hiroom: {len(manifest['scenes'])} scenes ready")
            except Exception as error:  # reported, never hidden; other datasets continue
                notes.append(f"hiroom: skipped, preparation failed: {type(error).__name__}: {error}")
                night.say(notes[-1])
            continue
        if dataset == "scannetpp" and not (ROOT / RECEIPT).exists():
            notes.append(f"scannetpp: skipped, licence receipt {RECEIPT} is not on this machine.")
            night.say(notes[-1])
            continue
        stage = ["scripts/stage_development.py", "--datasets", dataset, "--seeds", "42"]
        if not night.step(f"{dataset} stage seed-42 images", *stage, timeout=3600):
            notes.append(f"{dataset}: skipped, images could not be staged.")
            continue
        plans.append((dataset, *build(dataset)))

    def report():
        lines = ["Accurate-ground-truth datasets: mix arms and uniform-size curve (seed 42, exploratory)"]
        for dataset, _, _, scenes, _, curve in plans:
            for model in args.models:
                lines += table(dataset, model, scenes, curve)
        lines += [""] + notes + ["Small scene counts: read directions and large effects, not small differences."]
        text = "\n".join(lines)
        (WORK / "reports").mkdir(parents=True, exist_ok=True)
        (WORK / "reports" / "accurate.txt").write_text(text + "\n")
        return text

    # Model-major order: both datasets for Pi3 first, then VGGT, then the optional DA3 attempt.
    for model in args.models:
        if model == "da3" and not (ROOT / "checkpoints/da3/receipt.json").exists():
            if not night.step("da3 weights download", "scripts/download_checkpoint.py", "--model", "da3", timeout=3600):
                notes.append("da3: skipped, weights could not be downloaded.")
                continue
        for dataset, views, views_hash, scenes, arms, _ in plans:
            night.study(f"{model} {dataset}", model, config_for(dataset, model, views, views_hash, scenes, arms))
            report()
    text = report()
    print(text)
    if args.push:
        (ROOT / "reports/salvation").mkdir(parents=True, exist_ok=True)
        (ROOT / "reports/salvation/accurate.txt").write_text(text + "\n")
        for command in (["git", "add", "reports/salvation/accurate.txt"],
                        ["git", "commit", "-m", "Add accurate-ground-truth resolution report"], ["git", "push"]):
            subprocess.run(command, cwd=ROOT)


if __name__ == "__main__":
    main()
