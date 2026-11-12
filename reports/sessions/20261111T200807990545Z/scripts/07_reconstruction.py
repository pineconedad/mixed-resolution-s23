"""CPU-only reconstruction of one saved S23 cell using pinned DA3 protocols."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Saved runs/s23/.../result.json")
    parser.add_argument("--data-root", required=True, type=Path, help="Native dataset root")
    parser.add_argument("--views", type=Path, help="Saved materialized scenes/views JSON, not seed manifest")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--upstream", type=Path, help="Pinned Depth-Anything-3 checkout")
    parser.add_argument("--gt-root", type=Path, help="HiRoom fused_pcd directory")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from mixres.eval.benchmark_recon import reconstruct_result

    try:
        report = reconstruct_result(
            args.result, data_root=args.data_root, views_path=args.views,
            project_root=args.project_root,
            upstream=args.upstream or args.project_root / "third_party/Depth-Anything-3",
            gt_root=args.gt_root)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Reconstruction failed: {error}\n")
    print(f"{report['status']}: {args.result.with_name('recon.json')}")
    if report["status"] != "completed":
        parser.exit(1, "See recon.json and recon.log for explicit failures.\n")


if __name__ == "__main__":
    main()
