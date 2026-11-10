"""Download the official DA3 development archives locally, not on a paid GPU."""
import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import hf_hub_download
from mixres.io import atomic_json, file_hash, read_json

REVISION = "cb98821d5d4f721704f11bafde0403826b82b638"
FILES = {
    "eth3d": (15101568883, "2a12b961e72566e194d707db02f2be65686ff12ad5f4347cb16b4c266b2d6917"),
    "7scenes": (3553806786, "4fc12afe9922c975068abd8918b897580926a932b718ad7646f6822205ef5b88"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(FILES), default=["eth3d", "7scenes"])
    args = parser.parse_args()
    root = Path("data/benchmark_archives")
    root.mkdir(parents=True, exist_ok=True)
    report_path = Path("reports/data/development-downloads.json")
    report = read_json(report_path) if report_path.exists() else {"repo": "depth-anything/DA3-BENCH", "revision": REVISION, "datasets": {}}
    needed = sum(FILES[name][0] for name in args.datasets if not (root / f"{name}.zip").exists())
    if shutil.disk_usage(root).free < needed + 20 * 1024**3:
        raise SystemExit("Insufficient local disk headroom; no archive download started")
    for name in args.datasets:
        size, expected = FILES[name]
        report["datasets"][name] = {"status": "downloading", "expected_bytes": size, "expected_sha256": expected}
        atomic_json(report_path, report)
        try:
            path = Path(hf_hub_download("depth-anything/DA3-BENCH", f"{name}.zip", repo_type="dataset",
                                       revision=REVISION, local_dir=root, token=False))
            if path.stat().st_size != size or file_hash(path) != expected:
                raise RuntimeError("Official archive checksum mismatch")
            report["datasets"][name].update(status="verified", verified_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(report_path, report)
            print(f"{name}: official benchmark archive verified; held-out data unopened", flush=True)
        except Exception as error:
            report["datasets"][name].update(status="failed", error_type=type(error).__name__)
            atomic_json(report_path, report)
            raise SystemExit("Development archive download failed; see status report") from None
