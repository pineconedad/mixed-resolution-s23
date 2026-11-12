"""Verify pinned DA3 archives locally, with license and held-out access gates."""
import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.data.benchmark import ARCHIVES, HUB_REVISION, PROJECT_ROOT, require_dataset_access
from mixres.io import atomic_json, file_hash, read_json

REVISION = HUB_REVISION
FILES = {
    "eth3d": (15101568883, "2a12b961e72566e194d707db02f2be65686ff12ad5f4347cb16b4c266b2d6917"),
    "7scenes": (3553806786, "4fc12afe9922c975068abd8918b897580926a932b718ad7646f6822205ef5b88"),
}
FILES.update({name: (ARCHIVES[name]["bytes"], ARCHIVES[name]["sha256"])
              for name in ("scannetpp", "dtu", "hiroom")})


def main(datasets, *, root=None, permission_receipt=None, metadata_only=False):
    # Public release metadata does not grant permission to access the dataset.
    if not metadata_only:
        for name in datasets:
            require_dataset_access(name, PROJECT_ROOT, permission_receipt)
    if metadata_only:
        for name in datasets:
            size, sha = FILES[name]
            print(f"{name}.zip revision={REVISION} bytes={size} sha256={sha}")
        return
    from huggingface_hub import hf_hub_download

    root = Path(root or PROJECT_ROOT / "data/benchmark_archives")
    root.mkdir(parents=True, exist_ok=True)
    report_path = PROJECT_ROOT / "reports/data/development-downloads.json"
    report = read_json(report_path) if report_path.exists() else {"repo": "depth-anything/DA3-BENCH", "revision": REVISION, "datasets": {}}
    if report.get("repo") != "depth-anything/DA3-BENCH" or report.get("revision") != REVISION:
        raise RuntimeError("Download receipt does not match the pinned release")
    needed = sum(FILES[name][0] for name in datasets if not (root / f"{name}.zip").exists())
    if shutil.disk_usage(root).free < needed + 20 * 1024**3:
        raise SystemExit("Insufficient local disk headroom; no archive download started")
    for name in datasets:
        size, expected = FILES[name]
        report["datasets"][name] = {"status": "downloading", "expected_bytes": size, "expected_sha256": expected}
        if name == "scannetpp":
            report["datasets"][name]["permission_receipt_sha256"] = file_hash(permission_receipt)
        atomic_json(report_path, report)
        try:
            path = Path(hf_hub_download("depth-anything/DA3-BENCH", f"{name}.zip", repo_type="dataset",
                                       revision=REVISION, local_dir=root, token=False))
            if path.stat().st_size != size or file_hash(path) != expected:
                raise RuntimeError("Official archive checksum mismatch")
            report["datasets"][name].update(status="verified", verified_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(report_path, report)
            print(f"{name}: official benchmark archive checksum verified; archive not extracted", flush=True)
        except Exception as error:
            report["datasets"][name].update(status="failed", error_type=type(error).__name__)
            atomic_json(report_path, report)
            raise SystemExit("Benchmark archive download failed; see status report") from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(FILES), default=["eth3d", "7scenes"])
    parser.add_argument("--root", type=Path, help="Archive download directory")
    parser.add_argument("--metadata-only", action="store_true", help="Print pinned public metadata without dataset access")
    parser.add_argument("--scannetpp-permission-receipt", type=Path,
                        help="Explicit original-license permission JSON; see 00_make_views.py --help")
    args = parser.parse_args()
    main(args.datasets, root=args.root, permission_receipt=args.scannetpp_permission_receipt,
         metadata_only=args.metadata_only)
