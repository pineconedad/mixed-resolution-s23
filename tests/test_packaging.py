import subprocess
from pathlib import Path


def test_dataset_ignore_does_not_hide_source_package():
    root = Path(__file__).resolve().parents[1]
    sources = ["mixres/data/__init__.py", "mixres/data/resize.py", "mixres/data/sampler.py", "mixres/data/remote_zip.py"]
    for source in sources:
        assert (root / source).is_file()
        result = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", source], capture_output=True)
        assert result.returncode == 1, f"Source package must not be ignored: {source}"
        result = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", source], capture_output=True)
        assert result.returncode == 0, f"Source package must be tracked: {source}"
