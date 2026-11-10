"""Record the independent monitor environment, without environment-variable values."""
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixres.io import atomic_json, file_hash

root = Path(__file__).resolve().parents[1]
binary = root / "artifacts/runpodctl-darwin-arm64"
atomic_json(root / "reports/control/ops-environment.json", {
    "python": sys.version, "platform": platform.platform(),
    "mcp": importlib.metadata.version("mcp"),
    "runpodctl_version": subprocess.check_output([str(binary), "version"], text=True).strip(),
    "runpodctl_sha256": file_hash(binary),
    "requirements": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()})
print("Recorded independent operations environment without credentials")
