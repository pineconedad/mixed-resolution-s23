import json
import platform
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mixres.io import atomic_json

output = Path("env")
output.mkdir(exist_ok=True)
(output / "actual-requirements.lock").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
atomic_json(output / "actual-environment.json", {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "submodules": subprocess.check_output(["git", "submodule", "status"], text=True).splitlines()})
