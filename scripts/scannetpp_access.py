"""Local-only approval handoff; never handles the user's site credentials."""
import json
import secrets
import subprocess
import sys
import threading
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mixres.io import atomic_json

DIRECTORY = ROOT / "artifacts/permissions"
RECEIPT = DIRECTORY / "scannetpp.json"
NONCE = secrets.token_hex(24)
STATE = {"phase": "waiting_for_your_approval_confirmation"}


def continue_download():
    try:
        STATE["phase"] = "downloading_authorized_archive"
        with (DIRECTORY / "scannetpp-download.log").open("w") as log:
            subprocess.run([sys.executable, "scripts/download_benchmark.py", "--datasets", "scannetpp",
                            "--scannetpp-permission-receipt", str(RECEIPT)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        STATE["phase"] = "extracting_verified_archive"
        destination = ROOT / "third_party/Depth-Anything-3/workspace/benchmark_dataset"
        with zipfile.ZipFile(ROOT / "data/benchmark_archives/scannetpp.zip") as archive:
            if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                   for name in archive.namelist()):
                raise ValueError("Unsafe archive member")
            archive.extractall(destination)
        STATE["phase"] = "archive_downloaded_verified_and_extracted"
    except Exception as error:
        STATE.update(phase="download_or_extraction_failed", error=type(error).__name__)
    atomic_json(DIRECTORY / "scannetpp-status.json", STATE)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        page = f"""<!doctype html><html><meta charset="utf-8"><title>ScanNet++ access</title>
<style>body{{font:17px system-ui;max-width:760px;margin:48px auto;padding:0 24px;line-height:1.6}}
button{{font:inherit;padding:12px}}label{{display:block;margin:18px 0}}code{{font-size:14px}}</style>
<h1>ScanNet++ access</h1><p>This page runs only on your Mac. It does not collect credentials.</p>
<ol><li><a href="https://kaldir.vc.in.tum.de/scannetpp/register" target="_blank">Register with ScanNet++</a> using your institutional email.</li>
<li>Log in, create an application and accept the dataset Terms of Use.</li>
<li>Wait for the official dashboard or email to say your application is approved. Registration alone is not approval.</li>
<li>Confirm below once approved. The pinned processed archive will download and extract automatically. No chat message is needed.</li></ol>
<p><a href="https://kaldir.vc.in.tum.de/scannetpp/dashboard" target="_blank">Open official dashboard</a></p>
<form method="post"><input type="hidden" name="nonce" value="{NONCE}">
<label><input type="checkbox" required name="approved" value="yes"> My ScanNet++ application is approved.</label>
<label><input type="checkbox" required name="terms" value="yes"> I personally accepted the dataset Terms of Use for this research.</label>
<button>Confirm approved access and continue automatically</button></form>
<p>Local state: <code>{STATE['phase']}</code>. Refresh this page to check download status.</p></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode())

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        fields = parse_qs(self.rfile.read(min(size, 4096)).decode())
        if (fields.get("nonce") != [NONCE] or fields.get("approved") != ["yes"]
                or fields.get("terms") != ["yes"]):
            self.send_error(400, "Both approval and terms acceptance are required")
            return
        if STATE["phase"] == "waiting_for_your_approval_confirmation":
            atomic_json(RECEIPT, {"dataset": "scannetpp", "permission_granted": True,
                        "terms_accepted": True, "accepted_by": "local_account_user",
                        "accepted_utc": datetime.now(timezone.utc).isoformat(),
                        "license_url": "https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf",
                        "evidence": "User personally confirmed official application approval and terms acceptance on local approval form"})
            STATE["phase"] = "approval_received"
            threading.Thread(target=continue_download, daemon=True).start()
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


if __name__ == "__main__":
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    server = HTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    atomic_json(DIRECTORY / "access-page.json", {"url": url, "receipt": str(RECEIPT)})
    subprocess.run(["open", "https://kaldir.vc.in.tum.de/scannetpp/register", url], check=True)
    server.serve_forever()
