"""Budget decisions and backup-bound termination. No cloud resource creation."""
import math
import tarfile
import json
from dataclasses import dataclass

from mixres.io import file_hash


@dataclass(frozen=True)
class Budget:
    balance_at_start: float
    hourly_total: float
    reserve_usd: float = 3.0
    session_cap_usd: float = 1.0
    max_seconds: float = 3600
    backup_margin_seconds: float = 600

    def __post_init__(self):
        values = (self.balance_at_start, self.hourly_total, self.reserve_usd,
                  self.session_cap_usd, self.max_seconds, self.backup_margin_seconds)
        if not all(math.isfinite(value) for value in values) or min(values) < 0 or self.hourly_total == 0:
            raise ValueError("Budget requires finite nonnegative values and a positive hourly rate")
        if self.reserve_usd < 3 or self.session_cap_usd <= 0 or self.max_seconds <= self.backup_margin_seconds:
            raise ValueError("Keep the $3 reserve and enough time for backup")

    @property
    def hard_seconds(self):
        spend = min(self.session_cap_usd, max(0, self.balance_at_start - self.reserve_usd))
        return min(self.max_seconds, spend / self.hourly_total * 3600)

    def stop_reason(self, elapsed_seconds, live_balance, *, balance_read_ok=True):
        if not balance_read_ok or live_balance is None or not math.isfinite(live_balance):
            return "balance_unverified"
        if elapsed_seconds < 0 or not math.isfinite(elapsed_seconds):
            return "invalid_clock"
        safe_balance = min(live_balance, self.balance_at_start - self.hourly_total * elapsed_seconds / 3600)
        backup_cost = self.hourly_total * self.backup_margin_seconds / 3600
        if safe_balance <= self.reserve_usd + backup_cost:
            return "credit_reserve"
        if elapsed_seconds >= max(0, self.hard_seconds - self.backup_margin_seconds):
            return "session_limit"
        return None


def verified_backup(archive, expected_hash, pod_id):
    if file_hash(archive) != expected_hash:
        raise ValueError("Archive checksum mismatch")
    import hashlib
    from pathlib import PurePosixPath

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive paths")
        if any(not m.isfile() or PurePosixPath(m.name).is_absolute() or ".." in PurePosixPath(m.name).parts for m in members):
            raise ValueError("Unsafe archive member")
        manifest = json.load(tar.extractfile("session-manifest.json"))
        if manifest.get("pod_id") != pod_id:
            raise ValueError("Backup is not bound to this pod")
        if set(names) != set(manifest["files"]) | {"session-manifest.json"}:
            raise ValueError("Archive membership mismatch")
        for name, expected in manifest["files"].items():
            digest = hashlib.sha256()
            with tar.extractfile(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError("Backup member checksum mismatch")
    return {"pod_id": pod_id, "archive_sha256": expected_hash,
            "all_members_verified": True, "member_count": len(manifest["files"]),
            "session_id": manifest.get("session_id"),
            "job_quiescent": manifest.get("job_quiescent") is True}
