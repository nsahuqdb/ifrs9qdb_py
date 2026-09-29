"""
Governance around a run: snapshots, the audit log and approval.

None of this changes a number. All of it exists so that months later someone
can answer three questions without guesswork:

    what produced this figure   -> the snapshot frozen with the run
    who did what, and when      -> the audit log
    who signed it off           -> the approval record

A run's snapshot is taken BEFORE it is priced, not after, so it records the
configuration that was actually used rather than whatever the file says today.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

__all__ = ["take_snapshot", "read_snapshot", "compare_snapshots",
           "AuditLog", "approve_run", "read_approval", "APPROVAL_STAGES"]


# ============================================================ snapshots =====
def _hash_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def take_snapshot(run_dir, config_dir=None, static_dir=None,
                  run_config_file=None, run_config: dict | None = None,
                  snapshot_meta: dict | None = None) -> dict:
    """Freeze the configuration a run used, with a hash per file.

    Copying is not enough on its own: a copy can be edited afterwards and look
    original. The hashes are what make the snapshot evidence rather than a
    convenience.

    As R's phase 1 freezes it: config/ (model, inputs, overlays,
    suppressions) PLUS the run config, config.yml, beside them -- it names the
    model and the gating policy the run ran under -- and static/. A config
    version already carries its own config.yml in its config/ folder; a live
    run's lives at the project root and comes in as ``run_config_file`` (or,
    for a run given a config dict and no file, is written from
    ``run_config``). ``config_used.yml`` records what was copied from where,
    as R writes it.
    """
    import yaml
    run_dir = Path(run_dir)
    dest = run_dir / "config_used"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"taken": datetime.now().isoformat(timespec="seconds"),
                "by": _who(), "files": {}}

    for label, src in (("config", config_dir), ("static", static_dir)):
        if not src:
            continue
        src = Path(src)
        if not src.is_dir():
            continue
        target = dest / label
        target.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.iterdir()):
            if not p.is_file():
                continue
            shutil.copy(p, target / p.name)
            manifest["files"][f"{label}/{p.name}"] = {
                "sha256_16": _hash_file(p), "bytes": p.stat().st_size}

    rc_target = dest / "config" / "config.yml"
    rc_source = None
    if not rc_target.exists():
        rc_target.parent.mkdir(parents=True, exist_ok=True)
        if run_config_file and Path(run_config_file).is_file():
            shutil.copy(Path(run_config_file), rc_target)
            rc_source = str(run_config_file)
        elif isinstance(run_config, dict):
            rc_target.write_text(yaml.safe_dump(run_config, sort_keys=False,
                                                allow_unicode=True),
                                 encoding="utf-8")
            rc_source = "(the run's config dict)"
        if rc_target.exists():
            manifest["files"]["config/config.yml"] = {
                "sha256_16": _hash_file(rc_target),
                "bytes": rc_target.stat().st_size}

    marker = {
        "schema_version": "1.0",
        "kind": "snapshot" if snapshot_meta else "live",
        "snapshot_label": (snapshot_meta or {}).get("label"),
        "source_config": str(config_dir) if config_dir else None,
        "source_run_cfg": rc_source or "(in source_config)",
        "source_static": str(static_dir) if static_dir else None,
        "frozen_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (dest / "config_used.yml").write_text(
        yaml.safe_dump(marker, sort_keys=False), encoding="utf-8")
    (dest / "snapshot.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_snapshot(run_dir) -> dict | None:
    p = Path(run_dir) / "config_used" / "snapshot.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def compare_snapshots(run_a, run_b) -> pd.DataFrame:
    """What changed in the configuration between two runs.

    Comparing hashes rather than contents answers the question people actually
    ask -- "did anything change?" -- without a diff nobody reads.
    """
    a, b = read_snapshot(run_a), read_snapshot(run_b)
    if a is None or b is None:
        return pd.DataFrame()
    fa, fb = a.get("files", {}), b.get("files", {})
    rows = []
    for name in sorted(set(fa) | set(fb)):
        ha = fa.get(name, {}).get("sha256_16")
        hb = fb.get(name, {}).get("sha256_16")
        if ha == hb:
            status = "unchanged"
        elif ha is None:
            status = "added"
        elif hb is None:
            status = "removed"
        else:
            status = "CHANGED"
        rows.append({"file": name, "run_a": ha or "—", "run_b": hb or "—",
                     "status": status})
    return pd.DataFrame(rows)


# ============================================================ audit log =====
def _who() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


@dataclass
class AuditLog:
    """An append-only record of what was done to a run.

    Append-only on purpose: an audit trail that can be edited is not one. Each
    entry carries the actor and the time, and nothing here ever rewrites an
    earlier line.
    """
    path: Path

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: str, detail: str = "", **extra) -> dict:
        entry = {"time": datetime.now().isoformat(timespec="seconds"),
                 "who": _who(), "action": action, "detail": detail, **extra}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A corrupt line must not hide the rest of the trail.
                out.append({"time": "", "who": "", "action": "unreadable entry",
                            "detail": line[:200]})
        return out

    def to_frame(self) -> pd.DataFrame:
        e = self.entries()
        return pd.DataFrame(e) if e else pd.DataFrame(
            columns=["time", "who", "action", "detail"])


# ============================================================= approval =====
# Two sign-offs, because one person should not be able to release a provision
# alone. Risk owns the model view, Finance owns the booking.
APPROVAL_STAGES = ("risk", "finance")


def approve_run(run_dir, stage: str, approver: str, comment: str = "",
                validation_passed: bool | None = None) -> dict:
    """Record a sign-off.

    An approval is refused while validation is failing. The point of a gate is
    that it stops something; one that can be waved through on a bad run is
    decoration.
    """
    if stage not in APPROVAL_STAGES:
        raise ValueError(f"stage must be one of {APPROVAL_STAGES}")
    if not approver:
        raise ValueError("an approval needs a named approver")
    if validation_passed is False:
        raise ValueError(
            "This run has failing error-level checks. Resolve them, or record "
            "an explicit suppression with a reason, before approving.")

    run_dir = Path(run_dir)
    p = run_dir / "approval.json"
    current = read_approval(run_dir) or {"approvals": {}}
    current["approvals"][stage] = {
        "approver": approver, "comment": comment,
        "time": datetime.now().isoformat(timespec="seconds"),
        "recorded_by": _who(),
    }
    current["complete"] = all(s in current["approvals"] for s in APPROVAL_STAGES)
    p.write_text(json.dumps(current, indent=2))

    AuditLog(run_dir / "audit.jsonl").record(
        "approval", f"{stage} sign-off by {approver}", stage=stage,
        approver=approver, comment=comment)
    return current


def read_approval(run_dir) -> dict | None:
    p = Path(run_dir) / "approval.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def approval_status(run_dir) -> dict:
    """Where a run stands, in the form a queue screen needs."""
    a = read_approval(run_dir) or {"approvals": {}}
    got = a.get("approvals", {})
    return {
        "complete": bool(a.get("complete")),
        "outstanding": [s for s in APPROVAL_STAGES if s not in got],
        "approvals": got,
    }
