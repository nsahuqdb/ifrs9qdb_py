"""What code produced a run.

Captured separately from the config snapshot, because the two move at
different rates. Together, (snapshot_id, code_sha) is the full reproducibility
key: the config says what the run was told, the SHA says what read it.

Everything here degrades quietly. A deployment need not be a git checkout, and
a missing SHA is a gap in the record rather than a reason to refuse a run.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["get_current_code_sha", "code_status", "compare_code_to_snapshot"]


def _git(args, cwd=None) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def get_current_code_sha(cwd=None) -> str | None:
    """The working tree's commit, or None when this is not a git checkout."""
    return _git(["rev-parse", "HEAD"], cwd) or None


def code_status(cwd=None) -> dict:
    """SHA, branch, whether the tree is dirty, and when it was last committed.

    ``dirty`` is the one that matters at close: a run produced from a modified
    working tree cannot be reproduced from its SHA, and the record should say
    so rather than implying it can.
    """
    sha = _git(["rev-parse", "HEAD"], cwd)
    if not sha:
        return {"sha": None, "dirty": None, "branch": None,
                "last_commit_at": None, "available": False}
    return {
        "sha": sha,
        "dirty": bool(_git(["status", "--porcelain"], cwd)),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or None,
        "last_commit_at": _git(["log", "-1", "--format=%cI"], cwd) or None,
        "available": True,
    }


def compare_code_to_snapshot(snapshot_meta, cwd=None) -> dict:
    """Whether the deployed code is the code a snapshot was created with.

    Used at the start of a run, so somebody reproducing an old quarter against
    newer code knows that is what they are doing. It reports rather than
    refuses: running old config against new code is sometimes exactly the
    intention.
    """
    recorded = (snapshot_meta or {}).get("code_sha_at_creation")
    current = get_current_code_sha(cwd)
    if not recorded or not current:
        return {"match": None, "current": current, "snapshot": recorded,
                "message": "no SHA recorded on one side; cannot compare"}
    if current == recorded:
        return {"match": True, "current": current, "snapshot": recorded,
                "message": ""}
    label = (snapshot_meta or {}).get("label") or "<unlabelled>"
    return {
        "match": False, "current": current, "snapshot": recorded,
        "message": (f"snapshot '{label}' was created with code {recorded[:12]}; "
                    f"the deployed code is {current[:12]}. Output may differ "
                    "from what that snapshot produced."),
    }
