"""The project-wide audit log: one JSON line per event, append-only.

R's R/audit_log.R, event for event. Every meaningful action -- a run starting,
pausing for overrides, finishing; a validation stage; a pre-run check; a config
version created, edited or promoted; a suppression added; a run approved or
rejected; an export -- appends one line to ``logs/etl_audit.jsonl``. The R app's
Audit log page reads that file, so the two engines writing the same events to
the same file is what lets one page show both.

The per-run ``audit.jsonl`` a run keeps beside its outputs is separate and
stays: that is the history of ONE run, this is the history of the project.

Where the file goes, first match wins:

    set_audit_log_path(path)     what the app sets at startup
    IFRS9_AUDIT_LOG              environment
    logs/etl_audit.jsonl         relative to the working directory, as R's
"""
from __future__ import annotations

import contextlib
import getpass
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

__all__ = ["audit_log_path", "set_audit_log_path", "audit_event",
           "read_audit_log", "audit_suspended", "EVENT_LABELS",
           "event_label", "event_summary"]

_PATH: Path | None = None
_SUSPENDED = False


def set_audit_log_path(path) -> None:
    """Point the log somewhere else (the app pins it under its project root)."""
    global _PATH
    _PATH = Path(path) if path else None


def audit_log_path() -> Path:
    if _PATH is not None:
        return _PATH
    return Path(os.environ.get("IFRS9_AUDIT_LOG", "logs/etl_audit.jsonl"))


def _user() -> str:
    u = os.environ.get("IFRS9_USER")
    if u:
        return u
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


@contextlib.contextmanager
def audit_suspended():
    """Record nothing inside the block.

    A pre-run readiness check runs the pipeline into a temporary folder; its
    run_start and validation events would read as a real run in the log, so it
    suspends logging and records one event of its own -- as R's does.
    """
    global _SUSPENDED
    before = _SUSPENDED
    _SUSPENDED = True
    try:
        yield
    finally:
        _SUSPENDED = before


def audit_event(payload: dict) -> Path | None:
    """Append one event. ``payload`` must carry ``event``.

    ``ts`` and ``user`` are filled in when absent. A failure to write is
    swallowed: an audit line that cannot be written must not fail the run that
    produced it (the run's own reports still record what happened).
    """
    if _SUSPENDED:
        return None
    if not isinstance(payload, dict) or not payload.get("event"):
        raise ValueError("audit_event: payload must be a dict with an 'event'")
    entry = dict(payload)
    entry.setdefault("ts", datetime.now().astimezone()
                     .strftime("%Y-%m-%dT%H:%M:%S%z"))
    entry.setdefault("user", _user())
    path = audit_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=_json_default) + "\n")
    except OSError:
        return None
    return path


def _json_default(v):
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    try:
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
    except Exception:
        pass
    return str(v)


def read_audit_log(path=None) -> pd.DataFrame:
    """Every event as a row, the union of their fields as columns.

    A corrupt line is kept as an ``unreadable`` event rather than dropped: a
    gap in an audit trail is worse than a line nobody can parse.
    """
    p = Path(path) if path else audit_log_path()
    if not p.is_file():
        return pd.DataFrame(columns=["ts", "event", "user", "run_id"])
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"ts": "", "event": "unreadable", "user": "",
                         "raw": line[:200]})
    df = pd.DataFrame(rows)
    for c in ("ts", "event", "user", "run_id"):
        if c not in df.columns:
            df[c] = None
    return df


# ------------------------------------------------------------ presentation --
EVENT_LABELS = {
    "run_start": "Run started",
    "run_finish": "Run finished",
    "run_unofficial": "Unofficial run",
    "run_pending_checker": "Official run (pending approval)",
    "run_phase1_complete": "Paused for overrides",
    "run_export": "Outputs exported",
    "run_overrides_applied": "Overrides applied",
    "run_approved": "Run approved",
    "run_rejected": "Run rejected",
    "validation_summary": "Validation",
    "pre_run_check": "Pre-run check",
    "pre_run_readiness": "Pricing readiness",
    "snapshot_create": "Version created",
    "snapshot_clone": "Version cloned",
    "snapshot_edit": "Config edited",
    "snapshot_promote": "Status changed",
    "suppression_add": "Suppression added",
    "overlay_saved": "Overlay saved",
    "overlay_status": "Overlay status changed",
    "overlay_applied": "Overlay applied to run",
    "overlay_removed": "Overlay removed from run",
    "run_cancelled": "Run cancelled",
}


def event_label(ev) -> str:
    return EVENT_LABELS.get(str(ev), str(ev))


def _s(row: dict, k: str) -> str:
    v = row.get(k)
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        # a frame of mixed events stores counts as floats (NaN elsewhere)
        if v.is_integer():
            return str(int(v))
    s = str(v)
    return "" if s in ("nan", "None", "NA") else s


def event_summary(row: dict) -> str:
    """One readable line per event -- R's .audit_summary(), case for case."""
    ev = _s(row, "event")
    s = lambda k: _s(row, k)                                    # noqa: E731
    or_ = lambda x, alt: x if x else alt                        # noqa: E731
    if ev == "run_start":
        sc = s("ecl_scenario")
        return (f"Run started on config version '{or_(s('snapshot'), 'default config')}'"
                + (f" — scenario: {sc}" if sc and sc != "weighted" else ""))
    if ev == "run_finish":
        d = s("duration_seconds")
        try:
            dtxt = f" in {float(d):.0f}s" if d else ""
        except ValueError:
            dtxt = ""
        return f"Run finished — {or_(s('n_outputs'), '?')} output files{dtxt}"
    if ev == "run_export":
        return "Outputs downloaded"
    if ev == "run_overrides_applied":
        n = s("n_overrides")
        if n and n != "0":
            parts = [f"{s(k)} {lbl}" for k, lbl in
                     (("n_rating_overrides", "rating"),
                      ("n_stage_overrides", "stage"),
                      ("n_restructuring_overrides", "restructuring"))
                     if s(k) and s(k) != "0"]
            return (f"Applied {n} override{'' if n == '1' else 's'}"
                    + (f" ({', '.join(parts)})" if parts else ""))
        return "No overrides applied"
    if ev == "run_unofficial":
        n = or_(s("n_overrides"), "0")
        sc = s("ecl_scenario")
        return (f"Unofficial run finished — {or_(s('n_outputs'), '?')} output "
                f"files, {n} override{'' if n == '1' else 's'}"
                + (f"; scenario: {sc}" if sc and sc != "weighted" else "")
                + " (auto-approved)")
    if ev == "run_pending_checker":
        n = or_(s("n_overrides"), "0")
        return (f"Official run finished — {or_(s('n_outputs'), '?')} output "
                f"files, {n} override{'' if n == '1' else 's'}; pending approval")
    if ev == "run_phase1_complete":
        return (f"Phase 1 done ({or_(s('n_customers_lending'), '?')} lending "
                f"customers, {or_(s('n_accounts_investments'), '?')} investment "
                "accounts) — paused for overrides")
    if ev == "validation_summary":
        bits = []
        if s("n_error") and s("n_error") != "0":
            bits.append(f"{s('n_error')} errors")
        if s("n_warn") and s("n_warn") != "0":
            bits.append(f"{s('n_warn')} warnings")
        return (f"{or_(s('stage'), 'Validation')} checks: {or_(s('n_pass'), '?')} "
                f"of {or_(s('n_total'), '?')} passed"
                + (f" ({', '.join(bits)})" if bits else ""))
    if ev == "pre_run_check":
        return (f"Pre-run check: {or_(s('n_pass'), '?')} of "
                f"{or_(s('n_total'), '?')} checks passed")
    if ev == "pre_run_readiness":
        # R's field names: contracts, no_ecl, blank_in_lic, with_gaps
        return (f"Pricing readiness: {or_(s('contracts'), '?')} contracts, "
                f"{or_(s('no_ecl'), '0')} with no ECL, "
                f"{or_(s('blank_in_lic'), '0')} blank in LIC, "
                f"{or_(s('with_gaps'), '0')} priced with a gap")
    if ev == "snapshot_create":
        p = s("parent")
        return (f"Created version '{s('snapshot')}'"
                + (f" based on '{p}'" if p else " from the default config"))
    if ev == "snapshot_clone":
        return f"Cloned version '{s('cloned_from')}' into new draft '{s('snapshot')}'"
    if ev == "snapshot_edit":
        rel = Path(s("relpath")).name if s("relpath") else "a file"
        return (f"Edited {rel} in version '{s('snapshot')}'"
                + (f" — now {s('n_rows')} rows" if s("n_rows") else ""))
    if ev == "snapshot_promote":
        rank = {"draft": 1, "tested": 2, "pending_final": 3, "pending": 3,
                "approved": 4, "rejected": 0, "archived": 0}
        f, t = s("from_status"), s("to_status")
        back = t in rank and f in rank and rank[t] < rank[f]
        return (f"Version '{s('snapshot')}' {'moved back to' if back else 'moved to'} "
                f"{or_(t, '?')} (was {or_(f, '?')})")
    if ev == "suppression_add":
        return (f"Check '{s('validator_id')}' suppressed until "
                f"{or_(s('valid_until'), '?')}")
    if ev in ("run_approved", "run_rejected"):
        return (f"Run {'approved' if ev == 'run_approved' else 'rejected'}"
                + (f" \u2014 {s('reason')}" if s("reason") else ""))
    if ev == "run_cancelled":
        return "Run cancelled at the review pause; its partial folder was removed"
    if ev == "overlay_saved":
        return f"Overlay '{s('overlay_id')}' saved ({or_(s('n_rules'), '?')} rule(s))"
    if ev == "overlay_status":
        return f"Overlay '{s('overlay_id')}' moved to {or_(s('to_status'), s('status'))}"
    if ev == "overlay_applied":
        return f"Overlay '{s('overlay_id')}' applied to {or_(s('run_id'), 'a run')}"
    if ev == "overlay_removed":
        return f"Overlay '{s('overlay_id')}' removed from {or_(s('run_id'), 'a run')}"
    std = {"ts", "event", "user", "run_id"}
    extras = [f"{k}={_s(row, k)}" for k in row if k not in std and _s(row, k)]
    return ", ".join(extras)
