"""Validator findings the bank has explicitly accepted.

A suppressed finding still RUNS and is still recorded. What changes is that it
stops gating: its effective severity becomes INFO, so a known and accepted
condition does not block a close.

That is deliberate, and the alternative is worse. Without an auditable way to
say "we know, and it is fine here", people learn to ignore a permanently red
screen, and the one finding that mattered goes with it.

Every suppression therefore carries a REASON and an APPROVER, and may carry an
expiry. Nothing is ever deleted: the file is the audit trail.

File format, at ``config/validation_suppressions.yml`` or inside a config
snapshot:

    schema_version: "1.0"
    suppressions:
      - validator_id: INPUT_AccountMaster_dpd_nonneg
        reason: "Two contracts carry legacy negative DPD from a 2019
                 data-entry error, tracked in ticket #1234."
        approved_by: priya
        approved_at: 2026-04-15T10:00:00+0300
        valid_until: 2026-12-31       # optional

Suppressions do NOT carry forward to a new config snapshot. A new snapshot
starts clean, so an exception accepted for one quarter has to be accepted again
rather than quietly becoming permanent.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import pandas as pd
import yaml

__all__ = ["EMPTY_SUPPRESSIONS", "load_suppressions", "active_suppression_ids",
           "add_suppression", "suppression_reasons"]

_FIELDS = ("validator_id", "reason", "approved_by", "approved_at", "valid_until")

EMPTY_SUPPRESSIONS = pd.DataFrame({c: pd.Series(dtype="object") for c in _FIELDS})


def load_suppressions(path) -> pd.DataFrame:
    """Read the file. A missing or empty file is the normal case, not an error.

    A malformed file returns empty rather than raising: a validation run that
    cannot start because its exception list will not parse is strictly worse
    than one that runs with nothing suppressed and shows every finding.
    """
    if path is None:
        return EMPTY_SUPPRESSIONS.copy()
    p = Path(path)
    if not p.is_file():
        return EMPTY_SUPPRESSIONS.copy()
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return EMPTY_SUPPRESSIONS.copy()

    entries = raw.get("suppressions") or []
    if not isinstance(entries, list) or not entries:
        return EMPTY_SUPPRESSIONS.copy()

    rows = []
    for s in entries:
        if not isinstance(s, dict):
            continue
        rows.append({c: ("" if s.get(c) is None else str(s.get(c)))
                     for c in _FIELDS})
    if not rows:
        return EMPTY_SUPPRESSIONS.copy()
    return pd.DataFrame(rows, columns=list(_FIELDS))


def active_suppression_ids(suppressions, as_of=None) -> list[str]:
    """The ids in force today.

    An entry with no ``valid_until`` never expires; one with a date stops
    applying the day after it. An unparseable date is treated as no expiry,
    which keeps a typo from silently un-suppressing a finding the bank thinks
    it has accepted -- the finding would reappear with no explanation.
    """
    if suppressions is None or len(suppressions) == 0:
        return []
    as_of = as_of or _dt.date.today()
    if isinstance(as_of, _dt.datetime):
        as_of = as_of.date()

    out, seen = [], set()
    for _, r in pd.DataFrame(suppressions).iterrows():
        vid = str(r.get("validator_id") or "").strip()
        if not vid or vid in seen:
            continue
        raw = str(r.get("valid_until") or "").strip()
        if raw:
            try:
                until = pd.to_datetime(raw, errors="raise").date()
                if until < as_of:
                    continue
            except Exception:
                pass
        seen.add(vid)
        out.append(vid)
    return out


def suppression_reasons(suppressions, as_of=None) -> dict[str, str]:
    """``{validator_id: reason}`` for the ids in force, as the runner wants."""
    active = set(active_suppression_ids(suppressions, as_of))
    if not active:
        return {}
    out = {}
    for _, r in pd.DataFrame(suppressions).iterrows():
        vid = str(r.get("validator_id") or "").strip()
        if vid in active and vid not in out:
            out[vid] = str(r.get("reason") or "")
    return out


def _who() -> str:
    return (os.environ.get("IFRS9_USER") or os.environ.get("USER")
            or os.environ.get("USERNAME") or "unknown")


def add_suppression(path, validator_id: str, reason: str,
                    approved_by: str | None = None,
                    valid_until: str | None = None, audit=None) -> Path:
    """Append a suppression and write the file atomically.

    The reason and the approver are REQUIRED. A suppression without them is an
    unexplained silence in the audit trail, which is the thing this file exists
    to prevent. Omitting ``approved_by`` falls back to the logged-in user;
    passing it blank raises, because substituting a name nobody chose is worse
    than refusing.

    The write goes to a temporary file and is then renamed, so an interrupted
    write cannot leave a half-parsed exception list behind.
    """
    validator_id = str(validator_id or "").strip()
    reason = str(reason or "").strip()
    # OMITTING the approver falls back to the logged-in user; passing a BLANK
    # one is a mistake, and quietly substituting the OS user for it would put
    # a name in the audit trail that nobody chose.
    approved_by = _who() if approved_by is None else str(approved_by).strip()
    if not validator_id:
        raise ValueError("validator_id is required")
    if not reason:
        raise ValueError("reason is required - it is the audit trail")
    if not approved_by:
        raise ValueError("approved_by is required - it is the audit trail")

    p = Path(path)
    existing = {}
    if p.is_file():
        try:
            existing = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            existing = {}
    existing.setdefault("schema_version", "1.0")
    entries = existing.get("suppressions")
    if not isinstance(entries, list):
        entries = []

    entry = {
        "validator_id": validator_id,
        "reason": reason,
        "approved_by": approved_by,
        "approved_at": _dt.datetime.now().astimezone().strftime(
            "%Y-%m-%dT%H:%M:%S%z"),
    }
    if valid_until:
        entry["valid_until"] = str(valid_until)
    entries.append(entry)
    existing["suppressions"] = entries

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    tmp.replace(p)

    if audit is not None:
        try:
            audit.write("suppression_add", validator_id=validator_id,
                        reason=reason, approved_by=approved_by,
                        valid_until=valid_until or "")
        except Exception:
            pass
    return p
