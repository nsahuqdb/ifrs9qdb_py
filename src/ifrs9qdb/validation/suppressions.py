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

A suppression is ended, not deleted: ``remove_suppression`` sets its
``valid_until`` to yesterday and records who ended it, when and why, so the
entry stays in the file as the record of what applied until then.

FINDINGS ACCEPTED FOR ONE RUN are the other kind of exception: the pipeline
page's "accept for this run". They have the same effect on that run -- the
finding is recorded, its effective severity is INFO -- but nothing is written
to this file, so the next run asks again.

Either way the run says what was accepted and why. Its
``reports/accepted_findings.csv`` lists every finding accepted in it: those
accepted for the run, and the standing suppressions that took effect (the
check failed and was recorded as suppressed), each with its reason, who
accepted or approved it and when. validation.md ends with the same list, the
manifest carries it, and the audit log has a ``finding_accepted`` event for
each, with the run id.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import pandas as pd
import yaml

__all__ = ["EMPTY_SUPPRESSIONS", "load_suppressions", "active_suppression_ids",
           "add_suppression", "remove_suppression", "suppression_reasons",
           "ACCEPTED_FIELDS", "RECORD_FIELDS", "normalise_accepted_findings",
           "accepted_reasons", "accepted_findings_record",
           "accepted_findings_markdown"]

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
    from ..audit_log import audit_event
    audit_event({"event": "suppression_add", "validator_id": validator_id,
                 "reason": reason, "approved_by": approved_by,
                 "valid_until": valid_until or None})
    return p


def _now() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def remove_suppression(path, validator_id: str, reason: str,
                       removed_by: str | None = None, as_of=None) -> int:
    """End the suppressions in force for ``validator_id`` -- never delete.

    Each active entry for the id gets ``valid_until`` set to the day before
    ``as_of`` (today), so it no longer applies from today in either engine,
    and ``removed_by``, ``removed_at`` and ``removal_reason`` saying who
    ended it and why. The file stays the record of what applied until then.
    Returns the number of entries ended; 0 when none was in force.
    """
    validator_id = str(validator_id or "").strip()
    reason = str(reason or "").strip()
    removed_by = _who() if removed_by is None else str(removed_by).strip()
    if not validator_id:
        raise ValueError("validator_id is required")
    if not reason:
        raise ValueError("reason is required - it is the audit trail")
    if not removed_by:
        raise ValueError("removed_by is required - it is the audit trail")
    p = Path(path)
    if not p.is_file():
        return 0
    try:
        existing = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"cannot read {p}: {exc}") from exc
    entries = existing.get("suppressions")
    if not isinstance(entries, list):
        return 0
    as_of = as_of or _dt.date.today()
    if isinstance(as_of, _dt.datetime):
        as_of = as_of.date()
    yesterday = (as_of - _dt.timedelta(days=1)).isoformat()
    ended = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        if str(e.get("validator_id") or "").strip() != validator_id:
            continue
        one = pd.DataFrame([{c: ("" if e.get(c) is None else str(e.get(c)))
                             for c in _FIELDS}])
        if not active_suppression_ids(one, as_of):
            continue
        e["valid_until"] = yesterday
        e["removed_by"] = removed_by
        e["removed_at"] = _now()
        e["removal_reason"] = reason
        ended += 1
    if not ended:
        return 0
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    tmp.replace(p)
    from ..audit_log import audit_event
    audit_event({"event": "suppression_remove", "validator_id": validator_id,
                 "reason": reason, "removed_by": removed_by,
                 "n_entries": ended})
    return ended


# ------------------------------------------------- accepted for one run ----
ACCEPTED_FIELDS = ("validator_id", "reason", "accepted_by", "accepted_at")


def normalise_accepted_findings(accepted, user: str | None = None) -> pd.DataFrame:
    """Findings accepted for one run, as one frame.

    ``accepted`` is a list of dicts, a DataFrame, or ``{validator_id:
    reason}``. Every entry needs a reason (ValueError otherwise -- it is the
    audit trail); ``accepted_by`` falls back to ``user``, then the acting
    user, and ``accepted_at`` to now. A repeated id keeps its first entry.
    """
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in ACCEPTED_FIELDS})
    if accepted is None:
        return empty
    if isinstance(accepted, pd.DataFrame):
        rows = accepted.to_dict(orient="records")
    elif isinstance(accepted, dict):
        rows = [{"validator_id": k, "reason": v} for k, v in accepted.items()]
    else:
        rows = list(accepted)
    out, seen = [], set()
    for r in rows:
        r = dict(r or {})
        vid = str(r.get("validator_id") or "").strip()
        if not vid or vid in seen:
            continue
        reason = str(r.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"{vid}: a reason is required - it is the audit trail")
        by = str(r.get("accepted_by") or "").strip() or (user or _who())
        at = str(r.get("accepted_at") or "").strip() or _now()
        seen.add(vid)
        out.append({"validator_id": vid, "reason": reason, "accepted_by": by,
                    "accepted_at": at})
    if not out:
        return empty
    return pd.DataFrame(out, columns=list(ACCEPTED_FIELDS))


def accepted_reasons(accepted) -> dict[str, str]:
    """``{validator_id: reason}`` for findings accepted for one run -- merged
    into the suppressions the runner applies to that run only."""
    a = normalise_accepted_findings(accepted)
    return {r["validator_id"]: f"Accepted for this run by {r['accepted_by']}: "
                               f"{r['reason']}"
            for r in a.to_dict(orient="records")}


RECORD_FIELDS = ("validator_id", "severity", "source", "reason", "accepted_by",
                 "accepted_at", "valid_until", "in_effect")


def accepted_findings_record(accepted, issues, standing=None,
                             as_of=None) -> pd.DataFrame:
    """reports/accepted_findings.csv: every finding accepted in a run, and why.

    One row per finding accepted for the run (``source`` "run"), whether or
    not its check failed, and one per standing suppression that took effect
    -- its check failed and was recorded as suppressed -- (``source``
    "standing", with the approver, the date and the expiry from
    validation_suppressions.yml). ``in_effect``: the acceptance changed the
    run, i.e. the check failed and its effective severity became INFO.
    """
    issues = list(issues or [])
    sev = {i.id: str(i.severity) for i in issues}
    hit = {i.id for i in issues if not i.passed and i.suppressed}
    rows = []
    for r in normalise_accepted_findings(accepted).to_dict(orient="records"):
        vid = r["validator_id"]
        rows.append({"validator_id": vid, "severity": sev.get(vid, ""),
                     "source": "run", "reason": r["reason"],
                     "accepted_by": r["accepted_by"],
                     "accepted_at": r["accepted_at"], "valid_until": "",
                     "in_effect": vid in hit})
    if standing is not None and len(standing):
        active, seen = set(active_suppression_ids(standing, as_of)), set()
        for r in pd.DataFrame(standing).to_dict(orient="records"):
            vid = str(r.get("validator_id") or "").strip()
            if vid not in active or vid not in hit or vid in seen:
                continue
            seen.add(vid)
            rows.append({"validator_id": vid, "severity": sev.get(vid, ""),
                         "source": "standing",
                         "reason": str(r.get("reason") or ""),
                         "accepted_by": str(r.get("approved_by") or ""),
                         "accepted_at": str(r.get("approved_at") or ""),
                         "valid_until": str(r.get("valid_until") or ""),
                         "in_effect": True})
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in RECORD_FIELDS})
    return pd.DataFrame(rows, columns=list(RECORD_FIELDS))


def accepted_findings_markdown(record: pd.DataFrame) -> str:
    """The section validation.md ends with when anything was accepted."""
    if record is None or len(record) == 0:
        return ""
    lines = [f"\n## Accepted findings ({len(record)})\n"]
    for r in record.to_dict(orient="records"):
        sev = f" [{r['severity']}]" if r["severity"] else ""
        if r["source"] == "run":
            who = f"accepted for this run by {r['accepted_by']} at {r['accepted_at']}"
        else:
            who = (f"standing suppression approved by {r['accepted_by']} at "
                   f"{r['accepted_at']}"
                   + (f", valid until {r['valid_until']}" if r["valid_until"] else ""))
        note = "" if r["in_effect"] else " (not in effect: the check did not fail)"
        lines.append(f"- `{r['validator_id']}`{sev} {who}{note}")
        lines.append(f"  - _Reason:_ {r['reason']}")
    return "\n".join(lines) + "\n"
