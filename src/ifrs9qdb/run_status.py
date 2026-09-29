"""Maker-checker: whether a run has been signed, and by whom.

This is the same file the R engine writes, at ``reports/run_status.yml``, in
the same format. That matters more than it looks: the two engines are meant to
share a ``runs/`` folder, and a run written by one has to be readable and
approvable by the other. A different format here would quietly fork the
approval queue in two.

The state machine is small and deliberately strict:

    official run   -> pending_checker -> approved
                                      -> rejected
    unofficial run -> unofficial                     (terminal, no approval)

Only ``pending_checker`` transitions. A run that is already approved cannot be
re-approved, and one that was rejected does not go back into the queue -- a
rejected run is re-run, not re-argued.

Two rules carry the control, and both refuse rather than warn:

  * **A reason is required.** An approval with no reason records that somebody
    clicked, not that somebody decided.
  * **Separation of duties.** Whoever ran the pipeline cannot also approve it,
    when the config asks for that. Rejection is always allowed -- the person
    who built a run should be able to withdraw it.
"""
from __future__ import annotations

import datetime as _dt
import getpass
import json
import os
from pathlib import Path

import pandas as pd
import yaml

__all__ = ["RUN_TYPES", "STATUSES", "TERMINAL_STATUSES", "PENDING_STATUSES",
           "read_run_status", "write_run_status", "init_run_status",
           "normalise_status", "maker_for_run", "transition_run",
           "approve_run_status", "reject_run_status",
           "list_runs_pending_approval", "list_runs_decided",
           "annotate_runs_with_status", "approval_config", "manifest_path"]

RUN_TYPES = ("official", "unofficial")
STATUSES = ("pending_checker", "approved", "rejected", "unofficial", "unknown")
TERMINAL_STATUSES = ("approved", "rejected", "unofficial")
# `unknown` is here on purpose: a run with no status file must SURFACE for
# diagnosis rather than be filtered silently out of the queue.
PENDING_STATUSES = ("pending_checker", "pending_approval", "unknown")


def _now() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _who() -> str:
    for env in ("IFRS9_USER", "USER", "USERNAME"):
        v = os.environ.get(env)
        if v:
            return v
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _status_path(run_dir) -> Path:
    return Path(run_dir) / "reports" / "run_status.yml"


def normalise_status(status) -> str:
    """The current name for a status. ``pending_approval`` is the old one."""
    if not status:
        return "unknown"
    s = str(status).strip()
    return "pending_checker" if s == "pending_approval" else s


def read_run_status(run_dir) -> dict | None:
    p = _status_path(run_dir)
    if not p.is_file():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or None
    except Exception:
        return None


def write_run_status(run_dir, meta: dict) -> Path:
    """Write the status file atomically.

    Through a temporary file and a rename, because a half-written status file
    is a run that has left the queue without being decided.
    """
    p = _status_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def init_run_status(run_dir, run_id: str, run_type: str = "official",
                    by: str | None = None, ecl_scenario: str = "weighted",
                    overrides_applied: dict | None = None,
                    snapshot_label: str | None = None) -> dict:
    """Create the status file a completed run starts life with.

    An official run lands as ``pending_checker`` and must be approved. An
    unofficial one lands terminal, because a scenario or what-if run is not a
    number anybody books.
    """
    if run_type not in RUN_TYPES:
        raise ValueError(f"run_type must be one of {RUN_TYPES}")
    if run_type == "official" and ecl_scenario != "weighted":
        raise ValueError(
            "An official run must use the weighted ECL. Run a single scenario "
            "as an unofficial run.")

    official = run_type == "official"
    status = "pending_checker" if official else "unofficial"
    reason = ("Run completed; awaiting checker approval." if official
              else "Unofficial run — no approval required (terminal).")
    meta = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "run_type": run_type,
        "ecl_scenario": ecl_scenario,
        "transitions": [{"at": _now(), "by": by or _who(), "to": status,
                         "reason": reason}],
        "overrides_applied": overrides_applied or
        {"rating": 0, "stage": 0, "restructuring": 0},
        "snapshot_label": snapshot_label,
    }
    write_run_status(run_dir, meta)
    return meta


def manifest_path(run_dir) -> Path | None:
    """The run's manifest, wherever the engine that wrote it put one.

    The R engine writes ``reports/manifest.json``; earlier Python runs wrote it
    at the run root. Both are read, because the two engines are meant to share
    a runs/ folder and a reader that knows only its own convention silently
    sees nothing.
    """
    d = Path(run_dir)
    for candidate in (d / "reports" / "manifest.json", d / "manifest.json"):
        if candidate.is_file():
            return candidate
    return None


def maker_for_run(run_dir) -> str | None:
    """Who ran the pipeline, from the manifest. Used for separation of duties."""
    p = manifest_path(run_dir)
    if p is None:
        return None
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    # The R engine nests it under `run`; the Python writes it at the top.
    user = m.get("user") or (m.get("run") or {}).get("user")
    return str(user) if user else None


def approval_config(config_path=None) -> dict:
    """The project's ``approval:`` block, defaulting to not enforcing.

    Defaulting to off is deliberate: a control that appears by surprise, with
    no config, is one nobody agreed to. It is turned on explicitly.
    """
    p = Path(config_path) if config_path else Path.cwd() / "config.yml"
    if not p.is_file():
        return {"enforce_separation_of_duties": False}
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"enforce_separation_of_duties": False}
    block = cfg.get("approval") or {}
    return {"enforce_separation_of_duties":
            bool(block.get("enforce_separation_of_duties"))}


def transition_run(run_dir, target: str, by: str, reason: str,
                   config_path=None, audit=None) -> dict:
    """Move a run to ``approved`` or ``rejected``.

    Refuses rather than warns, in all four cases below, because each one is a
    control: a warning that can be scrolled past is not a gate.
    """
    if target not in ("approved", "rejected"):
        raise ValueError("a transition target must be 'approved' or 'rejected'")
    if not reason or not str(reason).strip():
        raise ValueError("a reason is required when approving or rejecting")
    by = str(by or "").strip()
    if not by:
        raise ValueError("a named approver is required")

    meta = read_run_status(run_dir)
    if meta is None:
        raise FileNotFoundError(
            f"no reports/run_status.yml for {run_dir}; the run cannot be "
            "approved because there is nothing recording that it is awaiting "
            "approval")

    current = normalise_status(meta.get("status"))
    if current != "pending_checker":
        raise ValueError(
            f"this run is {current!r}; only a run awaiting a checker can be "
            "approved or rejected")

    if target == "approved" and \
            approval_config(config_path)["enforce_separation_of_duties"]:
        maker = maker_for_run(run_dir)
        if maker and maker.strip().lower() == by.lower():
            raise PermissionError(
                f"separation of duties is enforced: {by!r} ran this pipeline "
                "and cannot also approve it. A different user must approve.")

    meta["status"] = target
    meta.setdefault("transitions", []).append(
        {"at": _now(), "by": by, "to": target, "reason": str(reason).strip()})
    write_run_status(run_dir, meta)

    if audit is not None:
        try:
            audit.record(f"run_{target}", reason,
                         run_id=meta.get("run_id"), user=by)
        except Exception:
            pass
    from .audit_log import audit_event
    audit_event({"event": f"run_{target}", "run_id": meta.get("run_id"),
                 "user": by, "reason": str(reason).strip()})
    return meta


def approve_run_status(run_dir, approver: str, reason: str, **kw) -> dict:
    return transition_run(run_dir, "approved", approver, reason, **kw)


def reject_run_status(run_dir, rejecter: str, reason: str, **kw) -> dict:
    """Rejection is never blocked by separation of duties.

    The person who built a run has to be able to withdraw it; making them find
    somebody else to do that achieves nothing.
    """
    return transition_run(run_dir, "rejected", rejecter, reason, **kw)


# ------------------------------------------------------------- listings ----
def _run_dirs(runs_dir) -> list[Path]:
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def _summarise(run_dir: Path) -> dict:
    meta = read_run_status(run_dir) or {}
    trans = meta.get("transitions") or []
    first = trans[0] if trans else {}
    last = trans[-1] if trans else {}
    ov = meta.get("overrides_applied") or {}
    return {
        "run_id": meta.get("run_id") or run_dir.name,
        "path": str(run_dir),
        "status": normalise_status(meta.get("status")),
        "run_type": meta.get("run_type") or "",
        "ecl_scenario": meta.get("ecl_scenario") or "",
        "requested_by": first.get("by") or "",
        "requested_at": first.get("at") or "",
        "request_comment": first.get("reason") or "",
        "decided_by": last.get("by") or "" if len(trans) > 1 else "",
        "decided_at": last.get("at") or "" if len(trans) > 1 else "",
        "decision_comment": last.get("reason") or "" if len(trans) > 1 else "",
        "n_overrides_total": sum(int(v or 0) for v in ov.values()
                                 if isinstance(v, (int, float))),
        "snapshot_label": meta.get("snapshot_label") or "",
    }


_COLUMNS = ["run_id", "path", "status", "run_type", "ecl_scenario",
            "requested_by", "requested_at", "request_comment", "decided_by",
            "decided_at", "decision_comment", "n_overrides_total",
            "snapshot_label"]


def _frame(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in _COLUMNS})
    return pd.DataFrame(rows, columns=_COLUMNS)


def list_runs_pending_approval(runs_dir) -> pd.DataFrame:
    """Runs awaiting a checker, plus any with no status file at all.

    A run with no status file is included as ``unknown`` rather than skipped.
    It is the case most worth seeing: something wrote a run and did not record
    that it needs approving, so filtering it out hides exactly the run nobody
    is looking at.
    """
    rows = []
    for d in _run_dirs(runs_dir):
        s = _summarise(d)
        if s["status"] in ("pending_checker", "unknown"):
            rows.append(s)
    return _frame(rows)


def list_runs_decided(runs_dir) -> pd.DataFrame:
    rows = [s for s in (_summarise(d) for d in _run_dirs(runs_dir))
            if s["status"] in ("approved", "rejected")]
    return _frame(rows)


def annotate_runs_with_status(runs, runs_dir=None) -> pd.DataFrame:
    """Add the approval columns to a frame of runs, without dropping any.

    A left join, deliberately: a run missing from the status listing still
    belongs on the screen.
    """
    df = pd.DataFrame(runs).copy()
    if len(df) == 0:
        return df
    key = "run_id" if "run_id" in df.columns else df.columns[0]
    extra = []
    for _, r in df.iterrows():
        d = Path(r["path"]) if "path" in df.columns and r.get("path") \
            else Path(runs_dir or ".") / str(r[key])
        extra.append(_summarise(d))
    add = pd.DataFrame(extra, columns=_COLUMNS).drop(columns=["path"])
    for c in add.columns:
        if c != "run_id":
            df[c] = add[c].to_numpy()
        elif "run_id" not in df.columns:
            df["run_id"] = add[c].to_numpy()
    return df
