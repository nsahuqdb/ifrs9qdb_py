"""Finding runs and reading what they recorded -- R's R/run_discovery.R.

A run is a folder under ``runs/`` holding ``Output/`` and ``reports/``. Both
engines write the same layout, and the R app and the Python app are meant to
share one ``runs/`` folder, so everything here reads either engine's runs:
the manifest in R's schema (``run``, ``snapshot``, ``run_metadata``,
``outputs``) and the earlier Python one (``run_id``, ``user``, ``created`` at
the top level) come back in the same shape.

Read-only. Nothing here writes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml

__all__ = ["runs_dir_default", "read_run_manifest", "manifest_summary",
           "read_run_validation", "read_run_reconciliation", "list_run_outputs",
           "read_run_overrides", "read_run_readiness", "read_input_source",
           "list_runs", "RUN_COLUMNS", "output_dir", "manifest_inputs",
           "augment_manifest_run_metadata", "archived_code_for"]


def archived_code_for(calc_id: str | None, root=None):
    """The archived code a run with this calculator version should execute,
    or None to run the live code -- R's make_calc_run_env() decision.

    None when no version is named (the active default), when the version has
    no archive, or when it is the active version and the live code already
    matches its registered fingerprint (the live code IS that version).
    """
    from .calculator_versions import (calc_version_code_dir,
                                      compute_code_fingerprint,
                                      get_calculator_version,
                                      read_calculator_versions)
    if not calc_id:
        return None
    d = calc_version_code_dir(calc_id, root)
    if d is None or not any(Path(d).rglob("*.py")):
        return None
    reg = read_calculator_versions(root)
    if reg.get("active") == calc_id:
        entry = get_calculator_version(calc_id, root) or {}
        if (entry.get("code_hash") or "") == (compute_code_fingerprint() or "x"):
            return None
    return Path(d)

RUN_COLUMNS = [
    "run_id", "path", "started_at", "finished_at", "duration_seconds", "user",
    "hostname", "code_sha", "snapshot_label", "snapshot_status", "n_outputs",
    "n_validation_failures", "has_reconciliation", "has_readiness",
    "run_type", "ecl_scenario", "run_purpose", "portfolio_date",
    "config_version", "calculator_version", "calculator_label",
    "calculator_matches_registered", "status", "approver", "approved_at",
    "engine", "has_report",
]


def runs_dir_default(runs_dir=None) -> Path:
    if runs_dir:
        return Path(runs_dir)
    return Path(os.environ.get("IFRS9_RUNS_DIR", "runs"))


def output_dir(run_path) -> Path | None:
    """The run's Output folder, whatever it is called (or the run root when
    the CSVs sit there directly)."""
    run = Path(run_path)
    for name in ("Output", "output", "OUTPUT"):
        if (run / name).is_dir():
            return run / name
    if (run / "FinalEclReport.csv").is_file():
        return run
    return None


def _manifest_path(run_path) -> Path | None:
    run = Path(run_path)
    for p in (run / "reports" / "manifest.json", run / "manifest.json"):
        if p.is_file():
            return p
    return None


def read_run_manifest(run_path) -> dict | None:
    """The manifest as written, or None."""
    p = _manifest_path(run_path)
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _na(v):
    """R writes missing values as the string "NA"; read them as None."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip() in ("NA", "", "null", "None", "nan"):
        return None
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, list) and len(v) == 1:
        return _na(v[0])
    return v


def manifest_summary(m: dict | None, run_path=None) -> dict:
    """One flat record from either manifest schema."""
    m = m or {}
    run = m.get("run") or {}
    meta = m.get("run_metadata") or {}
    snap = m.get("snapshot") or {}
    calc = m.get("calculator") or {}
    started = _na(run.get("started_at"))
    finished = _na(run.get("finished_at")) or _na(m.get("created"))
    outputs = m.get("outputs")
    if isinstance(outputs, dict):
        n_outputs = len(outputs)
    else:
        od = output_dir(run_path) if run_path else None
        n_outputs = len(list(od.glob("*.csv"))) if od is not None else 0
    rv = _na(run.get("r_version"))
    engine = "R" if rv else ("Python" if (run.get("python_version")
                                          or m.get("engine_version")) else None)
    dur = _na(run.get("duration_seconds"))
    try:
        dur = float(dur) if dur is not None else None
    except (TypeError, ValueError):
        dur = None
    match = _na(meta.get("calculator_matches_registered"))
    if match is None:
        match = _na(calc.get("matches_registered"))
    if isinstance(match, str):
        match = {"TRUE": True, "FALSE": False, "true": True,
                 "false": False}.get(match)
    return {
        "run_id": _na(run.get("run_id")) or _na(m.get("run_id"))
        or (Path(run_path).name if run_path else None),
        "started_at": started or finished,
        "finished_at": finished,
        "duration_seconds": dur,
        "user": _na(run.get("user")) or _na(m.get("user")),
        "hostname": _na(run.get("hostname")),
        "code_sha": _na(run.get("code_sha"))
        or _na((m.get("code") or {}).get("sha")),
        "snapshot_label": _na(snap.get("label")) if snap else None,
        "snapshot_status": _na(snap.get("status")) if snap else None,
        "snapshot_code_sha": _na(snap.get("code_sha_at_creation")) if snap else None,
        "n_outputs": int(n_outputs),
        "run_type": _na(meta.get("run_type")),
        "ecl_scenario": _na(meta.get("ecl_scenario")) or "weighted",
        "run_purpose": _na(meta.get("run_purpose")),
        "portfolio_date": _na(meta.get("portfolio_date")),
        "config_version": _na(meta.get("config_version"))
        or (_na(snap.get("label")) if snap else None),
        "calculator_version": _na(meta.get("calculator_version"))
        or _na(calc.get("id")),
        "calculator_label": _na(meta.get("calculator_label"))
        or _na(calc.get("label")),
        "calculator_code_hash": _na(meta.get("calculator_code_hash"))
        or _na(calc.get("code_hash")),
        "calculator_matches_registered": match,
        "input_dir": _na(m.get("input_dir")),
        "reporting_date": _na(m.get("reporting_date")),
        "engine": engine,
        "engine_detail": rv or _na(run.get("engine"))
        or (f"ifrs9qdb {m.get('engine_version')}" if m.get("engine_version")
            else None),
    }


def manifest_inputs(m: dict | None) -> pd.DataFrame:
    """The manifest's inputs block as a table (R: one entry per input)."""
    inputs = (m or {}).get("inputs")
    if not inputs:
        return pd.DataFrame(columns=["input", "file", "file_found", "exists",
                                     "size_bytes", "sha256"])
    rows = []
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            v = v or {}
            rows.append({"input": k, "file": _na(v.get("file")),
                         "file_found": _na(v.get("file_found")),
                         "exists": _na(v.get("exists")),
                         "size_bytes": _na(v.get("size_bytes")),
                         "sha256": (_na(v.get("sha256")) or "")[:16]})
    elif isinstance(inputs, list):
        for v in inputs:
            rows.append({k: _na(x) for k, x in (v or {}).items()})
    return pd.DataFrame(rows)


def read_run_validation(run_path) -> pd.DataFrame | None:
    for p in (Path(run_path) / "reports" / "validation.csv",
              Path(run_path) / "validation.csv"):
        if p.is_file():
            try:
                return pd.read_csv(p, dtype=str, keep_default_na=False)
            except Exception:
                return None
    return None


def read_run_reconciliation(run_path) -> dict:
    rep = Path(run_path) / "reports"
    md = rep / "reconciliation.md"
    miss = rep / "mismatches"
    out = {"md_path": str(md) if md.is_file() else None,
           "markdown": md.read_text(encoding="utf-8") if md.is_file() else None,
           "mismatches_dir": str(miss) if miss.is_dir() else None,
           "mismatches": []}
    if miss.is_dir():
        out["mismatches"] = [{"file": p.name, "path": str(p),
                              "size_bytes": p.stat().st_size}
                             for p in sorted(miss.glob("*.csv"))]
    return out


def list_run_outputs(run_path) -> list[Path]:
    od = output_dir(run_path)
    if od is None:
        return []
    return sorted(od.glob("*.csv"))


def read_run_overrides(run_path) -> dict[str, pd.DataFrame]:
    """Each overrides/*.csv of the run, by file name."""
    d = Path(run_path) / "overrides"
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.csv")):
        try:
            out[p.name] = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception:
            out[p.name] = None
    return out


def read_run_readiness(run_path) -> dict | None:
    """reports/readiness.csv, readiness_funnel.csv and readiness.md, or None."""
    rep = Path(run_path) / "reports"
    rc = rep / "readiness.csv"
    if not rc.is_file():
        return None
    out = {"contracts": None, "funnel": None, "markdown": None}
    try:
        out["contracts"] = pd.read_csv(rc, dtype=str, keep_default_na=False)
    except Exception:
        pass
    fc = rep / "readiness_funnel.csv"
    if fc.is_file():
        try:
            out["funnel"] = pd.read_csv(fc, dtype=str, keep_default_na=False)
        except Exception:
            pass
    md = rep / "readiness.md"
    if md.is_file():
        out["markdown"] = md.read_text(encoding="utf-8")
    return out


def read_input_source(run_path) -> dict | None:
    p = Path(run_path) / "reports" / "input_source.yml"
    if not p.is_file():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or None
    except Exception:
        return None


def augment_manifest_run_metadata(run_path, run_metadata: dict) -> bool:
    """R's augment_manifest_run_metadata(): stamp the operator's run metadata
    -- type, purpose, portfolio date, config and calculator version -- into an
    existing manifest.

    The app knows this whichever code produced the run, so it records it even
    when an archived (older) calculator wrote a manifest without it.
    """
    p = _manifest_path(run_path)
    if p is None or not run_metadata:
        return False
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    keep = ("run_type", "ecl_scenario", "run_purpose", "portfolio_date",
            "config_version", "calculator_version", "calculator_label",
            "calculator_code_hash", "calculator_matches_registered")
    block = dict(m.get("run_metadata") or {})
    block.update({k: run_metadata.get(k) for k in keep if k in run_metadata})
    m["run_metadata"] = block
    p.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return True


def list_runs(runs_dir=None) -> pd.DataFrame:
    """Every run with a manifest, newest first -- R's list_runs(), plus the
    approval status, the approver and when it was decided."""
    from .run_status import normalise_status, read_run_status
    root = runs_dir_default(runs_dir)
    if not root.is_dir():
        return pd.DataFrame(columns=RUN_COLUMNS)
    rows = []
    for s in sorted(p for p in root.iterdir() if p.is_dir()
                    and not p.name.startswith(".")):
        m = read_run_manifest(s)
        od = output_dir(s)
        if m is None and od is None:
            continue
        summ = manifest_summary(m, s)
        v = read_run_validation(s)
        n_fail = 0
        if v is not None and len(v) and "passed" in v.columns:
            n_fail = int((v["passed"].astype(str).str.upper() != "TRUE").sum())
        rs = read_run_status(s)
        status, approver, approved_at = None, None, None
        if rs:
            status = normalise_status(rs.get("status"))
            for t in rs.get("transitions") or []:
                if normalise_status(t.get("to")) == status:
                    approver, approved_at = t.get("by"), t.get("at")
        if summ["run_type"] is None and rs:
            summ["run_type"] = rs.get("run_type")
        rows.append({
            **{k: summ.get(k) for k in RUN_COLUMNS if k in summ},
            # The folder is the run's address; a manifest copied in from
            # another deployment can carry a different id.
            "run_id": s.name,
            "path": str(s),
            "n_validation_failures": n_fail,
            "has_reconciliation": (s / "reports" / "reconciliation.md").is_file(),
            "has_readiness": (s / "reports" / "readiness.csv").is_file(),
            "status": status or "unknown",
            "approver": approver,
            "approved_at": approved_at,
            "has_report": od is not None and (od / "FinalEclReport.csv").is_file(),
        })
    if not rows:
        return pd.DataFrame(columns=RUN_COLUMNS)
    df = pd.DataFrame(rows)
    for c in RUN_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[RUN_COLUMNS]
    key = df["started_at"].fillna("").astype(str)
    return df.iloc[key.argsort()[::-1]].reset_index(drop=True)
