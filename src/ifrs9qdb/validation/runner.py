"""
Validating a completed run.

Reads what the run produced and applies every check that can see it, then
reports the whole picture rather than stopping at the first failure -- a
half-finished validation is worse than none, because it looks like a clean bill.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .checks import (CROSS_FILE_VALIDATORS, DERIVED_VALIDATORS,
                     INPUT_VALIDATORS, TRANSFORM_VALIDATORS)
from .framework import ValidationResult, combine, run_suite

__all__ = ["validate_run", "read_suppressions"]


def read_suppressions(path) -> dict:
    """Accepted findings, by id, each with the reason it was accepted.

    A suppression without a reason is not recorded: the point is an auditable
    exception, not a quieter screen.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        import yaml
        data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}
    items = data.get("suppressions", data) if isinstance(data, dict) else {}
    out = {}
    for k, v in (items or {}).items():
        reason = v.get("reason") if isinstance(v, dict) else str(v)
        if reason:
            out[str(k)] = str(reason)
    return out


def validate_run(run_dir, inputs=None, suppressions=None) -> ValidationResult:
    """Run every applicable suite over a completed run."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "Output" if (run_dir / "Output").is_dir() else run_dir
    supp = suppressions if suppressions is not None else read_suppressions(
        run_dir / "suppressions.yml")

    results = []
    if inputs is not None:
        results.append(run_suite("Input", INPUT_VALIDATORS,
                                 {"inputs": inputs}, supp))

    results.append(run_suite("Transform", TRANSFORM_VALIDATORS,
                             {"out_dir": out_dir}, supp))

    report_path = out_dir / "FinalEclReport.csv"
    if report_path.is_file():
        from ..analytics import normalise
        rep = normalise(pd.read_csv(report_path, low_memory=False))
        results.append(run_suite("Derived", DERIVED_VALIDATORS,
                                 {"report": rep}, supp))

    results.append(run_suite("Cross-file", CROSS_FILE_VALIDATORS,
                             {"out_dir": out_dir}, supp))

    combined = combine(*results)
    try:
        (run_dir / "validation.json").write_text(json.dumps(
            {"summary": combined.summary(),
             "issues": [i.as_dict() for i in combined.issues]}, indent=2))
    except OSError:
        pass
    return combined
