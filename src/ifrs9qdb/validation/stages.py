"""Running the staged suites, and writing the report a person reads.

The suites are grouped by what they can see, which is also when they can run:

    INPUT      the raw extracts, plus cross-file agreement and config coverage
    TRANSFORM  the book after it has been shaped, before pricing
    DERIVED    the curves and weights the engine prices against
    READY      the written files LIC reads, before pricing: will every
               contract get an ECL, and from complete inputs (readiness.py)
    REPORT     after pricing: every contract has a row, every blank explained

Everything runs, including the checks that pass. A report showing only
failures cannot be read as evidence that anything was checked, which is what a
reviewer actually needs at close.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd

from .checks_cross_file import (CONFIG_COVERAGE_VALIDATORS,
                                CROSS_FILE_STAGE_VALIDATORS)
from .checks_derived import DERIVED_STAGE_VALIDATORS
from .checks_input import INPUT_STAGE_VALIDATORS
from .checks_transform import TRANSFORM_STAGE_VALIDATORS
from .framework import Severity, ValidationResult, combine, run_suite

__all__ = ["INPUT_STAGE", "TRANSFORM_STAGE", "DERIVED_STAGE", "STAGE_SUITES",
           "validate_stages", "validation_frame", "write_validation_reports",
           "effective_severity"]

# The INPUT stage carries the cross-file and config-coverage checks too: all
# three see only the raw extracts and the static reference, and grouping them
# together is what the R report does.
INPUT_STAGE = (INPUT_STAGE_VALIDATORS + CROSS_FILE_STAGE_VALIDATORS
               + CONFIG_COVERAGE_VALIDATORS)
TRANSFORM_STAGE = TRANSFORM_STAGE_VALIDATORS
DERIVED_STAGE = DERIVED_STAGE_VALIDATORS

STAGE_SUITES = {
    "INPUT": INPUT_STAGE,
    "TRANSFORM": TRANSFORM_STAGE,
    "DERIVED": DERIVED_STAGE,
}

_ORDER = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}


def effective_severity(issue) -> str:
    """What the finding counts as for gating.

    A suppressed failure is recorded at its real severity and gates at INFO,
    so the run can proceed without the finding disappearing from the report.
    """
    if issue.suppressed:
        return Severity.INFO
    return issue.severity


def validate_stages(*, inputs=None, static=None, trans_lending=None,
                    lending_view=None, trans_investments=None,
                    investment_view=None, ltpo=None, stpd=None,
                    internal_weights=None, external_weights=None,
                    mev_weights=None, reporting_date=None,
                    header_strip_log=None, suppressions=None,
                    stages=None) -> ValidationResult:
    """Run the staged suites over whatever of the run is available.

    Every argument is optional. A suite whose inputs are absent still runs and
    its checks pass trivially, which keeps the report's shape stable across
    runs -- a check that vanishes when its data is missing looks the same as a
    check that was never written.
    """
    # Every check reads the canonical columns R's checks read, with SQL*Plus
    # junk rows already stripped -- see schema.py for why this matters.
    if inputs is not None:
        from .schema import canonicalise
        inputs = canonicalise(inputs)
        if header_strip_log is None:
            header_strip_log = inputs.strip_log
    # The TRANSFORM checks read R's intermediates -- trans_l, cm_view,
    # trans_i, inv_view -- so those are built in R's shape from the frames
    # the ETL produced (r_frames.py).
    trans_l = cm_view = trans_i = inv_view = None
    if inputs is not None and trans_lending is not None:
        from .r_frames import lending_frames
        trans_l, cm_view = lending_frames(trans_lending, inputs, static)
    if inputs is not None and trans_investments is not None:
        from .r_frames import investment_frames
        trans_i, inv_view = investment_frames(trans_investments, inputs, static)
    args = {
        "trans_l": trans_l, "cm_view": cm_view,
        "trans_i": trans_i, "inv_view": inv_view,
        "inputs": inputs, "static": static,
        "trans_lending": trans_lending, "lending_view": lending_view,
        "trans_investments": trans_investments,
        "investment_view": investment_view,
        "ltpo": ltpo, "stpd": stpd,
        "internal_weights": internal_weights,
        "external_weights": external_weights, "mev_weights": mev_weights,
        "reporting_date": reporting_date,
        "header_strip_log": header_strip_log,
    }
    supp = suppressions or {}
    want = stages or list(STAGE_SUITES)
    return combine(*[run_suite(name, STAGE_SUITES[name], args, supp)
                     for name in want if name in STAGE_SUITES])


def validation_frame(result: ValidationResult) -> pd.DataFrame:
    """The findings as the CSV the R engine writes, column for column."""
    rows = []
    for issue in result.issues:
        stage = issue.id.split("_", 1)[0]
        stage = {"INPUT": "INPUT", "XFILE": "INPUT", "CONFIG": "INPUT",
                 "TRANS": "TRANSFORM", "DERIVED": "DERIVED",
                 "STATIC": "INPUT", "READY": "READY",
                 "REPORT": "REPORT"}.get(stage, stage)
        rows.append({
            "stage": stage,
            "id": issue.id,
            "severity": issue.severity,
            "effective_severity": effective_severity(issue),
            "context": issue.context,
            "description": issue.description,
            "rationale": issue.rationale,
            "remediation": issue.remediation,
            "suppressed": issue.suppressed,
            "passed": issue.passed,
            "message": issue.detail,
        })
    return pd.DataFrame(rows, columns=[
        "stage", "id", "severity", "effective_severity", "context",
        "description", "rationale", "remediation", "suppressed", "passed",
        "message"])


def _markdown(result: ValidationResult, frame: pd.DataFrame) -> str:
    failed = [i for i in result.issues if not i.passed]
    counts = {s: sum(1 for i in failed if effective_severity(i) == s)
              for s in (Severity.ERROR, Severity.WARN, Severity.INFO)}
    n_supp = sum(1 for i in result.issues if i.suppressed)
    lines = [
        "# Validation Report",
        f"Generated: {_dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}",
        f"Total checks: {len(result.issues)}",
        f"Passed: {len(result.issues) - len(failed)}  |  "
        f"ERROR: {counts[Severity.ERROR]}  |  WARN: {counts[Severity.WARN]}  |  "
        f"INFO: {counts[Severity.INFO]}  |  Suppressed: {n_supp}",
    ]
    for stage in ("INPUT", "TRANSFORM", "DERIVED", "READY", "REPORT"):
        sub = frame[frame["stage"] == stage]
        if sub.empty:
            continue
        lines.append(f"\n## {stage} ({len(sub)} checks)\n")
        bad = sub[~sub["passed"]].copy()
        if bad.empty:
            lines.append("- Everything checked, nothing to report.")
            continue
        bad["_o"] = bad["effective_severity"].map(_ORDER).fillna(9)
        for _, r in bad.sort_values("_o").iterrows():
            flag = " (suppressed)" if r["suppressed"] else ""
            lines.append(f"- **[{r['severity']}]**{flag} `{r['id']}` "
                         f"({r['context']}) {r['description']}")
            if r["message"]:
                lines.append(f"  - {r['message']}")
            if r["rationale"]:
                lines.append(f"  - _Rationale:_ {r['rationale']}")
            if r["remediation"]:
                lines.append(f"  - _Remediation:_ {r['remediation']}")
    return "\n".join(lines) + "\n"


def write_validation_reports(result: ValidationResult, reports_dir) -> dict:
    """Write validation.csv and validation.md, as the R engine does.

    Both, because they answer different questions: the CSV is what a reviewer
    filters and the Markdown is what they read.
    """
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    frame = validation_frame(result)
    csv_path, md_path = d / "validation.csv", d / "validation.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_markdown(result, frame), encoding="utf-8")
    return {"csv": csv_path, "md": md_path, "summary": result.summary()}
