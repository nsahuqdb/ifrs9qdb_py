"""Before a run: are the inputs fit to run, and will every contract be priced?

Two checks the app runs before it lets anybody start a run -- R's
pre_run_check() and pre_run_readiness(), with the same inputs, the same
validators and the same audit events.

``pre_run_check`` is quick (seconds): the configuration and static checks and
every INPUT-stage validator -- the file checks, the field checks, the
cross-file checks (an allocation pointing at a collateral record that does not
exist is one) and the config-coverage checks -- against the static reference
of the config version the run will use, with its suppressions applied.

``pre_run_readiness`` is the slow one (about as long as phase 1 plus the
curves): it builds the LIC input files into a temporary folder, assesses
readiness on exactly what LIC would read, and stops before pricing. It answers
"will every row get a number?" -- which contracts would come out with no ECL,
which LIC would leave blank, which are priced from incomplete inputs -- before
a run is committed to. Nothing is added to runs/, and the audit log records one
pre_run_readiness event rather than a run.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from .audit_log import audit_event, audit_suspended

__all__ = ["pre_run_check", "pre_run_readiness", "apply_input_extract_date",
           "CONFIG_FILE_FOR",
           "where_to_fix"]

# Which static file a CONFIG_ coverage finding is fixed in -- the `config`
# detail R's validators carry, which the app turns into "Fix in CONFIG: ...".
CONFIG_FILE_FOR = {
    "CONFIG_product_portfolio_coverage": "product_portfolio_mapping.csv",
    "CONFIG_off_balance_products_coverage": "off_balance_products.csv",
    "CONFIG_internal_rating_coverage": "master_rating_scale.csv",
    "CONFIG_portfolio_referential": "portfolios.csv",
    "CONFIG_collateral_type_coverage": "collateral_types.csv",
    "CONFIG_industry_sector_coverage": "industry_sector_mapping.csv",
}


def where_to_fix(check_id: str, context: str = "") -> str:
    """R's one-line "where to look" for a finding."""
    cfg = CONFIG_FILE_FOR.get(str(check_id))
    if cfg:
        return (f"Fix in CONFIG: edit {cfg} in the Config manager (draft "
                "version), then approve and re-run")
    if context:
        return f"Look in INPUT file: {context}"
    return ""


def _extract_date(src) -> str | None:
    """The reporting date the inputs carry, as R's
    resolve_input_extract_date() takes it."""
    from .validation._helpers import resolve_input_extract_date
    from .validation.schema import canonicalise
    d = resolve_input_extract_date(canonicalise(src))
    return None if d is None else d.strftime("%Y-%m-%d")


def apply_input_extract_date(run_config, extract_date) -> dict | None:
    """R's apply_input_extract_date(): the inputs' date overlays config.yml's
    run.extract_date (input wins; the config value is only the fallback when
    the inputs carry no date). Returns a copy; the caller's dict is untouched."""
    import copy
    if not isinstance(run_config, dict):
        return run_config
    if not extract_date:
        return run_config
    rc = copy.deepcopy(run_config)
    if not isinstance(rc.get("run"), dict):
        rc["run"] = {}
    rc["run"]["extract_date"] = str(extract_date)
    return rc


def pre_run_check(input_dir, static_dir=None, config_dir=None,
                  run_config: dict | None = None, base_dir=None,
                  suppressions_path=None, reporting_date=None,
                  record: bool = True, include_preflight: bool = True) -> dict:
    """R's pre_run_check(): config + static + INPUT validators, suppressed.

    Returns ``results`` (validation.csv's columns plus ``where``),
    ``extract_date`` (what the inputs carry -- the app pre-fills the portfolio
    date with it), ``strip_log`` (junk rows removed per file) and a count
    summary. ``record`` writes the pre_run_check audit event.
    ``include_preflight=False`` runs the INPUT validators alone -- the R app's
    data-quality preview on "Validate inputs".
    """
    from .etl.read_inputs import read_all_inputs
    from .etl.static_ref import load_static_reference
    from .validation import (PREFLIGHT_VALIDATORS, combine, load_suppressions,
                             run_suite, suppression_reasons, validate_stages,
                             validation_frame)
    from .validation.schema import canonicalise
    src = read_all_inputs(input_dir)
    static = load_static_reference(static_dir)
    sp = Path(suppressions_path) if suppressions_path else (
        Path(config_dir) / "validation_suppressions.yml" if config_dir else None)
    supp = suppression_reasons(load_suppressions(sp)) if sp else {}
    ext = _extract_date(src)
    rd = reporting_date or ext
    rep_date = None
    if rd:
        t = pd.to_datetime(rd)
        rep_date = f"{t.month}/{t.day}/{t.year}"
    inp = validate_stages(inputs=src, static=static, reporting_date=rep_date,
                          suppressions=supp, stages=["INPUT"])
    if include_preflight:
        # As R's pre_run_check(): the config checks see the run config the
        # run would use, i.e. with the inputs' reporting date applied.
        pre = run_suite("PREFLIGHT", PREFLIGHT_VALIDATORS,
                        {"static": static,
                         "run_config": apply_input_extract_date(run_config, ext),
                         "base_dir": base_dir}, supp)
        res = combine(pre, inp)
    else:
        res = inp
    frame = validation_frame(res)
    frame["where"] = [where_to_fix(i, c) for i, c in
                      zip(frame["id"], frame["context"])]
    passed = frame["passed"].astype(bool)
    active = ~passed & ~frame["suppressed"].astype(bool)
    summary = {"checks": int(len(frame)), "passed": int(passed.sum()),
               "errors": int((active & (frame["effective_severity"] == "ERROR")).sum()),
               "warnings": int((active & (frame["effective_severity"] == "WARN")).sum()),
               "info": int((active & (frame["effective_severity"] == "INFO")).sum()),
               "suppressed": int(frame["suppressed"].astype(bool).sum())}
    if record:
        audit_event({"event": "pre_run_check", "n_total": summary["checks"],
                     "n_pass": summary["passed"],
                     "n_fail": summary["checks"] - summary["passed"]})
    return {"results": frame, "summary": summary, "extract_date": ext,
            "strip_log": dict(canonicalise(src).strip_log),
            "missing": list(src.missing)}


def pre_run_readiness(input_dir, static_dir=None, config_dir=None,
                      reporting_date=None, work_dir=None, keep: bool = False,
                      progress=None) -> dict:
    """R's pre_run_readiness(): build the LIC files, assess, stop before pricing.

    Every gate records rather than stops (policy ``warn``), so the answer
    covers the whole book. Returns ``summary`` (contracts and exposure by
    predicted outcome), ``contracts`` (one row per contract, with its outcome
    and the reasons), ``funnel`` (rows in, rows out, per file), ``validation``
    (every check up to READY), ``steps`` and ``error``.
    """
    from .etl.pipeline import run_etl
    from .runs import read_run_readiness, read_run_validation
    work = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="pre_run_readiness_"))
    try:
        with audit_suspended():
            r = run_etl(input_dir, work, reporting_date=reporting_date,
                        run_id="pre_run_readiness", static_dir=static_dir,
                        config_dir=config_dir, progress=progress,
                        on_validation_error="warn", stop_before_pricing=True)
        run = work / "pre_run_readiness"
        rd = read_run_readiness(run) or {}
        v = read_run_validation(run)
        s = r.readiness or {"contracts": 0}
        audit_event({"event": "pre_run_readiness",
                     "contracts": s.get("contracts", 0),
                     "no_ecl": (s.get("No ECL") or {}).get("contracts", 0),
                     "blank_in_lic": (s.get("Blank in LIC") or {}).get("contracts", 0),
                     "with_gaps": (s.get("Priced - check") or {}).get("contracts", 0)})
        return {"ok": bool(r.ok), "error": r.error, "summary": s,
                "contracts": rd.get("contracts"), "funnel": rd.get("funnel"),
                "markdown": rd.get("markdown"), "validation": v,
                "steps": r.steps, "work_dir": str(work)}
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
