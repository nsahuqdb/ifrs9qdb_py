"""Pre-flight: the static reference and the run config, before a run starts.

These do not belong in the run suite. They answer a question that comes
earlier -- is the configuration the run is about to use intact? -- and the
answer decides whether starting the run is worth anything at all.

They catch a corrupted CSV, a header-only file, a column renamed by a manual
edit, and a path in config.yml that points at nothing.

Ids match the R package exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._helpers import col, fail, ok
from .framework import Severity, Validator

__all__ = ["STATIC_VALIDATORS", "CONFIG_PATH_VALIDATORS", "PREFLIGHT_VALIDATORS"]

# Optional static files are exempt from the "must have rows" rule.
REQUIRED_STATIC = [
    "off_balance_products", "industry_sector_mapping",
    "collective_assessment_rules", "product_portfolio_mapping",
    "staging_thresholds", "master_rating_scale", "scenario_severity",
    "ttc_pd_table", "portfolios",
]

REQUIRED_PATHS = ("input_dir", "output_dir", "static_dir", "model_config",
                  "model_inputs")
OPTIONAL_PATHS = ("reference_outputs", "runs_dir", "snapshot_dir")


def _presence_validators() -> list[Validator]:
    out = []
    for key in REQUIRED_STATIC:
        def check(static=None, _k=key):
            if static is None:
                return ok()
            df = static.get(_k) if hasattr(static, "get") else None
            if df is None:
                return fail(0, f"{_k}.csv did not load")
            if len(df) == 0:
                return fail(0, f"{_k}.csv loaded with no data rows - a "
                               "header-only file usually means a copy or an "
                               "encoding step truncated it")
            return ok()

        out.append(Validator(
            id=f"STATIC_{key}_present", severity=Severity.ERROR,
            description=f"{key}.csv is present and non-empty", fn=check,
            context="static",
            rationale="The static reference is what the model is calibrated "
                      "against. A missing table does not stop the run; it "
                      "makes every lookup against it return nothing.",
            remediation=f"Restore {key}.csv from the config snapshot.",
            tags=("preflight",), suppressible=False))
    return out


def _v_rating_scale_has_defaults(static=None):
    if static is None:
        return ok()
    scale = static.get("master_rating_scale")
    if scale is None or len(scale) == 0:
        return ok()
    h = pd.to_numeric(col(scale, "hierarchy"), errors="coerce")
    if h is None:
        return fail(0, "master_rating_scale.csv has no hierarchy column")
    present = set(h.dropna().astype(int))
    missing = sorted(set(range(1, 22)) - present)
    if not missing:
        return ok()
    return fail(len(missing),
                f"master_rating_scale.csv is missing hierarchy bucket(s) "
                f"{missing}; every bucket needs a row or its curve has no "
                "rating to attach to",
                examples=[str(m) for m in missing[:10]])


def _v_ttc_in_unit_interval(static=None):
    if static is None:
        return ok()
    ttc = static.get("ttc_pd_table")
    if ttc is None or len(ttc) == 0:
        return ok()
    x = pd.to_numeric(col(ttc, "ttc_pd"), errors="coerce")
    if x is None:
        return fail(0, "ttc_pd_table.csv has no ttc_pd column")
    bad = x.isna() | (x < 0) | (x > 1)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} TTC PD value(s) are not probabilities in [0, 1]. Note "
                   "that ZERO is valid and must be kept: the engine "
                   "short-circuits it to a zero curve, but the rating still "
                   "needs a bucket",
                examples=[str(v) for v in x[bad].head(10)])


def _v_scenario_probs_sum(static=None):
    if static is None:
        return ok()
    scen = static.get("scenario_severity")
    if scen is None or len(scen) == 0:
        return ok()
    p = col(scen, "probability", "weight", "scenario_probability")
    if p is None:
        return ok()            # this file carries severities, not weights
    x = pd.to_numeric(p, errors="coerce").dropna()
    if x.empty:
        return ok()
    total = float(x.sum())
    if abs(total - 1.0) < 0.001:
        return ok()
    return fail(1, f"scenario probabilities sum to {total:.4f}, not 1.000")


def _v_required_paths_exist(run_config=None, base_dir=None):
    if not run_config:
        return ok()
    paths = (run_config.get("paths") or {}) if isinstance(run_config, dict) else {}
    base = Path(base_dir) if base_dir else Path.cwd()
    missing = []
    for key in REQUIRED_PATHS:
        raw = paths.get(key)
        if not raw:
            missing.append(f"{key} (not set)")
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = base / p
        if not p.exists():
            missing.append(f"{key} -> {raw}")
    if not missing:
        return ok()
    return fail(len(missing),
                f"{len(missing)} required path(s) in config.yml do not resolve",
                examples=missing[:10])


def _v_optional_paths_resolve(run_config=None, base_dir=None):
    if not run_config:
        return ok()
    paths = (run_config.get("paths") or {}) if isinstance(run_config, dict) else {}
    base = Path(base_dir) if base_dir else Path.cwd()
    missing = []
    for key in OPTIONAL_PATHS:
        raw = paths.get(key)
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = base / p
        if not p.exists():
            missing.append(f"{key} -> {raw}")
    if not missing:
        return ok()
    return fail(len(missing),
                f"{len(missing)} optional path(s) are set but do not resolve; "
                "a path that is set and wrong is worse than one left unset",
                examples=missing[:10])


def _v_run_block_complete(run_config=None):
    if not run_config:
        return ok()
    run = (run_config.get("run") or {}) if isinstance(run_config, dict) else {}
    required = ("extract_date",)
    missing = [k for k in required if not run.get(k)]
    if not missing:
        return ok()
    return fail(len(missing),
                "the run: block in config.yml is missing " + ", ".join(missing),
                examples=missing)


STATIC_VALIDATORS: list[Validator] = _presence_validators() + [
    Validator("STATIC_master_rating_scale_has_default", Severity.WARN,
              "master_rating_scale.csv contains every hierarchy bucket",
              _v_rating_scale_has_defaults, context="static",
              tags=("preflight",),
              rationale="The hierarchy is the PD bucket key. A missing bucket "
                        "has a curve and no rating that maps to it.",
              remediation="Restore the missing rows from the config snapshot."),
    Validator("STATIC_ttc_pd_in_unit_interval", Severity.ERROR,
              "All ttc_pd_table values are probabilities in [0, 1]",
              _v_ttc_in_unit_interval, context="static", tags=("preflight",),
              rationale="The probit transform is undefined outside [0, 1], so "
                        "a bad value does not raise - it produces NaN and "
                        "prices the bucket to zero.",
              remediation="Correct ttc_pd_table.csv and cut a new config "
                          "version.",
              suppressible=False),
    Validator("STATIC_scenario_probs_sum_to_one", Severity.WARN,
              "Scenario probabilities sum to ~1 (within 0.001)",
              _v_scenario_probs_sum, context="static", tags=("preflight",),
              rationale="Weights that do not sum to one rescale every PD in "
                        "the same direction, which looks like a calibration "
                        "change rather than an error.",
              remediation="Adjust the weights to sum to 1.000."),
]

CONFIG_PATH_VALIDATORS: list[Validator] = [
    Validator("CONFIG_required_paths_exist", Severity.ERROR,
              "Required paths in config.yml resolve to existing locations",
              _v_required_paths_exist, context="config", tags=("preflight",),
              rationale="A run that starts with a bad path fails partway "
                        "through, after writing some of its outputs.",
              remediation="Correct the paths block in config.yml.",
              suppressible=False),
    Validator("CONFIG_optional_paths_resolve", Severity.WARN,
              "Optional paths resolve to existing locations, if set",
              _v_optional_paths_resolve, context="config", tags=("preflight",),
              rationale="A path that is set and wrong is worse than one left "
                        "unset: the feature it enables silently does nothing.",
              remediation="Correct or remove the path."),
    Validator("CONFIG_run_block_complete", Severity.ERROR,
              "The run: block in config.yml is complete",
              _v_run_block_complete, context="config", tags=("preflight",),
              rationale="Without an extract date the run dates itself from "
                        "today, which silently re-ages every contract if a run "
                        "is repeated a week later.",
              remediation="Set run.extract_date in config.yml.",
              suppressible=False),
]

PREFLIGHT_VALIDATORS = STATIC_VALIDATORS + CONFIG_PATH_VALIDATORS
