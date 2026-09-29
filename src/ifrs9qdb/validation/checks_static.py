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

# R's build_config_validators(): only these must exist before a run. The
# runtime locations (input_dir, output_dir, runs_dir) may legitimately not
# exist yet -- inputs arrive by upload or a data-drop folder, runs/ is created
# on the first run -- so their absence is not an error.
REQUIRED_PATHS = ("static_dir", "variable_dictionary", "models", "model_inputs")
OPTIONAL_PATHS = ("input_dir", "reference_outputs", "data_drop_root")
# config.yml's run: block must name the model and carry the reporting date
# (the inputs' EXTRACTDA fills it before the check, as in R).
REQUIRED_RUN_KEYS = ("internal_model", "extract_date")


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


def _resolve_path(raw, base_dir) -> Path:
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (Path(base_dir) if base_dir else Path.cwd()) / p
    return p


def _v_required_paths_exist(run_config=None, base_dir=None):
    """R: static_dir and the model / dictionary / inputs YAMLs must exist.
    Relative paths resolve against config.yml's own folder, as R's
    load_run_config() resolves them."""
    if not run_config:
        return ok()
    paths = run_config.get("paths") if isinstance(run_config, dict) else None
    if not isinstance(paths, dict):
        return fail(1, "config.yml has no `paths:` block")
    missing = []
    for key in REQUIRED_PATHS:
        raw = paths.get(key)
        if raw is None or not isinstance(raw, str) or not raw:
            missing.append(f"paths${key} = (unset)")
            continue
        if not _resolve_path(raw, base_dir).exists():
            missing.append(f"paths${key} = '{raw}'")
    if not missing:
        return ok()
    return fail(len(missing), "Missing/unresolvable required paths: "
                + "; ".join(missing), examples=missing)


def _v_optional_paths_resolve(run_config=None, base_dir=None):
    if not run_config:
        return ok()
    paths = run_config.get("paths") if isinstance(run_config, dict) else None
    if not isinstance(paths, dict):
        return ok()
    missing = []
    for key in OPTIONAL_PATHS:
        raw = paths.get(key)
        if raw is None or not isinstance(raw, str) or not raw:
            continue
        if not _resolve_path(raw, base_dir).exists():
            missing.append(f"paths${key} = '{raw}'")
    if not missing:
        return ok()
    return fail(len(missing), "Optional paths set but not found: "
                + "; ".join(missing), examples=missing)


def _v_run_block_complete(run_config=None):
    if not run_config:
        return ok()
    run = run_config.get("run") if isinstance(run_config, dict) else None
    if not isinstance(run, dict):
        return fail(1, "config.yml has no `run:` block")
    missing = [k for k in REQUIRED_RUN_KEYS
               if run.get(k) is None or str(run.get(k)) == ""]
    if not missing:
        return ok()
    return fail(len(missing), "Missing run.* keys: " + ", ".join(missing),
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
              "Required paths in config.yml's `paths:` block resolve to "
              "existing files / directories",
              _v_required_paths_exist, context="config", tags=("preflight",),
              rationale="Some path entries describe runtime locations "
                        "(input_dir, output_dir, runs_dir) that may "
                        "legitimately not exist yet - e.g. when inputs are "
                        "uploaded via the app, or the runs/ folder is created "
                        "on first run. But static_dir + the model/dictionary "
                        "YAMLs MUST exist; their absence indicates a broken "
                        "project.",
              remediation="Ensure static_dir and the model/dictionary YAML "
                          "paths in config.yml exist on disk.",
              suppressible=False),
    Validator("CONFIG_optional_paths_resolve", Severity.WARN,
              "Optional paths (input_dir, reference_outputs) resolve to "
              "existing locations, if set",
              _v_optional_paths_resolve, context="config", tags=("preflight",),
              rationale="input_dir is used for the configured-source flow; if "
                        "it doesn't exist, you must use the upload or "
                        "data-drop flow. reference_outputs enables "
                        "reconciliation against the Excel tool output; if set "
                        "but missing, no reconciliation will run.",
              remediation="Either point at the right location, or leave unset "
                          "if you don't need it."),
    Validator("CONFIG_run_block_complete", Severity.ERROR,
              "config.yml's `run:` block has the keys phase1 expects "
              "(internal_model, extract_date)",
              _v_run_block_complete, context="config", tags=("preflight",),
              rationale="The pipeline picks the model from run$internal_model "
                        "and uses run$extract_date to time-align the input "
                        "snapshot. Missing either produces a confusing failure "
                        "during model resolution. The inputs' EXTRACTDA fills "
                        "extract_date before this check, so it fails on the "
                        "date only when neither the inputs nor config.yml "
                        "carry one.",
              remediation="Add internal_model and extract_date under config.yml "
                          "run:",
              suppressible=False),
]

PREFLIGHT_VALIDATORS = STATIC_VALIDATORS + CONFIG_PATH_VALIDATORS
