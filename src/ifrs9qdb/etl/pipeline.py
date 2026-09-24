"""
The ETL run.

Reads the source extracts, builds the LIC input files, and writes a run folder.
Each step reports what it produced so a partial run is legible rather than
mysterious: the port is not finished, and a run that quietly wrote twelve of
eighteen files without saying so would be worse than one that stops.

A run never overwrites an existing folder. Reproducing a quarter means being
able to go back to exactly what was produced, and an in-place rewrite destroys
that.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .customer import derive_customer_flags, investment_customer_ids
from .lending import (build_account_master, transform_investments,
                      transform_lending)
from .lifetime import build_lifetime_parameter_other
from .read_inputs import read_all_inputs
from .macro import build_stpd_from_static, compute_internal_scenario_weights
from .reference import build_reference_files
from .static_ref import load_static_reference
from .transform import (pick, transform_allocation, transform_collateral,
                        transform_customer_master, transform_origination,
                        transform_origination_investments,
                        transform_staging_flags, write_outputs)

__all__ = ["RunResult", "run_etl", "next_run_id", "PRODUCED", "PENDING"]

# What the port produces today, and what it does not. Stated here rather than
# discovered at the end of a run.
PRODUCED = [
    "Collateral.csv", "AccountCollateralAllocation.csv",
    "CustomerMaster_1.csv", "CustomerMaster_2.csv",
    "CustomerStagingFlag_1.csv", "CustomerStagingFlag_2.csv",
    "AccountMaster_1.csv", "AccountMaster_2.csv",
    "Origination_1.csv", "Origination_2.csv",
    "LifeTimeParameterOther.csv", "StPD.csv",
]
REFERENCE = ["Portfolios.csv", "Ratings.csv", "RatingTypes.csv",
             "PortfolioRatingType.csv", "CollateralType.csv", "FxRate.csv"]
PENDING: list[str] = []


@dataclass
class RunResult:
    ok: bool
    run_id: str
    run_dir: Path
    output_dir: Path
    steps: list[dict] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None
    traceback: str = ""
    validation: dict | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "run_id": self.run_id,
            "run_dir": str(self.run_dir), "output_dir": str(self.output_dir),
            "steps": self.steps, "written": self.written,
            "pending": self.pending, "seconds": round(self.seconds, 1),
            "error": self.error, "traceback": self.traceback,
            "validation": self.validation,
        }


def next_run_id(runs_dir) -> str:
    """The next sequential id. Sequential, not a timestamp, because people
    refer to runs by number in review meetings."""
    runs_dir = Path(runs_dir)
    n = 0
    if runs_dir.is_dir():
        for p in runs_dir.iterdir():
            if p.is_dir() and p.name.startswith("run_"):
                try:
                    n = max(n, int(p.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
    return f"run_{n + 1:05d}"


def run_etl(input_dir, runs_dir, reporting_date=None, run_id=None,
            static_dir=None, config_dir=None, progress=None) -> RunResult:
    """Build a run from the source extracts.

    ``progress`` is called with (step, message) so a caller can show what is
    happening -- reading 117,000 collateral allocations is not instant, and a
    silent minute looks like a hang.
    """
    started = time.time()
    input_dir = Path(input_dir)
    runs_dir = Path(runs_dir)
    run_id = run_id or next_run_id(runs_dir)
    run_dir = runs_dir / run_id
    out_dir = run_dir / "Output"

    def say(step, msg):
        if progress:
            progress(step, msg)

    result = RunResult(ok=False, run_id=run_id, run_dir=run_dir,
                       output_dir=out_dir)

    if run_dir.exists():
        result.error = (f"{run_dir} already exists. A run is never overwritten, "
                        "so that a past quarter can always be reproduced.")
        return result
    if not input_dir.is_dir():
        result.error = f"No input directory at {input_dir}"
        return result

    try:
        say("read", "Reading the source extracts…")
        src = read_all_inputs(input_dir)
        result.steps.append({
            "step": "Read inputs", "ok": src.ok,
            "detail": (f"{len(src.tables)} of {len(src.tables) + len(src.missing)} "
                       f"files read"),
            "missing": src.missing,
        })
        if not src.ok:
            result.error = ("Missing source files: " + ", ".join(src.missing))
            return result

        am, am2 = src["AccountMaster"], src["AccountMasterInvestments"]
        ref = reporting_date or _infer_reporting_date(am)
        # Not strftime("%-m/..."): that is a glibc extension and raises
        # "Invalid format string" on Windows.
        _r = pd.to_datetime(ref)
        extract_date = f"{_r.month}/{_r.day}/{_r.year}"
        say("date", f"Reporting date {extract_date}")

        out_dir.mkdir(parents=True, exist_ok=True)
        tables: dict[str, pd.DataFrame] = {}

        say("lending", "Transforming the lending book…")
        # The portfolio is a LOOKUP from the product type, not the account
        # type itself. Without the mapping every product lands in its own
        # "portfolio" and no PD curve resolves, because the curves are keyed on
        # the six real portfolios.
        static = load_static_reference(static_dir)
        view = transform_lending(am, static.get("product_portfolio_mapping"),
                                 customers=src["CustomerMaster"])
        tables["AccountMaster_1.csv"] = build_account_master(view, extract_date)
        result.steps.append({"step": "Lending accounts", "ok": True,
                             "detail": f"{len(view):,} contracts"})

        say("investments", "Transforming the investment book…")
        inv = transform_investments(am2)
        tables["AccountMaster_2.csv"] = build_account_master(inv, extract_date,
                                                             investments=True)
        result.steps.append({"step": "Investment accounts", "ok": True,
                             "detail": f"{len(inv):,} accounts"})

        say("collateral", "Collateral and allocations…")
        tables["Collateral.csv"] = transform_collateral(src["Collateral"])
        tables["AccountCollateralAllocation.csv"] = transform_allocation(
            src["AccountCollateralAllocation"])
        result.steps.append({
            "step": "Collateral", "ok": True,
            "detail": (f"{len(tables['Collateral.csv']):,} items, "
                       f"{len(tables['AccountCollateralAllocation.csv']):,} allocations")})

        say("customers", "Customer master and staging flags…")
        inv_ids = investment_customer_ids(am2)
        tables["CustomerMaster_1.csv"] = transform_customer_master(
            src["CustomerMaster"], extract_date=extract_date)
        tables["CustomerMaster_2.csv"] = transform_customer_master(
            None, inv_ids, extract_date, investments=True)

        sf_ids = pick(src["CustomerStagingFlag"], "CUSTOMERID")
        flags = derive_customer_flags(am, src["CustomerStagingFlag"])
        if sf_ids is not None:
            flags = (flags.set_index("customer_id")
                     .reindex(sf_ids.astype(str).str.strip()).reset_index())
            flags.columns = ["customer_id"] + list(flags.columns[1:])
        tables["CustomerStagingFlag_1.csv"] = transform_staging_flags(
            None, flags, extract_date)
        tables["CustomerStagingFlag_2.csv"] = transform_staging_flags(
            None, pd.DataFrame({"customer_id": inv_ids}), extract_date,
            investments=True)
        result.steps.append({
            "step": "Customers", "ok": True,
            "detail": (f"{len(tables['CustomerMaster_1.csv']):,} lending, "
                       f"{len(inv_ids):,} investment")})

        say("origination", "Origination…")
        tables["Origination_1.csv"] = transform_origination(
            src["Origination"], view["contract_id_raw"])
        tables["Origination_2.csv"] = transform_origination_investments(
            src["OriginationInvestments"], am2, extract_date)
        result.steps.append({"step": "Origination", "ok": True,
                             "detail": f"{len(tables['Origination_1.csv']):,} contracts"})

        say("curves", "Building the EAD curves…")
        lpo = build_lifetime_parameter_other(
            src["RepaymentSchedule"], am, ref, extract_date,
            dict(zip(view["contract_id_raw"], view["contract_id"])),
            contracts=set(view["contract_id_raw"]))
        tables["LifeTimeParameterOther.csv"] = lpo
        result.steps.append({
            "step": "EAD curves", "ok": True,
            "detail": (f"{lpo['ContractId'].nunique():,} contracts, "
                       f"{len(lpo):,} monthly points")})

        say("stpd", "Building the PD term structures…")
        stpd = _build_stpd(static, config_dir, extract_date)
        if stpd is not None and len(stpd):
            tables["StPD.csv"] = stpd
            result.steps.append({
                "step": "PD curves", "ok": True,
                "detail": (f"{stpd['PDBucketDim1'].nunique()} buckets x "
                           f"{stpd['PortfolioCode'].nunique()} portfolios x "
                           f"{stpd['MonthLifetime'].nunique()} months")})
        else:
            result.steps.append({"step": "PD curves", "ok": False,
                                 "detail": "no model config found"})

        say("static", "Building the reference files…")
        ref = build_reference_files(static, extract_date)
        tables.update(ref)
        result.steps.append({"step": "Reference files", "ok": bool(ref),
                             "detail": f"{len(ref)} files"})

        say("write", "Writing the LIC input files…")
        written = write_outputs(out_dir, tables)
        result.written = sorted(written)
        result.pending = [f for f in PENDING if f not in result.written]

        say("snapshot", "Freezing the configuration…")
        try:
            from ..governance import AuditLog, take_snapshot
            cfg_src = Path(config_dir) if config_dir else \
                Path(__file__).parent.parent / "config"
            snap = take_snapshot(run_dir, cfg_src,
                                 static_dir or Path(__file__).parent.parent / "static")
            AuditLog(run_dir / "audit.jsonl").record(
                "run", f"ETL run from {input_dir}", input_dir=str(input_dir),
                reporting_date=extract_date)
            result.steps.append({"step": "Snapshot", "ok": True,
                                 "detail": f"{len(snap['files'])} files frozen"})
        except Exception as exc:
            result.steps.append({"step": "Snapshot", "ok": False,
                                 "detail": f"{type(exc).__name__}: {exc}"})

        say("price", "Pricing the book…")
        try:
            from .report import build_final_ecl_report
            rep = build_final_ecl_report(run_dir, run_id_label=run_id)
            ecl = pd.to_numeric(rep["Cla Amount Onbal"], errors="coerce")
            exp = pd.to_numeric(rep["Exposure On Bal"], errors="coerce")
            result.written.append("FinalEclReport.csv")
            result.steps.append({
                "step": "ECL report", "ok": True,
                "detail": (f"{len(rep):,} contracts, exposure "
                           f"{exp.sum():,.0f}, ECL {ecl.sum():,.0f} "
                           f"({100 * ecl.sum() / max(exp.sum(), 1):.2f}%)")})
        except Exception as exc:
            # The inputs are written and usable even if pricing fails, so the
            # run is not discarded -- the failure is reported against the step.
            result.steps.append({"step": "ECL report", "ok": False,
                                 "detail": f"{type(exc).__name__}: {exc}"})

        say("validate", "Validating the run…")
        try:
            from ..validation import (load_suppressions, suppression_reasons,
                                      validate_stages,
                                      write_validation_reports)
            model_cfg, model_inputs = load_model_config(config_dir)
            iw = ew = mw = None
            if model_cfg and model_inputs:
                from .macro import (external_gcc_forecast,
                                    gcc_weighted_history,
                                    resolve_external_scenario_weights,
                                    resolve_internal_scenario_weights)
                try:
                    iw = resolve_internal_scenario_weights(model_inputs, static)
                    gh = gcc_weighted_history(
                        static.get("gcc_real_gdp_growth"),
                        static.get("gcc_gdp_current_prices"))
                    ew = resolve_external_scenario_weights(
                        model_inputs, static, gh,
                        external_gcc_forecast(model_inputs, static, 5))
                    mw = [c["weight"] for c in
                          model_cfg["models"]["internal_v4_production"]
                          ["mev_components"]]
                except Exception:
                    pass
            supp_path = run_dir / "config_used" / "config" / \
                "validation_suppressions.yml"
            supp = suppression_reasons(load_suppressions(supp_path))
            findings = validate_stages(
                inputs=src, static=static, trans_lending=view,
                lending_view=flags, trans_investments=inv,
                investment_view=inv, ltpo=lpo, stpd=stpd,
                internal_weights=iw, external_weights=ew, mev_weights=mw,
                reporting_date=extract_date, suppressions=supp)
            write_validation_reports(findings, run_dir / "reports")
            summary = findings.summary()
            result.validation = summary
            result.steps.append({
                "step": "Validation", "ok": summary["errors"] == 0,
                "detail": (f"{summary['checks']} checks, "
                           f"{summary['errors']} error(s), "
                           f"{summary['warnings']} warning(s), "
                           f"{summary['suppressed']} suppressed")})
        except Exception as exc:
            # A validation failure must not discard a run that completed: the
            # outputs are still written and the failure is reported against
            # its own step.
            result.steps.append({"step": "Validation", "ok": False,
                                 "detail": f"{type(exc).__name__}: {exc}"})

        _write_manifest(run_dir, run_id, input_dir, extract_date, result)
        result.ok = True
        result.steps.append({
            "step": "Write outputs", "ok": True,
            "detail": f"{len(result.written)} files written, "
                      f"{len(result.pending)} still to port"})
    except Exception as exc:
        # A failed run must say WHERE it failed, not just what the exception
        # was. "Invalid format string" tells nobody which step was running.
        import traceback
        step = result.steps[-1]["step"] if result.steps else "reading the inputs"
        result.error = (f"Failed after '{step}': {type(exc).__name__}: {exc}")
        result.traceback = traceback.format_exc()
        result.steps.append({"step": "FAILED", "ok": False,
                             "detail": result.error})
    finally:
        result.seconds = time.time() - started
    return result


def load_model_config(config_dir=None):
    """The model config a run should use: its own frozen copy, else the package's.

    A run reproduces a past quarter only if it reads the config THAT run
    froze, so the run's own directory always wins.
    """
    import yaml

    search = []
    if config_dir:
        search.append(Path(config_dir))
    search += [Path.cwd() / "config", Path(__file__).parent.parent / "config"]
    for d in search:
        if (d / "model.yml").is_file() and (d / "model_inputs.yml").is_file():
            return (yaml.safe_load((d / "model.yml").read_text(encoding="utf-8")),
                    yaml.safe_load((d / "model_inputs.yml").read_text(
                        encoding="utf-8")))
    return None, None


def _build_stpd(static, config_dir, extract_date: str):
    """The PD term structures.

    Reads the model config from the run's own frozen copy when one is given,
    and otherwise from the package. Weights are COMPUTED from the non-oil GDP
    forecast rather than read, because the config runs on
    `auto_non_oil_gdp_cdf` and the explicit block in the file is a rounded
    snapshot of that calculation.
    """
    import yaml

    model, inputs_yml = load_model_config(config_dir)
    if model is None or inputs_yml is None:
        return None

    weights = None
    try:
        fcb = inputs_yml["mev_forecasts"]["forecasts"]
        n = int(inputs_yml.get("internal_scenario_weights", {})
                .get("n_forecast_years", 2))
        gdp = [float(fcb[y][0]) for y in sorted(fcb, key=lambda k: int(k))][:n]
        hist = static["non_oil_gdp_history"]["value"].to_numpy()
        weights = compute_internal_scenario_weights(hist, gdp,
                                                    static["scenario_severity"])
    except Exception:
        weights = None
    return build_stpd_from_static(static, model, inputs_yml, extract_date,
                                  scenario_weights=weights)


def _infer_reporting_date(accounts: pd.DataFrame):
    """The extract date carried on the accounts, not today's date.

    Using the run date would silently re-age every contract if a run were
    repeated a week later.
    """
    col = pick(accounts, "EXTRACTDA", "EXTRACTDATE", "ExtractDate")
    if col is None:
        return pd.Timestamp.today().normalize()
    d = pd.to_datetime(col, errors="coerce", format="mixed").dropna()
    return d.max() if len(d) else pd.Timestamp.today().normalize()


def _copy_reference(static_dir, out_dir: Path) -> list[str]:
    """The six reference files, copied unchanged from the static reference."""
    src = Path(static_dir) if static_dir else None
    copied = []
    mapping = {
        "Portfolios.csv": "portfolios.csv",
        "Ratings.csv": "master_rating_scale.csv",
        "CollateralType.csv": "collateral_types.csv",
        "FxRate.csv": "fx_rates.csv",
    }
    if src and src.is_dir():
        for target, source in mapping.items():
            p = src / source
            if p.is_file():
                shutil.copy(p, out_dir / target)
                copied.append(target)
    return copied


def _write_manifest(run_dir: Path, run_id: str, input_dir: Path,
                    extract_date: str, result: RunResult) -> None:
    """What produced this run.

    Written so a number can be traced back months later: which inputs, which
    engine version, what was still pending at the time.
    """
    from .. import __version__
    # The declared calculator version AND a fingerprint of the code that
    # actually ran. Either alone is insufficient: the version can claim what
    # the code is not, and a bare hash names nothing.
    try:
        from ..calculator_versions import calculator_version_for_run
        calculator = calculator_version_for_run(root=run_dir.parent.parent)
    except Exception:
        calculator = None
    try:
        from ..code_version import code_status
        code = code_status()
    except Exception:
        code = None

    manifest = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "engine_version": __version__,
        "calculator": calculator,
        "code": code,
        "input_dir": str(input_dir),
        "reporting_date": extract_date,
        "files_written": result.written,
        "files_pending": result.pending,
        "validation": result.validation,
        "steps": result.steps,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                      default=str))
