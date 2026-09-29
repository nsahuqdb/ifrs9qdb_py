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

import numpy as np
import pandas as pd

from ..audit_log import audit_event
from .customer import derive_customer_flags, investment_customer_ids
from .lending import (build_account_master, transform_investments,
                      transform_lending)
from .lifetime import build_lifetime_parameter_other
from .read_inputs import read_all_inputs
from .macro import build_stpd_from_static
from .reference import build_reference_files
from .static_ref import load_static_reference
from .transform import (pick, transform_allocation, transform_collateral,
                        transform_customer_master, transform_origination,
                        transform_origination_investments,
                        transform_staging_flags, write_outputs,
                        build_origination_rows)

__all__ = ["RunResult", "PhaseState", "run_etl", "run_etl_phase1",
           "run_etl_phase2", "normalise_overrides", "next_run_id", "PRODUCED",
           "PENDING", "OVERRIDE_KINDS"]

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
    readiness: dict | None = None
    overrides_applied: dict | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "run_id": self.run_id,
            "run_dir": str(self.run_dir), "output_dir": str(self.output_dir),
            "steps": self.steps, "written": self.written,
            "pending": self.pending, "seconds": round(self.seconds, 1),
            "error": self.error, "traceback": self.traceback,
            "validation": self.validation,
            "readiness": self.readiness,
            "overrides_applied": self.overrides_applied,
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


@dataclass
class PhaseState:
    """Everything phase 1 hands to phase 2 -- R's phase-1 ``state`` list.

    Held in memory between the two calls: in the app, between the moment the
    run pauses for review and the moment somebody presses Continue. ``done``
    is set when phase 1 already ended the run (a gate stopped it, or it could
    not start); ``result`` is then final and phase 2 must not be called.
    """
    run_id: str
    run_dir: Path
    out_dir: Path
    input_dir: Path
    runs_dir: Path
    result: RunResult
    started: float
    started_at: datetime
    static_dir: object = None
    config_dir: object = None
    cfg_src: Path | None = None
    reporting_date: object = None
    extract_date: str = ""
    user: str | None = None
    run_type: str = "unofficial"
    ecl_scenario: str = "weighted"
    run_purpose: str | None = None
    portfolio_date: str | None = None
    snapshot_meta: dict | None = None
    calculator_version: str | None = None
    input_source: dict | None = None
    run_config: dict | None = None
    # config.yml's run.internal_model: which PD model prices the run
    model_id: str | None = None
    project_root: Path | None = None
    on_validation_error: str = "warn"
    stop_before_pricing: bool = False
    progress: object = None
    src: object = None
    static: dict | None = None
    supp: dict | None = None
    stage_results: list = field(default_factory=list)
    view: pd.DataFrame | None = None
    inv: pd.DataFrame | None = None
    flags: pd.DataFrame | None = None
    tables: dict = field(default_factory=dict)
    cm_view: pd.DataFrame | None = None
    inv_view: pd.DataFrame | None = None
    done: bool = False

    def say(self, step, msg):
        if self.progress:
            self.progress(step, msg)

    def customer_view(self) -> pd.DataFrame:
        """The customer-level view the override page shows: R's cm_view,
        one row per CustomerMaster customer."""
        cols = ["customer_id", "customer_name", "rating_final", "stage_final",
                "restructuring_final", "watchlist_status", "exposure_total",
                "dpd_status"]
        if self.cm_view is None:
            return pd.DataFrame(columns=cols)
        return self.cm_view[[c for c in cols if c in self.cm_view.columns]].copy()


def run_etl(input_dir, runs_dir, reporting_date=None, run_id=None,
            static_dir=None, config_dir=None, progress=None,
            run_type: str = "unofficial", ecl_scenario: str = "weighted",
            user: str | None = None,
            on_validation_error: str = "warn",
            stop_before_pricing: bool = False,
            overrides: dict | None = None, **meta) -> RunResult:
    """Build a run from the source extracts: phase 1, then phase 2.

    ``progress`` is called with (step, message) so a caller can show what is
    happening -- reading 117,000 collateral allocations is not instant, and a
    silent minute looks like a hang.

    ``run_type`` decides whether the run enters the approval queue. It defaults
    to ``unofficial`` -- terminal, no approval -- because a run is not a number
    anybody books until somebody says it is. An official run must use the
    weighted ECL; a single scenario is by definition not the reported figure.

    VALIDATION GATES, in the order R's phased run applies them:

        INPUT      the raw extracts, before anything is transformed
        TRANSFORM  the shaped book
        DERIVED    the EAD and PD curves
        READY      the files LIC reads, BEFORE the book is priced: which
                   contracts will come out with no ECL, which LIC would blank,
                   and which are priced from incomplete inputs
        REPORT     after pricing: every contract has a row and every blank
                   ECL was predicted by READY

    ``on_validation_error`` is R's run.on_validation_error: ``stop`` ends the
    run at the first gate with an unsuppressed ERROR, before anything is
    priced; ``warn`` (the app's setting) records it and carries on; ``ignore``
    carries on silently. Either way every finding is in reports/.

    ``stop_before_pricing`` runs everything up to and including READY and
    stops: the pre-run readiness check, which answers "will every row get a
    number?" before a run is committed to.

    ``overrides`` are applied between the phases, as the app's pause step
    does (see ``run_etl_phase2``). ``meta`` carries the run metadata phase 1
    records: run_purpose, portfolio_date, snapshot_meta, calculator_version,
    input_source, run_config, project_root.
    """
    state = run_etl_phase1(input_dir, runs_dir, reporting_date=reporting_date,
                           run_id=run_id, static_dir=static_dir,
                           config_dir=config_dir, progress=progress,
                           run_type=run_type, ecl_scenario=ecl_scenario,
                           user=user, on_validation_error=on_validation_error,
                           stop_before_pricing=stop_before_pricing, **meta)
    if state.done:
        return state.result
    return run_etl_phase2(state, overrides=overrides)


def run_etl_phase1(input_dir, runs_dir, reporting_date=None, run_id=None,
                   static_dir=None, config_dir=None, progress=None,
                   run_type: str = "unofficial", ecl_scenario: str = "weighted",
                   user: str | None = None, on_validation_error: str = "warn",
                   stop_before_pricing: bool = False,
                   run_purpose: str | None = None,
                   portfolio_date: str | None = None,
                   snapshot_meta: dict | None = None,
                   calculator_version: str | None = None,
                   input_source: dict | None = None,
                   run_config: dict | None = None,
                   project_root=None) -> PhaseState:
    """Phase 1: read, validate INPUT, transform, validate TRANSFORM -- then stop.

    R's run_etl_phase1(). What comes back carries the customer-level view
    (``state.customer_view()``) the app shows for review, so a rating, stage or
    restructuring decision the model cannot see can be recorded -- with a
    reason -- before anything is priced. ``run_etl_phase2`` finishes the run.
    """
    if on_validation_error not in ("stop", "warn", "ignore"):
        on_validation_error = "warn"
    started = time.time()
    input_dir = Path(input_dir)
    runs_dir = Path(runs_dir)
    run_id = run_id or next_run_id(runs_dir)
    run_dir = runs_dir / run_id
    out_dir = run_dir / "Output"
    result = RunResult(ok=False, run_id=run_id, run_dir=run_dir,
                       output_dir=out_dir)
    st = PhaseState(
        run_id=run_id, run_dir=run_dir, out_dir=out_dir, input_dir=input_dir,
        runs_dir=runs_dir, result=result, started=started,
        started_at=datetime.now().astimezone(), static_dir=static_dir,
        config_dir=config_dir, user=user, run_type=run_type,
        ecl_scenario=ecl_scenario, run_purpose=run_purpose,
        portfolio_date=(str(portfolio_date) if portfolio_date else None),
        snapshot_meta=snapshot_meta, calculator_version=calculator_version,
        input_source=input_source, run_config=run_config,
        project_root=Path(project_root) if project_root else runs_dir.parent,
        on_validation_error=on_validation_error,
        stop_before_pricing=stop_before_pricing, progress=progress)

    if run_dir.exists():
        result.error = (f"{run_dir} already exists. A run is never overwritten, "
                        "so that a past quarter can always be reproduced.")
        st.done = True
        return st
    if not input_dir.is_dir():
        result.error = f"No input directory at {input_dir}"
        st.done = True
        return st

    audit_event({"event": "run_start", "run_id": run_id,
                 "snapshot": (snapshot_meta or {}).get("label"),
                 "ecl_scenario": ecl_scenario, "code_sha": _code_sha(),
                 "config_path": str(config_dir) if config_dir else None,
                 "started_at": st.started_at.strftime("%Y-%m-%dT%H:%M:%S%z")})
    try:
        # R resolves the model before it reads anything and stops when
        # config.yml names one the registry does not hold, or a component
        # names a variable the dictionary does not define.
        from .model_registry import model_id_from_run_config, resolve_model
        st.model_id = model_id_from_run_config(run_config)
        mc_check, _ = load_model_config(config_dir)
        if mc_check is not None:
            try:
                resolve_model(mc_check, st.model_id)
            except ValueError as exc:
                result.error = f"Model configuration: {exc}"
                result.steps.append({"step": "Model", "ok": False,
                                     "detail": str(exc)})
                st.done = True
                return st

        st.say("read", "Reading the source extracts…")
        src = read_all_inputs(input_dir)
        st.src = src
        result.steps.append({
            "step": "Read inputs", "ok": src.ok,
            "detail": (f"{len(src.tables)} of {len(src.tables) + len(src.missing)} "
                       f"files read"),
            "missing": src.missing,
        })
        if not src.ok:
            result.error = ("Missing source files: " + ", ".join(src.missing))
            st.done = True
            return st

        from ..validation import (load_suppressions, suppression_reasons,
                                  validate_stages)
        st.cfg_src = Path(config_dir) if config_dir else \
            Path(__file__).parent.parent / "config"
        st.supp = suppression_reasons(load_suppressions(
            st.cfg_src / "validation_suppressions.yml"))
        gate = _Gate(on_validation_error, result, st.stage_results, run_id)

        am, am2 = src["AccountMaster"], src["AccountMasterInvestments"]
        # The dates, as R takes them. The run's reporting date -- the stamp on
        # every output file, the anchor of the EAD and PD curves and of the
        # maturity extension -- is the AccountMaster EXTRACTDA most rows carry
        # (R's resolve_input_extract_date()); config.yml's run.extract_date is
        # only the fallback for inputs that carry none (R's
        # apply_input_extract_date). The transforms extend a lapsed maturity
        # from that same date, and from the latest EXTRACTDA of their own file
        # only when the run has none (R's run_reporting_date()), so one stray
        # row cannot move part of the calculation onto another date --
        # INPUT_extract_date_matches_run_cfg reports any such row.
        # R reads every date from the schema-typed inputs (read_all_inputs
        # types EXTRACTDA before anything looks at it), so the same typed
        # tables are used here.
        from ..prerun import apply_input_extract_date
        from ..validation._helpers import (latest_extract_date,
                                           resolve_input_extract_date)
        from ..validation.schema import canonicalise
        typed = canonicalise(src)
        adopted = resolve_input_extract_date(typed)
        cfg_date = _config_extract_date(run_config)
        ref = (reporting_date or adopted or cfg_date
               or _infer_reporting_date(am))
        if reporting_date or adopted is not None:
            st.run_config = apply_input_extract_date(
                run_config, pd.Timestamp(ref).strftime("%Y-%m-%d"))
        run_date = reporting_date or adopted or cfg_date
        lend_ref = run_date or latest_extract_date(typed["AccountMaster"])
        inv_ref = run_date or latest_extract_date(
            typed["AccountMasterInvestments"])
        st.reporting_date = ref
        # Not strftime("%-m/..."): that is a glibc extension and raises
        # "Invalid format string" on Windows.
        _r = pd.to_datetime(ref)
        extract_date = f"{_r.month}/{_r.day}/{_r.year}"
        st.extract_date = extract_date
        st.say("date", f"Reporting date {extract_date}")

        out_dir.mkdir(parents=True, exist_ok=True)
        if input_source:
            try:
                from ..acquisition import record_input_source
                record_input_source(run_dir, input_source.get("kind", "configured"),
                                    input_source.get("details") or {})
            except Exception:
                pass
        tables = st.tables
        static = load_static_reference(static_dir)
        st.static = static

        st.say("validate", "Validating the source extracts…")
        v_in = validate_stages(inputs=src, static=static,
                               reporting_date=extract_date, suppressions=st.supp,
                               run_config=st.run_config, stages=["INPUT"])
        if not gate("INPUT", v_in):
            _halt(result, run_dir, run_id, input_dir, extract_date,
                  st.stage_results, user, started, state=st)
            st.done = True
            return st

        st.say("lending", "Transforming the lending book…")
        # The portfolio is a LOOKUP from the product type, not the account
        # type itself. Without the mapping every product lands in its own
        # "portfolio" and no PD curve resolves, because the curves are keyed on
        # the six real portfolios.
        view = transform_lending(am, static.get("product_portfolio_mapping"),
                                 customers=src["CustomerMaster"],
                                 industry=src["IndustryCode"], static=static,
                                 reporting_date=lend_ref)
        st.view = view
        tables["AccountMaster_1.csv"] = build_account_master(view, extract_date)
        result.steps.append({"step": "Lending accounts", "ok": True,
                             "detail": f"{len(view):,} contracts"})

        st.say("investments", "Transforming the investment book…")
        inv = transform_investments(am2, static=static, reporting_date=inv_ref)
        st.inv = inv
        tables["AccountMaster_2.csv"] = build_account_master(inv, extract_date,
                                                             investments=True)
        result.steps.append({"step": "Investment accounts", "ok": True,
                             "detail": f"{len(inv):,} accounts"})

        st.say("collateral", "Collateral and allocations…")
        tables["Collateral.csv"] = transform_collateral(src["Collateral"],
                                                        extract_date)
        from ..runconfig import allocation_percentage_unit
        tables["AccountCollateralAllocation.csv"] = transform_allocation(
            src["AccountCollateralAllocation"], extract_date,
            unit=allocation_percentage_unit(st.run_config))
        result.steps.append({
            "step": "Collateral", "ok": True,
            "detail": (f"{len(tables['Collateral.csv']):,} items, "
                       f"{len(tables['AccountCollateralAllocation.csv']):,} allocations")})

        st.say("customers", "Customer master and staging flags…")
        inv_ids = investment_customer_ids(am2)
        tables["CustomerMaster_1.csv"] = transform_customer_master(
            src["CustomerMaster"], extract_date=extract_date)
        tables["CustomerMaster_2.csv"] = transform_customer_master(
            None, inv_ids, extract_date, investments=True)

        # One staging-flag row per CustomerMaster customer, in CustomerMaster
        # order -- the key R's lending view (and so its writer) uses. Keying on
        # the staging extract instead dropped the flags of any customer the
        # extract left out: LIC then had no default flag for a customer the
        # ETL had itself staged as defaulted. A customer with no row in the
        # extract gets the computed flags with the watchlist flag FALSE, as R
        # writes it (XFILE_AM_customer_in_staging_flags reports the gap).
        from ..ids import as_id as _as_id
        cm_ids = pick(src["CustomerMaster"], "CUSTOMERID")
        sf_ids = pick(src["CustomerStagingFlag"], "CUSTOMERID")
        key_ids = cm_ids if cm_ids is not None else sf_ids
        flags = derive_customer_flags(am, src["CustomerStagingFlag"])
        if key_ids is not None:
            flags = (flags.set_index("customer_id")
                     .reindex(_as_id(key_ids)).reset_index())
            flags.columns = ["customer_id"] + list(flags.columns[1:])
            for c in ("is_default", "is_watchlist", "is_local1", "is_local3",
                      "is_restructured"):
                if c in flags.columns:
                    flags[c] = flags[c].astype(object).where(
                        flags[c].notna(), False).astype(bool)
        st.flags = flags
        tables["CustomerStagingFlag_1.csv"] = transform_staging_flags(
            None, flags, extract_date)
        tables["CustomerStagingFlag_2.csv"] = transform_staging_flags(
            None, pd.DataFrame({"customer_id": inv_ids}), extract_date,
            investments=True)
        result.steps.append({
            "step": "Customers", "ok": True,
            "detail": (f"{len(tables['CustomerMaster_1.csv']):,} lending, "
                       f"{len(inv_ids):,} investment")})

        st.say("origination", "Origination…")
        # Built from the account masters, not the origination extract: R
        # writes one blank row per account under its account-master id.
        tables["Origination_1.csv"] = build_origination_rows(
            extract_date, view["contract_id"])
        tables["Origination_2.csv"] = build_origination_rows(
            extract_date, inv["contract_id"] if len(inv) else [])
        result.steps.append({"step": "Origination", "ok": True,
                             "detail": f"{len(tables['Origination_1.csv']):,} contracts"})

        st.say("validate", "Validating the transformed book…")
        v_tr = validate_stages(inputs=src, static=static, trans_lending=view,
                               lending_view=flags, trans_investments=inv,
                               investment_view=inv, suppressions=st.supp,
                               stages=["TRANSFORM"])
        if not gate("TRANSFORM", v_tr):
            _halt(result, run_dir, run_id, input_dir, extract_date,
                  st.stage_results, user, started, state=st)
            st.done = True
            return st

        # The customer view the review step shows, in R's shape.
        try:
            from ..validation.r_frames import investment_frames, lending_frames
            from ..validation.schema import canonicalise
            canon = canonicalise(src)
            _, st.cm_view = lending_frames(view, canon, static)
            _, st.inv_view = investment_frames(inv, canon, static)
        except Exception:
            st.cm_view = st.cm_view if st.cm_view is not None else None

        # A paused run is legible on disk: what was found so far.
        try:
            _write_validation(result, run_dir, st.stage_results, quiet=True)
        except Exception:
            pass
        audit_event({"event": "run_phase1_complete", "run_id": run_id,
                     "n_customers_lending": 0 if st.cm_view is None
                     else int(len(st.cm_view)),
                     "n_accounts_investments": 0 if st.inv_view is None
                     else int(len(st.inv_view)),
                     "paused_for_overrides": True})
        return st
    except Exception as exc:
        # A failed run must say WHERE it failed, not just what the exception
        # was. "Invalid format string" tells nobody which step was running.
        import traceback
        step = result.steps[-1]["step"] if result.steps else "reading the inputs"
        result.error = (f"Failed after '{step}': {type(exc).__name__}: {exc}")
        result.traceback = traceback.format_exc()
        result.steps.append({"step": "FAILED", "ok": False,
                             "detail": result.error})
        result.seconds = time.time() - started
        st.done = True
        return st


OVERRIDE_KINDS = {"rating": "override_rating", "stage": "override_stage",
                  "restructuring": "override_restructuring"}


def normalise_overrides(overrides, state: PhaseState) -> dict[str, pd.DataFrame]:
    """R's .normalise_override_frame(), for each of the three kinds.

    Each override is a customer id, the new value and a REQUIRED reason; who
    and when default to the current user and now; the prior value is what the
    model calculated. A frame, a list of dicts or None is accepted per kind,
    with the value under ``value`` or R's ``override_<kind>``.
    """
    overrides = overrides or {}
    user = state.user or _who()
    now = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    cm = state.cm_view if state.cm_view is not None else pd.DataFrame(
        columns=["customer_id", "rating_final", "stage_final",
                 "restructuring_final"])
    prior_col = {"rating": "rating_final", "stage": "stage_final",
                 "restructuring": "restructuring_final"}
    cols = ["customer_id", "value", "reason", "created_by", "created_at",
            "prior_value", "source_run_id"]
    out = {}
    for kind, value_col in OVERRIDE_KINDS.items():
        raw = overrides.get(kind)
        df = pd.DataFrame(raw) if raw is not None else pd.DataFrame()
        if len(df) == 0:
            out[kind] = pd.DataFrame(columns=cols)
            continue
        vcol = value_col if value_col in df.columns else "value"
        if vcol not in df.columns or "customer_id" not in df.columns:
            raise ValueError(f"{kind} overrides need customer_id and a value")
        cid = df["customer_id"].astype(str).str.strip()
        reason = (df["reason"] if "reason" in df.columns
                  else pd.Series([""] * len(df))).fillna("").astype(str)
        if (reason.str.strip() == "").any():
            raise ValueError("Every override row must have a non-empty reason")
        by = (df["created_by"] if "created_by" in df.columns
              else pd.Series([""] * len(df))).fillna("").astype(str)
        at = (df["created_at"] if "created_at" in df.columns
              else pd.Series([""] * len(df))).fillna("").astype(str)
        lut = dict(zip(cm["customer_id"].astype(str),
                       cm[prior_col[kind]].fillna("").astype(str))) \
            if prior_col[kind] in cm.columns else {}
        prior = cid.map(lambda c: lut.get(c, ""))
        out[kind] = pd.DataFrame({
            "customer_id": cid.to_numpy(),
            "value": df[vcol].astype(str).str.strip().to_numpy(),
            "reason": reason.to_numpy(),
            "created_by": by.where(by != "", user).to_numpy(),
            "created_at": at.where(at != "", now).to_numpy(),
            "prior_value": prior.to_numpy(),
            "source_run_id": state.run_id,
        }, columns=cols)
    return out


def _put(df: pd.DataFrame, column: str, mask: pd.Series, values) -> None:
    """Set ``column`` where ``mask`` holds, by replacing the whole column.

    Whole-column rather than ``.loc`` assignment: under copy-on-write a string
    column can be backed by a read-only array, and writing into it in place
    raises "assignment destination is read-only".
    """
    m = mask.to_numpy(dtype=bool)
    if column in df.columns:
        was_bool = pd.api.types.is_bool_dtype(df[column])
        col = df[column].astype(object).to_numpy(copy=True)
    else:
        was_bool = False
        col = np.full(len(df), None, dtype=object)
    col[m] = np.asarray(values, dtype=object)
    df[column] = col.astype(bool) if was_bool else col


def _apply_overrides(state: PhaseState, ov: dict[str, pd.DataFrame]) -> None:
    """Write each override to the lending view, the staging flags and the
    customer view -- R's phase 2, column for column.

        rating         every contract of the customer takes the new rating
                       (AccountMaster_1.Rating reads rating_worst)
        stage          stage_final; Stage 3 is IsDefault, Stage 2 IsLocal3
        restructuring  restructuring_final; IsLocal1 is "Restructured"
    """
    view, flags, cm = state.view, state.flags, state.cm_view

    def lookup(frame, kind):
        lut = dict(zip(ov[kind]["customer_id"], ov[kind]["value"]))
        cid = frame["customer_id"].astype(str)
        hit = cid.isin(lut)
        return hit, cid[hit].map(lut).to_numpy(dtype=object)

    if len(ov["rating"]):
        if view is not None:
            hit, new = lookup(view, "rating")
            _put(view, "rating_worst", hit, new)
        if cm is not None:
            hit, new = lookup(cm, "rating")
            _put(cm, "rating_final", hit, new)
    if len(ov["stage"]):
        if flags is not None:
            hit, new = lookup(flags, "stage")
            _put(flags, "stage_final", hit, new)
            _put(flags, "is_default", hit, new == "Stage 3")
            _put(flags, "is_local3", hit, new == "Stage 2")
        if cm is not None:
            hit, new = lookup(cm, "stage")
            _put(cm, "stage_final", hit, new)
    if len(ov["restructuring"]):
        if flags is not None:
            hit, new = lookup(flags, "restructuring")
            _put(flags, "is_local1", hit, new == "Restructured")
            if "is_restructured" in flags.columns:
                _put(flags, "is_restructured", hit, new == "Restructured")
        if cm is not None:
            hit, new = lookup(cm, "restructuring")
            _put(cm, "restructuring_final", hit, new)
    if len(ov["rating"]):
        state.tables["AccountMaster_1.csv"] = build_account_master(
            view, state.extract_date)
    if len(ov["stage"]) or len(ov["restructuring"]):
        state.tables["CustomerStagingFlag_1.csv"] = transform_staging_flags(
            None, flags, state.extract_date)


def run_etl_phase2(state: PhaseState, overrides: dict | None = None) -> RunResult:
    """Phase 2: apply the reviewed overrides, build the curves, validate,
    write the LIC files, check readiness, price.

    R's run_etl_phase2(). The overrides are persisted to
    ``<run>/overrides/{rating,stage,restructuring}_overrides.csv`` -- always,
    even empty, so the run is self-describing -- and counted in
    run_status.yml, which the approval queue reads.
    """
    if state.done:
        return state.result
    result, run_dir, out_dir = state.result, state.run_dir, state.out_dir
    run_id, input_dir, extract_date = state.run_id, state.input_dir, \
        state.extract_date
    static, src, supp, tables = state.static, state.src, state.supp, state.tables
    user, started = state.user, state.started
    gate = _Gate(state.on_validation_error, result, state.stage_results, run_id)
    applied = {"rating": 0, "stage": 0, "restructuring": 0}
    try:
        # ---- overrides ------------------------------------------------------
        ov = normalise_overrides(overrides, state)
        ov_dir = run_dir / "overrides"
        ov_dir.mkdir(parents=True, exist_ok=True)
        for kind, df in ov.items():
            df.to_csv(ov_dir / f"{kind}_overrides.csv", index=False)
            applied[kind] = int(len(df))
        audit_event({"event": "run_overrides_applied", "run_id": run_id,
                     "n_overrides": sum(applied.values()),
                     "n_rating_overrides": applied["rating"],
                     "n_stage_overrides": applied["stage"],
                     "n_restructuring_overrides": applied["restructuring"]})
        if sum(applied.values()):
            _apply_overrides(state, ov)
            result.steps.append({
                "step": "Overrides", "ok": True,
                "detail": (f"{sum(applied.values())} applied (rating "
                           f"{applied['rating']}, stage {applied['stage']}, "
                           f"restructuring {applied['restructuring']})")})
        view, inv, am = state.view, state.inv, src["AccountMaster"]

        state.say("curves", "Building the EAD curves…")
        lpo = build_lifetime_parameter_other(
            src["RepaymentSchedule"], am, state.reporting_date, extract_date,
            dict(zip(view["contract_id_raw"], view["contract_id"])),
            contracts=set(view["contract_id_raw"]))
        tables["LifeTimeParameterOther.csv"] = lpo
        result.steps.append({
            "step": "EAD curves", "ok": True,
            "detail": (f"{lpo['ContractId'].nunique():,} contracts, "
                       f"{len(lpo):,} monthly points")})

        state.say("stpd", "Building the PD term structures…")
        stpd = _build_stpd(static, state.config_dir, extract_date,
                           model_id=state.model_id)
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

        state.say("static", "Building the reference files…")
        refs = build_reference_files(static, extract_date)
        tables.update(refs)
        result.steps.append({"step": "Reference files", "ok": bool(refs),
                             "detail": f"{len(refs)} files"})

        state.say("validate", "Validating the EAD and PD curves…")
        from ..validation import validate_stages
        model_cfg, model_inputs = load_model_config(state.config_dir)
        iw = ew = mw = None
        if model_cfg and model_inputs:
            from .macro import (external_gcc_forecast, gcc_weighted_history,
                                resolve_external_scenario_weights,
                                resolve_internal_scenario_weights)
            try:
                iw = resolve_internal_scenario_weights(model_inputs, static)
                gh = gcc_weighted_history(static.get("gcc_real_gdp_growth"),
                                          static.get("gcc_gdp_current_prices"))
                ew = resolve_external_scenario_weights(
                    model_inputs, static, gh,
                    external_gcc_forecast(model_inputs, static, 5))
                from .model_registry import mev_model_weights, resolve_model
                mw = mev_model_weights(resolve_model(model_cfg, state.model_id),
                                       model_inputs)
            except Exception:
                pass
        v_der = validate_stages(inputs=src, static=static, ltpo=lpo, stpd=stpd,
                                trans_lending=view, internal_weights=iw,
                                external_weights=ew, mev_weights=mw,
                                suppressions=supp, stages=["DERIVED"])
        if not gate("DERIVED", v_der):
            return _halt(result, run_dir, run_id, input_dir, extract_date,
                         state.stage_results, user, started, state=state,
                         applied=applied)

        state.say("write", "Writing the LIC input files…")
        written = write_outputs(out_dir, tables)
        result.written = sorted(written)
        result.pending = [f for f in PENDING if f not in result.written]

        state.say("snapshot", "Freezing the configuration…")
        try:
            from ..governance import AuditLog, take_snapshot
            rc_file = None
            if (state.run_config is not None and not state.snapshot_meta
                    and state.project_root
                    and (Path(state.project_root) / "config.yml").is_file()):
                rc_file = Path(state.project_root) / "config.yml"
            snap = take_snapshot(run_dir, state.cfg_src,
                                 state.static_dir
                                 or Path(__file__).parent.parent / "static",
                                 run_config_file=rc_file,
                                 run_config=state.run_config,
                                 snapshot_meta=state.snapshot_meta)
            AuditLog(run_dir / "audit.jsonl").record(
                "run", f"ETL run from {input_dir}", input_dir=str(input_dir),
                reporting_date=extract_date)
            result.steps.append({"step": "Snapshot", "ok": True,
                                 "detail": f"{len(snap['files'])} files frozen"})
        except Exception as exc:
            result.steps.append({"step": "Snapshot", "ok": False,
                                 "detail": f"{type(exc).__name__}: {exc}"})

        # ---- READINESS: will every row get a number? -------------------
        state.say("readiness", "Checking every contract can be priced…")
        from ..validation.framework import run_suite
        from ..validation.readiness import (READY_STAGE_VALIDATORS,
                                            REPORT_STAGE_VALIDATORS,
                                            assess_readiness,
                                            check_report_against_readiness,
                                            write_readiness_reports)
        readiness = assess_readiness(run_dir, static=static,
                                     model_cfg=model_cfg, src=src)
        write_readiness_reports(readiness, run_dir / "reports")
        result.readiness = readiness.summary() if readiness.ok else None
        v_ready = run_suite("READY", READY_STAGE_VALIDATORS,
                            {"readiness": readiness}, supp)
        if readiness.ok:
            s_ = readiness.summary()
            result.steps.append({
                "step": "Readiness", "ok": s_["No ECL"]["contracts"] == 0
                and s_["Blank in LIC"]["contracts"] == 0,
                "detail": (f"{s_['contracts']:,} contracts: "
                           f"{s_['Priced']['contracts']:,} ready, "
                           f"{s_['Priced - check']['contracts']:,} with a gap, "
                           f"{s_['Blank in LIC']['contracts']:,} blank in LIC, "
                           f"{s_['No ECL']['contracts']:,} with no ECL")})
        if not gate("READY", v_ready) or state.stop_before_pricing:
            if state.stop_before_pricing and not result.error:
                result.ok = True
                result.steps.append({"step": "Stopped before pricing",
                                     "ok": True,
                                     "detail": "pre-run readiness check only"})
            return _halt(result, run_dir, run_id, input_dir, extract_date,
                         state.stage_results, user, started,
                         keep_ok=state.stop_before_pricing, state=state,
                         applied=applied)

        state.say("price", "Pricing the book…")
        rep = None
        try:
            from .report import build_final_ecl_report
            rep = build_final_ecl_report(run_dir, run_id_label=run_id,
                                         static=static, model_cfg=model_cfg)
            ecl = pd.to_numeric(rep["Cla Amount Onbal"], errors="coerce")
            exp = pd.to_numeric(rep["Exposure On Bal"], errors="coerce")
            result.written.append("FinalEclReport.csv")
            result.steps.append({
                "step": "ECL report", "ok": True,
                "detail": (f"{len(rep):,} contracts, exposure "
                           f"{exp.sum():,.0f}, ECL {ecl.sum():,.0f} "
                           f"({100 * ecl.sum() / max(exp.sum(), 1):.2f}%), "
                           f"{int(ecl.isna().sum()):,} without an ECL")})
        except Exception as exc:
            # The inputs are written and usable even if pricing fails, so the
            # run is not discarded -- the failure is reported against the step.
            result.steps.append({"step": "ECL report", "ok": False,
                                 "detail": f"{type(exc).__name__}: {exc}"})

        if rep is not None:
            chk = check_report_against_readiness(rep, readiness)
            v_rep = run_suite("REPORT", REPORT_STAGE_VALIDATORS,
                              {"report_check": chk}, supp)
            gate("REPORT", v_rep, halt=False, audit=False)
            state.say("scenarios", "Pricing each scenario…")
            _price_scenarios(result, run_dir, out_dir, run_id, static,
                             model_cfg, model_inputs, extract_date,
                             model_id=state.model_id)

        _write_validation(result, run_dir, state.stage_results)

        _write_manifest(run_dir, run_id, input_dir, extract_date, result,
                        user=user, state=state, model_cfg=model_cfg)

        # The approval queue reads this. Written last, and its failure is
        # reported loudly: without it a completed run never appears for
        # sign-off and simply sits there.
        try:
            from ..run_status import init_run_status
            init_run_status(run_dir, run_id, run_type=state.run_type, by=user,
                            ecl_scenario=state.ecl_scenario,
                            overrides_applied=applied,
                            snapshot_label=(state.snapshot_meta or {}).get("label"))
            result.steps.append({
                "step": "Run status", "ok": True,
                "detail": f"{state.run_type}; "
                          + ("awaiting checker approval"
                             if state.run_type == "official"
                             else "terminal, no approval required")})
        except Exception as exc:
            result.steps.append({
                "step": "Run status", "ok": False,
                "detail": (f"{type(exc).__name__}: {exc} - the run will NOT "
                           "appear in the approval queue")})
        n_out = len(list(out_dir.glob("*.csv")))
        audit_event({"event": "run_pending_checker"
                     if state.run_type == "official" else "run_unofficial",
                     "run_id": run_id, "run_type": state.run_type,
                     "ecl_scenario": state.ecl_scenario, "n_outputs": n_out,
                     "n_overrides": sum(applied.values())})
        result.ok = True
        result.steps.append({
            "step": "Write outputs", "ok": True,
            "detail": f"{len(result.written)} files written, "
                      f"{len(result.pending)} still to port"})
        audit_event({"event": "run_finish", "run_id": run_id,
                     "snapshot": (state.snapshot_meta or {}).get("label"),
                     "code_sha": _code_sha(),
                     "duration_seconds": round(time.time() - started, 1),
                     "output_dir": str(out_dir),
                     "manifest_path": str(run_dir / "reports" / "manifest.json"),
                     "n_outputs": n_out,
                     "finished_at": datetime.now().astimezone()
                     .strftime("%Y-%m-%dT%H:%M:%S%z")})
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
    result.overrides_applied = applied
    return result


def _code_sha() -> str | None:
    try:
        from ..code_version import get_current_code_sha
        return get_current_code_sha()
    except Exception:
        return None


def _who() -> str:
    import getpass
    import os
    u = (os.environ.get("IFRS9_USER") or os.environ.get("USER")
         or os.environ.get("USERNAME"))
    if u:
        return u
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _price_scenarios(result, run_dir: Path, out_dir: Path, run_id: str,
                     static, model_cfg, model_inputs, extract_date: str,
                     model_id: str | None = None) -> None:
    """Every run also prices the book under each scenario on its own.

    R's phase 2 writes, per scenario, StPD_<scenario>.csv (the curve set with
    that scenario weighted 1 on both scales) and
    FinalEclReport_scenario_<scenario>.csv priced against it, beside the
    probability-weighted report -- the per-scenario provisions the analytics
    compare. The scenario curve files keep R's in-memory schema
    (extract_date, portfolio_code, ...), exactly as R writes them.
    """
    import re
    if not (model_cfg and model_inputs):
        return
    scen = static.get("scenario_severity")
    if scen is None or len(scen) == 0 or "scenario" not in scen.columns:
        return
    from .macro import build_stpd_from_static
    from .report import build_final_ecl_report
    names = [str(x) for x in scen["scenario"]]
    iso = pd.to_datetime(extract_date).strftime("%Y-%m-%d")
    done = []
    for sc in names:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", sc)
        try:
            one_hot = {x: float(x == sc) for x in names}
            st = build_stpd_from_static(static, model_cfg, model_inputs,
                                        extract_date, scenario_weights=one_hot,
                                        model_id=model_id)
            snake = pd.DataFrame({
                "extract_date": iso,
                "portfolio_code": st["PortfolioCode"],
                "pd_bucket_dim1": st["PDBucketDim1"],
                "pd_bucket_dim2": "",
                "month_lifetime": st["MonthLifetime"],
                "pd_lifetime": st["PDLifetime"],
            })
            snake.to_csv(out_dir / f"StPD_{safe}.csv", index=False)
            build_final_ecl_report(run_dir, run_id_label=run_id, static=static,
                                   model_cfg=model_cfg, stpd=snake,
                                   out_name=f"FinalEclReport_scenario_{safe}.csv")
            result.written += [f"StPD_{safe}.csv",
                               f"FinalEclReport_scenario_{safe}.csv"]
            done.append(sc)
        except Exception as exc:
            result.steps.append({"step": f"Scenario {sc}", "ok": False,
                                 "detail": f"{type(exc).__name__}: {exc}"})
    if done:
        result.steps.append({"step": "Scenario provisions", "ok": True,
                             "detail": f"{len(done)} scenario report(s): "
                                       + ", ".join(done)})


class _Gate:
    """R's .gate_on_validation(): record a stage's findings and decide.

    Gates on the EFFECTIVE severity, so a suppressed ERROR does not stop the
    run while still being recorded in the report.
    """

    def __init__(self, policy: str, result: RunResult, stage_results: list,
                 run_id: str | None = None):
        self.policy = policy
        self.result = result
        self.stage_results = stage_results
        self.run_id = run_id

    def __call__(self, stage: str, res, halt: bool = True,
                 audit: bool = True) -> bool:
        from ..validation.stages import effective_severity
        self.stage_results.append(res)
        active = [i for i in res.issues if not i.passed and not i.suppressed]
        n_err = sum(1 for i in active if effective_severity(i) == "ERROR")
        n_warn = sum(1 for i in active if effective_severity(i) == "WARN")
        n_info = sum(1 for i in active if effective_severity(i) == "INFO")
        n_supp = sum(1 for i in res.issues if i.suppressed)
        if audit:
            # R's .gate_on_validation() logs every gated stage; REPORT is not
            # gated there, so it is not logged here either.
            audit_event({"event": "validation_summary", "run_id": self.run_id,
                         "stage": stage, "n_total": len(res.issues),
                         "n_pass": sum(1 for i in res.issues if i.passed),
                         "n_error": n_err, "n_warn": n_warn, "n_info": n_info,
                         "n_suppressed": n_supp, "policy": self.policy})
        self.result.steps.append({
            "step": f"Validation: {stage}", "ok": n_err == 0,
            "detail": (f"{len(res.issues)} checks, {n_err} error(s), "
                       f"{n_warn} warning(s), {n_supp} suppressed"),
            "errors": [i.id for i in active if effective_severity(i) == "ERROR"][:10]})
        if n_err and self.policy == "stop" and halt:
            ids = [i.id for i in active if effective_severity(i) == "ERROR"]
            self.result.error = (f"[{stage}] {n_err} ERROR-level validation "
                                 f"failure(s): {', '.join(ids[:5])}. The run "
                                 "stopped before pricing (on_validation_error: "
                                 "stop). See reports/validation.md.")
            return False
        return True


def _write_validation(result: RunResult, run_dir: Path, stage_results: list,
                      quiet: bool = False):
    from ..validation import combine, write_validation_reports
    if not stage_results:
        return
    findings = combine(*stage_results)
    write_validation_reports(findings, run_dir / "reports")
    summary = findings.summary()
    result.validation = summary
    if quiet:
        return
    result.steps.append({
        "step": "Validation", "ok": summary["errors"] == 0,
        "detail": (f"{summary['checks']} checks, {summary['errors']} error(s), "
                   f"{summary['warnings']} warning(s), "
                   f"{summary['suppressed']} suppressed")})


def _halt(result: RunResult, run_dir: Path, run_id: str, input_dir,
          extract_date: str, stage_results: list, user, started: float,
          keep_ok: bool = False, state=None, applied=None) -> RunResult:
    """Stop at a gate: write what was found, then return without pricing."""
    _write_validation(result, run_dir, stage_results)
    try:
        _write_manifest(run_dir, run_id, input_dir, extract_date, result,
                        user=user, state=state)
    except Exception:
        pass
    result.ok = bool(keep_ok and not result.error)
    result.seconds = time.time() - started
    if applied is not None:
        result.overrides_applied = applied
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


def _build_stpd(static, config_dir, extract_date: str,
                model_id: str | None = None):
    """The PD term structures.

    Reads the model config from the run's own frozen copy when one is given,
    and otherwise from the package.

    The weights are left to ``build_stpd_from_static``, which resolves them
    SEPARATELY for the two rating scales -- the internal one from the non-oil
    GDP forecast, the external one from the GCC growth path. Computing one set
    here and passing it applied the internal weights to the external scale as
    well, which left the Investments and Banks and FIs curves out by up to
    0.012 against the reference run while every internal curve stayed exact.
    """
    model, inputs_yml = load_model_config(config_dir)
    if model is None or inputs_yml is None:
        return None
    return build_stpd_from_static(static, model, inputs_yml, extract_date,
                                  model_id=model_id)


def _config_extract_date(run_config):
    """config.yml's run.extract_date, when it is set and parses -- R's
    fallback for inputs that carry none, read as R's normalise_extract_date()
    reads it."""
    if not isinstance(run_config, dict):
        return None
    run = run_config.get("run")
    v = run.get("extract_date") if isinstance(run, dict) else None
    if v in (None, ""):
        return None
    from ..dates import normalise_extract_dates
    t = normalise_extract_dates([v]).iloc[0]
    return None if pd.isna(t) else pd.Timestamp(t)


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


def _file_sha256(path: Path) -> str | None:
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _csv_rows(path: Path) -> int | None:
    """Data rows in a CSV: its lines, less the header -- R's .csv_rows()."""
    try:
        with open(path, "rb") as fh:
            n = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))
        return max(0, n - 1)
    except OSError:
        return None


def _manifest_inputs(input_dir: Path) -> dict:
    """R's .manifest_inputs(): each expected file, what was found, its hash."""
    from .read_inputs import INPUT_SPECS, resolve_input_path
    out = {}
    for spec in INPUT_SPECS:
        cands = [spec.file] + list(getattr(spec, "file_alternatives", []) or [])
        try:
            found = resolve_input_path(Path(input_dir), spec)
        except Exception:
            found = None
        if found is None:
            out[spec.name] = {"file": spec.file, "file_found": None,
                              "candidates": cands, "exists": False,
                              "size_bytes": None, "sha256": None}
        else:
            found = Path(found)
            out[spec.name] = {"file": spec.file, "file_found": found.name,
                              "candidates": cands, "exists": True,
                              "size_bytes": found.stat().st_size,
                              "sha256": _file_sha256(found)}
    return out


def _manifest_outputs(out_dir: Path) -> dict:
    """R's .manifest_outputs(): every CSV written, with rows, size and hash."""
    if not out_dir.is_dir():
        return {}
    return {p.name: {"n_rows": _csv_rows(p), "size_bytes": p.stat().st_size,
                     "sha256": _file_sha256(p)}
            for p in sorted(out_dir.glob("*.csv"))}


def _write_manifest(run_dir: Path, run_id: str, input_dir: Path,
                    extract_date: str, result: RunResult,
                    user: str | None = None, state: PhaseState | None = None,
                    model_cfg: dict | None = None) -> None:
    """What produced this run, in the R engine's manifest schema.

    R's write_manifest(): ``run`` (who, when, where, code), ``snapshot`` (the
    config version, when one was used), ``run_metadata`` (type, purpose,
    portfolio date, config and calculator version), ``config``, ``inputs``
    (each file found and its hash) and ``outputs`` (each file written, its
    rows and hash). Both apps' run pages read it, so a run made by either
    engine lists the same way. The Python-only fields the earlier schema
    carried (engine version, steps, validation summary) are kept alongside.
    """
    import platform
    import socket
    import sys

    from .. import __version__
    root = (state.project_root if state is not None and state.project_root
            else run_dir.parent.parent)
    # The declared calculator version AND a fingerprint of the code that
    # actually ran. Either alone is insufficient: the version can claim what
    # the code is not, and a bare hash names nothing.
    try:
        from ..calculator_versions import calculator_version_for_run
        calculator = calculator_version_for_run(
            id=(state.calculator_version if state is not None else None) or None,
            root=root)
    except Exception:
        calculator = None
    try:
        from ..code_version import code_status
        code = code_status()
    except Exception:
        code = None

    if not user:
        user = _who()
    started_at = (state.started_at if state is not None
                  else datetime.now().astimezone())
    finished_at = datetime.now().astimezone()
    snap = (state.snapshot_meta if state is not None else None) or None
    if model_cfg is None and state is not None:
        try:
            model_cfg, _ = load_model_config(state.config_dir)
        except Exception:
            model_cfg = None
    # R records the RESOLVED model -- the one config.yml selected, with its
    # components -- not the whole registry.
    resolved_model = None
    if model_cfg:
        try:
            from .model_registry import resolve_model
            resolved_model = resolve_model(
                model_cfg, state.model_id if state is not None else None)
        except Exception:
            resolved_model = None

    manifest = {
        "schema_version": "1.0",
        "run": {
            "run_id": run_id,
            "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "finished_at": finished_at.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "python_version": f"Python {platform.python_version()}",
            "engine": f"ifrs9qdb (python) {__version__}",
            "hostname": socket.gethostname(),
            "user": user,
            "code_sha": (code or {}).get("sha"),
        },
        "snapshot": ({k: snap.get(k) for k in
                      ("label", "status", "created_at", "created_by", "parent",
                       "code_sha_at_creation")} if snap else None),
        "run_metadata": {
            "run_type": state.run_type if state is not None else None,
            "ecl_scenario": state.ecl_scenario if state is not None else "weighted",
            "run_purpose": state.run_purpose if state is not None else None,
            "portfolio_date": state.portfolio_date if state is not None else None,
            "config_version": (snap or {}).get("label"),
            "calculator_version": (calculator or {}).get("id")
            or (state.calculator_version if state is not None else None),
            "calculator_label": (calculator or {}).get("label"),
            "calculator_code_hash": (calculator or {}).get("code_hash"),
            "calculator_matches_registered":
                (calculator or {}).get("matches_registered"),
        },
        "config": {"model_config": resolved_model,
                   "run_config": (state.run_config if state is not None else None)
                   or {}},
        "inputs": _manifest_inputs(input_dir),
        "overrides": result.overrides_applied or {},
        "outputs": _manifest_outputs(run_dir / "Output"),
        "messages": [f"{st.get('step')}: {st.get('detail', '')}"
                     for st in result.steps],
        # ---- the earlier Python schema, kept for its readers ------------
        "run_id": run_id,
        # Who ran it. The approval helpers read this to enforce separation of
        # duties, so a run with no maker recorded cannot be checked against
        # its approver.
        "user": user,
        "created": finished_at.isoformat(timespec="seconds"),
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
    # reports/, which is where the R engine writes it and therefore where
    # anything reading a run of either engine's making will look.
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                      default=_json_default))


def _json_default(v):
    try:
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
    except Exception:
        pass
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.isoformat()
    return str(v)
