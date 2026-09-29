"""Pricing readiness: will every row get a number, and if not, why not.

The input checks ask whether each file is well formed. This asks the question
that matters to whoever signs the provision: FED THESE FILES, DOES EVERY
CONTRACT COME OUT WITH AN ECL, AND IS THAT ECL BUILT FROM COMPLETE INPUTS?

It runs on the Output folder -- the files LIC reads -- after the ETL has
written them and BEFORE the book is priced, and it reuses the engine's own
lookups (``etl.report.engine_frame`` and its pricing context), so a contract it
calls priceable is exactly one the engine then prices.

WHAT A GAP DOES, FIELD BY FIELD
    Every row below is a check here (READY_*), and each one says what LIC does
    and what the R and Python engines do, because they are not always the same.

    Field                       Gap                      Effect
    --------------------------  -----------------------  -------------------------------
    ContractId                  repeated                 the contract is priced twice
    Rating -> Ratings.Hierarchy blank / not on the scale no PD curve: ECL BLANK (Stage 1/2)
    (Portfolio, bucket) -> StPD no curve                 ECL BLANK (Stage 1/2)
    EIR                         blank                    ECL BLANK (Stage 1/2)
    EIR                         < 0.1% or > 30%          discounting wrong (units?)
    OnBalance                   blank                    priced as a zero exposure
    OnBalance                   negative                 a NEGATIVE provision
    Allocation -> Collateral    record missing           LIC: NaN coverage, whole
                                                         contract's ECL BLANK;
                                                         R/Python: allocation ignored
    Collateral type             not in CollateralType    no haircut: no benefit (R/Py)
    Collateral value            0 / blank, type would    benefit lost, LGD rises
                                give a benefit
    Allocation share            blank or outside [0,1]   benefit lost / overstated
    MaturityDate                blank / unparseable      horizon floored to 3 months
    MaturityDate                on/before the extract    3-month bullet (as LIC)
    Schedule (LTPO) curve       stops > 12m before       lifetime ECL understated --
                                maturity, balance left   the two-digit-year wrap
    CustomerStagingFlag         no row for customer      staged on DPD alone
    PastDueDays                 blank                    treated as not past due
    PaymentTypeId               no fallback rule         EAD held flat (bullet)
    AccountType                 not mapped               priced as Business Finance

The result is a per-contract table (reports/readiness.csv), a row funnel from
the raw extracts to the priced book, and the READY_* findings, which gate the
run like any other validation stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .framework import Severity, Validator

__all__ = ["Readiness", "assess_readiness", "READY_STAGE_VALIDATORS",
           "REPORT_STAGE_VALIDATORS", "write_readiness_reports",
           "READINESS_REASONS", "READINESS_REMEDIATION",
           "check_report_against_readiness"]

# code -> (validator id, severity, outcome it forces, what happens)
READINESS_REASONS: dict[str, tuple[str, str, str, str]] = {
    "DUPLICATE_CONTRACT": (
        "READY_contract_unique", Severity.ERROR, "Priced - check",
        "the contract appears more than once in the account masters, so it "
        "is priced (and counted) more than once"),
    "NO_PD_CURVE": (
        "READY_pd_curve", Severity.ERROR, "No ECL",
        "no PD curve resolves for the rating and portfolio, so the ECL is "
        "BLANK in LIC and in both engines"),
    "EIR_MISSING": (
        "READY_eir_present", Severity.ERROR, "No ECL",
        "no EIR to discount with, so the ECL is BLANK"),
    "EIR_RANGE": (
        "READY_eir_range", Severity.WARN, "Priced - check",
        "the EIR is outside 0.1%-30%, so the discounting is wrong (usually "
        "a rate in the wrong units)"),
    "EXPOSURE_MISSING": (
        "READY_exposure_present", Severity.ERROR, "Priced - check",
        "OnBalance is blank, so the contract is priced as a zero exposure"),
    "EXPOSURE_NEGATIVE": (
        "READY_exposure_nonneg", Severity.WARN, "Priced - check",
        "OnBalance is negative, which produces a negative provision"),
    "ORPHAN_ALLOCATION": (
        "READY_collateral_allocation_links", Severity.ERROR, "Blank in LIC",
        "an allocation points at a collateral record that is not in "
        "Collateral.csv: LIC returns NaN coverage and a BLANK ECL for the "
        "whole contract; R and Python ignore that allocation"),
    "COLLATERAL_TYPE_UNMAPPED": (
        "READY_collateral_type_mapped", Severity.WARN, "Priced - check",
        "allocated collateral has a type with no haircut in "
        "CollateralType.csv: R and Python give it no benefit; how LIC treats "
        "an unmapped type is not documented, and if it behaves as it does for "
        "a missing collateral record (NaN coverage) these contracts would come "
        "out BLANK in LIC"),
    "COLLATERAL_VALUE_MISSING": (
        "READY_collateral_value_present", Severity.WARN, "Priced - check",
        "allocated collateral of a type that WOULD reduce the loss has no "
        "value, so the benefit is lost and the LGD rises"),
    "ALLOCATION_SHARE_INVALID": (
        "READY_allocation_share_valid", Severity.WARN, "Priced - check",
        "an allocation share is blank or outside 0-100%, so the benefit is "
        "lost or overstated"),
    "MATURITY_MISSING": (
        "READY_maturity_present", Severity.WARN, "Priced - check",
        "no usable maturity date, so the horizon is floored to 3 months "
        "(Stage 2 lifetime loss understated)"),
    "MATURITY_PAST": (
        "READY_maturity_after_extract", Severity.INFO, "Priced",
        "the maturity is on or before the extract date (the ETL extends "
        "earlier ones by a year), so it is priced as a 3-month bullet, as LIC "
        "does"),
    "EAD_CURVE_TRUNCATED": (
        "READY_ead_curve_complete", Severity.ERROR, "Priced - check",
        "the repayment-schedule EAD curve stops more than 12 months before "
        "maturity with balance still outstanding, so lifetime ECL is "
        "understated (the signature of schedule dates read as 19xx)"),
    "NO_STAGING_FLAGS": (
        "READY_staging_flags_present", Severity.WARN, "Priced - check",
        "the customer has no CustomerStagingFlag row, so watchlist, default "
        "and local flags are unknown and the stage rests on DPD alone"),
    "DPD_MISSING": (
        "READY_dpd_present", Severity.WARN, "Priced - check",
        "PastDueDays is blank, so the contract is treated as not past due"),
    "ZERO_EXPOSURE_SCHEDULE": (
        "READY_zero_exposure_schedule", Severity.INFO, "Priced",
        "OnBalance is zero but the schedule carries exposure, so the ECL comes "
        "from the schedule and is not capped"),
    "PAYMENT_TYPE_DEFAULT": (
        "READY_payment_type_rule", Severity.INFO, "Priced",
        "no repayment schedule and no fallback rule for the portfolio and "
        "payment type, so the EAD is held flat (bullet)"),
    "PORTFOLIO_UNMAPPED": (
        "READY_portfolio_mapped", Severity.WARN, "Priced - check",
        "the account type is not in product_portfolio_mapping, so the "
        "contract is priced as Business Finance"),
}

_OUTCOME_ORDER = {"No ECL": 0, "Blank in LIC": 1, "Priced - check": 2, "Priced": 3}

# A schedule curve counts as truncated when it stops this many months short of
# maturity with at least this share of its first-month balance outstanding.
TRUNCATION_MONTHS = 12
TRUNCATION_BALANCE_SHARE = 0.05
EIR_MIN, EIR_MAX = 0.001, 0.30


@dataclass
class Readiness:
    """The assessment: per contract, the funnel, and what it adds up to."""
    ok: bool
    contracts: pd.DataFrame = field(default_factory=pd.DataFrame)
    funnel: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocations: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def with_reason(self, code: str) -> pd.DataFrame:
        if self.contracts.empty:
            return self.contracts
        return self.contracts[self.contracts["reasons"].str.contains(
            rf"\b{code}\b", regex=True)]

    def summary(self) -> dict:
        c = self.contracts
        if c.empty:
            return {"contracts": 0}
        out = {"contracts": int(len(c)),
               "exposure": float(pd.to_numeric(c["on_balance"], errors="coerce").sum())}
        for k in _OUTCOME_ORDER:
            sub = c[c["outcome"] == k]
            out[k] = {"contracts": int(len(sub)),
                      "exposure": float(pd.to_numeric(sub["on_balance"],
                                                      errors="coerce").sum())}
        return out


def _num(x):
    return pd.to_numeric(pd.Series(x), errors="coerce")


def _na(v) -> str:
    """A value as R prints it in a message: a missing one is NA."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return str(v)


def assess_readiness(run_or_out_dir, static=None, model_cfg=None, src=None,
                     dpd_stage2_threshold: float = 60) -> Readiness:
    """Assess a written Output folder before it is priced.

    ``src`` (the raw InputSet, optional) adds the raw-extract end of the row
    funnel; without it the funnel starts at the Output files.
    """
    from ..etl.report import (_Ctx, _chr, _col, _num as _n, _portfolio_map,
                              engine_frame, read_output_csv)
    from ..engine import resolve_ead_shape

    d = Path(run_or_out_dir)
    out_dir = d / "Output" if (d / "Output").is_dir() else d
    if static is None:
        try:
            from ..etl.static_ref import load_static_reference
            frozen = d / "config_used" / "static"
            static = load_static_reference(frozen if frozen.is_dir() else None)
        except Exception:
            static = None
    if model_cfg is None:
        try:
            from ..etl.pipeline import load_model_config
            frozen = d / "config_used" / "config"
            model_cfg, _ = load_model_config(frozen if frozen.is_dir() else None)
        except Exception:
            model_cfg = None

    need = ["AccountMaster_1.csv", "Ratings.csv", "StPD.csv"]
    missing = [f for f in need if not (out_dir / f).is_file()]
    if missing:
        return Readiness(ok=False, missing=missing)

    pmap = _portfolio_map(static)
    ctx = _Ctx(out_dir, model_cfg=model_cfg)
    books = [b for b in (engine_frame(out_dir, "_1", pmap, dpd_stage2_threshold),
                         engine_frame(out_dir, "_2", pmap, dpd_stage2_threshold))
             if b is not None]
    con = pd.concat(books, ignore_index=True)
    n = len(con)
    reasons: list[list[str]] = [[] for _ in range(n)]
    notes: list[dict] = [{} for _ in range(n)]

    def add(mask, code, note=None):
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            reasons[i].append(code)
            if note is not None:
                v = note(i) if callable(note) else note
                if v:
                    notes[i][code] = v

    stage = con["stage"].to_numpy()
    # Stage 3 is booked at the balance (or at zero): its number does not
    # depend on the PD curve, the EIR, the collateral, the maturity or the
    # EAD curve, so gaps in those are pricing gaps only for Stage 1 and 2.
    # The INPUT checks still report the underlying data defect for every row.
    live = stage < 3

    # ---- identity --------------------------------------------------------
    dup = con["contract"].notna() & con["contract"].duplicated(keep=False)
    add(dup, "DUPLICATE_CONTRACT")

    # ---- PD curve -------------------------------------------------------
    rt_code = np.where(con["rating_type"] == "External", "2", "1")
    bucket = [ctx.rat2bucket.get(f"{t} ::> {r}") for t, r in zip(rt_code, con["rating"])]
    con["pd_bucket"] = bucket
    has_curve = np.array([ctx.pd_curve(p, r, t) is not None for p, r, t in
                          zip(con["portfolio"], con["rating"], con["rating_type"])])
    con["has_pd_curve"] = has_curve

    def pd_note(i):
        r = con["rating"].iat[i]
        if r is None or str(r).strip() == "":
            return "rating is blank"
        if bucket[i] is None:
            return f"rating '{r}' is not on the {con['rating_type'].iat[i]} scale"
        return (f"no StPD curve for portfolio '{con['portfolio'].iat[i]}' "
                f"bucket {bucket[i]}")
    add(live & ~has_curve, "NO_PD_CURVE", pd_note)

    # ---- EIR --------------------------------------------------------------
    eir = _num(con["eir"])
    add(live & eir.isna().to_numpy(), "EIR_MISSING")
    add(live & (eir.notna() & ((eir < EIR_MIN) | (eir > EIR_MAX))).to_numpy(),
        "EIR_RANGE", lambda i: f"EIR {eir.iat[i]:.6g}")

    # ---- exposure -----------------------------------------------------------
    onb = _num(con["on_balance"])
    add(onb.isna(), "EXPOSURE_MISSING")
    add(onb < 0, "EXPOSURE_NEGATIVE", lambda i: f"OnBalance {onb.iat[i]:,.2f}")

    # ---- collateral links -----------------------------------------------------
    alloc = read_output_csv(out_dir, "AccountCollateralAllocation.csv")
    coll = read_output_csv(out_dir, "Collateral.csv")
    ctype = read_output_csv(out_dir, "CollateralType.csv")
    alloc_stats = {}
    n_alloc = np.zeros(n, dtype=int)
    con["allocations"] = 0
    if alloc is not None and len(alloc):
        a = pd.DataFrame({"contract": _chr(_col(alloc, "ContractId")),
                          "coll": _chr(_col(alloc, "CollateralId")),
                          "share": _n(_col(alloc, "AllocationPercentage"))})
        cids = set(_chr(_col(coll, "CollateralId")).dropna()) if coll is not None else set()
        c_type = (pd.Series(_chr(_col(coll, "CollateralTypeId")).to_numpy(),
                            index=_chr(_col(coll, "CollateralId")).to_numpy())
                  if coll is not None else pd.Series(dtype=object))
        c_type = c_type[~c_type.index.duplicated(keep="first")]
        c_val = (pd.Series(_n(_col(coll, "CollateralValue")).to_numpy(),
                           index=_chr(_col(coll, "CollateralId")).to_numpy())
                 if coll is not None else pd.Series(dtype=float))
        c_val = c_val[~c_val.index.duplicated(keep="first")]
        hc = (pd.Series(_n(_col(ctype, "HaircutGeneral")).to_numpy(),
                        index=_chr(_col(ctype, "CollateralTypeId")).to_numpy())
              if ctype is not None else pd.Series(dtype=float))
        hc = hc[~hc.index.duplicated(keep="first")]
        a["orphan"] = a["coll"].notna() & ~a["coll"].isin(cids)
        a["type"] = a["coll"].map(c_type)
        a["value"] = a["coll"].map(c_val).astype(float)
        a["haircut"] = a["type"].map(hc).astype(float)
        a["type_unmapped"] = ~a["orphan"] & a["type"].notna() & a["haircut"].isna()
        a["value_missing"] = (~a["orphan"] & ~a["type_unmapped"]
                              & (a["haircut"] < 1) & ~(a["value"] > 0))
        a["share_bad"] = a["share"].isna() | (a["share"] < 0) | (a["share"] > 1.0001)
        blank_contract = a["contract"].isna()
        alloc_stats = {
            "rows": int(len(a)),
            "blank_contract_rows": int(blank_contract.sum()),
            "blank_contract_collateral_n": int(a.loc[blank_contract, "coll"]
                                               .dropna().nunique()),
            "blank_contract_collateral": sorted(a.loc[blank_contract, "coll"]
                                                .dropna().unique().tolist())[:50],
            "orphan_rows": int(a["orphan"].sum()),
            "orphan_collateral": sorted(a.loc[a["orphan"], "coll"].dropna()
                                        .unique().tolist())[:50],
            "unknown_contract_rows": int((a["contract"].notna()
                                          & ~a["contract"].isin(set(con["contract"]))).sum()),
        }
        g = a[a["contract"].notna()].groupby("contract")
        per = pd.DataFrame({
            "allocations": g.size(),
            "orphan": g["orphan"].sum(),
            "type_unmapped": g["type_unmapped"].sum(),
            "value_missing": g["value_missing"].sum(),
            "share_bad": g["share_bad"].sum(),
        })
        per_c = per.reindex(con["contract"]).fillna(0).astype(int)
        n_alloc = per_c["allocations"].to_numpy()
        con["allocations"] = n_alloc
        orphan_ids = a[a["orphan"]].groupby("contract")["coll"].apply(
            lambda s: ", ".join(sorted(s.dropna().unique())[:5]))
        add(live & (per_c["orphan"].to_numpy() > 0), "ORPHAN_ALLOCATION",
            lambda i: "missing collateral " + str(orphan_ids.get(con["contract"].iat[i], "")))
        tu = a[a["type_unmapped"]]
        types = tu.groupby("contract")["type"].apply(
            lambda s: ", ".join(sorted(s.dropna().unique())))
        valued = tu.assign(v=tu["value"] > 0).groupby("contract")["v"].sum()
        add(live & (per_c["type_unmapped"].to_numpy() > 0), "COLLATERAL_TYPE_UNMAPPED",
            lambda i: ("type " + str(types.get(con["contract"].iat[i], "")) + ", "
                       + str(int(valued.get(con["contract"].iat[i], 0)))
                       + " of them with a positive value"))
        alloc_stats["type_unmapped_types"] = sorted(tu["type"].dropna().unique().tolist())
        alloc_stats["type_unmapped_valued_rows"] = int((tu["value"] > 0).sum())
        add(live & (per_c["value_missing"].to_numpy() > 0), "COLLATERAL_VALUE_MISSING")
        add(live & (per_c["share_bad"].to_numpy() > 0), "ALLOCATION_SHARE_INVALID")

    # ---- maturity and the EAD curve ------------------------------------------
    raw_m = _num(con["raw_months"])
    add(live & con["d_maturity"].isna().to_numpy(), "MATURITY_MISSING")
    past = (con["d_maturity"].notna() & con["d_extract"].notna()
            & (con["d_maturity"] <= con["d_extract"])).to_numpy()
    add(live & past, "MATURITY_PAST")

    sched_len = np.array([len(ctx.schedules.get(c, [])) if c is not None else 0
                          for c in con["contract"]])
    first = np.array([ctx.schedules[c][0] if c in ctx.schedules and len(ctx.schedules[c])
                      else np.nan for c in con["contract"]])
    last = np.array([ctx.schedules[c][-1] if c in ctx.schedules and len(ctx.schedules[c])
                     else np.nan for c in con["contract"]])
    con["schedule_months"] = sched_len
    share_left = np.where(np.abs(first) > 0, last / np.where(first == 0, np.nan, first), np.nan)
    # What the contract's stage actually prices over: 12 months for Stage 1,
    # to maturity for Stage 2. A curve cut short beyond the horizon in use
    # does not move this run's number (INPUT_RS_dates_plausible still lists
    # every affected schedule).
    months_needed = np.where(stage == 1, np.minimum(12, raw_m.fillna(0).to_numpy()),
                             raw_m.fillna(0).to_numpy())
    slack = np.where(stage == 1, 0, TRUNCATION_MONTHS)
    short = months_needed - sched_len
    trunc = (live & (sched_len > 0) & (short > slack)
             & (share_left > TRUNCATION_BALANCE_SHARE))
    add(trunc, "EAD_CURVE_TRUNCATED",
        lambda i: (f"curve ends at month {sched_len[i]}, maturity is "
                   f"{int(raw_m.iat[i])} months away, {share_left[i]:.0%} of the "
                   "balance still outstanding"))
    add(live & (onb == 0).to_numpy() & (sched_len > 0) & (np.nan_to_num(first) > 0),
        "ZERO_EXPOSURE_SCHEDULE")

    # ---- staging inputs -----------------------------------------------------
    add(~con["has_flags_row"].fillna(False).to_numpy(), "NO_STAGING_FLAGS")
    add(_num(con["dpd"]).isna().to_numpy(), "DPD_MISSING")

    # ---- EAD fallback shape and the portfolio mapping -----------------------------
    rules, default = ctx.rules
    matched_rule = np.array([
        any((rp is not None and rp == pf and rt == str(pt)) or
            (rp is None and rt == str(pt)) for rp, rt, _ in rules)
        for pf, pt in zip(con["portfolio"], con["payment_type"])])
    add(live & (sched_len == 0) & ~matched_rule & (raw_m.fillna(0).to_numpy() > 3),
        "PAYMENT_TYPE_DEFAULT",
        lambda i: f"{con['portfolio'].iat[i]} / payment type {_na(con['payment_type'].iat[i])}")
    unmapped = (con["suffix"] == "_1") & ~con["account_type"].isin(set(pmap))
    add(unmapped.to_numpy(), "PORTFOLIO_UNMAPPED",
        lambda i: f"account type '{_na(con['account_type'].iat[i])}'")

    # ---- outcome ----------------------------------------------------------------
    outcome, sev = [], []
    for i in range(n):
        o = "Priced"
        for code in reasons[i]:
            forced = READINESS_REASONS[code][2]
            if _OUTCOME_ORDER[forced] < _OUTCOME_ORDER[o]:
                o = forced
        outcome.append(o)
    con["outcome"] = outcome
    con["reasons"] = [" ".join(r) for r in reasons]
    con["detail"] = ["; ".join(f"{k}: {v}" for k, v in x.items()) for x in notes]
    con["_notes"] = notes
    con["engine_prices"] = ~con["outcome"].eq("No ECL")

    table = pd.DataFrame({
        "ContractId": con["contract"], "Book": con["book"],
        "CustomerId": con["customer"], "AccountType": con["account_type"],
        "Portfolio": con["portfolio"], "Rating": con["rating"],
        "RatingType": con["rating_type"], "PdBucket": con["pd_bucket"],
        "PredictedStage": con["stage"], "OnBalance": onb,
        "EIR": eir, "MaturityDate": con["d_maturity"].dt.strftime("%Y-%m-%d"),
        "MonthsToMaturity": raw_m, "ScheduleMonths": con["schedule_months"],
        "Allocations": con["allocations"],
        "Outcome": con["outcome"], "Reasons": con["reasons"],
        "Detail": con["detail"],
    })
    con = con.assign(on_balance=onb)

    funnel = _funnel(out_dir, con, alloc_stats, src)
    return Readiness(ok=True, contracts=con, funnel=funnel,
                     allocations=alloc_stats, table=table)


def _funnel(out_dir: Path, con: pd.DataFrame, alloc_stats: dict, src) -> pd.DataFrame:
    """Rows from the raw extracts to the priced book, file by file.

    ``raw_rows`` is what the extract held, ``junk_rows_stripped`` the SQL*Plus
    repeated headers, footers and blank rows removed on reading, and ``lost``
    what is left over once those are accounted for: rows the ETL did not carry
    into the file LIC reads.
    """
    from ..etl.report import read_output_csv
    if src is not None:
        from .schema import canonicalise
        src = canonicalise(src)
    log = dict(getattr(src, "strip_log", None) or {}) if src is not None else {}
    rows = []
    for key, out, label in [
            ("AccountMaster", "AccountMaster_1.csv", "Lending contracts"),
            ("AccountMasterInvestments", "AccountMaster_2.csv", "Investment accounts"),
            ("Collateral", "Collateral.csv", "Collateral records"),
            ("AccountCollateralAllocation", "AccountCollateralAllocation.csv",
             "Collateral allocations"),
            ("CustomerStagingFlag", "CustomerStagingFlag_1.csv", "Staging-flag rows"),
            ("RepaymentSchedule", None, "Repayment-schedule rows")]:
        o = read_output_csv(out_dir, out) if out else None
        clean = stripped = None
        if src is not None:
            t = src.tables.get(key) if hasattr(src, "tables") else None
            if t is not None:
                clean = int(len(t))
                stripped = sum(int(v) for k, v in log.items()
                               if str(k).split(".")[0] == key)
        out_n = None if o is None else int(len(o))
        rows.append({"step": label,
                     "raw_rows": None if clean is None else clean + stripped,
                     "junk_rows_stripped": stripped,
                     "output_rows": out_n,
                     "lost": (clean - out_n) if (clean is not None and out_n is not None)
                     else None})
    f = pd.DataFrame(rows)

    onb = pd.to_numeric(con["on_balance"], errors="coerce")
    for label, mask in [
            ("Contracts in the Output (both books)", pd.Series(True, index=con.index)),
            ("  will be priced", con["outcome"] != "No ECL"),
            ("    of which Stage 3 (priced at the balance)",
             (con["stage"] == 3) & (con["outcome"] != "No ECL")),
            ("    of which with a gap to review", con["outcome"] == "Priced - check"),
            ("  will be BLANK in LIC (engine prices them)", con["outcome"] == "Blank in LIC"),
            ("  will get NO ECL", con["outcome"] == "No ECL")]:
        f = pd.concat([f, pd.DataFrame([{
            "step": label, "raw_rows": None, "junk_rows_stripped": None,
            "output_rows": int(mask.sum()), "lost": None,
            "exposure": float(onb[mask].sum())}])], ignore_index=True)
    if alloc_stats:
        f = pd.concat([f, pd.DataFrame([
            {"step": "Allocations with a blank ContractId (collateral applied to nothing)",
             "output_rows": alloc_stats.get("blank_contract_rows", 0)},
            {"step": "Allocations to a collateral record that does not exist",
             "output_rows": alloc_stats.get("orphan_rows", 0)},
            {"step": "Allocations to a contract not in the account masters",
             "output_rows": alloc_stats.get("unknown_contract_rows", 0)}])],
            ignore_index=True)
    return f


# --------------------------------------------------------------- validators ---
def _reason_check(code: str):
    vid, sev, outcome, text = READINESS_REASONS[code]

    def check(readiness=None):
        if readiness is None or not readiness.ok:
            return {"passed": True}
        sub = readiness.with_reason(code)
        if sub.empty:
            return {"passed": True}
        expo = float(pd.to_numeric(sub["on_balance"], errors="coerce").sum())
        stage_mix = sub["stage"].value_counts().sort_index()
        mix = ", ".join(f"Stage {int(k)}: {int(v)}" for k, v in stage_mix.items())
        ex = sub["contract"].astype(str).head(10).tolist()
        eg = [(c, nts.get(code)) for c, nts in zip(sub["contract"], sub["_notes"])
              if nts.get(code)]
        msg = (f"{len(sub):,} contract(s), exposure {expo:,.0f} ({mix}): {text}."
               + (f" e.g. {eg[0][0]}: {eg[0][1]}" if eg else ""))
        return {"passed": False, "count": int(len(sub)), "detail": msg,
                "examples": ex}
    return check


def _v_funnel(readiness=None):
    if readiness is None or not readiness.ok:
        return {"passed": True}
    f = readiness.funnel
    lost = f[pd.to_numeric(f["lost"], errors="coerce").fillna(0) > 0]
    if lost.empty:
        return {"passed": True}
    parts = [f"{r['step']}: {int(r['raw_rows']):,} raw, "
             f"{int(r['junk_rows_stripped']):,} junk rows stripped -> "
             f"{int(r['output_rows']):,} written ({int(r['lost']):,} not carried)"
             for _, r in lost.iterrows()]
    return {"passed": False, "count": int(pd.to_numeric(lost["lost"]).sum()),
            "detail": "; ".join(parts)}


def _v_blank_contract_allocations(readiness=None):
    if readiness is None or not readiness.ok:
        return {"passed": True}
    n = int(readiness.allocations.get("blank_contract_rows", 0))
    if n == 0:
        return {"passed": True}
    ids = readiness.allocations.get("blank_contract_collateral", [])
    k = int(readiness.allocations.get("blank_contract_collateral_n", len(ids)))
    return {"passed": False, "count": n,
            "detail": (f"{n:,} allocation row(s) have a blank ContractId, so "
                       f"{k:,} collateral record(s) are allocated to no "
                       "contract and give no benefit to anything"),
            "examples": [str(x) for x in ids[:10]]}


def _validators() -> list[Validator]:
    out = [Validator(
        "READY_rows_carried", Severity.ERROR,
        "Every raw row reaches the files LIC reads",
        _v_funnel, context="Output",
        rationale=("A row the ETL drops is a contract that is never priced and "
                   "never reported: nothing downstream can notice it is gone."),
        remediation=("Compare the raw extract with the Output file named in the "
                     "message and find which transformation step dropped rows."),
        tags=("pre_run", "readiness"))]
    for code, (vid, sev, outcome, text) in READINESS_REASONS.items():
        out.append(Validator(
            vid, sev,
            {"DUPLICATE_CONTRACT": "Every contract appears once across the account masters",
             "NO_PD_CURVE": "Every Stage 1/2 contract resolves a PD curve",
             "EIR_MISSING": "Every Stage 1/2 contract has an EIR",
             "EIR_RANGE": "Every EIR is between 0.1% and 30%",
             "EXPOSURE_MISSING": "Every contract has an OnBalance",
             "EXPOSURE_NEGATIVE": "No contract has a negative OnBalance",
             "ORPHAN_ALLOCATION": "Every allocation of a priced contract finds its collateral record",
             "COLLATERAL_TYPE_UNMAPPED": "Every allocated collateral type has a haircut",
             "COLLATERAL_VALUE_MISSING": "Allocated collateral that could reduce the loss has a value",
             "ALLOCATION_SHARE_INVALID": "Every allocation share is present and within 0-100%",
             "MATURITY_MISSING": "Every contract has a usable maturity date",
             "MATURITY_PAST": "Contracts past maturity are identified",
             "EAD_CURVE_TRUNCATED": "Every repayment-schedule EAD curve runs to maturity",
             "NO_STAGING_FLAGS": "Every customer has a staging-flag row",
             "DPD_MISSING": "Every contract has PastDueDays",
             "ZERO_EXPOSURE_SCHEDULE": "Zero-balance contracts carrying a schedule are identified",
             "PAYMENT_TYPE_DEFAULT": "Every unscheduled contract has an EAD fallback rule",
             "PORTFOLIO_UNMAPPED": "Every account type maps to a portfolio",
             }[code],
            _reason_check(code), context="Output",
            rationale=f"If not: {text}.",
            remediation=_REMEDIATION[code],
            tags=("pre_run", "readiness")))
    out.append(Validator(
        "READY_allocation_contract_present", Severity.WARN,
        "Every allocation row names a contract",
        _v_blank_contract_allocations, context="AccountCollateralAllocation",
        rationale=("An allocation with no contract applies its collateral to "
                   "nothing: the value is in the file and reduces no loss."),
        remediation=("Fill the ContractId at source, or remove the rows if the "
                     "collateral genuinely secures nothing."),
        tags=("pre_run", "readiness")))
    return out


_REMEDIATION = {
    "DUPLICATE_CONTRACT": "De-duplicate the account master at source; one row per contract.",
    "NO_PD_CURVE": ("Add the rating to master_rating_scale.csv (or correct it at "
                    "source), and check the portfolio mapping for the account type."),
    "EIR_MISSING": "Supply the EIR at source; the ETL's account-type fallback needs at least one rate per type.",
    "EIR_RANGE": "Check the rate's units at source: the extract carries EIR in percent (4.2 = 4.2%).",
    "EXPOSURE_MISSING": "Supply ONBALANCE at source.",
    "EXPOSURE_NEGATIVE": "Confirm the credit balance at source; a negative exposure should not be provisioned.",
    "ORPHAN_ALLOCATION": ("Add the missing collateral record to the Collateral extract, or "
                          "remove the allocation. LIC will not price the contract until this is fixed."),
    "COLLATERAL_TYPE_UNMAPPED": "Add the type to collateral_types.csv with its QCB haircut.",
    "COLLATERAL_VALUE_MISSING": "Send the collateral ids to the collateral unit to fix the appraisal value.",
    "ALLOCATION_SHARE_INVALID": "Correct the allocation percentage at source.",
    "MATURITY_MISSING": "Supply MATURITYDATE at source.",
    "MATURITY_PAST": "None needed; listed so expired facilities are visible.",
    "EAD_CURVE_TRUNCATED": ("Fix the repayment-schedule dates at source (two-digit years "
                            "read as 19xx); the listed contracts' schedules end early."),
    "NO_STAGING_FLAGS": "Add the customer to the CustomerStagingFlag extract.",
    "DPD_MISSING": "Supply PASTDUEDAYS at source.",
    "ZERO_EXPOSURE_SCHEDULE": "Confirm whether the facility is closed; if so, drop its schedule.",
    "PAYMENT_TYPE_DEFAULT": "Add a rule for the portfolio and payment type under ecl.ead_fallback in model.yml.",
    "PORTFOLIO_UNMAPPED": "Add the account type to product_portfolio_mapping.csv.",
}

# What to do about each reason, for the pages that list them.
READINESS_REMEDIATION = _REMEDIATION

READY_STAGE_VALIDATORS: list[Validator] = _validators()


# ------------------------------------------------------ after pricing ---------
def check_report_against_readiness(report: pd.DataFrame,
                                   readiness: Readiness | None) -> dict:
    """What the priced report shows, against what readiness predicted."""
    ecl = pd.to_numeric(report["Cla Amount Onbal"], errors="coerce")
    ids = report["Contract Id"].astype(str)
    out = {"rows": int(len(report)), "blank": int(ecl.isna().sum()),
           "blank_ids": ids[ecl.isna()].tolist()}
    if readiness is not None and readiness.ok:
        con = readiness.contracts
        expected = set(con.loc[con["outcome"] == "No ECL", "contract"].astype(str))
        blank = set(out["blank_ids"])
        out["unexpected_blank"] = sorted(blank - expected)
        out["expected_but_priced"] = sorted(expected - blank)
        want = set(con["contract"].astype(str))
        out["missing_rows"] = sorted(want - set(ids))
        out["extra_rows"] = sorted(set(ids) - want)
    return out


def _v_report_rows(report_check=None):
    if report_check is None:
        return {"passed": True}
    miss = report_check.get("missing_rows", [])
    extra = report_check.get("extra_rows", [])
    if not miss and not extra:
        return {"passed": True}
    return {"passed": False, "count": len(miss) + len(extra),
            "detail": (f"{len(miss)} contract(s) in the Output have no report row; "
                       f"{len(extra)} report row(s) match no Output contract"),
            "examples": (miss + extra)[:10]}


def _v_report_ecl(report_check=None):
    if report_check is None:
        return {"passed": True}
    n = report_check.get("blank", 0)
    if n == 0:
        return {"passed": True}
    unexpected = report_check.get("unexpected_blank", [])
    return {"passed": False, "count": n,
            "detail": (f"{n:,} report row(s) have no ECL; "
                       f"{len(unexpected)} of them were NOT predicted by the "
                       "readiness check" + (" (investigate)" if unexpected else
                                            " (all explained by READY_* findings)")),
            "examples": (unexpected or report_check.get("blank_ids", []))[:10]}


REPORT_STAGE_VALIDATORS: list[Validator] = [
    Validator("REPORT_rows_complete", Severity.ERROR,
              "Every contract in the Output has exactly one report row",
              _v_report_rows, context="FinalEclReport",
              rationale="A contract with no row is a provision nobody sees.",
              remediation="Re-run the report; if it persists, the account masters "
                          "and the report disagree on contract ids.",
              tags=("post_run",)),
    Validator("REPORT_ecl_populated", Severity.ERROR,
              "Every report row carries an ECL",
              _v_report_ecl, context="FinalEclReport",
              rationale=("A blank ECL is a contract the provision silently leaves "
                         "out. Blanks the readiness check predicted are explained "
                         "there; any other blank is a defect."),
              remediation="Resolve the READY_* findings listed for these contracts.",
              tags=("post_run",)),
]


# ------------------------------------------------------------- writing --------
def write_readiness_reports(readiness: Readiness, reports_dir) -> dict:
    """reports/readiness.csv (per contract), readiness_funnel.csv, readiness.md."""
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    if not readiness.ok:
        (d / "readiness.md").write_text(
            "# Pricing readiness\n\nNot assessed: missing "
            + ", ".join(readiness.missing) + "\n", encoding="utf-8")
        return {"md": d / "readiness.md"}
    t = readiness.table.copy()
    t["_o"] = t["Outcome"].map(_OUTCOME_ORDER)
    t = t.sort_values(["_o", "ContractId"], kind="mergesort").drop(columns="_o")
    t.to_csv(d / "readiness.csv", index=False)
    readiness.funnel.to_csv(d / "readiness_funnel.csv", index=False)
    paths["csv"] = d / "readiness.csv"
    paths["funnel"] = d / "readiness_funnel.csv"

    s = readiness.summary()
    lines = ["# Pricing readiness", "",
             f"{s['contracts']:,} contracts, exposure {s['exposure']:,.0f}.", "",
             "| Outcome | Contracts | Exposure |", "|---|---:|---:|"]
    for k in _OUTCOME_ORDER:
        lines.append(f"| {k} | {s[k]['contracts']:,} | {s[k]['exposure']:,.0f} |")
    lines += ["", "## Why", ""]
    c = readiness.contracts
    for code, (vid, sev, outcome, text) in READINESS_REASONS.items():
        sub = readiness.with_reason(code)
        if sub.empty:
            continue
        expo = float(pd.to_numeric(sub["on_balance"], errors="coerce").sum())
        lines.append(f"- **[{sev}] {vid}** - {len(sub):,} contract(s), exposure "
                     f"{expo:,.0f}: {text}.")
    lines += ["", "## Row funnel", "", "| Step | Raw | Stripped | Output | Lost |",
              "|---|---:|---:|---:|---:|"]
    for _, r in readiness.funnel.iterrows():
        def f(v):
            return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{int(v):,}"
        lines.append(f"| {r['step']} | {f(r.get('raw_rows'))} | "
                     f"{f(r.get('junk_rows_stripped'))} | {f(r.get('output_rows'))} | "
                     f"{f(r.get('lost'))} |")
    (d / "readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["md"] = d / "readiness.md"
    return paths
