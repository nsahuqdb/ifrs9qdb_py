"""
The checks themselves.

Grouped by what they can see, because that decides when they can run:

    input       the raw extracts, before anything is derived
    transform   the LIC input files, after the ETL
    derived     the priced report
    cross-file  agreement BETWEEN files, which is where most real problems live

The cross-file group matters most and is the easiest to skip. A collateral
allocation pointing at a collateral record that does not exist passes every
single-file check and then makes LIC return NaN coverage, which zeroes a
contract's ECL without any error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .framework import Severity, Validator, validator

__all__ = ["INPUT_VALIDATORS", "TRANSFORM_VALIDATORS", "DERIVED_VALIDATORS",
           "CROSS_FILE_VALIDATORS", "ALL_VALIDATORS"]


def _ok(**kw):
    return {"passed": True, **kw}


def _fail(count: int, detail: str, examples=None):
    return {"passed": False, "count": int(count), "detail": detail,
            "examples": list(examples or [])[:10]}


def _col(df, *names):
    if df is None or len(df) == 0:
        return None
    low = {"".join(ch for ch in str(c).lower() if ch.isalnum()): c
           for c in df.columns}
    for n in names:
        hit = low.get("".join(ch for ch in n.lower() if ch.isalnum()))
        if hit is not None:
            return df[hit]
    return None


# ============================================================== input =======
def _v_inputs_present(inputs):
    missing = getattr(inputs, "missing", [])
    if missing:
        return _fail(len(missing), "not delivered: " + ", ".join(missing),
                     missing)
    return _ok(detail=f"{len(inputs.tables)} files read")


def _v_accounts_not_empty(inputs):
    n = len(inputs["AccountMaster"])
    return _ok(detail=f"{n:,} rows") if n else _fail(0, "AccountMaster is empty")


def _v_contract_ids_unique(inputs):
    from ..etl.lending import drop_repeated_headers
    am = drop_repeated_headers(inputs["AccountMaster"])
    cid = _col(am, "ContractId")
    if cid is None:
        return _fail(0, "no ContractId column")
    dup = cid[cid.duplicated()].astype(str).unique()
    if len(dup):
        return _fail(len(dup), f"{len(dup)} contract ids appear more than once",
                     dup)
    return _ok(detail=f"{len(cid):,} unique")


def _v_balances_numeric(inputs):
    from ..etl.lending import drop_repeated_headers
    am = drop_repeated_headers(inputs["AccountMaster"])
    bal = pd.to_numeric(_col(am, "OnBalance"), errors="coerce")
    bad = bal.isna().sum()
    if bad:
        return _fail(bad, f"{bad:,} rows have a non-numeric on-balance amount")
    return _ok(detail=f"{len(bal):,} rows")


def _v_negative_balances(inputs):
    from ..etl.lending import drop_repeated_headers
    am = drop_repeated_headers(inputs["AccountMaster"])
    bal = pd.to_numeric(_col(am, "OnBalance"), errors="coerce")
    neg = (bal < 0).sum()
    if neg:
        return _fail(neg, f"{neg:,} contracts carry a negative balance")
    return _ok()


def _v_maturity_present(inputs):
    from ..etl.lending import drop_repeated_headers
    am = drop_repeated_headers(inputs["AccountMaster"])
    mat = pd.to_datetime(_col(am, "MaturityDate", "MATURITYDAT"),
                         errors="coerce", format="mixed")
    miss = mat.isna().sum()
    if miss:
        return _fail(miss, f"{miss:,} contracts have no usable maturity date")
    return _ok()


def _v_matured_contracts(inputs):
    from ..etl.lending import drop_repeated_headers
    am = drop_repeated_headers(inputs["AccountMaster"])
    mat = pd.to_datetime(_col(am, "MaturityDate", "MATURITYDAT"),
                         errors="coerce", format="mixed")
    ext = pd.to_datetime(_col(am, "ExtractDate", "EXTRACTDA"),
                         errors="coerce", format="mixed")
    if mat is None or ext is None or ext.isna().all():
        return _ok(detail="no extract date to compare against")
    ref = ext.max()
    past = ((mat < ref) & mat.notna()).sum()
    if past:
        return _fail(past, f"{past:,} contracts matured on or before {ref.date()}")
    return _ok()


def _v_collateral_values(inputs):
    v = pd.to_numeric(_col(inputs["Collateral"], "CollateralValue"),
                      errors="coerce")
    if v is None:
        return _fail(0, "no CollateralValue column")
    bad = ((v <= 0) | v.isna()).sum()
    if bad:
        return _fail(bad, f"{bad:,} collateral records have no positive value")
    return _ok(detail=f"{len(v):,} records")


INPUT_VALIDATORS = [
    Validator("IN001", Severity.ERROR, "All twelve source extracts delivered",
              _v_inputs_present, context="input",
              rationale="A missing file silently removes a portfolio or a "
                        "whole dimension from the run.",
              remediation="Ask IT to re-run the missing extract."),
    Validator("IN002", Severity.ERROR, "AccountMaster is not empty",
              _v_accounts_not_empty, context="input",
              rationale="An empty account master produces a run with no book.",
              remediation="Check the extract query returned rows."),
    Validator("IN003", Severity.ERROR, "Contract ids are unique",
              _v_contract_ids_unique, context="input",
              rationale="A duplicate id makes every downstream join ambiguous "
                        "and can double-count a provision.",
              remediation="Identify the duplicated ids with IT."),
    Validator("IN004", Severity.ERROR, "On-balance amounts are numeric",
              _v_balances_numeric, context="input",
              rationale="A non-numeric balance becomes NaN and prices to zero.",
              remediation="Check for text or thousands separators in the source."),
    Validator("IN005", Severity.WARN, "No negative balances",
              _v_negative_balances, context="input",
              rationale="A negative exposure gives a negative provision, which "
                        "nets off real losses elsewhere.",
              remediation="Confirm with Finance whether these are credit balances."),
    Validator("IN006", Severity.WARN, "Every contract has a maturity date",
              _v_maturity_present, context="input",
              rationale="Without a maturity the engine falls back to a minimum "
                        "term, understating lifetime loss.",
              remediation="Ask IT to populate the maturity date."),
    Validator("IN007", Severity.INFO, "No contracts already matured",
              _v_matured_contracts, context="input",
              rationale="A matured facility still carries a balance here, so "
                        "either it should have been closed or the date is stale.",
              remediation="Confirm whether these are genuinely still open."),
    Validator("IN008", Severity.WARN, "Collateral records carry a value",
              _v_collateral_values, context="input",
              rationale="A zero-valued collateral record contributes nothing "
                        "but still consumes an allocation.",
              remediation="Review the collateral extract with Credit."),
]


# ========================================================== transform =======
def _v_outputs_written(out_dir):
    from pathlib import Path
    from ..etl.pipeline import PRODUCED
    d = Path(out_dir)
    missing = [f for f in PRODUCED if not (d / f).is_file()]
    if missing:
        return _fail(len(missing), "not written: " + ", ".join(missing), missing)
    return _ok(detail=f"{len(PRODUCED)} files")


def _v_account_master_rows(out_dir):
    from pathlib import Path
    p = Path(out_dir) / "AccountMaster_1.csv"
    if not p.is_file():
        return _fail(0, "AccountMaster_1.csv not written")
    n = len(pd.read_csv(p, low_memory=False))
    return _ok(detail=f"{n:,} contracts") if n else _fail(0, "no rows")


def _v_ratings_resolve(out_dir):
    """Every contract's rating must exist on its own scale.

    A rating that does not resolve has no PD bucket, so the contract prices to
    nothing and disappears from the provision without an error anywhere.
    """
    from pathlib import Path
    d = Path(out_dir)
    am, rt = d / "AccountMaster_1.csv", d / "Ratings.csv"
    if not (am.is_file() and rt.is_file()):
        return _fail(0, "AccountMaster_1.csv or Ratings.csv missing")
    a = pd.read_csv(am, low_memory=False)
    r = pd.read_csv(rt)
    known = set(r["Rating"].astype(str))
    used = a["Rating"].astype(str)
    bad = used[~used.isin(known) & used.ne("") & used.ne("nan")]
    if len(bad):
        return _fail(len(bad), f"{len(bad):,} contracts carry a rating not on "
                               "any scale", bad.unique())
    return _ok(detail=f"{used.nunique()} distinct ratings")


TRANSFORM_VALIDATORS = [
    Validator("TR001", Severity.ERROR, "All LIC input files written",
              _v_outputs_written, context="transform",
              rationale="LIC reads a fixed set; a missing file stops the tool.",
              remediation="Check the run log for the step that failed."),
    Validator("TR002", Severity.ERROR, "AccountMaster_1 has rows",
              _v_account_master_rows, context="transform",
              rationale="An empty account file means no book to price.",
              remediation="Check the junk-row stripping did not remove everything."),
    Validator("TR003", Severity.ERROR, "Every rating resolves to a scale",
              _v_ratings_resolve, context="transform",
              rationale="An unresolved rating has no PD bucket, so the "
                        "contract prices to nothing and vanishes from the "
                        "provision silently.",
              remediation="Add the missing grade to the master rating scale."),
]


# ============================================================ derived =======
def _v_report_written(report):
    if report is None or len(report) == 0:
        return _fail(0, "no ECL report")
    return _ok(detail=f"{len(report):,} contracts")


def _v_everything_priced(report):
    ecl = pd.to_numeric(report["ecl"], errors="coerce")
    unpriced = ecl.isna().sum()
    if unpriced:
        return _fail(unpriced, f"{unpriced:,} contracts have no ECL")
    return _ok(detail=f"{len(ecl):,} priced")


def _v_ecl_within_exposure(report):
    ecl = pd.to_numeric(report["ecl"], errors="coerce")
    exp = pd.to_numeric(report["exposure"], errors="coerce")
    over = ((ecl > exp) & (exp > 0)).sum()
    if over:
        return _fail(over, f"{over:,} contracts are provisioned above their "
                           "own balance")
    return _ok()


def _v_stage3_fully_provisioned(report):
    """Under stage3_method = full_outstanding, Stage 3 books the whole balance.

    A Stage 3 contract at partial coverage means the setting is not what was
    expected, which changes the provision by a large amount.
    """
    s3 = report[report["stage"] == 3]
    if len(s3) == 0:
        return _ok(detail="no Stage 3 contracts")
    ecl = pd.to_numeric(s3["ecl"], errors="coerce")
    exp = pd.to_numeric(s3["exposure"], errors="coerce")
    partial = ((exp > 0) & (ecl < exp * 0.999)).sum()
    if partial:
        return _fail(partial, f"{partial:,} Stage 3 contracts are below full "
                              "coverage")
    return _ok(detail=f"{len(s3):,} Stage 3 contracts at full coverage")


def _v_no_negative_ecl(report):
    ecl = pd.to_numeric(report["ecl"], errors="coerce")
    neg = (ecl < 0).sum()
    return _fail(neg, f"{neg:,} contracts have a negative provision") if neg else _ok()


def _v_coverage_plausible(report):
    exp = pd.to_numeric(report["exposure"], errors="coerce").sum()
    ecl = pd.to_numeric(report["ecl"], errors="coerce").sum()
    if exp <= 0:
        return _fail(0, "total exposure is zero")
    cov = 100 * ecl / exp
    if not (0.1 <= cov <= 60):
        return _fail(0, f"book coverage is {cov:.2f}%, outside the plausible "
                        "range of 0.1% to 60%")
    return _ok(detail=f"{cov:.2f}%")


DERIVED_VALIDATORS = [
    Validator("DE001", Severity.ERROR, "ECL report produced",
              _v_report_written, context="derived",
              rationale="Without it there is nothing to report.",
              remediation="Check the pricing step in the run log."),
    Validator("DE002", Severity.WARN, "Every contract priced",
              _v_everything_priced, context="derived",
              rationale="An unpriced contract contributes nothing, so the "
                        "provision is understated by exactly its share.",
              remediation="Usually a missing PD curve or an unresolved rating."),
    Validator("DE003", Severity.ERROR, "No contract provisioned above balance",
              _v_ecl_within_exposure, context="derived",
              rationale="A provision above the exposure cannot be right and "
                        "will be queried.",
              remediation="Check ecl.cap_ecl_at_exposure is on."),
    Validator("DE004", Severity.WARN, "Stage 3 at full coverage",
              _v_stage3_fully_provisioned, context="derived",
              rationale="The QDB basis books Stage 3 at the full outstanding; "
                        "partial coverage means the setting is not what was "
                        "expected.",
              remediation="Check ecl.stage3_method."),
    Validator("DE005", Severity.ERROR, "No negative provisions",
              _v_no_negative_ecl, context="derived",
              rationale="A negative provision nets off real losses elsewhere.",
              remediation="Check for negative exposures in the source."),
    Validator("DE006", Severity.WARN, "Book coverage is plausible",
              _v_coverage_plausible, context="derived",
              rationale="A coverage far outside the usual range points at a "
                        "calibration or mapping problem rather than a real "
                        "change in credit quality.",
              remediation="Compare against the prior quarter before signing."),
]


# ========================================================= cross-file =======
def _v_allocations_resolve(out_dir):
    """Every allocation must point at a collateral record that exists.

    This is the one that matters most. LIC treats missing collateral as NaN
    rather than zero coverage, so a single orphan reference poisons an entire
    contract's ECL -- and nothing errors.
    """
    from pathlib import Path
    d = Path(out_dir)
    alloc, coll = d / "AccountCollateralAllocation.csv", d / "Collateral.csv"
    if not (alloc.is_file() and coll.is_file()):
        return _fail(0, "allocation or collateral file missing")
    a = pd.read_csv(alloc, low_memory=False)
    c = pd.read_csv(coll, low_memory=False)
    known = set(c["CollateralId"].astype(str))
    used = a["CollateralId"].astype(str)
    orphan = used[~used.isin(known)]
    if len(orphan):
        return _fail(len(orphan),
                     f"{len(orphan):,} allocations point at a collateral "
                     "record that does not exist",
                     orphan.unique())
    return _ok(detail=f"{len(a):,} allocations resolve")


def _v_allocation_contracts_exist(out_dir):
    from pathlib import Path
    d = Path(out_dir)
    alloc, am = d / "AccountCollateralAllocation.csv", d / "AccountMaster_1.csv"
    if not (alloc.is_file() and am.is_file()):
        return _fail(0, "allocation or account file missing")
    a = pd.read_csv(alloc, low_memory=False)
    m = pd.read_csv(am, low_memory=False)
    known = set(m["ContractId"].astype(str))
    used = a["ContractId"].astype(str)
    orphan = used[~used.isin(known)]
    if len(orphan):
        return _fail(len(orphan.unique()),
                     f"{len(orphan.unique()):,} contracts in the allocation "
                     "file are not in the account master",
                     orphan.unique())
    return _ok()


def _v_curves_cover_the_book(out_dir):
    from pathlib import Path
    d = Path(out_dir)
    lpo, am = d / "LifeTimeParameterOther.csv", d / "AccountMaster_1.csv"
    if not (lpo.is_file() and am.is_file()):
        return _fail(0, "curve or account file missing")
    l = pd.read_csv(lpo, low_memory=False)
    m = pd.read_csv(am, low_memory=False)
    have = set(l["ContractId"].astype(str))
    total = len(m)
    without = total - m["ContractId"].astype(str).isin(have).sum()
    pct = 100 * without / max(total, 1)
    if pct > 40:
        return _fail(without, f"{without:,} contracts ({pct:.0f}%) have no "
                              "supplied EAD curve and fall back to the "
                              "parametric shape")
    return _ok(detail=f"{without:,} of {total:,} on the parametric fallback "
                      f"({pct:.0f}%)")


def _v_pd_curves_cover_buckets(out_dir):
    from pathlib import Path
    d = Path(out_dir)
    stpd, am = d / "StPD.csv", d / "AccountMaster_1.csv"
    rt = d / "Ratings.csv"
    if not (stpd.is_file() and am.is_file() and rt.is_file()):
        return _fail(0, "StPD, account or ratings file missing")
    s = pd.read_csv(stpd, low_memory=False)
    m = pd.read_csv(am, low_memory=False)
    r = pd.read_csv(rt)
    hier = dict(zip(r["Rating"].astype(str), r["Hierarchy"]))
    need = {(str(p), hier.get(str(rating)))
            for p, rating in zip(m["PortfolioCode"], m["Rating"])
            if hier.get(str(rating)) is not None}
    have = {(str(p), int(b)) for p, b in zip(s["PortfolioCode"], s["PDBucketDim1"])}
    missing = {x for x in need if (x[0], int(x[1])) not in have}
    if missing:
        return _fail(len(missing),
                     f"{len(missing)} portfolio/bucket combinations used by "
                     "the book have no PD curve",
                     [f"{p}|{int(b)}" for p, b in list(missing)[:10]])
    return _ok(detail=f"{len(need)} combinations covered")


def _v_customers_consistent(out_dir):
    from pathlib import Path
    d = Path(out_dir)
    am, cm = d / "AccountMaster_1.csv", d / "CustomerMaster_1.csv"
    if not (am.is_file() and cm.is_file()):
        return _fail(0, "account or customer file missing")
    m = pd.read_csv(am, low_memory=False)
    c = pd.read_csv(cm, low_memory=False)
    known = set(c["CustomerId"].astype(str))
    used = m["CustomerId"].astype(str)
    orphan = used[~used.isin(known)].unique()
    if len(orphan):
        return _fail(len(orphan), f"{len(orphan):,} customers hold contracts "
                                  "but are not in the customer master",
                     orphan)
    return _ok()


CROSS_FILE_VALIDATORS = [
    Validator("XF001", Severity.ERROR, "Collateral allocations resolve",
              _v_allocations_resolve, context="cross-file",
              rationale="LIC treats missing collateral as NaN rather than "
                        "zero coverage, so ONE orphan reference zeroes an "
                        "entire contract's ECL without any error.",
              remediation="Ask IT to include the referenced collateral, or "
                          "remove the allocation."),
    Validator("XF002", Severity.WARN, "Allocations point at real contracts",
              _v_allocation_contracts_exist, context="cross-file",
              rationale="An allocation to a contract that is not in the book "
                        "is carried for nothing and hides a stale extract.",
              remediation="Confirm the two extracts were taken at the same time."),
    Validator("XF003", Severity.INFO, "Supplied EAD curve coverage",
              _v_curves_cover_the_book, context="cross-file",
              rationale="Contracts without a schedule fall back to a "
                        "parametric shape, which is an assumption rather than "
                        "a contractual profile.",
              remediation="None needed if the share is the usual one; a jump "
                          "means the schedule extract lost rows."),
    Validator("XF004", Severity.ERROR, "PD curves cover every bucket in use",
              _v_pd_curves_cover_buckets, context="cross-file",
              rationale="A portfolio and bucket with no curve prices to "
                        "nothing, removing those contracts from the provision "
                        "silently.",
              remediation="Check the StPD build covered every portfolio."),
    Validator("XF005", Severity.WARN, "Every borrower is in the customer master",
              _v_customers_consistent, context="cross-file",
              rationale="A missing customer loses its rating and staging "
                        "flags, so its contracts stage and price on defaults.",
              remediation="Ask IT to re-run the customer extract."),
]

ALL_VALIDATORS = (INPUT_VALIDATORS + TRANSFORM_VALIDATORS
                  + DERIVED_VALIDATORS + CROSS_FILE_VALIDATORS)
