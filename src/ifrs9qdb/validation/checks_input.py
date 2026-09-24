"""Input-stage checks: the raw extracts, before anything is derived.

These run first because everything after them is built on what they check. A
duplicate contract id here becomes a double-counted provision six steps later,
with nothing in between to say where it came from.

Ids match the R package exactly, so a suppression approved against the R engine
applies to this one and a finding can be compared across the two.
"""
from __future__ import annotations

import pandas as pd

from ..etl.read_inputs import INPUT_SPECS
from ..ids import as_id
from ._helpers import (blank_detail, col, dup_detail, examples_of, fail,
                       fk_detail, has, numeric_detail, ok, parse_any_date)
from .framework import Severity, Validator

__all__ = ["INPUT_STAGE_VALIDATORS"]

_KNOWN_CURRENCIES = {"QAR", "USD"}


# --------------------------------------------------------------- presence ---
def _presence_validators() -> list[Validator]:
    """One per expected extract. Generated, because there are twelve of them
    and a hand-written list drifts from the loader's registry."""
    out = []
    for spec in INPUT_SPECS:
        name = spec.name
        fname = getattr(spec, "file", name)

        def check(inputs, _k=name, _f=fname):
            if has(inputs, _k):
                return ok()
            return fail(0, f"{_k} is missing or has zero rows (expected {_f})")

        out.append(Validator(
            id=f"INPUT_{name}_present",
            severity=Severity.ERROR,
            description=f"{fname} is present and non-empty",
            fn=check,
            context=name,
            rationale=("A missing file silently removes a portfolio or a whole "
                       "dimension from the run. The loader accepts either Office "
                       "Open XML or Oracle SQL*Plus HTML, so a failure here means "
                       "the extract did not arrive at all."),
            remediation=(f"Confirm {fname} is in the input directory with at "
                         "least one data row, and re-run the SQL extract if not."),
            tags=("pre_run",),
            suppressible=False,          # never silently ignore a missing file
        ))
    return out


# ------------------------------------------------------------ definitions ---
def _unique(key: str, table: str, column: str, label: str | None = None):
    def check(inputs, _t=table, _c=column):
        if not has(inputs, _t):
            return ok()
        c = col(inputs[_t], _c)
        if c is None:
            return fail(0, f"{_c} column not found in {_t}")
        return dup_detail(c, _c)
    return check


def _nonblank(table: str, column: str):
    def check(inputs, _t=table, _c=column):
        if not has(inputs, _t):
            return ok()
        c = col(inputs[_t], _c)
        if c is None:
            return fail(0, f"{_c} column not found in {_t}")
        return blank_detail(c, _c)
    return check


def _numeric_nonneg(table: str, column: str):
    def check(inputs, _t=table, _c=column):
        if not has(inputs, _t):
            return ok()
        c = col(inputs[_t], _c)
        if c is None:
            return fail(0, f"{_c} column not found in {_t}")
        return numeric_detail(c, _c)
    return check


def _date_parses(table: str, column: str):
    def check(inputs, _t=table, _c=column):
        if not has(inputs, _t):
            return ok()
        raw = col(inputs[_t], _c)
        if raw is None:
            return fail(0, f"{_c} column not found in {_t}")
        parsed = parse_any_date(raw)
        bad = parsed.isna() & pd.Series(raw).notna() & \
            (pd.Series(raw).astype(str).str.strip() != "")
        n = int(bad.sum())
        if n == 0:
            return ok()
        return fail(n, f"{n} unparseable {_c} value(s)",
                    examples=examples_of(raw, bad))
    return check


def _v_maturity_after_open(inputs):
    if not has(inputs, "AccountMaster"):
        return ok()
    df = inputs["AccountMaster"]
    o, m = col(df, "OPENDATE", "open_date"), col(df, "MATURITYDATE", "maturity_date")
    if o is None or m is None:
        return fail(0, "OpenDate or MaturityDate column missing")
    od, md = parse_any_date(o), parse_any_date(m)
    bad = od.notna() & md.notna() & (md < od)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} contract(s) mature before they open",
                examples=examples_of(col(df, "CONTRACTID", "contract_id"), bad))


def _v_account_customer_fk(inputs):
    if not (has(inputs, "AccountMaster") and has(inputs, "CustomerMaster")):
        return ok()
    child = col(inputs["AccountMaster"], "CUSTOMERID", "customer_id")
    parent = col(inputs["CustomerMaster"], "CUSTOMERID", "customer_id")
    if child is None or parent is None:
        return fail(0, "CustomerId column missing on one side")
    return fk_detail(child, parent, "AccountMaster.CustomerId", "CustomerMaster")


def _v_aca_percent_range(inputs):
    if not has(inputs, "AccountCollateralAllocation"):
        return ok()
    c = col(inputs["AccountCollateralAllocation"],
            "ALLOCATIONPERCENTAGE", "allocation_percentage")
    if c is None:
        return fail(0, "AllocationPercentage column not found")
    x = pd.to_numeric(c, errors="coerce")
    bad = x.notna() & ((x < 0) | (x > 100))
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} allocation(s) outside 0-100% in raw units",
                examples=examples_of(c, bad))


def _v_aca_contract_fk(inputs):
    if not (has(inputs, "AccountCollateralAllocation") and has(inputs, "AccountMaster")):
        return ok()
    child = col(inputs["AccountCollateralAllocation"], "CONTRACTID", "contract_id")
    parent = col(inputs["AccountMaster"], "CONTRACTID", "contract_id")
    if child is None or parent is None:
        return fail(0, "ContractId column missing on one side")
    return fk_detail(child, parent, "allocation ContractId", "AccountMaster")


def _v_aca_total_per_contract(inputs):
    if not has(inputs, "AccountCollateralAllocation"):
        return ok()
    df = inputs["AccountCollateralAllocation"]
    cid = col(df, "CONTRACTID", "contract_id")
    pct = pd.to_numeric(col(df, "ALLOCATIONPERCENTAGE", "allocation_percentage"),
                        errors="coerce")
    if cid is None or pct is None:
        return fail(0, "ContractId or AllocationPercentage column missing")
    tot = pd.DataFrame({"c": as_id(cid), "p": pct}).groupby("c")["p"].sum()
    over = tot[tot > 100.1]
    if over.empty:
        return ok()
    return fail(len(over), f"{len(over)} contract(s) allocated above 100%",
                examples=[f"{k}={v:.2f}%" for k, v in over.head(10).items()])


def _v_aca_pair_unique(inputs):
    if not has(inputs, "AccountCollateralAllocation"):
        return ok()
    df = inputs["AccountCollateralAllocation"]
    a = col(df, "COLLATERALID", "collateral_id")
    b = col(df, "CONTRACTID", "contract_id")
    if a is None or b is None:
        return fail(0, "CollateralId or ContractId column missing")
    pair = as_id(a) + "|" + as_id(b)
    return dup_detail(pair, "(CollateralId, ContractId)")


def _v_rs_coverage(inputs):
    if not (has(inputs, "RepaymentSchedule") and has(inputs, "AccountMaster")):
        return ok()
    rs = as_id(col(inputs["RepaymentSchedule"], "CONTRACTID", "contract_id"))
    am = col(inputs["AccountMaster"], "CONTRACTID", "contract_id")
    if am is None:
        return fail(0, "AccountMaster ContractId column missing")
    have = set(rs)
    want = as_id(am)
    want = want[want != ""]
    missing = sorted(set(want) - have)
    if not missing:
        return ok()
    return fail(len(missing),
                f"{len(missing)} lending contract(s) have no repayment schedule; "
                "these fall back to the parametric EAD shape",
                examples=missing[:10])


def _v_currency_known(inputs):
    if not has(inputs, "AccountMaster"):
        return ok()
    c = col(inputs["AccountMaster"], "CURRENCY", "currency", "CURRENCYCODE")
    if c is None:
        return ok()
    s = pd.Series(c).astype(str).str.strip().str.upper()
    bad = s.notna() & (s != "") & ~s.isin(_KNOWN_CURRENCIES | {"NAN", ""})
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} row(s) carry a currency outside "
                   f"{sorted(_KNOWN_CURRENCIES)}",
                examples=sorted(set(s[bad]))[:10])


def _v_investment_rating_known(inputs, static=None):
    if not has(inputs, "AccountMasterInvestments"):
        return ok()
    c = col(inputs["AccountMasterInvestments"], "RATING", "rating")
    if c is None or static is None:
        return ok()
    scale = static.get("master_rating_scale")
    if scale is None or len(scale) == 0:
        return ok()
    ext = scale[scale[scale.columns[1]].astype(str).str.lower().str.startswith("ext")]
    known = set(ext[ext.columns[0]].astype(str).str.strip())
    known |= set(pd.Series(col(ext, "external_equivalent")
                           if col(ext, "external_equivalent") is not None
                           else []).astype(str).str.strip())
    s = pd.Series(c).astype(str).str.strip()
    bad = (s != "") & ~s.isin(known | {"nan", ""})
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} investment rating(s) are not in the external scale",
                examples=sorted(set(s[bad]))[:10])


def _extract_dates(inputs):
    seen = {}
    for spec in INPUT_SPECS:
        if not has(inputs, spec.name):
            continue
        c = col(inputs[spec.name], "EXTRACTDA", "EXTRACTDATE", "extract_date")
        if c is None:
            continue
        d = parse_any_date(c).dropna()
        if len(d):
            seen[spec.name] = d.max().date()
    return seen


def _v_consistent_extract_date(inputs):
    seen = _extract_dates(inputs)
    if len(set(seen.values())) <= 1:
        return ok()
    return fail(len(set(seen.values())),
                "input files disagree on the extract date",
                examples=[f"{k}={v}" for k, v in sorted(seen.items())][:10])


def _v_extract_date_matches_cfg(inputs, reporting_date=None):
    if reporting_date is None:
        return ok()
    seen = _extract_dates(inputs)
    if not seen:
        return ok()
    want = pd.to_datetime(reporting_date, errors="coerce")
    if pd.isna(want):
        return ok()
    want = want.date()
    off = {k: v for k, v in seen.items() if v != want}
    if not off:
        return ok()
    return fail(len(off), f"{len(off)} file(s) do not carry the reporting date "
                          f"{want}",
                examples=[f"{k}={v}" for k, v in sorted(off.items())][:10])


def _v_duplicate_headers_stripped(inputs, header_strip_log=None):
    if not header_strip_log:
        return ok()
    total = sum(int(v) for v in dict(header_strip_log).values())
    if total == 0:
        return ok()
    return fail(total,
                f"{total} repeated header row(s) were stripped while reading; "
                "SQL*Plus repeats its headings every thousand rows and they "
                "arrive as data",
                examples=[f"{k}={v}" for k, v in dict(header_strip_log).items()][:10])


# ------------------------------------------------------------------ suite ---
def _v(id, severity, description, fn, context, rationale, remediation,
       suppressible=True):
    return Validator(id=id, severity=severity, description=description, fn=fn,
                     context=context, rationale=rationale,
                     remediation=remediation, tags=("pre_run",),
                     suppressible=suppressible)


INPUT_STAGE_VALIDATORS: list[Validator] = _presence_validators() + [
    _v("INPUT_AccountMaster_contractid_unique", Severity.ERROR,
       "Every CONTRACTID in AccountMaster is unique",
       _unique("", "AccountMaster", "CONTRACTID"), "AccountMaster",
       "ContractId is the primary key for the lending pipeline. A duplicate "
       "double-counts exposure and makes every downstream join ambiguous.",
       "Find the duplicated contracts with IT. If they are deliberate "
       "(multi-currency variants), the extract should be filtered to one row "
       "per ContractId before this pipeline reads it.",
       suppressible=False),
    _v("INPUT_AccountMaster_contractid_nonblank", Severity.ERROR,
       "Every row in AccountMaster has a non-blank CONTRACTID",
       _nonblank("AccountMaster", "CONTRACTID"), "AccountMaster",
       "A blank contract id cannot be joined to anything, so the row is priced "
       "against no curve and no collateral.",
       "Check the extract for trailing or partial rows."),
    _v("INPUT_AccountMaster_onbalance_nonneg", Severity.WARN,
       "ONBALANCE in AccountMaster is numeric and non-negative",
       _numeric_nonneg("AccountMaster", "ONBALANCE"), "AccountMaster",
       "A non-numeric balance becomes NaN and prices to zero; a negative one "
       "produces a negative provision.",
       "Confirm the amounts with IT, checking for thousands separators or a "
       "currency symbol in the column."),
    _v("INPUT_AccountMaster_opendate_parses", Severity.ERROR,
       "OPENDATE column in AccountMaster parses as a date",
       _date_parses("AccountMaster", "OPENDATE"), "AccountMaster",
       "Months on book is measured from the open date; an unparseable value "
       "drops the contract out of every vintage analysis.",
       "Check the extract's date format - two formats in one file is common."),
    _v("INPUT_AccountMaster_maturity_after_open", Severity.WARN,
       "MaturityDate >= OpenDate for every contract",
       _v_maturity_after_open, "AccountMaster",
       "A maturity before the open date gives a negative term, which produces "
       "a nonsensical EAD curve.",
       "Correct the maturity date at source, or add a contract-level override "
       "if the business has accepted these as legacy records."),
    _v("INPUT_AccountMaster_customer_fk", Severity.WARN,
       "Every CUSTOMERID in AccountMaster exists in CustomerMaster",
       _v_account_customer_fk, "AccountMaster",
       "The lending rating is a CUSTOMER attribute. A contract whose customer "
       "is missing gets no rating, so no PD bucket resolves and it prices to "
       "zero.",
       "Ask IT whether the customer extract is filtered differently from the "
       "account extract."),
    _v("INPUT_AccountMaster_currency_known", Severity.WARN,
       "Currency codes are in {QAR, USD}",
       _v_currency_known, "AccountMaster",
       "An unknown currency has no FX rate, so the exposure is carried at "
       "face value in the wrong unit.",
       "Add the currency to fx_rates.csv, or confirm the code is a typo."),
    _v("INPUT_CustomerMaster_customerid_unique", Severity.ERROR,
       "Every CUSTOMERID in CustomerMaster is unique",
       _unique("", "CustomerMaster", "CUSTOMERID"), "CustomerMaster",
       "A duplicate customer makes the rating lookup ambiguous, and which row "
       "wins is an accident of ordering.",
       "De-duplicate at source."),
    _v("INPUT_ACA_allocation_in_percent_range", Severity.WARN,
       "AllocationPercentage (raw, in % units) in [0, 100]",
       _v_aca_percent_range, "AccountCollateralAllocation",
       "The source writes 10.09 for ten per cent. A value outside the range "
       "usually means the column has already been converted, which would "
       "divide the collateral by a hundred again.",
       "Check the extract's units before changing anything downstream."),
    _v("INPUT_ACA_contract_fk", Severity.WARN,
       "Every ContractId in AccountCollateralAllocation exists in AccountMaster",
       _v_aca_contract_fk, "AccountCollateralAllocation",
       "An allocation pointing at no account is collateral that will never be "
       "applied - it silently reduces nothing.",
       "Check whether the account extract is filtered more tightly than the "
       "allocation extract."),
    _v("INPUT_ACA_total_allocation_per_contract", Severity.WARN,
       "Sum of allocation_percentage per ContractId <= 100.1% (raw % units)",
       _v_aca_total_per_contract, "AccountCollateralAllocation",
       "More than 100% allocated to one contract over-states its collateral "
       "and under-states the provision.",
       "Review the allocations for the named contracts."),
    _v("INPUT_ACA_pair_unique", Severity.ERROR,
       "Every (COLLATERALID, CONTRACTID) pair is unique",
       _v_aca_pair_unique, "AccountCollateralAllocation",
       "A repeated pair counts the same collateral twice against the same "
       "contract.",
       "De-duplicate the allocation extract."),
    _v("INPUT_RS_coverage", Severity.WARN,
       "Every active lending ContractId has a RepaymentSchedule row",
       _v_rs_coverage, "RepaymentSchedule",
       "Without a schedule the contract has no supplied EAD curve and falls "
       "back to the parametric shape, which is materially different for "
       "amortising products.",
       "Expected for revolving and off-balance products. Investigate if a term "
       "loan appears here."),
    _v("INPUT_AccountMasterInvestments_rating_known", Severity.WARN,
       "RATING in AccountMasterInvestments is in the external scale (or blank)",
       _v_investment_rating_known, "AccountMasterInvestments",
       "An unrecognised agency grade resolves to no bucket, so the investment "
       "prices to zero.",
       "Add the grade to master_rating_scale.csv if it is genuine."),
    _v("INPUT_AccountMasterInvestments_contractid_unique", Severity.ERROR,
       "Every CONTRACTID in AccountMasterInvestments is unique",
       _unique("", "AccountMasterInvestments", "CONTRACTID"),
       "AccountMasterInvestments",
       "A duplicate double-counts the holding.",
       "De-duplicate at source.", suppressible=False),
    _v("INPUT_AccountMasterInvestments_contractid_non_blank", Severity.ERROR,
       "Every row in AccountMasterInvestments has a non-blank CONTRACTID",
       _nonblank("AccountMasterInvestments", "CONTRACTID"),
       "AccountMasterInvestments",
       "A blank id cannot be joined to anything.",
       "Check the extract for trailing rows."),
    _v("INPUT_AccountMasterInvestments_onbalance_numeric_nonneg", Severity.ERROR,
       "ONBALANCE in AccountMasterInvestments is numeric and non-negative",
       _numeric_nonneg("AccountMasterInvestments", "ONBALANCE"),
       "AccountMasterInvestments",
       "A non-numeric holding prices to zero.",
       "Check the extract for formatting in the amount column."),
    _v("INPUT_AccountMasterInvestments_opendate_parses", Severity.WARN,
       "OPENDATE column in AccountMasterInvestments parses as a date",
       _date_parses("AccountMasterInvestments", "OPENDATE"),
       "AccountMasterInvestments",
       "Two date formats appear in this file - ExtractDate is unpadded and the "
       "others are zero-padded - so a single-format parse drops rows.",
       "Check the extract's date formats."),
    _v("INPUT_CustomerMasterInvestments_customerid_unique", Severity.ERROR,
       "Every CUSTOMERID in CustomerMasterInvestments is unique",
       _unique("", "CustomerMasterInvestments", "CUSTOMERID"),
       "CustomerMasterInvestments",
       "A duplicate counterparty makes the rating lookup ambiguous.",
       "De-duplicate at source."),
    _v("INPUT_CustomerStagingFlag_customerid_unique", Severity.ERROR,
       "Every CUSTOMERID in CustomerStagingFlag is unique",
       _unique("", "CustomerStagingFlag", "CUSTOMERID"), "CustomerStagingFlag",
       "A duplicate makes the staging flags ambiguous, and staging decides the "
       "whole provision for that customer.",
       "De-duplicate at source.", suppressible=False),
    _v("INPUT_CustomerStagingFlagInvestments_customerid_unique", Severity.ERROR,
       "Every CUSTOMERID in CustomerStagingFlagInvestments is unique",
       _unique("", "CustomerStagingFlagInvestments", "CUSTOMERID"),
       "CustomerStagingFlagInvestments",
       "A duplicate makes the staging flags ambiguous.",
       "De-duplicate at source."),
    _v("INPUT_Collateral_collateralid_unique", Severity.ERROR,
       "Every COLLATERALID in Collateral is unique",
       _unique("", "Collateral", "COLLATERALID"), "Collateral",
       "A duplicate collateral record is counted once per row when the "
       "allocations are joined, over-stating cover.",
       "De-duplicate at source.", suppressible=False),
    _v("INPUT_IndustryCode_customerid_unique", Severity.ERROR,
       "Every CUSTOMERID in IndustryCode is unique (one industry per customer)",
       _unique("", "IndustryCode", "CUSTOMERID"), "IndustryCode",
       "Two industries for one customer duplicates the customer's exposure in "
       "every sector concentration figure.",
       "De-duplicate at source."),
    _v("INPUT_Origination_contractid_unique", Severity.ERROR,
       "Every CONTRACTID in Origination is unique",
       _unique("", "Origination", "CONTRACTID"), "Origination",
       "A duplicate makes the origination rating lookup ambiguous, which "
       "drives the SICR comparison.",
       "De-duplicate at source."),
    _v("INPUT_OriginationInvestments_contractid_unique", Severity.ERROR,
       "Every CONTRACTID in OriginationInvestments is unique",
       _unique("", "OriginationInvestments", "CONTRACTID"),
       "OriginationInvestments",
       "A duplicate makes the origination lookup ambiguous.",
       "De-duplicate at source."),
    _v("INPUT_consistent_extract_date", Severity.WARN,
       "All inputs resolve to the same EXTRACTDA date",
       _v_consistent_extract_date, "",
       "Files from different extract dates describe different books. Mixing "
       "them produces a run that reconciles to nothing.",
       "Re-run the extracts together."),
    _v("INPUT_extract_date_matches_run_cfg", Severity.ERROR,
       "The input files carry the run's reporting date",
       _v_extract_date_matches_cfg, "",
       "Running last quarter's extracts under this quarter's date produces a "
       "run that looks current and is not.",
       "Point the run at the right input directory, or correct the reporting "
       "date."),
    _v("INPUT_duplicate_headers_stripped", Severity.WARN,
       "No repeated header rows were found in the input files",
       _v_duplicate_headers_stripped, "",
       "SQL*Plus repeats its headings every thousand rows. Left in, they "
       "become contracts with a customer id of \"CUSTOMERID\".",
       "Informational: the reader strips them. A large count means the export "
       "settings changed."),
]
