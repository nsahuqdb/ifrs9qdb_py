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
                       extract_date_column, extract_date_counts, fk_detail,
                       has, numeric_detail, ok,
                       parse_any_date, r_parse_dates,
                       resolve_input_extract_date, text)
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
        cands = [spec.file] + list(getattr(spec, "alternatives", ()) or ())
        fname = cands[0] if len(cands) == 1 else \
            f"{cands[0]} (or {', '.join(cands[1:])})"

        def check(inputs, _k=name, _c=", ".join(cands)):
            if has(inputs, _k):
                return ok()
            return fail(0, f"{_k} is missing or has zero rows (tried: {_c})")

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


def _nonblank(table: str, column: str, template: str):
    """R: blank = NA or empty after trimming; message per R's wording."""
    def check(inputs, _t=table, _c=column, _tpl=template):
        if not has(inputs, _t):
            return ok()
        c = col(inputs[_t], _c, "contract_id", "account_id")
        if c is None:
            return fail(0, "CONTRACTID column not found")
        bad = text(c) == ""
        n = int(bad.sum())
        if n == 0:
            return ok()
        return fail(n, _tpl.format(n=n))
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


def _v_am_onbalance(inputs):
    """R: every value that is not a number counts, blanks included."""
    if not has(inputs, "AccountMaster"):
        return ok()
    c = col(inputs["AccountMaster"], "ONBALANCE", "on_balance")
    if c is None:
        return fail(0, "ONBALANCE column not found")
    x = pd.to_numeric(text(c).str.replace(",", "", regex=False), errors="coerce")
    n_na, n_neg = int(x.isna().sum()), int((x < 0).sum())
    if n_na == 0 and n_neg == 0:
        return ok()
    return fail(n_na + n_neg, f"{n_na} non-numeric, {n_neg} negative",
                examples=examples_of(c, x.isna() | (x < 0)))


def _v_ami_onbalance(inputs):
    """R: blanks count as zero here (the downstream behaviour); text does not."""
    if not has(inputs, "AccountMasterInvestments"):
        return ok()
    c = col(inputs["AccountMasterInvestments"], "ONBALANCE", "on_balance")
    if c is None:
        return fail(0, "ONBALANCE column not found")
    t = text(c)
    blank = (t == "") | t.str.lower().isin(["na", "nan"])
    x = pd.to_numeric(t.str.replace(",", "", regex=False), errors="coerce")
    x[blank] = 0.0
    n_bad, n_neg = int((x.isna() & ~blank).sum()), int((x < 0).sum())
    if n_bad == 0 and n_neg == 0:
        return ok()
    return fail(n_bad + n_neg,
                f"{n_bad} non-numeric, {n_neg} negative ONBALANCE values")


def _anchor_date(inputs):
    """The reporting date the run adopts, as R's resolve_input_extract_date()
    takes it (see that helper for how a mixed file resolves)."""
    return resolve_input_extract_date(inputs)


def _v_am_eir_present(inputs):
    if not has(inputs, "AccountMaster"):
        return ok()
    c = col(inputs["AccountMaster"], "eir", "EIR")
    if c is None:
        return fail(0, "EIR column not found")
    x = pd.to_numeric(text(c), errors="coerce")
    bad = x.isna() | (x == 0)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} contract(s) have a zero or blank EIR; the ETL fills "
                   "them with the account-type average EIR (V4 rule)",
                examples=examples_of(col(inputs["AccountMaster"], "contract_id",
                                         "CONTRACTID"), bad))


def _eir_plausible(table, id_col):
    def check(inputs, _t=table, _id=id_col):
        if not has(inputs, _t):
            return ok()
        c = col(inputs[_t], "eir", "EIR")
        if c is None:
            return fail(0, "EIR column not found")
        x = pd.to_numeric(text(c), errors="coerce")
        bad = x.notna() & (x != 0) & ((x < 0.1) | (x > 30))
        n = int(bad.sum())
        if n == 0:
            return ok()
        ids = text(col(inputs[_t], _id, "contract_id"))[bad].head(5).tolist()
        vals = [f"{v:g}" for v in x[bad].head(5)]
        return fail(n, f"{n} EIR value(s) outside 0.1%-30% (the extract carries "
                       "EIR in percent): " + ", ".join(
                           f"{i}={v}" for i, v in zip(ids, vals)),
                    examples=ids)
    return check


def _v_am_maturity_valid(inputs, static=None):
    if not has(inputs, "AccountMaster"):
        return ok()
    df = inputs["AccountMaster"]
    c = col(df, "maturity_date", "MATURITYDATE", "MATURITYDAT")
    if c is None:
        return fail(0, "MATURITYDATE column not found")
    d = parse_any_date(c)
    blank = d.isna()
    anchor = _anchor_date(inputs)
    before = (d < anchor) if anchor is not None else pd.Series(False, index=d.index)
    n_blank, n_before = int(blank.sum()), int(before.sum())
    if n_blank == 0 and n_before == 0:
        return ok()
    days = 365
    try:
        th = static.get("staging_thresholds") if static is not None else None
        if th is not None:
            k = text(col(th, "key"))
            v = pd.to_numeric(col(th, "value")[k == "maturity_extension_days"],
                              errors="coerce").dropna()
            if len(v):
                days = int(v.iloc[0])
    except Exception:
        pass
    return fail(n_blank + n_before,
                f"{n_blank} contract(s) have a blank or unparseable MATURITYDATE "
                "(priced over the 3-month minimum horizon); "
                f"{n_before} mature before the extract date (the ETL extends "
                f"them to the extract date + {days} days)",
                examples=examples_of(col(df, "contract_id", "CONTRACTID"),
                                     blank | before))


def _v_am_dpd_valid(inputs):
    if not has(inputs, "AccountMaster"):
        return ok()
    c = col(inputs["AccountMaster"], "past_due_days", "PASTDUEDAYS")
    if c is None:
        return fail(0, "PASTDUEDAYS column not found")
    x = pd.to_numeric(text(c), errors="coerce")
    n_na, n_neg = int(x.isna().sum()), int((x < 0).sum())
    if n_na + n_neg == 0:
        return ok()
    return fail(n_na + n_neg,
                f"{n_na} blank or non-numeric PASTDUEDAYS (the ETL treats them "
                f"as 0 days past due), {n_neg} negative")


def _v_rs_dates_plausible(inputs):
    """A payment date before its posting date: the two-digit-year wrap.

    Excel's default century window reads a two-digit year 30-99 as 19xx, so a
    schedule running to 2030-2043 arrives as 1930-1943. The EAD-curve builder
    drops every row before the reporting date, which cuts those contracts'
    curves off at the last good date.
    """
    if not has(inputs, "RepaymentSchedule"):
        return ok()
    df = inputs["RepaymentSchedule"]
    sd = parse_any_date(col(df, "start_date", "START_DAT"))
    pdt = col(df, "post_date", "POST_DATE")
    pdt = parse_any_date(pdt) if pdt is not None else pd.Series(pd.NaT, index=sd.index)
    bad = (sd.notna() & pdt.notna() & (sd < pdt)) | (sd.notna() & (sd.dt.year < 1950))
    n = int(bad.sum())
    if n == 0:
        return ok()
    cid = text(col(df, "contract_id", "KEY_1"))
    k = int(cid[bad].nunique())
    yrs = sd[bad].dt.year
    ex = cid[bad].drop_duplicates().head(5).tolist()
    return fail(n, f"{n} schedule row(s) across {k} contract(s) have a payment "
                   f"date before the posting date ({int(yrs.min())}-{int(yrs.max())}): "
                   "two-digit years read as 19xx. These rows are dropped when "
                   "the EAD curve is built, so those contracts' curves stop "
                   "early and their lifetime ECL is understated. e.g. "
                   + ", ".join(ex), examples=ex)


def _scale_labels(static):
    """Every label on the master rating scale, internal and external -- the
    list the ETL's rating chain matches a customer rating against."""
    ms = static.get("master_rating_scale") if static is not None else None
    if ms is None or len(ms) == 0:
        return None
    return set(text(col(ms, "rating"))) - {""}


def _fallback(static, segment, default):
    try:
        fb = static.get("segment_fallback_ratings")
        hit = text(col(fb, "fallback_rating"))[text(col(fb, "segment")) == segment]
        return hit.iloc[0] if len(hit) else default
    except Exception:
        return default


def _v_cm_rating_known(inputs, static=None):
    """A customer rating the chain does not recognise is treated as unrated.

    The lending rating comes from CustomerMaster.RATING, matched EXACTLY
    against the master scale. 'Unrated' falls through to the collective-
    assessment rule or the segment fallback by design; so does a typo such as
    'QDB 99' or 'QDB9', silently. Nothing else checked these values: the
    rating-coverage check reads AccountMaster's rating column, which the
    lending extract leaves blank.
    """
    if not has(inputs, "CustomerMaster") or static is None:
        return ok()
    labels = _scale_labels(static)
    if labels is None:
        return ok()
    df = inputs["CustomerMaster"]
    r = text(col(df, "rating", "RATING"))
    bad = (r != "") & (r != "Unrated") & ~r.isin(labels)
    n = int(bad.sum())
    if n == 0:
        return ok()
    ids = text(col(df, "customer_id", "CUSTOMERID"))[bad]
    shown = [f"{i}={v}" for i, v in zip(ids.head(10), r[bad].head(10))]
    return fail(n, f"{n} customer(s) carry a RATING that is neither on the "
                   "master rating scale nor 'Unrated': " + ", ".join(shown)
                + "; the ETL treats them as unrated (collective-assessment or "
                  "segment fallback rating)",
                examples=list(ids.head(10)))


def _v_cm_rating_fallback(inputs, static=None):
    if not has(inputs, "CustomerMaster") or static is None:
        return ok()
    labels = _scale_labels(static)
    if labels is None:
        return ok()
    df = inputs["CustomerMaster"]
    r = text(col(df, "rating", "RATING"))
    n_unr, n_blank = int((r == "Unrated").sum()), int((r == "").sum())
    n_other = int(((r != "") & (r != "Unrated") & ~r.isin(labels)).sum())
    n = n_unr + n_blank + n_other
    if n == 0:
        return ok()
    return fail(n, f"{n} of {len(r)} customer(s) are rated by fallback: {n_unr} "
                   f"'Unrated', {n_blank} blank, {n_other} not on the scale. The "
                   "ETL takes their rating from the sector collective-assessment "
                   "rule or the segment fallback ("
                   f"{_fallback(static, 'Unrated Customer (Internal Rating)', 'QDB 5')}; "
                   f"Al Dhameen {_fallback(static, 'Al Dhameen Customers', 'QDB 6')})")


def _v_ami_rating_present(inputs, static=None):
    if not has(inputs, "AccountMasterInvestments"):
        return ok()
    df = inputs["AccountMasterInvestments"]
    r = text(col(df, "rating", "RATING"))
    n = int((r == "").sum())
    if n == 0:
        return ok()
    ids = text(col(df, "account_id", "contract_id", "CONTRACTID"))[r == ""]
    return fail(n, f"{n} investment(s) have a blank RATING; the ETL assigns the "
                   "segment fallback rating "
                   f"{_fallback(static, 'Investment Portfolio', 'Baa3')}: "
                + ", ".join(ids.head(10)), examples=list(ids.head(10)))


def _v_ami_dpd_ignored(inputs):
    """Days past due on the investment book never reach the stage.

    The ETL writes AccountMaster_2.PastDueDays as 0 (the V4 convention, kept
    by both engines) and the investment book is staged on rating
    deterioration alone, so a holding in arrears is not moved to Stage 2 or 3
    by its arrears. That is methodology, not a data error -- this check makes
    sure it is at least seen.
    """
    if not has(inputs, "AccountMasterInvestments"):
        return ok()
    df = inputs["AccountMasterInvestments"]
    c = col(df, "past_due_days", "PASTDUEDAYS")
    if c is None:
        return ok()
    x = pd.to_numeric(text(c), errors="coerce")
    bad = x.notna() & (x > 0)
    n = int(bad.sum())
    if n == 0:
        return ok()
    ids = text(col(df, "account_id", "contract_id", "CONTRACTID"))[bad].head(10)
    shown = ", ".join(f"{i}={v:g}" for i, v in zip(ids, x[bad].head(10)))
    return fail(n, f"{n} investment(s) are past due ({shown}); the investment "
                   "book is staged on rating deterioration only and is written "
                   "to LIC with PastDueDays 0, so days past due do not move "
                   "them to Stage 2 or 3", examples=list(ids))


def _unparsed_dates(df, column: str):
    """R's .unparsed_dates(): the values of a date column that do not read as
    a date, and up to five of them. On the schema-typed column such a value
    is already blank, so the count is what the typing recorded
    (schema_unread_values); on raw text the column is parsed here, with the
    schema's own parser."""
    c = col(df, column)
    if c is None:
        return None
    if pd.api.types.is_datetime64_any_dtype(c):
        u = (getattr(df, "attrs", {}).get("schema_unread") or {}).get(column)
        return (0, []) if u is None else (int(u["n"]), list(u.get("sample", [])))
    from ..dates import schema_parse_dates
    from .schema import _unread
    u = _unread(pd.Series(c), schema_parse_dates(c), "date", column)
    return (0, []) if u is None else (u["n"], u["sample"])


def _date_parses(table: str, column: str, template: str, required: bool = True):
    """R: a value present but unparseable. On the schema-typed column R reads,
    an unparseable date is already NA, so this counts what the schema layer
    could not type."""
    def check(inputs, _t=table, _c=column, _tpl=template, _req=required):
        if not has(inputs, _t):
            return ok()
        got = _unparsed_dates(inputs[_t], _c)
        if got is None:
            return fail(0, f"{_c} column not found") if _req else ok()
        n, sample = got
        if n == 0:
            return ok()
        return fail(n, _tpl.format(n=n), examples=sample)
    return check


def _v_maturity_after_open(inputs):
    if not has(inputs, "AccountMaster"):
        return ok()
    df = inputs["AccountMaster"]
    o = col(df, "open_date", "OPENDATE")
    m = col(df, "maturity_date", "MATURITYDATE", "MATURITYDAT")
    if o is None or m is None:
        return fail(0, "open_date or maturity_date column missing")
    o, m = parse_any_date(o), parse_any_date(m)
    n = int((o.notna() & m.notna() & (m < o)).sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} contracts have maturity_date < open_date")


def _v_account_customer_fk(inputs):
    if not (has(inputs, "AccountMaster") and has(inputs, "CustomerMaster")):
        return ok()
    a = col(inputs["AccountMaster"], "customer_id", "CUSTOMERID")
    c = col(inputs["CustomerMaster"], "customer_id", "CUSTOMERID")
    if a is None or c is None:
        return fail(0, "CustomerId column missing in one of the files")
    have = set(as_id(c))
    orphans = [v for v in pd.unique(as_id(a)) if v not in have]
    if not orphans:
        return ok()
    return fail(len(orphans), f"{len(orphans)} CustomerIds in AccountMaster have "
                              "no CustomerMaster row (e.g. "
                              + ", ".join(orphans[:5]) + ")",
                examples=orphans[:10])


def _v_aca_percent_range(inputs, run_config=None):
    """R: every blank counts as non-numeric; out of range beyond 100.0001 --
    or 1.000001 when run.allocation_percentage_unit is fraction."""
    if not has(inputs, "AccountCollateralAllocation"):
        return ok()
    c = col(inputs["AccountCollateralAllocation"], "allocation_percentage",
            "ALLOCATIONPERCENTAGE")
    if c is None:
        return fail(0, "AllocationPercentage column not found")
    from ..runconfig import allocation_divisor, allocation_percentage_unit
    x = pd.to_numeric(text(c), errors="coerce")
    top = allocation_divisor(x, allocation_percentage_unit(run_config))
    n_bad = int((x.notna() & ((x < 0) | (x > top * 1.000001))).sum())
    n_na = int(x.isna().sum())
    if n_bad == 0 and n_na == 0:
        return ok()
    return fail(n_bad + n_na, f"{n_bad} outside [0,{top:g}], {n_na} non-numeric")


def _v_aca_unit_consistent(inputs, run_config=None):
    """R: the file's values contradict run.allocation_percentage_unit -- every
    value at most 1 under percent, any value above 1 under fraction."""
    from ..runconfig import ALLOCATION_UNITS, allocation_percentage_unit
    unit = allocation_percentage_unit(run_config)
    if unit not in ALLOCATION_UNITS:
        return fail(1, f"run.allocation_percentage_unit is '{unit}'; use percent, "
                       "fraction or auto (the run guesses as auto meanwhile)")
    if unit == "auto" or not has(inputs, "AccountCollateralAllocation"):
        return ok()
    c = col(inputs["AccountCollateralAllocation"], "allocation_percentage",
            "ALLOCATIONPERCENTAGE")
    if c is None:
        return ok()
    x = pd.to_numeric(text(c), errors="coerce").dropna()
    if x.empty:
        return ok()
    if unit == "percent" and x.max() <= 1 and bool((x > 0).any()):
        return fail(len(x), f"every AllocationPercentage is at most 1 ({len(x)} "
                            f"value(s), max {_r_num(x.max())}): the file looks like "
                            "fractions, but run.allocation_percentage_unit is "
                            "percent, so each share would count a hundredth of itself")
    if unit == "fraction" and bool((x > 1).any()):
        n = int((x > 1).sum())
        return fail(n, f"{n} AllocationPercentage value(s) above 1 (max "
                       f"{_r_num(x.max())}): the file looks like percentages, but "
                       "run.allocation_percentage_unit is fraction, so those "
                       "shares would count up to a hundred times")
    return ok()


def _r_num(v) -> str:
    """A number as R's format() writes it: up to 7 significant digits, no
    trailing zeros, and scientific notation only when it is narrower than the
    fixed form (R's scipen = 0): 0.6, 60, 111892, but 1e-04."""
    import math
    x = float(v)
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else str(x)
    e = math.floor(math.log10(abs(x)))
    fixed = f"{x:.{max(0, 6 - e)}f}"
    if "." in fixed:
        fixed = fixed.rstrip("0").rstrip(".")
    mant, _, exp = f"{x:.6e}".partition("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    sci = f"{mant}e{exp}"
    return fixed if len(fixed) <= len(sci) else sci


def _v_aca_contract_fk(inputs, static=None):
    """Allocation ContractIds against AccountMaster, in BOTH id forms.

    The allocation extract already carries LIC's numeric ids (1264010963),
    while AccountMaster carries the source form (0000126FGG010963) that the
    ETL converts. Comparing raw against raw flagged 682 contracts as orphans
    on the June book that the ETL joins perfectly well -- and buried the one
    real finding, allocation rows with no ContractId at all, inside that
    count. Blank ids are reported separately.
    """
    if not (has(inputs, "AccountCollateralAllocation") and has(inputs, "AccountMaster")):
        return ok()
    child = col(inputs["AccountCollateralAllocation"], "contract_id", "CONTRACTID")
    parent = col(inputs["AccountMaster"], "contract_id", "CONTRACTID")
    if child is None or parent is None:
        return fail(0, "ContractId missing in one of the files")
    from ..etl.lending import apply_id_substitutions
    prod = None
    try:
        obp = static.get("off_balance_products") if static is not None else None
        if obp is not None and len(obp):
            prod = dict(zip(text(col(obp, "product_code")),
                            text(col(obp, "lic_input_code"))))
    except Exception:
        prod = None
    am_raw = text(parent)
    am_raw = am_raw[am_raw != ""]
    am_ids = set(as_id(am_raw)) | set(as_id(apply_id_substitutions(am_raw, prod)))
    aca = as_id(child)
    n_blank = int((aca == "").sum())
    orphans = [v for v in pd.unique(aca[aca != ""]) if v not in am_ids]
    if not orphans and n_blank == 0:
        return ok()
    return fail(len(orphans) + n_blank,
                f"{len(orphans)} ContractIds in ACA absent from AccountMaster; "
                f"{n_blank} allocation row(s) have a blank ContractId",
                examples=orphans[:10])


def _v_aca_total_per_contract(inputs, run_config=None):
    """Shares summed per contract -- expected above 100% for a contract
    secured by several items, since each share is a fraction of ITS
    collateral. Informational: double allocation of one item is what
    XFILE_ACA_allocation_sum_per_collateral catches."""
    if not has(inputs, "AccountCollateralAllocation"):
        return ok()
    df = inputs["AccountCollateralAllocation"]
    cid = col(df, "contract_id", "CONTRACTID")
    pc = col(df, "allocation_percentage", "ALLOCATIONPERCENTAGE")
    coll = col(df, "collateral_id", "COLLATERALID")
    if cid is None or pc is None:
        return fail(0, "ContractId or AllocationPercentage column missing")
    from ..runconfig import allocation_divisor, allocation_percentage_unit
    pct = pd.to_numeric(pc, errors="coerce")
    # in percent: a fraction file is scaled up to compare against 100
    pct = pct * (100 / allocation_divisor(pct, allocation_percentage_unit(run_config)))
    d = pd.DataFrame({"c": as_id(cid), "p": pct,
                      "k": as_id(coll) if coll is not None else ""})
    d = d[d["c"] != ""]
    g = d.groupby("c")
    tot = g["p"].sum()
    over = tot[tot > 100.1]
    if over.empty:
        return ok()
    single = int((g["k"].nunique()[over.index] == 1).sum())
    return fail(len(over),
                f"{len(over)} ContractIds have total allocation > 100% (e.g. "
                f"{', '.join(list(over.index[:5]))}); {single} of them draw on a "
                "single collateral item",
                examples=list(over.index[:10]))


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
    rs = set(as_id(col(inputs["RepaymentSchedule"], "contract_id", "KEY_1")))
    am = col(inputs["AccountMaster"], "contract_id", "CONTRACTID")
    if am is None:
        return fail(0, "AccountMaster CONTRACTID not found")
    want = pd.unique(as_id(am))
    missing = [v for v in want if v not in rs]
    if not missing:
        return ok()
    return fail(len(missing),
                f"{len(missing)} AccountMaster ContractIds have no "
                "RepaymentSchedule rows",
                examples=missing[:10])


def _v_currency_known(inputs, static=None):
    if not has(inputs, "AccountMaster"):
        return ok()
    c = col(inputs["AccountMaster"], "currency_code", "CURRENCYCODE", "CUR")
    if c is None:
        return fail(0, "currency_code column not found")
    vals = text(c).str.upper()
    vals = vals[vals != ""]
    fx = static.get("fx_rates") if static is not None else None
    known = set(text(col(fx, "currency_code")).str.upper()) if fx is not None \
        and len(fx) else _KNOWN_CURRENCIES
    bad = [v for v in pd.unique(vals) if v not in known]
    if not bad:
        return ok()
    return fail(len(bad), "Unknown currency codes: " + ", ".join(bad))


def _v_investment_rating_known(inputs, static=None):
    if not has(inputs, "AccountMasterInvestments"):
        return ok()
    c = col(inputs["AccountMasterInvestments"], "rating", "RATING")
    if c is None:
        return fail(0, "RATING column not found")
    vals = text(c)
    vals = vals[vals != ""]
    ms = static.get("master_rating_scale") if static is not None else None
    ext = set()
    if ms is not None and len(ms):
        ext = set(text(col(ms, "rating"))[text(col(ms, "rating_type")) == "External"])
    bad = [v for v in pd.unique(vals) if v not in ext]
    if not bad:
        return ok()
    return fail(len(bad), f"{len(bad)} unknown ratings: " + ", ".join(bad[:10]))


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
    """R: the distinct dates EXTRACTDA parses to across the files, in the
    order the files are read -- every distinct value, not a sample."""
    dates = []
    for spec in INPUT_SPECS:
        if not has(inputs, spec.name):
            continue
        c = extract_date_column(inputs[spec.name])
        if c is None:
            continue
        raw = pd.Series(c).dropna()
        if raw.empty:
            continue
        raw = raw[~raw.duplicated()]
        d = r_parse_dates(raw).dropna()
        for x in pd.unique(d.dt.strftime("%Y-%m-%d")):
            if x not in dates:
                dates.append(x)
    if len(dates) <= 1:
        return ok()
    return fail(len(dates), "Multiple distinct dates after parsing: "
                + ", ".join(dates))


def _v_extract_date_matches_cfg(inputs, reporting_date=None):
    """R: every row of every file carries the date the run adopts -- the
    AccountMaster EXTRACTDA most rows carry, not the config. A stray row is
    data from another vintage, and a file dated otherwise another extract."""
    anchor = _anchor_date(inputs)
    if anchor is None:
        return ok()
    parts, n_rows, files = [], 0, []
    for spec in INPUT_SPECS:
        if not has(inputs, spec.name):
            continue
        counts = extract_date_counts(inputs[spec.name])
        other = counts[counts["date"] != anchor].sort_values("date")
        if other.empty:
            continue
        parts.append(f"{spec.name}: " + ", ".join(
            f"{int(r)} row(s) dated {d.strftime('%Y-%m-%d')}"
            for d, r in zip(other["date"], other["rows"])))
        n_rows += int(other["rows"].sum())
        files.append(spec.name)
    if not parts:
        return ok()
    return fail(n_rows, "input files disagree on EXTRACTDA (reporting date "
                        f"{anchor.strftime('%Y-%m-%d')}): " + "; ".join(parts),
                examples=files)


def _v_extract_date_plausible(inputs):
    """R: the adopted reporting date is not before any contract's OPENDATE,
    nor after today -- the symptom of a stale, mistyped or mixed EXTRACTDA
    (a file split evenly between two dates is dated by the earlier, see
    resolve_input_extract_date)."""
    anchor = _anchor_date(inputs)
    if anchor is None:
        return ok()
    c = col(inputs["AccountMaster"], "open_date", "OPENDATE")
    od = (pd.Series(dtype="datetime64[ns]") if c is None
          else r_parse_dates(c))
    after = od.notna() & (od > anchor)
    today = pd.Timestamp.today().normalize()
    future = anchor > today
    n = int(after.sum())
    if n == 0 and not future:
        return ok()
    parts = []
    if n:
        parts.append(f"before the OPENDATE of {n} contract(s) (latest "
                     f"{od[after].max().strftime('%Y-%m-%d')})")
    if future:
        parts.append(f"after today ({today.strftime('%Y-%m-%d')})")
    return fail(n + int(future),
                f"reporting date {anchor.strftime('%Y-%m-%d')} is "
                + " and ".join(parts)
                + " - EXTRACTDA is stale, mistyped or mixed")


def _v_duplicate_headers_stripped(inputs, header_strip_log=None):
    log = {k: int(v) for k, v in dict(header_strip_log or {}).items() if int(v) > 0}
    if not log:
        return ok()
    parts = [f"{k} ({v})" for k, v in log.items()]
    return fail(sum(log.values()),
                f"Auto-removed repeated header row(s) from {len(log)} file(s): "
                + ", ".join(parts),
                examples=list(log)[:10])


def _v_values_typed(inputs):
    """R: every non-blank value in a date or number column reads as one --
    what the schema could not type, by file and column."""
    from .schema import schema_unread_values
    u = schema_unread_values(inputs)
    if u.empty:
        return ok()
    parts = [f"{f}.{src} {n} (e.g. {smp.replace(' | ', ', ')})"
             for f, src, n, smp in zip(u["file"], u["source"], u["n"], u["sample"])]
    return fail(int(u["n"].sum()),
                f"{int(u['n'].sum())} value(s) in {len(u)} column(s) could not be "
                "read and are treated as blank: " + "; ".join(parts),
                examples=[f"{f}.{src}" for f, src in zip(u["file"], u["source"])])


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
       _nonblank("AccountMaster", "CONTRACTID",
                 "{n} rows have blank ContractId"), "AccountMaster",
       "A blank contract id cannot be joined to anything, so the row is priced "
       "against no curve and no collateral.",
       "Check the extract for trailing or partial rows."),
    _v("INPUT_AccountMaster_onbalance_nonneg", Severity.WARN,
       "ONBALANCE in AccountMaster is numeric and non-negative",
       _v_am_onbalance, "AccountMaster",
       "A non-numeric balance becomes NaN and prices to zero; a negative one "
       "produces a negative provision.",
       "Confirm the amounts with IT, checking for thousands separators or a "
       "currency symbol in the column."),
    _v("INPUT_AccountMaster_opendate_parses", Severity.ERROR,
       "OPENDATE column in AccountMaster parses as a date",
       _date_parses("AccountMaster", "open_date",
                    "{n} unparseable OpenDate values"), "AccountMaster",
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
       "AllocationPercentage in [0, 100] ([0, 1] when "
       "run.allocation_percentage_unit is fraction)",
       _v_aca_percent_range, "AccountCollateralAllocation",
       "The source writes 10.09 for ten per cent. A value outside the range "
       "usually means the column has already been converted, which would "
       "divide the collateral by a hundred again.",
       "Check the extract's units before changing anything downstream."),
    _v("INPUT_ACA_allocation_unit_consistent", Severity.ERROR,
       "AllocationPercentage values are in the unit config.yml "
       "run.allocation_percentage_unit names",
       _v_aca_unit_consistent, "AccountCollateralAllocation",
       "The run divides each AllocationPercentage by 100 when the unit is "
       "percent and by 1 when it is fraction, and the share decides how much "
       "of the collateral's value reduces each contract's loss. A file in the "
       "other unit moves every collateral benefit by a factor of 100: "
       "fractions read as percentages lose almost all of it, percentages read "
       "as fractions overstate it a hundredfold.",
       "Set run.allocation_percentage_unit in config.yml to the unit the "
       "extract uses (percent: 57.25 for 57.25%; fraction: 0.5725), or have "
       "the extract re-delivered in the configured unit."),
    _v("INPUT_ACA_contract_fk", Severity.WARN,
       "Every ContractId in AccountCollateralAllocation exists in AccountMaster",
       _v_aca_contract_fk, "AccountCollateralAllocation",
       "An allocation pointing at no account is collateral that will never be "
       "applied - it silently reduces nothing.",
       "Check whether the account extract is filtered more tightly than the "
       "allocation extract."),
    _v("INPUT_ACA_total_allocation_per_contract", Severity.INFO,
       "Contracts whose allocation shares sum above 100% (expected when a "
       "contract is secured by several collateral items)",
       _v_aca_total_per_contract, "AccountCollateralAllocation",
       "Each share is a fraction of ONE collateral item, so a contract secured "
       "by several items legitimately sums above 100% (June 2026: all 846 such "
       "contracts draw on 2-72 items, none on a single one). Double allocation "
       "of one item is caught per collateral by "
       "XFILE_ACA_allocation_sum_per_collateral; this check was a WARN that "
       "fired on every multi-collateral contract.",
       "None unless a contract is listed as drawing on a single collateral "
       "item - that one is over-allocated."),
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
    _v("INPUT_RS_dates_plausible", Severity.ERROR,
       "No repayment-schedule payment date falls before its posting date",
       _v_rs_dates_plausible, "RepaymentSchedule",
       "A schedule date before its posting date is a two-digit year read as "
       "19xx (Excel maps 30-99 to 1930-1999, so 2030-2043 arrives as "
       "1930-1943). The EAD-curve builder drops those rows as history, so the "
       "contract's curve ends at the last correctly dated payment and its "
       "lifetime ECL is understated. June 2026: 25,466 rows, 1,124 contracts; "
       "repairing them raises Stage 2 ECL by 6.8%.",
       "Re-extract the schedule with four-digit years. Until then, suppress "
       "with a reason only if the understatement is accepted for this run."),
    _v("INPUT_AccountMaster_eir_present", Severity.WARN,
       "Every AccountMaster contract has a non-zero EIR",
       _v_am_eir_present, "AccountMaster",
       "A zero or blank EIR is replaced by the average EIR of the contract's "
       "account type (the V4 workbook rule), so the contract is discounted at "
       "a rate that is not its own.",
       "Supply the contract's EIR at source."),
    _v("INPUT_AccountMaster_eir_plausible", Severity.WARN,
       "Every AccountMaster EIR is between 0.1% and 30% (percent units)",
       _eir_plausible("AccountMaster", "contract_id"), "AccountMaster",
       "The extract carries EIR in percent (4.2 = 4.2%). A value like 0.0001 "
       "is effectively no discounting; one above 30 is usually a decimal "
       "written as a percent the other way round.",
       "Correct the rate at source."),
    _v("INPUT_AccountMaster_maturity_valid", Severity.WARN,
       "Every AccountMaster MATURITYDATE is present and after the extract date",
       _v_am_maturity_valid, "AccountMaster",
       "A blank maturity prices over the 3-month minimum horizon, which for a "
       "Stage 2 contract understates lifetime loss. A maturity before the "
       "extract date is moved to the extract date plus the configured "
       "extension (V4 rule) - expected for overdue facilities, but it means "
       "the priced maturity is not the contract's.",
       "Supply the maturity at source; review long-overdue facilities."),
    _v("INPUT_AccountMaster_dpd_valid", Severity.WARN,
       "Every AccountMaster PASTDUEDAYS is a non-negative number",
       _v_am_dpd_valid, "AccountMaster",
       "A blank PASTDUEDAYS is treated as 0 days past due, so a delinquent "
       "contract can be staged as performing.",
       "Supply PASTDUEDAYS at source."),
    _v("INPUT_AccountMasterInvestments_eir_plausible", Severity.WARN,
       "Every AccountMasterInvestments EIR is between 0.1% and 30% (percent units)",
       _eir_plausible("AccountMasterInvestments", "account_id"),
       "AccountMasterInvestments",
       "The investment is discounted at its EIR; a rate in the wrong units "
       "distorts the provision.",
       "Correct the rate at source."),
    _v("INPUT_CustomerMaster_rating_known", Severity.WARN,
       "Every CustomerMaster RATING is on the master scale or 'Unrated'",
       _v_cm_rating_known, "CustomerMaster",
       "The lending rating comes from CustomerMaster.RATING, matched exactly "
       "against the master scale. Anything else - a typo such as 'QDB 99' or "
       "'QDB9' - is treated as unrated and silently re-rated by the "
       "collective-assessment rule or the segment fallback, so the contract is "
       "priced on a rating nobody gave it.",
       "Correct the rating at source, or add the label to "
       "master_rating_scale.csv if it is genuine."),
    _v("INPUT_CustomerMaster_rating_fallback", Severity.INFO,
       "Customers rated by the collective-assessment rule or segment fallback",
       _v_cm_rating_fallback, "CustomerMaster",
       "An unrated customer is priced on a rating derived by rule (sector and "
       "days past due) or on the segment fallback. Expected for unrated "
       "customers; listed so the share of the book priced that way is visible.",
       "None required; review if the share moves materially between runs."),
    _v("INPUT_AccountMasterInvestments_rating_present", Severity.WARN,
       "Every AccountMasterInvestments RATING is present",
       _v_ami_rating_present, "AccountMasterInvestments",
       "A blank investment rating is replaced by the segment fallback (Baa3), "
       "so the holding is priced on a rating it was never given.",
       "Supply the agency rating at source."),
    _v("INPUT_AccountMasterInvestments_dpd_ignored", Severity.WARN,
       "No investment is past due (days past due do not stage investments)",
       _v_ami_dpd_ignored, "AccountMasterInvestments",
       "AccountMaster_2 is written with PastDueDays 0 and the investment book "
       "is staged on rating deterioration alone, so an investment in arrears "
       "stays in Stage 1 unless its rating has fallen.",
       "Review the listed holdings; stage them manually (stage override) if "
       "the arrears are a significant increase in credit risk."),
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
       _nonblank("AccountMasterInvestments", "CONTRACTID",
                 "{n} rows with blank CONTRACTID"),
       "AccountMasterInvestments",
       "A blank id cannot be joined to anything.",
       "Check the extract for trailing rows."),
    _v("INPUT_AccountMasterInvestments_onbalance_numeric_nonneg", Severity.ERROR,
       "ONBALANCE in AccountMasterInvestments is numeric and non-negative",
       _v_ami_onbalance,
       "AccountMasterInvestments",
       "A non-numeric holding prices to zero.",
       "Check the extract for formatting in the amount column."),
    _v("INPUT_AccountMasterInvestments_opendate_parses", Severity.WARN,
       "OPENDATE column in AccountMasterInvestments parses as a date",
       _date_parses("AccountMasterInvestments", "open_date",
                    "{n} unparseable OPENDATE values", required=False),
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
    _v("INPUT_extract_date_plausible", Severity.ERROR,
       "The reporting date the run adopts is not before any contract's "
       "OPENDATE, nor after today",
       _v_extract_date_plausible, "AccountMaster",
       "The reporting date is read from AccountMaster EXTRACTDA and anchors "
       "every output stamp, maturity extension, EAD and PD curve. A wrong "
       "EXTRACTDA fails no other check when every file carries it: a stale or "
       "mistyped date, or a file where as many rows carry an earlier date as "
       "the reporting date (the run adopts the date most rows carry, and the "
       "earliest on a tie), prices the whole book at the wrong date. A "
       "reporting date before contracts' opening dates, or after today, is "
       "the symptom.",
       "Check EXTRACTDA in the extract - every row should carry the reporting "
       "date - and re-export if it is stale, mistyped or mixed.",
       suppressible=False),
    _v("INPUT_values_typed", Severity.WARN,
       "Every non-blank value in a date or number column reads as a date or "
       "number",
       _v_values_typed, "",
       "Each input column is typed before any check or calculation sees it. A "
       "value the typing cannot read - a date in an unknown format, text in an "
       "amount, a number with a thousands separator - becomes blank, and "
       "the run treats it as missing: a blank ONBALANCE prices at zero, a "
       "blank MATURITYDATE at the 3-month minimum horizon, a blank allocation "
       "loses the collateral. The field checks count the blanks; this one "
       "names the values that were lost, so the extract can be fixed.",
       "Re-export the file with the column in its usual format: dates as "
       "M/D/YYYY, YYYY-MM-DD or DD-MON-YY, numbers without separators or text."),
    _v("INPUT_duplicate_headers_stripped", Severity.WARN,
       "No repeated header rows were found in the input files",
       _v_duplicate_headers_stripped, "",
       "SQL*Plus repeats its headings every thousand rows. Left in, they "
       "become contracts with a customer id of \"CUSTOMERID\".",
       "Informational: the reader strips them. A large count means the export "
       "settings changed."),
]
