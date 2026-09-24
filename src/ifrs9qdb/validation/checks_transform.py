"""Transform-stage checks: the book after it has been shaped, before pricing.

This is where the staging rule and the rating resolution can be checked as
STATEMENTS -- a customer over 90 days past due is Stage 3, a watchlisted one is
not Stage 1 -- rather than by reading the code that produced them.

The checks accept either the R column names or this package's, because the two
pipelines name the same quantity differently and a check should not depend on
which produced the frame.

Ids match the R package exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ids import as_id
from ._helpers import blank_detail, col, dup_detail, fail, ok
from .framework import Severity, Validator

__all__ = ["TRANSFORM_STAGE_VALIDATORS"]

_STAGES = {"Stage 1", "Stage 2", "Stage 3"}


def _empty(df) -> bool:
    return df is None or len(df) == 0


def _exposure(df):
    c = col(df, "exposure_amount", "exposure", "on_balance", "ONBALANCE")
    return None if c is None else pd.to_numeric(c, errors="coerce")


def _hierarchy(df, static, rating_col, rating_type):
    """The rating's position on its scale, from the frame or from the static."""
    c = col(df, "rating_hierarchy", "hierarchy", "bucket")
    if c is not None:
        return pd.to_numeric(c, errors="coerce")
    if static is None:
        return None
    scale = static.get("master_rating_scale")
    r = col(df, *rating_col)
    if scale is None or r is None:
        return None
    rt = col(scale, "rating_type")
    sub = scale
    if rt is not None:
        sub = scale[pd.Series(rt).astype(str).str.lower().str.startswith(
            rating_type[:3])]
    names = col(sub, "rating")
    hier = col(sub, "hierarchy")
    if names is None or hier is None:
        return None
    lut = dict(zip(pd.Series(names).astype(str).str.strip(),
                   pd.to_numeric(hier, errors="coerce")))
    return pd.Series(r).astype(str).str.strip().map(lut)


def _scale_names(static, rating_type: str) -> set[str]:
    if static is None:
        return set()
    scale = static.get("master_rating_scale")
    if scale is None or len(scale) == 0:
        return set()
    rt = col(scale, "rating_type")
    sub = scale
    if rt is not None:
        sub = scale[pd.Series(rt).astype(str).str.lower().str.startswith(
            rating_type[:3])]
    out = set(pd.Series(col(sub, "rating")).astype(str).str.strip())
    ext = col(sub, "external_equivalent")
    if ext is not None:
        out |= set(pd.Series(ext).astype(str).str.strip())
    return {v for v in out if v and v.lower() != "nan"}


# ------------------------------------------------------- trans_lending -----
def _v_lend_contract_unique(trans_lending=None):
    if _empty(trans_lending):
        return ok()
    return dup_detail(col(trans_lending, "contract_id", "CONTRACTID"), "contract_id")


def _v_lend_customer_populated(trans_lending=None):
    if _empty(trans_lending):
        return ok()
    return blank_detail(col(trans_lending, "customer_id", "CUSTOMERID"),
                        "customer_id")


def _v_lend_rating_populated(trans_lending=None):
    if _empty(trans_lending):
        return ok()
    c = col(trans_lending, "rating_worst", "rating_after_override", "rating")
    if c is None:
        return fail(0, "no rating column on the lending view")
    return blank_detail(c, "rating")


def _v_lend_rating_in_scale(trans_lending=None, static=None):
    if _empty(trans_lending) or static is None:
        return ok()
    known = _scale_names(static, "internal")
    if not known:
        return ok()
    c = col(trans_lending, "rating_worst", "rating_after_override", "rating")
    if c is None:
        return ok()
    s = pd.Series(c).astype(str).str.strip()
    bad = (s != "") & ~s.isin(known | {"nan"})
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} contract(s) carry a rating outside the internal scale",
                examples=sorted(set(s[bad]))[:10])


def _v_lend_hierarchy_range(trans_lending=None, static=None):
    if _empty(trans_lending):
        return ok()
    h = _hierarchy(trans_lending, static,
                   ("rating_worst", "rating_after_override", "rating"), "internal")
    if h is None:
        return ok()
    bad = h.notna() & ((h < 1) | (h > 21))
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} contract(s) have a rating hierarchy outside 1..21",
                examples=[str(v) for v in h[bad].unique()[:10]])


def _v_lend_exposure_nonneg(trans_lending=None):
    if _empty(trans_lending):
        return ok()
    x = _exposure(trans_lending)
    if x is None:
        return ok()
    bad = x.isna() | (x < 0)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} contract(s) have a missing or negative exposure")


def _v_lend_total_exposure_positive(trans_lending=None):
    if _empty(trans_lending):
        return ok()
    x = _exposure(trans_lending)
    if x is None:
        return ok()
    total = float(x.fillna(0).sum())
    if total > 0:
        return ok()
    return fail(0, f"total lending exposure is {total:,.2f}; the book priced to "
                   "nothing, which is almost always a failed join rather than "
                   "an empty book")


def _v_lend_dpd_nonneg(trans_lending=None):
    if _empty(trans_lending):
        return ok()
    c = col(trans_lending, "past_dues_days", "past_due_days", "dpd")
    if c is None:
        return ok()
    x = pd.to_numeric(c, errors="coerce")
    bad = x.notna() & (x < 0)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} contract(s) have negative days past due")


def _v_lend_product_type_known(trans_lending=None, static=None):
    if _empty(trans_lending) or static is None:
        return ok()
    m = static.get("product_portfolio_mapping")
    if m is None or len(m) == 0:
        return ok()
    known = set(pd.Series(col(m, "product_type", "account_type")
                          ).astype(str).str.strip())
    c = col(trans_lending, "account_type", "product_type")
    if c is None:
        return ok()
    s = pd.Series(c).astype(str).str.strip()
    bad = {v for v in s.unique() if v and v.lower() != "nan"} - known
    if not bad:
        return ok()
    return fail(int(s.isin(bad).sum()),
                f"{len(bad)} product type(s) are not in the mapping",
                examples=sorted(bad)[:10])


def _v_lend_portfolio_complete(trans_lending=None):
    if _empty(trans_lending):
        return ok()
    c = col(trans_lending, "portfolio_code", "portfolio")
    if c is None:
        return fail(0, "no portfolio column on the lending view")
    return blank_detail(c, "portfolio_code")


def _v_lend_overrides_filled(trans_lending=None):
    if _empty(trans_lending):
        return ok()
    missing = []
    for name in (("rating_worst", "rating_after_override"),
                 ("payment_type",), ("portfolio_code", "portfolio")):
        if col(trans_lending, *name) is None:
            missing.append(name[0])
    if not missing:
        return ok()
    return fail(len(missing),
                "the lending view is missing back-filled column(s): "
                + ", ".join(missing))


# ----------------------------------------------- lending portfolio view ----
def _v_pv_customer_unique(lending_view=None):
    if _empty(lending_view):
        return ok()
    return dup_detail(col(lending_view, "customer_id", "CUSTOMERID"), "customer_id")


def _v_pv_stage_in_set(lending_view=None):
    if _empty(lending_view):
        return ok()
    c = col(lending_view, "stage_final", "stage")
    if c is None:
        return ok()
    s = pd.Series(c).astype(str).str.strip()
    bad = ~s.isin(_STAGES)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} customer(s) have a stage outside {sorted(_STAGES)}",
                examples=sorted(set(s[bad]))[:10])


def _v_pv_dpd90_stage3(lending_view=None):
    if _empty(lending_view):
        return ok()
    d = col(lending_view, "worst_dpd", "past_dues_worst", "dpd")
    s = col(lending_view, "stage_final", "stage")
    if d is None or s is None:
        return ok()
    dpd = pd.to_numeric(d, errors="coerce")
    stage = pd.Series(s).astype(str).str.strip()
    bad = (dpd > 90) & (stage != "Stage 3")
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} customer(s) over 90 days past due are not Stage 3",
                examples=list(as_id(col(lending_view, "customer_id"))[bad][:10]))


def _v_pv_clean_low_dpd_stage1(lending_view=None, dpd_threshold: float = 60):
    if _empty(lending_view):
        return ok()
    d = pd.to_numeric(col(lending_view, "worst_dpd", "past_dues_worst", "dpd"),
                      errors="coerce")
    s = pd.Series(col(lending_view, "stage_final", "stage")).astype(str).str.strip()
    w = col(lending_view, "is_watchlist", "watchlist")
    r = col(lending_view, "is_restructured", "restructured", "is_local1")
    if d is None or s is None:
        return ok()
    watch = pd.Series(w).fillna(False).astype(bool) if w is not None \
        else pd.Series(False, index=s.index)
    restr = pd.Series(r).fillna(False).astype(bool) if r is not None \
        else pd.Series(False, index=s.index)
    clean = (d <= dpd_threshold) & ~watch & ~restr
    bad = clean & (s != "Stage 1")
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} clean, low-DPD customer(s) are staged above Stage 1",
                examples=list(as_id(col(lending_view, "customer_id"))[bad][:10]))


def _v_pv_watchlist_stage23(lending_view=None):
    if _empty(lending_view):
        return ok()
    w = col(lending_view, "is_watchlist", "watchlist")
    s = col(lending_view, "stage_final", "stage")
    if w is None or s is None:
        return ok()
    watch = pd.Series(w).fillna(False).astype(bool)
    stage = pd.Series(s).astype(str).str.strip()
    bad = watch & ~stage.isin({"Stage 2", "Stage 3"})
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} watchlisted customer(s) are still Stage 1",
                examples=list(as_id(col(lending_view, "customer_id"))[bad][:10]))


def _v_pv_restructured_stage23(lending_view=None):
    if _empty(lending_view):
        return ok()
    r = col(lending_view, "is_restructured", "restructured", "is_local1")
    s = col(lending_view, "stage_final", "stage")
    if r is None or s is None:
        return ok()
    restr = pd.Series(r).fillna(False).astype(bool)
    stage = pd.Series(s).astype(str).str.strip()
    bad = restr & ~stage.isin({"Stage 2", "Stage 3"})
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} restructured customer(s) are still Stage 1",
                examples=list(as_id(col(lending_view, "customer_id"))[bad][:10]))


def _v_pv_customer_count(lending_view=None, trans_lending=None):
    if _empty(lending_view) or _empty(trans_lending):
        return ok()
    a = len(lending_view)
    b = as_id(col(trans_lending, "customer_id")).nunique()
    if a == b:
        return ok()
    return fail(abs(a - b),
                f"the customer view has {a:,} rows against {b:,} distinct "
                "customers on the contracts")


def _v_pv_exposure_reconciles(lending_view=None, trans_lending=None):
    if _empty(lending_view) or _empty(trans_lending):
        return ok()
    v = col(lending_view, "exposure_total", "exposure", "on_balance")
    t = _exposure(trans_lending)
    if v is None or t is None:
        return ok()
    a = float(pd.to_numeric(v, errors="coerce").fillna(0).sum())
    b = float(t.fillna(0).sum())
    if abs(a - b) <= max(1.0, 1e-6 * max(abs(a), abs(b))):
        return ok()
    return fail(1, f"customer-level exposure {a:,.2f} does not reconcile to "
                   f"contract-level {b:,.2f} (difference {a - b:,.2f})")


# --------------------------------------------------- trans_investments -----
def _v_inv_account_unique(trans_investments=None):
    if _empty(trans_investments):
        return ok()
    return dup_detail(col(trans_investments, "account_id", "contract_id",
                          "CONTRACTID"), "account_id")


def _v_inv_rating_populated(trans_investments=None):
    if _empty(trans_investments):
        return ok()
    c = col(trans_investments, "rating_current", "rating")
    if c is None:
        return ok()
    return blank_detail(c, "rating_current")


def _v_inv_rating_in_scale(trans_investments=None, static=None):
    if _empty(trans_investments) or static is None:
        return ok()
    known = _scale_names(static, "external")
    if not known:
        return ok()
    c = col(trans_investments, "rating_current", "rating")
    if c is None:
        return ok()
    s = pd.Series(c).astype(str).str.strip()
    bad = (s != "") & ~s.isin(known | {"nan"})
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} investment(s) carry a grade outside the external scale",
                examples=sorted(set(s[bad]))[:10])


def _v_inv_hierarchy_range(trans_investments=None, static=None):
    if _empty(trans_investments):
        return ok()
    h = _hierarchy(trans_investments, static, ("rating_current", "rating"),
                   "external")
    if h is None:
        return ok()
    bad = h.notna() & ((h < 1) | (h > 21))
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} investment(s) have a rating hierarchy outside 1..21")


def _v_inv_exposure_nonneg(trans_investments=None):
    if _empty(trans_investments):
        return ok()
    x = _exposure(trans_investments)
    if x is None:
        return ok()
    bad = x.isna() | (x < 0)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} investment(s) have a missing or negative exposure")


def _v_inv_total_exposure_positive(trans_investments=None):
    if _empty(trans_investments):
        return ok()
    x = _exposure(trans_investments)
    if x is None:
        return ok()
    total = float(x.fillna(0).sum())
    if total > 0:
        return ok()
    return fail(0, f"total investment exposure is {total:,.2f}")


def _v_invpv_count(investment_view=None, trans_investments=None):
    if _empty(investment_view) or _empty(trans_investments):
        return ok()
    a, b = len(investment_view), len(trans_investments)
    if a == b:
        return ok()
    return fail(abs(a - b), f"the investment view has {a:,} rows against "
                            f"{b:,} transformed investments")


def _v_invpv_stage_in_set(investment_view=None):
    if _empty(investment_view):
        return ok()
    c = col(investment_view, "stage_final", "stage")
    if c is None:
        return ok()
    s = pd.Series(c).astype(str).str.strip()
    bad = ~s.isin(_STAGES)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} investment(s) have a stage outside {sorted(_STAGES)}",
                examples=sorted(set(s[bad]))[:10])


def _v_invpv_top_tier_stage1(investment_view=None, static=None):
    if _empty(investment_view):
        return ok()
    h = _hierarchy(investment_view, static, ("rating_current", "rating"),
                   "external")
    s = col(investment_view, "stage_final", "stage")
    if h is None or s is None:
        return ok()
    stage = pd.Series(s).astype(str).str.strip()
    bad = h.notna() & (h <= 4) & (stage != "Stage 1")
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} top-tier investment(s) (hierarchy <= 4) are staged "
                   "above Stage 1")


def _v(id, severity, description, fn, context, rationale, remediation):
    return Validator(id=id, severity=severity, description=description, fn=fn,
                     context=context, rationale=rationale,
                     remediation=remediation, tags=("transform",))


TRANSFORM_STAGE_VALIDATORS: list[Validator] = [
    _v("TRANS_LEND_contract_id_unique", Severity.ERROR,
       "Every contract_id in trans_lending is unique", _v_lend_contract_unique,
       "trans_lending",
       "A duplicate after the id transformation means two source contracts "
       "collapsed onto one id, which double-counts exposure.",
       "Check apply_id_substitutions against the source ids."),
    _v("TRANS_LEND_customer_id_populated", Severity.ERROR,
       "Every contract has a non-NA customer_id", _v_lend_customer_populated,
       "trans_lending",
       "The rating and the staging both come from the customer. A contract "
       "with no customer gets neither.",
       "Check the customer join in transform_lending()."),
    _v("TRANS_LEND_rating_populated", Severity.ERROR,
       "Every contract has a rating after the customer back-fill",
       _v_lend_rating_populated, "trans_lending",
       "The lending rating is a CUSTOMER attribute and is not on the account "
       "rows. If the join fails every rating is blank, no bucket resolves, and "
       "the run prices almost nothing while completing normally.",
       "Check that CustomerMaster was read and the ids join."),
    _v("TRANS_LEND_rating_in_internal_scale", Severity.ERROR,
       "Every rating is in the Internal portion of master_rating_scale",
       _v_lend_rating_in_scale, "trans_lending",
       "A grade off the scale resolves to no bucket.",
       "Add the grade to master_rating_scale.csv or correct it at source."),
    _v("TRANS_LEND_hierarchy_in_range", Severity.ERROR,
       "rating_hierarchy is in 1..21 for every contract", _v_lend_hierarchy_range,
       "trans_lending",
       "The hierarchy is the PD bucket key. Outside 1..21 there is no curve.",
       "Check master_rating_scale.csv for a bad hierarchy value."),
    _v("TRANS_LEND_exposure_nonneg", Severity.WARN,
       "exposure_amount is present and >= 0 for every contract",
       _v_lend_exposure_nonneg, "trans_lending",
       "A missing exposure prices to zero; a negative one gives a negative "
       "provision.",
       "Trace the contract back to ONBALANCE in the extract."),
    _v("TRANS_LEND_total_exposure_positive", Severity.ERROR,
       "Total lending exposure > 0", _v_lend_total_exposure_positive,
       "trans_lending",
       "A zero total is the signature of a failed join, not of an empty book.",
       "Check the account extract and the customer join."),
    _v("TRANS_LEND_dpd_nonneg", Severity.WARN,
       "past_dues_days >= 0 for every contract", _v_lend_dpd_nonneg,
       "trans_lending",
       "Negative days past due is a data-entry artefact and distorts staging.",
       "Correct at source, or suppress with a reason if legacy."),
    _v("TRANS_LEND_product_type_known", Severity.INFO,
       "account_type is in product_portfolio_mapping", _v_lend_product_type_known,
       "trans_lending",
       "An unmapped product falls to the default portfolio, which may not be "
       "the intended one.",
       "Extend product_portfolio_mapping.csv."),
    _v("TRANS_LEND_portfolio_mapping_complete", Severity.WARN,
       "Every contract resolves to a portfolio", _v_lend_portfolio_complete,
       "trans_lending",
       "The PD curves are keyed on the six real portfolios. A blank portfolio "
       "resolves to no curve.",
       "Extend product_portfolio_mapping.csv."),
    _v("TRANS_LEND_pass6_overrides_filled", Severity.ERROR,
       "The back-filled columns are present on the lending view",
       _v_lend_overrides_filled, "trans_lending",
       "The worst rating, the payment type and the portfolio are all derived "
       "rather than copied. A missing one silently changes every curve.",
       "Check transform_lending() completed its derivation passes."),

    _v("TRANS_LENDPV_customer_id_unique", Severity.ERROR,
       "Every customer_id in the lending portfolio view is unique",
       _v_pv_customer_unique, "lending_portfolio_view",
       "The view is one row per customer. A duplicate means the grouping "
       "failed and the staging flags are ambiguous.",
       "Check the groupby key in derive_customer_flags()."),
    _v("TRANS_LENDPV_stage_in_set", Severity.ERROR,
       "stage_final is in {Stage 1, Stage 2, Stage 3}", _v_pv_stage_in_set,
       "lending_portfolio_view",
       "Anything else is not a stage LIC will accept.",
       "Check apply_staging_rule()."),
    _v("TRANS_LENDPV_dpd_gt_90_implies_stage3", Severity.ERROR,
       "DPD > 90 implies stage_final = Stage 3", _v_pv_dpd90_stage3,
       "lending_portfolio_view",
       "This is the first rule in the staging order, and it is absolute: a "
       "defaulted customer cannot be pulled back to Stage 2 by also being "
       "watchlisted.",
       "Check the order of the conditions in apply_staging_rule()."),
    _v("TRANS_LENDPV_clean_low_dpd_stage1", Severity.WARN,
       "Clean, low-DPD customers are Stage 1", _v_pv_clean_low_dpd_stage1,
       "lending_portfolio_view",
       "A customer with no trigger at all should not be provisioned at "
       "lifetime. Rows here usually mean a flag is being read from the wrong "
       "column.",
       "Check the watchlist and restructuring joins."),
    _v("TRANS_LENDPV_watchlist_implies_stage_2_or_3", Severity.ERROR,
       "A watchlisted customer is Stage 2 or 3", _v_pv_watchlist_stage23,
       "lending_portfolio_view",
       "Watchlist is a significant-increase trigger by policy.",
       "Check the watchlist flag reaches the staging rule."),
    _v("TRANS_LENDPV_restructured_implies_stage_2_or_3", Severity.ERROR,
       "A restructured customer is Stage 2 or 3", _v_pv_restructured_stage23,
       "lending_portfolio_view",
       "Restructuring is a significant-increase trigger by policy.",
       "Check IsLocal1 in the staging extract is being read as restructuring."),
    _v("TRANS_LENDPV_customer_count_matches_trans", Severity.ERROR,
       "The view has one row per distinct customer on the contracts",
       _v_pv_customer_count, "lending_portfolio_view",
       "A shortfall means customers were dropped in the grouping and their "
       "facilities are staged against nothing.",
       "Check for blank customer ids before the groupby."),
    _v("TRANS_LENDPV_exposure_reconciles", Severity.ERROR,
       "Customer-level exposure equals contract-level exposure",
       _v_pv_exposure_reconciles, "lending_portfolio_view",
       "The two views describe the same book. A difference means one of them "
       "lost rows.",
       "Compare the row counts before and after the grouping."),

    _v("TRANS_INV_account_id_unique", Severity.ERROR,
       "Every account_id in trans_investments is unique", _v_inv_account_unique,
       "trans_investments",
       "Investment ids are surrogate sequential numbers; a duplicate means the "
       "sequence was generated twice.",
       "Check the surrogate id assignment."),
    _v("TRANS_INV_rating_populated", Severity.ERROR,
       "Every investment has a rating_current", _v_inv_rating_populated,
       "trans_investments",
       "Without a grade the holding has no bucket and prices to zero.",
       "Check the counterparty join."),
    _v("TRANS_INV_rating_in_external_scale", Severity.ERROR,
       "Every rating_current is in the External portion of master_rating_scale",
       _v_inv_rating_in_scale, "trans_investments",
       "The external book uses the agency scale, not the QDB ladder.",
       "Add the grade to master_rating_scale.csv."),
    _v("TRANS_INV_hierarchy_in_range", Severity.ERROR,
       "rating_hierarchy is in 1..21 for every investment",
       _v_inv_hierarchy_range, "trans_investments",
       "Outside 1..21 there is no curve. Note the two scales reuse the same "
       "numbers, so the lookup must be on (rating_type, rating).",
       "Check master_rating_scale.csv."),
    _v("TRANS_INV_exposure_nonneg", Severity.WARN,
       "exposure_amount is present and >= 0 for every investment",
       _v_inv_exposure_nonneg, "trans_investments",
       "A missing holding prices to zero.",
       "Trace back to ONBALANCE in the investment extract."),
    _v("TRANS_INV_total_exposure_positive", Severity.ERROR,
       "Total investment exposure > 0", _v_inv_total_exposure_positive,
       "trans_investments",
       "A zero total means the investment extract did not join.",
       "Check AccountMasterInvestments was read."),

    _v("TRANS_INVPV_account_count_matches_trans", Severity.ERROR,
       "The investment view has one row per transformed investment",
       _v_invpv_count, "investment_portfolio_view",
       "The file is keyed on the ACCOUNT, not the counterparty - 73 rows "
       "against 62 counterparties is correct and a shortfall is not.",
       "Check the surrogate id assignment."),
    _v("TRANS_INVPV_stage_in_set", Severity.ERROR,
       "stage_final is in {Stage 1, Stage 2, Stage 3}", _v_invpv_stage_in_set,
       "investment_portfolio_view",
       "Anything else is not a stage LIC will accept.",
       "Check the investment staging rule."),
    _v("TRANS_INVPV_top_tier_implies_stage1", Severity.WARN,
       "Top-tier investments (hierarchy <= 4) are Stage 1",
       _v_invpv_top_tier_stage1, "investment_portfolio_view",
       "An Aaa-to-Aa3 holding staged above 1 is almost always a rating that "
       "failed to resolve rather than a genuine deterioration.",
       "Check the counterparty rating join."),
]
