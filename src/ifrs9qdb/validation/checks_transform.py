"""TRANSFORM stage: the book after it has been shaped, before pricing.

A literal port of R/validators_transform.R -- the same 28 checks, in R's
order, reading the same columns of the same intermediates (built in R's shape
by ``r_frames``), with the same messages. The previous Python checks read
Python's own frames with fallbacks and passed wherever a column was missing;
on a book where a contract's customer is absent from CustomerMaster they
passed three checks R fails.

Args, as R names them: ``trans_l``, ``cm_view``, ``trans_i``, ``inv_view``,
``static``.
"""
from __future__ import annotations

import math

import pandas as pd

from ._helpers import col, dup_detail, text
from ._texts import STAGE_TEXTS
from .framework import Severity, Validator

__all__ = ["TRANSFORM_STAGE_VALIDATORS", "r_format"]


def r_format(x) -> str:
    """R's format() of a number: 7 significant digits, fixed unless
    scientific is narrower, trailing zeros dropped."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "NA"
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    decimals = max(0, 6 - e)
    fixed = f"{x:.{decimals}f}"
    if "." in fixed:
        fixed = fixed.rstrip("0").rstrip(".")
    mant, exp = f"{x:.6e}".split("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    sci = f"{mant}e{int(exp):+03d}"
    return fixed if len(fixed) <= len(sci) else sci


def _ok():
    return {"passed": True}


def _bad(message, count=0):
    return {"passed": False, "count": int(count), "detail": message}


def _c(df, name):
    """R's .get_col(): the column, or None when the frame or column is absent."""
    if df is None or name not in getattr(df, "columns", []):
        return None
    return df[name]


def _na(s) -> pd.Series:
    s = pd.Series(s)
    return s.isna() | s.astype(object).map(lambda v: v is None)


# ---------------------------------------------------------------- lending ---
def _lend_contract_unique(trans_l=None):
    ids = _c(trans_l, "contract_id")
    if ids is None:
        return _ok()
    ids = pd.Series(ids)
    dup = ids[ids.duplicated()]
    if dup.empty:
        return _ok()
    return _bad(f"{len(dup)} duplicate contract_id (e.g. "
                f"{', '.join(map(str, pd.unique(dup)[:5]))})", len(dup))


def _lend_customer_populated(trans_l=None):
    ids = _c(trans_l, "customer_id")
    if ids is None:
        return _ok()
    n = int((text(ids) == "").sum())
    return _ok() if n == 0 else _bad(f"{n} contracts have NA/blank customer_id", n)


def _lend_rating_populated(trans_l=None):
    r = _c(trans_l, "rating")
    if r is None:
        return _ok()
    n = int(_na(r).sum())
    return _ok() if n == 0 else _bad(f"{n} contracts have NA rating", n)


def _scale_set(static, rating_type):
    ms = static.get("master_rating_scale") if static is not None else None
    if ms is None:
        return set()
    return set(text(col(ms, "rating"))[text(col(ms, "rating_type")) == rating_type])


def _rating_in_scale(series, static, rating_type):
    if series is None:
        return _ok()
    ok_set = _scale_set(static, rating_type)
    vals = pd.Series(series)[~_na(series)]
    bad = [v for v in pd.unique(vals) if v not in ok_set]
    return _ok() if not bad else _bad("Unknown ratings: " + ", ".join(map(str, bad)),
                                      len(bad))


def _hierarchy_range(h, noun):
    if h is None:
        return _ok()
    h = pd.to_numeric(h, errors="coerce")
    n = int((h.notna() & ((h < 1) | (h > 21))).sum())
    return _ok() if n == 0 else _bad(f"{n} {noun} have hierarchy outside 1..21", n)


def _exposure_nonneg(e):
    if e is None:
        return _ok()
    e = pd.to_numeric(e, errors="coerce")
    n_na, n_neg = int(e.isna().sum()), int((e < 0).sum())
    if n_na == 0 and n_neg == 0:
        return _ok()
    return _bad(f"{n_na} NA, {n_neg} negative", n_na + n_neg)


def _total_exposure_positive(e):
    if e is None:
        return _ok()
    tot = float(pd.to_numeric(e, errors="coerce").sum())
    return _ok() if tot > 0 else _bad(f"Total exposure = {r_format(tot)}")


def _lend_dpd_nonneg(trans_l=None):
    d = _c(trans_l, "past_dues_days")
    if d is None:
        return _ok()
    n = int((pd.to_numeric(d, errors="coerce") < 0).sum())
    return _ok() if n == 0 else _bad(f"{n} contracts have negative DPD", n)


def _product_types(static):
    m = static.get("product_portfolio_mapping") if static is not None else None
    if m is None or len(m) == 0:
        return None
    return set(text(col(m, "product_type")))


def _lend_product_type_known(trans_l=None, static=None):
    a = _c(trans_l, "account_type")
    if a is None:
        return _ok()
    known = _product_types(static)
    if known is None:
        return _bad("product_portfolio_mapping is empty")
    obs = text(a)
    bad = [v for v in pd.unique(obs[obs != ""]) if v not in known]
    return _ok() if not bad else _bad(
        f"{len(bad)} product types not in product_portfolio_mapping: "
        + ", ".join(bad), len(bad))


def _lend_portfolio_complete(trans_l=None, static=None):
    a = _c(trans_l, "account_type")
    known = _product_types(static)
    if a is None or known is None:
        return _ok()
    obs = text(a)
    orphan = (obs != "") & ~obs.isin(known)
    n = int(orphan.sum())
    if n == 0:
        return _ok()
    counts = obs[orphan].value_counts().sort_index()
    return _bad(f"{n} contracts have product types not mapped to a portfolio: "
                + ", ".join(f"{k} ({int(v)})" for k, v in counts.items()), n)


def _lend_pass6(trans_l=None):
    r = _c(trans_l, "rating_after_override")
    d = _c(trans_l, "is_default_final")
    missing = []
    if r is None or bool(_na(r).any()):
        missing.append("rating_after_override")
    if d is None or bool(_na(d).any()):
        missing.append("is_default_final")
    return _ok() if not missing else _bad("Unfilled: " + ", ".join(missing))


# ------------------------------------------------------- lending view (cm) ---
def _pv_customer_unique(cm_view=None):
    ids = _c(cm_view, "customer_id")
    if ids is None:
        return _ok()
    n = int(pd.Series(ids).duplicated().sum())
    return _ok() if n == 0 else _bad(f"{n} duplicate customer_id", n)


def _stage_in_set(s):
    if s is None:
        return _ok()
    vals = pd.Series(s)[~_na(s)]
    bad = [v for v in pd.unique(vals) if v not in ("Stage 1", "Stage 2", "Stage 3")]
    return _ok() if not bad else _bad("Unknown stages: " + ", ".join(map(str, bad)),
                                      len(bad))


def _pv_dpd90(cm_view=None):
    s, d = _c(cm_view, "stage_final"), _c(cm_view, "dpd_status")
    if s is None or d is None:
        return _ok()
    d = pd.to_numeric(d, errors="coerce")
    n = int((d.notna() & (d > 90) & (pd.Series(s) != "Stage 3")).sum())
    return _ok() if n == 0 else _bad(
        f"{n} customers have DPD>90 but stage != Stage 3", n)


def _threshold(static, default=60.0):
    try:
        th = static.get("staging_thresholds")
        k = text(col(th, "key"))
        v = pd.to_numeric(col(th, "value")[k == "dpd_stage2_threshold_days"],
                          errors="coerce").dropna()
        return float(v.iloc[0]) if len(v) else default
    except Exception:
        return default


def _blank(x) -> pd.Series:
    x = pd.Series(x)
    return _na(x) | (x.astype(object) == "")


def _pv_clean_low_dpd(cm_view=None, static=None):
    s, d = _c(cm_view, "stage_final"), _c(cm_view, "dpd_status")
    r, w = _c(cm_view, "restructuring_final"), _c(cm_view, "watchlist_status")
    if s is None or d is None or r is None or w is None:
        return _ok()
    d = pd.to_numeric(d, errors="coerce")
    clean = d.notna() & (d <= _threshold(static)) & _blank(r) & _blank(w)
    n = int((clean & (pd.Series(s) != "Stage 1")).sum())
    return _ok() if n == 0 else _bad(f"{n} clean low-DPD customers not in Stage 1", n)


def _pv_watchlist(cm_view=None):
    s, f = _c(cm_view, "stage_final"), _c(cm_view, "watchlist_status")
    if s is None or f is None:
        return _ok()
    n = int(((pd.Series(f) == "Watchlist").fillna(False)
             & ~pd.Series(s).isin(["Stage 2", "Stage 3"])).sum())
    return _ok() if n == 0 else _bad(f"{n} watchlist customers not in Stage 2/3", n)


def _pv_restructured(cm_view=None):
    s, f = _c(cm_view, "stage_final"), _c(cm_view, "restructuring_final")
    if s is None or f is None:
        return _ok()
    n = int(((pd.Series(f) == "Restructured").fillna(False)
             & ~pd.Series(s).isin(["Stage 2", "Stage 3"])).sum())
    return _ok() if n == 0 else _bad(f"{n} restructured customers not in Stage 2/3", n)


def _pv_customer_count(trans_l=None, cm_view=None):
    if trans_l is None or cm_view is None:
        return _ok()
    n_cm = len(cm_view)
    n_trans = int(pd.Series(trans_l["customer_id"]).nunique(dropna=False))
    return _ok() if n_cm == n_trans else _bad(
        f"cm_view={n_cm}, distinct trans customers={n_trans}")


def _pv_exposure_reconciles(trans_l=None, cm_view=None):
    if trans_l is None or cm_view is None:
        return _ok()
    a = float(pd.to_numeric(_c(cm_view, "exposure_total"), errors="coerce").sum())
    b = float(pd.to_numeric(_c(trans_l, "exposure_amount"), errors="coerce").sum())
    tol = max(1e-2, abs(b) * 1e-9)
    if abs(a - b) <= tol:
        return _ok()
    return _bad(f"cm_view sum={r_format(a)}, trans sum={r_format(b)}, "
                f"diff={r_format(a - b)}")


# ------------------------------------------------------------ investments ---
def _inv_account_unique(trans_i=None):
    ids = _c(trans_i, "account_id")
    if ids is None:
        return _ok()
    return dup_detail(ids, "account_id")


def _inv_rating_populated(trans_i=None):
    r = _c(trans_i, "rating_current")
    if r is None:
        return _ok()
    n = int(_na(r).sum())
    return _ok() if n == 0 else _bad(f"{n} investments have NA rating_current", n)


def _invpv_count(trans_i=None, inv_view=None):
    if trans_i is None or inv_view is None:
        return _ok()
    return _ok() if len(inv_view) == len(trans_i) else _bad(
        f"inv_view={len(inv_view)}, trans_i={len(trans_i)}")


def _invpv_top_tier(trans_i=None, inv_view=None):
    h, s = _c(trans_i, "rating_hierarchy"), _c(inv_view, "stage_final")
    if h is None or s is None or len(inv_view) != len(trans_i):
        return _ok()
    h = pd.to_numeric(h, errors="coerce").reset_index(drop=True)
    s = pd.Series(s).reset_index(drop=True)
    n = int((h.notna() & (h <= 4) & (s != "Stage 1")).sum())
    return _ok() if n == 0 else _bad(f"{n} top-tier investments not in Stage 1", n)


def _v(id, severity, description, fn, context, rationale="", remediation=""):
    why, fix = STAGE_TEXTS.get(id, ("", ""))
    return Validator(id=id, severity=severity, description=description, fn=fn,
                     context=context, rationale=rationale or why,
                     remediation=remediation or fix, tags=("transform",))


def _lend_rating_scale(trans_l=None, static=None):
    return _rating_in_scale(_c(trans_l, "rating"), static, "Internal")


def _lend_hierarchy(trans_l=None):
    return _hierarchy_range(_c(trans_l, "rating_hierarchy"), "contracts")


def _lend_exposure(trans_l=None):
    return _exposure_nonneg(_c(trans_l, "exposure_amount"))


def _lend_total(trans_l=None):
    return _total_exposure_positive(_c(trans_l, "exposure_amount"))


def _pv_stage(cm_view=None):
    return _stage_in_set(_c(cm_view, "stage_final"))


def _inv_rating_scale(trans_i=None, static=None):
    return _rating_in_scale(_c(trans_i, "rating_current"), static, "External")


def _inv_hierarchy(trans_i=None):
    return _hierarchy_range(_c(trans_i, "rating_hierarchy"), "investments")


def _inv_exposure(trans_i=None):
    return _exposure_nonneg(_c(trans_i, "exposure_amount"))


def _inv_total(trans_i=None):
    return _total_exposure_positive(_c(trans_i, "exposure_amount"))


def _invpv_stage(inv_view=None):
    return _stage_in_set(_c(inv_view, "stage_final"))


TRANSFORM_STAGE_VALIDATORS: list[Validator] = [
    _v("TRANS_LEND_contract_id_unique", Severity.ERROR,
       "Every contract_id in trans_lending is unique",
       _lend_contract_unique, "trans_lending",
       "A contract that survives the transformation twice is priced twice."),
    _v("TRANS_LEND_customer_id_populated", Severity.ERROR,
       "Every contract has a non-NA customer_id",
       _lend_customer_populated, "trans_lending",
       "Rating, staging and contagion are all customer attributes."),
    _v("TRANS_LEND_rating_populated", Severity.ERROR,
       "Every contract has a non-NA rating after Pass 6 back-fill",
       _lend_rating_populated, "trans_lending",
       "A contract with no rating resolves no PD bucket."),
    _v("TRANS_LEND_rating_in_internal_scale", Severity.ERROR,
       "Every rating is in the Internal portion of master_rating_scale",
       _lend_rating_scale, "trans_lending",
       "A rating outside the internal scale has no hierarchy and no PD curve."),
    _v("TRANS_LEND_hierarchy_in_range", Severity.ERROR,
       "rating_hierarchy is in 1..21 for every contract",
       _lend_hierarchy, "trans_lending"),
    _v("TRANS_LEND_exposure_nonneg", Severity.WARN,
       "exposure_amount is non-NA and >= 0 for every contract",
       _lend_exposure, "trans_lending",
       "A blank balance is carried as 0 on the transformation; a negative one "
       "produces a negative provision."),
    _v("TRANS_LEND_total_exposure_positive", Severity.ERROR,
       "Total lending exposure > 0",
       _lend_total, "trans_lending"),
    _v("TRANS_LEND_dpd_nonneg", Severity.WARN,
       "past_dues_days >= 0 for every contract",
       _lend_dpd_nonneg, "trans_lending"),
    _v("TRANS_LEND_product_type_known", Severity.INFO,
       "account_type (product type) is in product_portfolio_mapping",
       _lend_product_type_known, "trans_lending",
       "When a product type is missing from the mapping, contracts of that type "
       "fall back to the default 'Business Finance' portfolio.",
       "Add the missing product types to product_portfolio_mapping.csv, or "
       "confirm the fallback is intended and suppress this validator."),
    _v("TRANS_LEND_portfolio_mapping_complete", Severity.WARN,
       "Every contract's account_type maps to a portfolio (no orphans)",
       _lend_portfolio_complete, "trans_lending",
       "Contracts of an unmapped product type fall through to the default "
       "'Business Finance' bucket and are mis-bucketed in AccountMaster_1.csv.",
       "Update product_portfolio_mapping.csv to cover every observed type."),
    _v("TRANS_LEND_pass6_overrides_filled", Severity.ERROR,
       "Pass 6 back-fills (rating_after_override, is_default_final) populated",
       _lend_pass6, "trans_lending",
       "The back-fill comes from the customer view, which is keyed on "
       "CustomerMaster: a contract whose customer is missing there has none."),
    _v("TRANS_LENDPV_customer_id_unique", Severity.ERROR,
       "Every customer_id in lending portfolio view is unique",
       _pv_customer_unique, "lending_portfolio_view"),
    _v("TRANS_LENDPV_stage_in_set", Severity.ERROR,
       "stage_final is in {Stage 1, Stage 2, Stage 3}",
       _pv_stage, "lending_portfolio_view"),
    _v("TRANS_LENDPV_dpd_gt_90_implies_stage3", Severity.ERROR,
       "DPD > 90 implies stage_final = Stage 3",
       _pv_dpd90, "lending_portfolio_view"),
    _v("TRANS_LENDPV_clean_low_dpd_stage1", Severity.WARN,
       "Clean low-DPD customers (DPD<=60, no watchlist, not restructured) -> Stage 1",
       _pv_clean_low_dpd, "lending_portfolio_view"),
    _v("TRANS_LENDPV_watchlist_implies_stage_2_or_3", Severity.ERROR,
       "watchlist_status='Watchlist' implies stage in {Stage 2, Stage 3}",
       _pv_watchlist, "lending_portfolio_view"),
    _v("TRANS_LENDPV_restructured_implies_stage_2_or_3", Severity.ERROR,
       "restructuring_final='Restructured' implies stage in {Stage 2, Stage 3}",
       _pv_restructured, "lending_portfolio_view"),
    _v("TRANS_LENDPV_customer_count_matches_trans", Severity.ERROR,
       "cm_view row count == distinct customer_id in trans_lending",
       _pv_customer_count, "lending_portfolio_view",
       "The customer view is keyed on CustomerMaster; a contract whose customer "
       "is not there is missing from it, and so from staging."),
    _v("TRANS_LENDPV_exposure_reconciles", Severity.ERROR,
       "Sum of cm_view exposure_total == sum of trans_lending exposure_amount",
       _pv_exposure_reconciles, "lending_portfolio_view"),
    _v("TRANS_INV_account_id_unique", Severity.ERROR,
       "Every account_id in trans_investments is unique",
       _inv_account_unique, "trans_investments"),
    _v("TRANS_INV_rating_populated", Severity.ERROR,
       "Every investment has a non-NA rating_current",
       _inv_rating_populated, "trans_investments"),
    _v("TRANS_INV_rating_in_external_scale", Severity.ERROR,
       "Every rating_current is in the External portion of master_rating_scale",
       _inv_rating_scale, "trans_investments"),
    _v("TRANS_INV_hierarchy_in_range", Severity.ERROR,
       "rating_hierarchy is in 1..21 for every investment",
       _inv_hierarchy, "trans_investments"),
    _v("TRANS_INV_exposure_nonneg", Severity.WARN,
       "exposure_amount is non-NA and >= 0 for every investment",
       _inv_exposure, "trans_investments"),
    _v("TRANS_INV_total_exposure_positive", Severity.ERROR,
       "Total investment exposure > 0",
       _inv_total, "trans_investments"),
    _v("TRANS_INVPV_account_count_matches_trans", Severity.ERROR,
       "inv_view row count == nrow(trans_investments)",
       _invpv_count, "investment_portfolio_view"),
    _v("TRANS_INVPV_stage_in_set", Severity.ERROR,
       "stage_final is in {Stage 1, Stage 2, Stage 3}",
       _invpv_stage, "investment_portfolio_view"),
    _v("TRANS_INVPV_top_tier_implies_stage1", Severity.WARN,
       "Top-tier investments (rating_hierarchy <= 4) -> Stage 1",
       _invpv_top_tier, "investment_portfolio_view"),
]
