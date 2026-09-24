"""Derived-stage checks: the curves and weights the engine prices against.

These are the last chance to catch a number that is well formed and wrong. A
scenario weighting that silently falls back to an equal split produces a curve
set that is monotonic, bounded, correctly shaped and out by a mean of 0.0058 --
so the checks here assert the things a wrong weighting does NOT preserve: that
the weights sum to one, that the two scales differ from each other, and that
the curves order correctly by rating.

Ids match the R package exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ids import as_id
from ._helpers import col, fail, ok
from .framework import Severity, Validator

__all__ = ["DERIVED_STAGE_VALIDATORS"]

LTPO_SCHEMA = ["ExtractDate", "ContractId", "MonthLifetime", "EADLifetime",
               "LGDLifetime", "PaymentScheduleLifetime", "TotalLimitLifetime"]
STPD_SCHEMA = ["ExtractDate", "PortfolioCode", "PDBucketDim1", "PDBucketDim2",
               "MonthLifetime", "PDLifetime"]
INTERNAL_PORTFOLIOS = ["Business Finance", "Off BS", "Al Dhameen", "Tasdeer"]
EXTERNAL_PORTFOLIOS = ["Banks and Fis", "Investments"]
N_BUCKETS, N_MONTHS = 21, 600
STPD_ROWS = 6 * N_BUCKETS * N_MONTHS          # 75,600


def _empty(df) -> bool:
    return df is None or len(df) == 0


def _schema(df, expected, name):
    if _empty(df):
        return ok()
    got = list(df.columns)
    if got == expected:
        return ok()
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    bits = []
    if missing:
        bits.append(f"missing {missing}")
    if extra:
        bits.append(f"unexpected {extra}")
    if not bits:
        bits.append(f"column ORDER differs: {got}")
    return fail(len(missing) + len(extra), f"{name} schema: " + "; ".join(bits))


def _one_extract_date(df, name):
    if _empty(df):
        return ok()
    c = col(df, "ExtractDate", "extract_date")
    if c is None:
        return fail(0, f"{name} has no ExtractDate column")
    vals = sorted({str(v).strip() for v in pd.Series(c).dropna().unique()})
    if len(vals) <= 1:
        return ok()
    return fail(len(vals), f"{name} carries {len(vals)} different extract dates",
                examples=vals[:10])


# ------------------------------------------------------------------ LTPO ---
def _v_ltpo_schema(ltpo=None):
    return _schema(ltpo, LTPO_SCHEMA, "LifeTimeParameterOther")


def _v_ltpo_extract_date_unique(ltpo=None):
    return _one_extract_date(ltpo, "LifeTimeParameterOther")


def _v_ltpo_ead_nonneg(ltpo=None):
    if _empty(ltpo):
        return ok()
    x = pd.to_numeric(col(ltpo, "EADLifetime"), errors="coerce")
    bad = x.isna() | (x < 0)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} curve point(s) have a missing or negative EAD")


def _v_ltpo_month_starts_at_zero(ltpo=None):
    if _empty(ltpo):
        return ok()
    cid = as_id(col(ltpo, "ContractId"))
    m = pd.to_numeric(col(ltpo, "MonthLifetime"), errors="coerce")
    first = pd.DataFrame({"c": cid, "m": m}).groupby("c")["m"].min()
    bad = first[first != 0]
    if bad.empty:
        return ok()
    return fail(len(bad), f"{len(bad)} contract(s) do not start at month 0",
                examples=[f"{k} starts at {v}" for k, v in bad.head(10).items()])


def _v_ltpo_months_contiguous(ltpo=None):
    if _empty(ltpo):
        return ok()
    d = pd.DataFrame({"c": as_id(col(ltpo, "ContractId")),
                      "m": pd.to_numeric(col(ltpo, "MonthLifetime"),
                                         errors="coerce")})
    g = d.groupby("c")["m"].agg(["min", "max", "count", "nunique"])
    bad = g[(g["min"] != 0) | (g["nunique"] != g["count"])
            | (g["max"] != g["count"] - 1)]
    if bad.empty:
        return ok()
    return fail(len(bad),
                f"{len(bad)} contract(s) have a gap, a duplicate or a bad start "
                "in their month sequence",
                examples=list(bad.index[:10]))


def _v_ltpo_ead_nonincreasing(ltpo=None):
    if _empty(ltpo):
        return ok()
    d = pd.DataFrame({"c": as_id(col(ltpo, "ContractId")),
                      "m": pd.to_numeric(col(ltpo, "MonthLifetime"),
                                         errors="coerce"),
                      "v": pd.to_numeric(col(ltpo, "EADLifetime"),
                                         errors="coerce")}).sort_values(["c", "m"])
    rises = d.groupby("c")["v"].apply(lambda s: bool((s.diff() > 1e-6).any()))
    up = rises[rises]
    if up.empty:
        return ok()
    return fail(len(up),
                f"{len(up)} curve(s) rise above their opening balance. For a "
                "term loan that is wrong; for a revolving or off-balance "
                "facility it is the schedule carrying committed but undrawn "
                "amounts, and it is why the engine caps ECL at exposure",
                examples=list(up.index[:10]))


def _v_ltpo_contracts_subset(ltpo=None, trans_lending=None):
    if _empty(ltpo) or _empty(trans_lending):
        return ok()
    have = set(as_id(col(trans_lending, "contract_id", "CONTRACTID")))
    used = as_id(col(ltpo, "ContractId"))
    orphan = sorted(set(used[used != ""]) - have)
    if not orphan:
        return ok()
    return fail(len(orphan),
                f"{len(orphan)} curve(s) are for contracts not on the book",
                examples=orphan[:10])


def _v_ltpo_month0_reconciles(ltpo=None, trans_lending=None):
    if _empty(ltpo) or _empty(trans_lending):
        return ok()
    m = pd.to_numeric(col(ltpo, "MonthLifetime"), errors="coerce")
    v = pd.to_numeric(col(ltpo, "EADLifetime"), errors="coerce")
    cid = as_id(col(ltpo, "ContractId"))
    zero = pd.DataFrame({"c": cid, "v": v})[m == 0]
    bal_col = col(trans_lending, "on_balance", "exposure_amount", "ONBALANCE")
    if bal_col is None:
        return ok()
    book = pd.DataFrame({"c": as_id(col(trans_lending, "contract_id",
                                        "CONTRACTID")),
                         "b": pd.to_numeric(bal_col, errors="coerce")})
    j = zero.merge(book, on="c", how="inner")
    if len(j) == 0:
        return ok()
    a, b = float(j["v"].fillna(0).sum()), float(j["b"].fillna(0).sum())
    if abs(a - b) <= max(1.0, 1e-6 * max(abs(a), abs(b))):
        return ok()
    return fail(1, f"month-0 EAD totals {a:,.2f} against an on-balance total of "
                   f"{b:,.2f} for the same contracts (difference {a - b:,.2f}). "
                   "Month 0 is today's outstanding from the account master, not "
                   "the schedule's first figure")


# ------------------------------------------------------------------ StPD ---
def _v_stpd_schema(stpd=None):
    return _schema(stpd, STPD_SCHEMA, "StPD")


def _v_stpd_row_count(stpd=None):
    if _empty(stpd):
        return ok()
    if len(stpd) == STPD_ROWS:
        return ok()
    return fail(abs(len(stpd) - STPD_ROWS),
                f"StPD has {len(stpd):,} rows against the expected "
                f"{STPD_ROWS:,} (6 portfolios x {N_BUCKETS} buckets x "
                f"{N_MONTHS} months)")


def _v_stpd_extract_date_unique(stpd=None):
    return _one_extract_date(stpd, "StPD")


def _v_stpd_portfolio_set(stpd=None):
    if _empty(stpd):
        return ok()
    seen = set(pd.Series(col(stpd, "PortfolioCode")).astype(str).str.strip())
    want = set(INTERNAL_PORTFOLIOS + EXTERNAL_PORTFOLIOS)
    missing, extra = sorted(want - seen), sorted(seen - want)
    if not missing and not extra:
        return ok()
    bits = []
    if missing:
        bits.append(f"missing {missing}")
    if extra:
        bits.append(f"unexpected {extra}")
    return fail(len(missing) + len(extra), "StPD portfolios: " + "; ".join(bits))


def _v_stpd_bucket_set(stpd=None):
    if _empty(stpd):
        return ok()
    d = pd.DataFrame({"p": pd.Series(col(stpd, "PortfolioCode")).astype(str),
                      "b": pd.to_numeric(col(stpd, "PDBucketDim1"),
                                         errors="coerce")})
    want = set(range(1, N_BUCKETS + 1))
    bad = {p: sorted(want - set(g["b"].dropna().astype(int)))
           for p, g in d.groupby("p")}
    bad = {p: m for p, m in bad.items() if m}
    if not bad:
        return ok()
    return fail(sum(len(m) for m in bad.values()),
                f"{len(bad)} portfolio(s) are missing buckets. A TTC PD of ZERO "
                "must still get a bucket - filtering on > 0 instead of >= 0 "
                "loses the three highest external grades",
                examples=[f"{p}: {m}" for p, m in list(bad.items())[:10]])


def _v_stpd_month_set(stpd=None):
    if _empty(stpd):
        return ok()
    d = pd.DataFrame({"p": pd.Series(col(stpd, "PortfolioCode")).astype(str),
                      "b": pd.to_numeric(col(stpd, "PDBucketDim1"), errors="coerce"),
                      "m": pd.to_numeric(col(stpd, "MonthLifetime"), errors="coerce")})
    g = d.groupby(["p", "b"])["m"].nunique()
    bad = g[g != N_MONTHS]
    if bad.empty:
        return ok()
    return fail(len(bad), f"{len(bad)} (portfolio, bucket) pair(s) do not carry "
                          f"all {N_MONTHS} months",
                examples=[f"{k}={v}" for k, v in bad.head(10).items()])


def _v_stpd_dim2_all_na(stpd=None):
    if _empty(stpd):
        return ok()
    c = col(stpd, "PDBucketDim2")
    if c is None:
        return ok()
    s = pd.Series(c)
    populated = s.notna() & (s.astype(str).str.strip() != "")
    n = int(populated.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} row(s) populate PDBucketDim2, which the output schema "
                   "leaves empty")


def _v_stpd_pd_finite(stpd=None):
    if _empty(stpd):
        return ok()
    x = pd.to_numeric(col(stpd, "PDLifetime"), errors="coerce")
    bad = x.isna() | ~np.isfinite(x.fillna(np.inf)) | (x < 0)
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} PD value(s) are missing, infinite or negative")


def _v_stpd_pd_bounded(stpd=None):
    if _empty(stpd):
        return ok()
    x = pd.to_numeric(col(stpd, "PDLifetime"), errors="coerce")
    bad = x > 1.001
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} PD value(s) exceed 1.001. The monthly conversion is a "
                   "running SUM rather than the survival formula, so it can "
                   "pass 1 and is capped; values above the cap mean the cap "
                   "did not apply",
                examples=[f"{v:.6f}" for v in x[bad].head(10)])


def _v_stpd_monotone(stpd=None):
    if _empty(stpd):
        return ok()
    d = pd.DataFrame({"p": pd.Series(col(stpd, "PortfolioCode")).astype(str),
                      "b": pd.to_numeric(col(stpd, "PDBucketDim1"), errors="coerce"),
                      "m": pd.to_numeric(col(stpd, "MonthLifetime"), errors="coerce"),
                      "v": pd.to_numeric(col(stpd, "PDLifetime"), errors="coerce")}
                     ).sort_values(["p", "b", "m"])
    drops = d.groupby(["p", "b"])["v"].apply(
        lambda s: bool((s.diff() < -1e-12).any()))
    bad = drops[drops]
    if bad.empty:
        return ok()
    return fail(len(bad), f"{len(bad)} curve(s) fall as maturity grows; a "
                          "cumulative default probability cannot decrease",
                examples=[str(k) for k in bad.index[:10]])


def _v_stpd_pd_increases_with_hierarchy(stpd=None):
    if _empty(stpd):
        return ok()
    d = pd.DataFrame({"p": pd.Series(col(stpd, "PortfolioCode")).astype(str),
                      "b": pd.to_numeric(col(stpd, "PDBucketDim1"), errors="coerce"),
                      "m": pd.to_numeric(col(stpd, "MonthLifetime"), errors="coerce"),
                      "v": pd.to_numeric(col(stpd, "PDLifetime"), errors="coerce")})
    probe = d[d["m"] == 12]
    if probe.empty:
        probe = d[d["m"] == d["m"].max()]
    bad = []
    for p, g in probe.groupby("p"):
        s = g.sort_values("b")["v"].to_numpy()
        if len(s) > 1 and (np.diff(s) < -1e-9).any():
            bad.append(p)
    if not bad:
        return ok()
    return fail(len(bad), f"{len(bad)} portfolio(s) do not order by rating: a "
                          "worse grade should carry a higher PD",
                examples=bad[:10])


def _identical_curves(stpd, portfolios, label):
    if _empty(stpd):
        return ok()
    d = pd.DataFrame({"p": pd.Series(col(stpd, "PortfolioCode")).astype(str).str.strip(),
                      "b": pd.to_numeric(col(stpd, "PDBucketDim1"), errors="coerce"),
                      "m": pd.to_numeric(col(stpd, "MonthLifetime"), errors="coerce"),
                      "v": pd.to_numeric(col(stpd, "PDLifetime"), errors="coerce")})
    have = [p for p in portfolios if p in set(d["p"])]
    if len(have) < 2:
        return ok()
    piv = d[d["p"].isin(have)].pivot_table(index=["b", "m"], columns="p",
                                           values="v")
    spread = piv.max(axis=1) - piv.min(axis=1)
    n = int((spread > 1e-12).sum())
    if n == 0:
        return ok()
    return fail(n, f"the {label} portfolios do not share one curve set "
                   f"({n:,} points differ). A curve belongs to the rating "
                   "SCALE, not to the portfolio",
                examples=[f"max spread {spread.max():.3e}"])


def _v_stpd_internal_identical(stpd=None):
    return _identical_curves(stpd, INTERNAL_PORTFOLIOS, "internal")


def _v_stpd_external_identical(stpd=None):
    return _identical_curves(stpd, EXTERNAL_PORTFOLIOS, "external")


def _v_stpd_zero_ttc_zero_curve(stpd=None, static=None):
    if _empty(stpd) or static is None:
        return ok()
    ttc = static.get("ttc_pd_table")
    scale = static.get("master_rating_scale")
    if ttc is None or scale is None:
        return ok()
    rt = pd.to_numeric(col(ttc, "rating_type"), errors="coerce")
    zero_names = set(pd.Series(col(ttc, "rating"))[
        (pd.to_numeric(col(ttc, "ttc_pd"), errors="coerce") == 0)
        & (rt == 2)].astype(str).str.strip())
    if not zero_names:
        return ok()
    hier = dict(zip(pd.Series(col(scale, "rating")).astype(str).str.strip(),
                    pd.to_numeric(col(scale, "hierarchy"), errors="coerce")))
    buckets = {int(hier[n]) for n in zero_names if n in hier and pd.notna(hier[n])}
    if not buckets:
        return ok()
    d = pd.DataFrame({"p": pd.Series(col(stpd, "PortfolioCode")).astype(str).str.strip(),
                      "b": pd.to_numeric(col(stpd, "PDBucketDim1"), errors="coerce"),
                      "v": pd.to_numeric(col(stpd, "PDLifetime"), errors="coerce")})
    sub = d[d["p"].isin(EXTERNAL_PORTFOLIOS) & d["b"].isin(buckets)]
    bad = sub[sub["v"].abs() > 1e-12]
    if bad.empty:
        return ok()
    return fail(len(bad), f"{len(bad):,} point(s) in a zero-TTC external bucket "
                          "carry a non-zero PD",
                examples=sorted({int(b) for b in bad["b"].unique()})[:10])


# ---------------------------------------------------------------- weights ---
def _weights_sum(weights, label, tol=1e-4):
    if weights is None:
        return ok()
    s = pd.Series(weights, dtype=float) if not isinstance(weights, pd.Series) \
        else weights.astype(float)
    if len(s) == 0:
        return ok()
    total = float(s.sum())
    if abs(total - 1.0) <= tol:
        return ok()
    return fail(1, f"{label} sum to {total:.6f}, not 1.0. An equal split is the "
                   "classic silent failure here: it is well formed and always "
                   "wrong")


def _weights_nonneg(weights, label):
    if weights is None:
        return ok()
    s = pd.Series(weights, dtype=float)
    bad = s[s < 0]
    if bad.empty:
        return ok()
    return fail(len(bad), f"{len(bad)} {label} are negative",
                examples=[f"{k}={v}" for k, v in bad.head(10).items()])


def _v_scen_internal_sum(internal_weights=None):
    return _weights_sum(internal_weights, "internal scenario weights",
                        tol=1e-3)


def _v_scen_internal_nonneg(internal_weights=None):
    return _weights_nonneg(internal_weights, "internal scenario weights")


def _v_scen_external_per_year_sum(external_weights=None):
    if not external_weights:
        return ok()
    py = external_weights.get("per_year") if isinstance(external_weights, dict) \
        else None
    if py is None or len(py) == 0:
        return ok()
    sums = pd.DataFrame(py).astype(float).sum(axis=1)
    bad = sums[(sums - 1.0).abs() > 1e-3]
    if bad.empty:
        return ok()
    return fail(len(bad), f"{len(bad)} external weight row(s) do not sum to 1.0",
                examples=[f"{k}={v:.6f}" for k, v in bad.head(10).items()])


def _v_scen_external_average_sum(external_weights=None):
    if not external_weights:
        return ok()
    avg = external_weights.get("average") if isinstance(external_weights, dict) \
        else None
    return _weights_sum(avg, "the external 'average' weight row", tol=1e-3)


def _v_scen_external_nonneg(external_weights=None):
    if not external_weights:
        return ok()
    if not isinstance(external_weights, dict):
        return ok()
    py = external_weights.get("per_year")
    if py is None or len(py) == 0:
        return ok()
    flat = pd.DataFrame(py).astype(float).to_numpy().ravel()
    n = int((flat < 0).sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} external scenario weight(s) are negative")


def _v_mev_sum(mev_weights=None):
    return _weights_sum(mev_weights, "MEV model weights", tol=1e-6)


def _v_mev_nonneg(mev_weights=None):
    return _weights_nonneg(mev_weights, "MEV model weights")


def _v(id, severity, description, fn, context, rationale, remediation):
    return Validator(id=id, severity=severity, description=description, fn=fn,
                     context=context, rationale=rationale,
                     remediation=remediation, tags=("derived",))


DERIVED_STAGE_VALIDATORS: list[Validator] = [
    _v("DERIVED_LTPO_schema", Severity.ERROR,
       "LifeTimeParameterOther has the expected 7-column schema",
       _v_ltpo_schema, "LifeTimeParameterOther",
       "LIC reads this file by column position as well as by name.",
       "Check LPO_COLUMNS in etl/lifetime.py."),
    _v("DERIVED_LTPO_extract_date_unique", Severity.ERROR,
       "LifeTimeParameterOther carries a single extract date",
       _v_ltpo_extract_date_unique, "LifeTimeParameterOther",
       "Two dates in one file means curves from two runs were mixed.",
       "Re-run the ETL from a single input set."),
    _v("DERIVED_LTPO_ead_nonneg", Severity.ERROR,
       "EADLifetime is present and >= 0", _v_ltpo_ead_nonneg,
       "LifeTimeParameterOther",
       "A negative exposure at default produces a negative provision.",
       "Check the repayment schedule for negative balances."),
    _v("DERIVED_LTPO_month_starts_at_zero", Severity.ERROR,
       "Every contract has a month_lifetime = 0 row",
       _v_ltpo_month_starts_at_zero, "LifeTimeParameterOther",
       "Month 0 is today's outstanding. Without it LIC has no starting "
       "exposure and prices the contract from the first scheduled payment.",
       "Check the month-0 row is written from the account master."),
    _v("DERIVED_LTPO_months_contiguous", Severity.ERROR,
       "Each contract's months form a contiguous 0..N-1 sequence",
       _v_ltpo_months_contiguous, "LifeTimeParameterOther",
       "A gap makes LIC interpolate across it; a duplicate double-counts that "
       "month's loss.",
       "Check the month bound is EXCLUSIVE: months 0 .. end_month - 1."),
    _v("DERIVED_LTPO_ead_nonincreasing", Severity.INFO,
       "EADLifetime does not rise within a contract", _v_ltpo_ead_nonincreasing,
       "LifeTimeParameterOther",
       "For a term loan a rising curve is wrong. For a revolving or "
       "off-balance facility it is the schedule carrying committed but undrawn "
       "amounts, which is why the engine caps ECL at exposure. Informational, "
       "so that nobody later 'fixes' a rising curve.",
       "No action for revolving products."),
    _v("DERIVED_LTPO_contracts_subset_of_trans", Severity.ERROR,
       "Every curve belongs to a contract on the book",
       _v_ltpo_contracts_subset, "LifeTimeParameterOther",
       "A curve for a contract that is not on the book is priced by LIC "
       "against nothing.",
       "Check the schedule filter against the account list."),
    _v("DERIVED_LTPO_total_month0_ead_reconciles", Severity.WARN,
       "Month-0 EAD reconciles to the on-balance total for covered contracts",
       _v_ltpo_month0_reconciles, "LifeTimeParameterOther",
       "Month 0 is today's outstanding from the account master, not the "
       "schedule's first figure. The two differ whenever a payment falls in "
       "the current month, and taking the schedule value understates it.",
       "Compare a contract with a payment this month."),

    _v("DERIVED_STPD_schema", Severity.ERROR,
       "StPD has the expected 6-column schema", _v_stpd_schema, "StPD",
       "LIC reads the file by position as well as by name.",
       "Check the StPD writer."),
    _v("DERIVED_STPD_row_count", Severity.ERROR,
       f"StPD has exactly {STPD_ROWS:,} rows", _v_stpd_row_count, "StPD",
       "6 portfolios x 21 buckets x 600 months. A short file means a bucket "
       "or a portfolio was dropped.",
       "Check the TTC filter uses >= 0 rather than > 0."),
    _v("DERIVED_STPD_extract_date_unique", Severity.ERROR,
       "StPD carries a single extract date", _v_stpd_extract_date_unique, "StPD",
       "Two dates means curves from two runs were mixed.",
       "Re-run the ETL from a single input set."),
    _v("DERIVED_STPD_portfolio_set_complete", Severity.ERROR,
       "All 6 portfolios are present (4 internal + 2 external)",
       _v_stpd_portfolio_set, "StPD",
       "A missing portfolio means every contract in it prices to zero.",
       "Check portfolios.csv and the rating_type column."),
    _v("DERIVED_STPD_bucket_set_complete", Severity.ERROR,
       "Each portfolio has all 21 buckets", _v_stpd_bucket_set, "StPD",
       "A TTC PD of ZERO must still get a bucket. Filtering on > 0 instead of "
       ">= 0 loses the three highest external grades and 3,600 rows.",
       "Check the TTC filter."),
    _v("DERIVED_STPD_month_set_complete", Severity.ERROR,
       "Each (portfolio, bucket) has all 600 months", _v_stpd_month_set, "StPD",
       "A short curve makes LIC extrapolate beyond its end.",
       "Check max_month in the monthly conversion."),
    _v("DERIVED_STPD_dim2_all_na", Severity.ERROR,
       "PDBucketDim2 is empty throughout", _v_stpd_dim2_all_na, "StPD",
       "The output schema leaves this column empty; populating it changes how "
       "LIC keys the curve.",
       "Check the StPD writer."),
    _v("DERIVED_STPD_pd_nonneg_finite", Severity.ERROR,
       "PDLifetime is present, finite and non-negative", _v_stpd_pd_finite,
       "StPD",
       "A NaN PD silently prices a whole bucket to zero.",
       "Check the probit chain for a TTC PD of 0 or 1."),
    _v("DERIVED_STPD_pd_within_workbook_bound", Severity.ERROR,
       "PDLifetime <= 1.001", _v_stpd_pd_bounded, "StPD",
       "The monthly conversion is a running SUM rather than the survival "
       "formula, so it can pass 1 and is capped. A value above the cap means "
       "the cap did not apply.",
       "Check the cap in convert_to_monthly_stpd()."),
    _v("DERIVED_STPD_pd_monotone_non_decreasing", Severity.ERROR,
       "PDLifetime does not fall as maturity grows", _v_stpd_monotone, "StPD",
       "A cumulative default probability cannot decrease.",
       "Check the cumulative step."),
    _v("DERIVED_STPD_pd_increases_with_hierarchy", Severity.WARN,
       "A worse rating carries a higher PD", _v_stpd_pd_increases_with_hierarchy,
       "StPD",
       "If the ordering inverts, the scale has been applied upside down - "
       "which is exactly what happens if the internal probit shift is used on "
       "the external book, where the factor is SUBTRACTED.",
       "Check which formula each rating type uses."),
    _v("DERIVED_STPD_internal_portfolios_identical", Severity.WARN,
       "The 4 internal portfolios share one curve set",
       _v_stpd_internal_identical, "StPD",
       "A curve belongs to the rating SCALE, not the portfolio. A difference "
       "means the two scales were mixed.",
       "Check the term structure is built once per scale."),
    _v("DERIVED_STPD_external_portfolios_identical", Severity.WARN,
       "The 2 external portfolios share one curve set",
       _v_stpd_external_identical, "StPD",
       "As above, for the external scale.",
       "Check the term structure is built once per scale."),
    _v("DERIVED_STPD_zero_ttc_zero_curve", Severity.WARN,
       "External buckets with TTC = 0 have an all-zero curve",
       _v_stpd_zero_ttc_zero_curve, "StPD",
       "The engine short-circuits a zero TTC to a zero curve, but the rating "
       "still needs a bucket in the output.",
       "Check the zero short-circuit in the term structure."),

    _v("DERIVED_SCEN_internal_weights_sum_to_one", Severity.ERROR,
       "Internal scenario weights sum to 1.0", _v_scen_internal_sum,
       "scenario_weights",
       "An equal split sums to one too, so this alone will not catch it - but "
       "a weighting that does NOT sum to one is always wrong.",
       "Check resolve_internal_scenario_weights()."),
    _v("DERIVED_SCEN_internal_weights_nonneg", Severity.ERROR,
       "Internal scenario weights are all >= 0", _v_scen_internal_nonneg,
       "scenario_weights",
       "A negative probability is not a probability.",
       "Check the band construction; the central scenario takes the residual."),
    _v("DERIVED_SCEN_external_per_year_sum_to_one", Severity.ERROR,
       "Each external per-year weight row sums to 1.0",
       _v_scen_external_per_year_sum, "scenario_weights",
       "The external scale uses a DIFFERENT weight vector for each year, "
       "because the regional forecast moves. Each row must still be a "
       "probability distribution.",
       "Check compute_external_scenario_weights_per_year()."),
    _v("DERIVED_SCEN_external_average_sum_to_one", Severity.ERROR,
       "The external 'average' row sums to 1.0", _v_scen_external_average_sum,
       "scenario_weights",
       "The average row applies to 45 of the 50 years in every external curve.",
       "Check it is the column-wise mean of the per-year rows."),
    _v("DERIVED_SCEN_external_weights_nonneg", Severity.ERROR,
       "External scenario weights are all >= 0", _v_scen_external_nonneg,
       "scenario_weights",
       "A negative probability is not a probability.",
       "Check the band construction."),
    _v("DERIVED_MEV_weights_sum_to_one", Severity.ERROR,
       "MEV model weights sum to 1.0", _v_mev_sum, "MEV",
       "Only Non-Oil GDP carries weight in the production model; real estate "
       "and domestic credit are weighted 0.0. That is the model, not a fault, "
       "but the three must still sum to one.",
       "Check the mev_components block in model.yml."),
    _v("DERIVED_MEV_weights_nonneg", Severity.ERROR,
       "MEV model weights are all >= 0", _v_mev_nonneg, "MEV",
       "A negative weight inverts that variable's contribution.",
       "Check model.yml."),
]
