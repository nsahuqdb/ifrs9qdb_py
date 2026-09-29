"""DERIVED stage: the curves and weights the engine prices against.

A literal port of R/validators_derived.R -- the same 29 checks, in R's order,
with R's tolerances and messages. The curve tables arrive in the Output shape
(ContractId, MonthLifetime, ...); ``_r_shape`` renames them to the snake_case
columns R's checks read, so a missing column fails the schema check exactly
as it does in R.

Three tolerances used to differ from R and now match it: the internal
scenario weights sum to 1 within 1e-4 (not 1e-3), the MEV weights within 1e-3
(not 1e-6), and the external non-negativity check covers the 'average' row as
well as the per-year rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._helpers import col, text
from .checks_transform import r_format
from ._texts import STAGE_TEXTS
from .framework import Severity, Validator

__all__ = ["DERIVED_STAGE_VALIDATORS"]

INTERNAL_PORTFOLIOS = ["Business Finance", "Off BS", "Al Dhameen", "Tasdeer"]
EXTERNAL_PORTFOLIOS = ["Banks and Fis", "Investments"]
LTPO_R = ["extract_date", "contract_id", "month_lifetime", "ead_lifetime",
          "lgd_lifetime", "payment_schedule_lifetime", "total_limit_lifetime"]
STPD_R = ["extract_date", "portfolio_code", "pd_bucket_dim1", "pd_bucket_dim2",
          "month_lifetime", "pd_lifetime"]


def _squash(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _r_shape(df, names):
    """The frame with its columns renamed to R's snake_case names."""
    if df is None:
        return None
    want = {_squash(n): n for n in names}
    ren = {c: want[_squash(c)] for c in df.columns if _squash(c) in want}
    return df.rename(columns=ren)


def _ok():
    return {"passed": True}


def _bad(message, count=0):
    return {"passed": False, "count": int(count), "detail": message}


def _num(s):
    return pd.to_numeric(pd.Series(s), errors="coerce")


def _dates(s) -> pd.Series:
    from ._helpers import parse_any_date
    return parse_any_date(pd.Series(s))


# ------------------------------------------------------------------ LTPO ---
def _ltpo_schema(ltpo=None):
    if ltpo is None:
        return _bad("ltpo is NULL")
    missing = [c for c in LTPO_R if c not in _r_shape(ltpo, LTPO_R).columns]
    return _ok() if not missing else _bad("Missing columns: " + ", ".join(missing))


def _ltpo_extract_date(ltpo=None):
    t = _r_shape(ltpo, LTPO_R)
    if t is None or "extract_date" not in t.columns:
        return _ok()
    d = _dates(t["extract_date"])
    uniq = list(pd.unique(d.dt.strftime("%Y-%m-%d").fillna("NA")))
    return _ok() if len(uniq) == 1 else _bad(
        f"{len(uniq)} distinct extract_dates: " + ", ".join(uniq[:5]), len(uniq))


def _ltpo_ead_nonneg(ltpo=None):
    t = _r_shape(ltpo, LTPO_R)
    if t is None or "ead_lifetime" not in t.columns:
        return _ok()
    e = _num(t["ead_lifetime"])
    n_na, n_neg = int(e.isna().sum()), int((e < 0).sum())
    return _ok() if n_na == 0 and n_neg == 0 else _bad(
        f"{n_na} NA, {n_neg} negative", n_na + n_neg)


def _ltpo_month_zero(ltpo=None):
    t = _r_shape(ltpo, LTPO_R)
    if t is None or len(t) == 0:
        return _ok()
    m = _num(t["month_lifetime"])
    have0 = set(t.loc[(m == 0).to_numpy(), "contract_id"].astype(str))
    ids = pd.unique(t["contract_id"].astype(str))
    missing = [c for c in ids if c not in have0]
    return _ok() if not missing else _bad(
        f"{len(missing)} contracts have no month-0 row (e.g. "
        + ", ".join(missing[:5]) + ")", len(missing))


def _ltpo_contiguous(ltpo=None):
    t = _r_shape(ltpo, LTPO_R)
    if t is None or len(t) == 0:
        return _ok()
    d = pd.DataFrame({"c": t["contract_id"].astype(str), "m": _num(t["month_lifetime"])})

    def ok_(m):
        s = np.sort(m.to_numpy())
        return len(s) > 0 and s[0] == 0 and np.array_equal(s, np.arange(len(s)))
    n_bad = int((~d.groupby("c")["m"].apply(ok_)).sum())
    return _ok() if n_bad == 0 else _bad(
        f"{n_bad} contracts have non-contiguous months", n_bad)


def _ltpo_nonincreasing(ltpo=None):
    t = _r_shape(ltpo, LTPO_R)
    if t is None or len(t) == 0:
        return _ok()
    d = pd.DataFrame({"c": t["contract_id"].astype(str),
                      "m": _num(t["month_lifetime"]), "e": _num(t["ead_lifetime"])})
    d = d.sort_values(["c", "m"], kind="mergesort")
    up = d.groupby("c")["e"].apply(lambda e: bool((e.diff().dropna() > 1e-6).any()))
    n_bad = int(up.sum())
    return _ok() if n_bad == 0 else _bad(
        f"{n_bad} contracts have an EAD rise between months — expected for "
        "revolving/off-balance products and contracts with interest/fee "
        "accrual baked into the schedule", n_bad)


def _ltpo_subset(ltpo=None, trans_l=None):
    t = _r_shape(ltpo, LTPO_R)
    if t is None or trans_l is None:
        return _ok()
    have = set(trans_l["contract_id"].astype(str))
    orphans = [c for c in pd.unique(t["contract_id"].astype(str)) if c not in have]
    return _ok() if not orphans else _bad(
        f"{len(orphans)} ltpo contracts not in trans_l (e.g. "
        + ", ".join(orphans[:5]) + ")", len(orphans))


def _ltpo_month0_reconciles(ltpo=None, trans_l=None):
    t = _r_shape(ltpo, LTPO_R)
    if t is None or trans_l is None:
        return _ok()
    m0 = t[(_num(t["month_lifetime"]) == 0).to_numpy()]
    ids = m0["contract_id"].astype(str)
    common = set(ids) & set(trans_l["contract_id"].astype(str))
    if not common:
        return _ok()
    a = float(_num(m0["ead_lifetime"])[ids.isin(common).to_numpy()].sum())
    tc = trans_l["contract_id"].astype(str)
    b = float(_num(trans_l["exposure_amount"])[tc.isin(common).to_numpy()].sum())
    tol = max(1.0, abs(b) * 1e-6)
    return _ok() if abs(a - b) <= tol else _bad(
        f"ltpo month-0 sum={r_format(a)}, trans exposure sum={r_format(b)}, "
        f"diff={r_format(a - b)}, contracts compared={len(common)}")


# ------------------------------------------------------------------ StPD ---
def _stpd(stpd):
    return _r_shape(stpd, STPD_R)


def _stpd_schema(stpd=None):
    if stpd is None:
        return _bad("stpd is NULL")
    missing = [c for c in STPD_R if c not in _stpd(stpd).columns]
    return _ok() if not missing else _bad("Missing columns: " + ", ".join(missing))


def _stpd_rows(stpd=None):
    if stpd is None:
        return _ok()
    return _ok() if len(stpd) == 75600 else _bad(
        f"nrow(stpd)={len(stpd)}, expected 75,600")


def _stpd_extract_date(stpd=None):
    s = _stpd(stpd)
    if s is None or "extract_date" not in s.columns:
        return _ok()
    n = int(pd.Series(s["extract_date"]).nunique(dropna=False))
    return _ok() if n == 1 else _bad(f"{n} distinct extract_dates", n)


def _stpd_portfolios(stpd=None):
    s = _stpd(stpd)
    if s is None:
        return _ok()
    expected = sorted(INTERNAL_PORTFOLIOS + EXTERNAL_PORTFOLIOS)
    observed = sorted(set(text(s["portfolio_code"])))
    if expected == observed:
        return _ok()
    return _bad(f"Expected {len(expected)} portfolios, got {len(observed)}. "
                f"Missing: {', '.join(p for p in expected if p not in observed)}. "
                f"Extra: {', '.join(p for p in observed if p not in expected)}")


def _stpd_buckets(stpd=None):
    s = _stpd(stpd)
    if s is None:
        return _ok()
    b = _num(s["pd_bucket_dim1"])
    per = b.groupby(text(s["portfolio_code"])).apply(
        lambda x: sorted(set(x.dropna().astype(int))) == list(range(1, 22)))
    n_bad = int((~per).sum())
    return _ok() if n_bad == 0 else _bad(f"{n_bad} portfolios missing buckets", n_bad)


def _stpd_months(stpd=None):
    s = _stpd(stpd)
    if s is None or len(s) == 0:
        return _ok()
    m = _num(s["month_lifetime"])
    n = s.groupby([text(s["portfolio_code"]), _num(s["pd_bucket_dim1"])]).size()
    bad = int((n != 600).sum())
    lo, hi = int(m.min()), int(m.max())
    if bad == 0 and lo == 1 and hi == 600:
        return _ok()
    return _bad(f"month range = [{lo}, {hi}]; {bad} (portfolio, bucket) groups "
                "don't have 600 rows", bad)


def _stpd_dim2(stpd=None):
    s = _stpd(stpd)
    if s is None:
        return _ok()
    if "pd_bucket_dim2" not in s.columns:
        return _bad("pd_bucket_dim2 missing")
    n = int((text(s["pd_bucket_dim2"]) != "").sum())
    return _ok() if n == 0 else _bad(f"{n} non-NA values in pd_bucket_dim2", n)


def _stpd_finite(stpd=None):
    s = _stpd(stpd)
    if s is None:
        return _ok()
    p = _num(s["pd_lifetime"])
    n_na = int(p.isna().sum())
    n_neg = int((p < 0).sum())
    n_inf = int(np.isinf(p.fillna(0)).sum())
    return _ok() if n_na + n_neg + n_inf == 0 else _bad(
        f"{n_na} NA, {n_neg} negative, {n_inf} infinite", n_na + n_neg + n_inf)


def _stpd_bound(stpd=None):
    s = _stpd(stpd)
    if s is None:
        return _ok()
    p = _num(s["pd_lifetime"])
    mx = float(p.max()) if p.notna().any() else float("nan")
    if np.isfinite(mx) and mx <= 1.001:
        return _ok()
    return _bad(f"max pd_lifetime = {mx:.6f}, {int((p > 1).sum())} rows above 1, "
                f"{int((p > 1.001).sum())} rows above 1.001. Likely cause: scenario "
                "weights summing to >1 (typo in workbook explicit_weights — switch "
                "to mode='auto_non_oil_gdp_cdf' in model_inputs.yml).",
                int((p > 1.001).sum()))


def _stpd_monotone(stpd=None):
    s = _stpd(stpd)
    if s is None or len(s) == 0:
        return _ok()
    d = pd.DataFrame({"p": text(s["portfolio_code"]), "b": _num(s["pd_bucket_dim1"]),
                      "m": _num(s["month_lifetime"]), "v": _num(s["pd_lifetime"])})
    d = d.sort_values(["p", "b", "m"], kind="mergesort")
    dec = d.groupby(["p", "b"])["v"].apply(lambda v: bool((v.diff().dropna() < -1e-12).any()))
    n_bad = int(dec.sum())
    return _ok() if n_bad == 0 else _bad(
        f"{n_bad} (portfolio, bucket) groups have a decreasing pd_lifetime", n_bad)


def _stpd_hierarchy(stpd=None):
    s = _stpd(stpd)
    if s is None or len(s) == 0:
        return _ok()
    d = pd.DataFrame({"p": text(s["portfolio_code"]), "b": _num(s["pd_bucket_dim1"]),
                      "m": _num(s["month_lifetime"]), "v": _num(s["pd_lifetime"])})
    months = [m for m in (12, 60, 120) if m in set(d["m"])]
    bad = []
    for m in months:
        for p in pd.unique(d["p"]):
            sub = d[(d["p"] == p) & (d["m"] == m)].sort_values("b", kind="mergesort")
            if bool((sub["v"].diff().dropna() < -1e-9).any()):
                bad.append(f"{p} @ m={m}")
    return _ok() if not bad else _bad("PD not monotone in hierarchy at: "
                                      + "; ".join(bad[:10]), len(bad))


def _stpd_identical(ports, label):
    def check(stpd=None):
        s = _stpd(stpd)
        if s is None:
            return _ok()
        sub = s[text(s["portfolio_code"]).isin(ports).to_numpy()]
        n = sub.groupby([_num(sub["pd_bucket_dim1"]), _num(sub["month_lifetime"])])[
            "pd_lifetime"].nunique()
        n_bad = int((n != 1).sum())
        return _ok() if n_bad == 0 else _bad(
            f"{n_bad} (bucket, month) cells differ across {label} portfolios", n_bad)
    return check


def _stpd_zero_ttc(stpd=None, static=None):
    s = _stpd(stpd)
    t = static.get("ttc_pd_table") if static is not None else None
    if s is None or t is None or len(t) == 0:
        return _ok()
    rt = text(col(t, "rating_type"))
    ext = t[(rt == "External") | (rt == "2")]
    if len(ext) == 0:
        return _ok()
    zero = [i + 1 for i, v in enumerate(pd.to_numeric(col(ext, "ttc_pd"),
                                                      errors="coerce")) if v == 0]
    if not zero:
        return _ok()
    sub = s[text(s["portfolio_code"]).isin(EXTERNAL_PORTFOLIOS).to_numpy()
            & _num(s["pd_bucket_dim1"]).isin(zero).to_numpy()]
    n = int((_num(sub["pd_lifetime"]) > 1e-12).sum())
    return _ok() if n == 0 else _bad(
        f"{n} non-zero pd_lifetime values for zero-TTC buckets "
        + ", ".join(map(str, zero)), n)


# --------------------------------------------------------------- weights ---
def _as_series(w) -> pd.Series | None:
    if w is None:
        return None
    if isinstance(w, pd.Series):
        return pd.to_numeric(w, errors="coerce")
    if isinstance(w, dict):
        return pd.to_numeric(pd.Series(w), errors="coerce")
    try:
        return pd.to_numeric(pd.Series(list(w)), errors="coerce")
    except TypeError:
        return None


def _scen_internal_sum(internal_weights=None):
    w = _as_series(internal_weights)
    if w is None or len(w) == 0:
        return _ok()
    s = float(w.sum())
    return _ok() if abs(s - 1) < 1e-4 else _bad(
        f"Sum = {s:.6f} (expected ~1.0; if explicit V4 weights, sum=1.0003 due to "
        "AE8 typo — switch model_inputs.yml mode to 'auto_non_oil_gdp_cdf')")


def _scen_internal_nonneg(internal_weights=None):
    w = _as_series(internal_weights)
    if w is None or len(w) == 0:
        return _ok()
    bad = [str(k) for k, v in w.items() if v < 0]
    return _ok() if not bad else _bad("Negative weights: " + ", ".join(bad), len(bad))


def _per_year(ew):
    if not isinstance(ew, dict) or ew.get("per_year") is None:
        return None
    return pd.DataFrame(ew["per_year"]).apply(pd.to_numeric, errors="coerce")


def _scen_external_per_year(external_weights=None):
    py = _per_year(external_weights)
    if py is None or len(py) == 0:
        return _ok()
    sums = py.sum(axis=1).to_numpy()
    bad = [str(i + 1) for i, v in enumerate(sums) if abs(v - 1) >= 1e-3]
    return _ok() if not bad else _bad("Years with row-sum != 1: " + ", ".join(bad),
                                      len(bad))


def _scen_external_average(external_weights=None):
    if external_weights is None:
        return _ok()
    avg = external_weights.get("average") if isinstance(external_weights, dict) \
        else external_weights
    w = _as_series(avg)
    if w is None or len(w) == 0:
        return _ok()
    s = float(w.sum())
    return _ok() if abs(s - 1) < 1e-3 else _bad(f"Sum = {s:.6f} (expected ~1.0)")


def _scen_external_nonneg(external_weights=None):
    if external_weights is None:
        return _ok()
    vals = []
    py = _per_year(external_weights)
    if py is not None:
        vals += list(py.to_numpy().ravel())
    avg = external_weights.get("average") if isinstance(external_weights, dict) \
        else external_weights
    w = _as_series(avg)
    if w is not None:
        vals += list(w)
    n = int(sum(1 for v in vals if pd.notna(v) and v < -1e-12))
    return _ok() if n == 0 else _bad(f"{n} negative weight values", n)


def _mev_sum(mev_weights=None):
    w = _as_series(mev_weights)
    if w is None or len(w) == 0:
        return _ok()
    s = float(w.sum())
    return _ok() if abs(s - 1) < 1e-3 else _bad(f"Sum = {s:.6f}")


def _mev_nonneg(mev_weights=None):
    w = _as_series(mev_weights)
    if w is None or len(w) == 0:
        return _ok()
    bad = [str(k) for k, v in w.items() if v < -1e-12]
    return _ok() if not bad else _bad("Negative MEV weights: " + ", ".join(bad),
                                      len(bad))


def _v(id, severity, description, fn, context):
    why, fix = STAGE_TEXTS.get(id, ("", ""))
    return Validator(id=id, severity=severity, description=description, fn=fn,
                     context=context, rationale=why, remediation=fix,
                     tags=("derived",))


DERIVED_STAGE_VALIDATORS: list[Validator] = [
    _v("DERIVED_LTPO_schema", Severity.ERROR,
       "ltpo has the expected 7-column schema", _ltpo_schema,
       "LifeTimeParameterOther"),
    _v("DERIVED_LTPO_extract_date_unique", Severity.ERROR,
       "ltpo has a single extract_date value", _ltpo_extract_date,
       "LifeTimeParameterOther"),
    _v("DERIVED_LTPO_ead_nonneg", Severity.ERROR,
       "ead_lifetime is non-NA and >= 0", _ltpo_ead_nonneg, "LifeTimeParameterOther"),
    _v("DERIVED_LTPO_month_starts_at_zero", Severity.ERROR,
       "Every contract has a month_lifetime=0 row", _ltpo_month_zero,
       "LifeTimeParameterOther"),
    _v("DERIVED_LTPO_months_contiguous", Severity.ERROR,
       "Each contract's months form a contiguous 0..N-1 sequence", _ltpo_contiguous,
       "LifeTimeParameterOther"),
    _v("DERIVED_LTPO_ead_nonincreasing", Severity.INFO,
       "ead_lifetime is non-increasing within each contract (term-loan principle; "
       "revolving/off-bal/accrual products allowed to grow)", _ltpo_nonincreasing,
       "LifeTimeParameterOther"),
    _v("DERIVED_LTPO_contracts_subset_of_trans", Severity.ERROR,
       "Every ltpo contract_id exists in trans_l", _ltpo_subset,
       "LifeTimeParameterOther"),
    _v("DERIVED_LTPO_total_month0_ead_reconciles", Severity.WARN,
       "Sum of month-0 EAD == sum of trans_l ONBALANCE for covered contracts",
       _ltpo_month0_reconciles, "LifeTimeParameterOther"),
    _v("DERIVED_STPD_schema", Severity.ERROR,
       "stpd has the expected 6-column schema", _stpd_schema, "StPD"),
    _v("DERIVED_STPD_row_count", Severity.ERROR,
       "stpd has exactly 6 portfolios × 21 buckets × 600 months = 75,600 rows",
       _stpd_rows, "StPD"),
    _v("DERIVED_STPD_extract_date_unique", Severity.ERROR,
       "stpd has a single extract_date value", _stpd_extract_date, "StPD"),
    _v("DERIVED_STPD_portfolio_set_complete", Severity.ERROR,
       "All 6 portfolios present (4 internal + 2 external)", _stpd_portfolios, "StPD"),
    _v("DERIVED_STPD_bucket_set_complete", Severity.ERROR,
       "Each portfolio has all 21 buckets (hierarchy 1..21)", _stpd_buckets, "StPD"),
    _v("DERIVED_STPD_month_set_complete", Severity.ERROR,
       "Each (portfolio, bucket) has all 600 months (1..600)", _stpd_months, "StPD"),
    _v("DERIVED_STPD_dim2_all_na", Severity.ERROR,
       "pd_bucket_dim2 is all NA (matches Excel output schema)", _stpd_dim2, "StPD"),
    _v("DERIVED_STPD_pd_nonneg_finite", Severity.ERROR,
       "pd_lifetime is non-NA, non-negative, and finite", _stpd_finite, "StPD"),
    _v("DERIVED_STPD_pd_within_workbook_bound", Severity.ERROR,
       "pd_lifetime <= 1.001 (allows FP slack but rejects real overflow)",
       _stpd_bound, "StPD"),
    _v("DERIVED_STPD_pd_monotone_non_decreasing", Severity.ERROR,
       "pd_lifetime non-decreasing within (portfolio, bucket) as month grows",
       _stpd_monotone, "StPD"),
    _v("DERIVED_STPD_pd_increases_with_hierarchy", Severity.WARN,
       "Worse rating (higher hierarchy) => higher pd_lifetime, fixing (portfolio, month)",
       _stpd_hierarchy, "StPD"),
    _v("DERIVED_STPD_internal_portfolios_identical", Severity.WARN,
       "4 internal portfolios (Business Finance, Off BS, Al Dhameen, Tasdeer) share "
       "identical curves", _stpd_identical(INTERNAL_PORTFOLIOS, "internal"), "StPD"),
    _v("DERIVED_STPD_external_portfolios_identical", Severity.WARN,
       "2 external portfolios (Banks and Fis, Investments) share identical curves",
       _stpd_identical(EXTERNAL_PORTFOLIOS, "external"), "StPD"),
    _v("DERIVED_STPD_zero_ttc_zero_curve", Severity.WARN,
       "External buckets with TTC=0 (Aaa/Aa1/Aa2) have entire pd_lifetime curve = 0",
       _stpd_zero_ttc, "StPD"),
    _v("DERIVED_SCEN_internal_weights_sum_to_one", Severity.ERROR,
       "Internal scenario weights sum to 1.0 within 1e-4", _scen_internal_sum,
       "scenario_weights"),
    _v("DERIVED_SCEN_internal_weights_nonneg", Severity.ERROR,
       "Internal scenario weights are all >= 0", _scen_internal_nonneg,
       "scenario_weights"),
    _v("DERIVED_SCEN_external_per_year_sum_to_one", Severity.ERROR,
       "External scenario weights: each year's row sums to ~1.0",
       _scen_external_per_year, "scenario_weights"),
    _v("DERIVED_SCEN_external_average_sum_to_one", Severity.ERROR,
       "External scenario weights: 'average' (year 6+) row sums to ~1.0",
       _scen_external_average, "scenario_weights"),
    _v("DERIVED_SCEN_external_weights_nonneg", Severity.ERROR,
       "External scenario weights are all >= 0", _scen_external_nonneg,
       "scenario_weights"),
    _v("DERIVED_MEV_weights_sum_to_one", Severity.ERROR,
       "MEV model weights sum to ~1.0", _mev_sum, "MEV"),
    _v("DERIVED_MEV_weights_nonneg", Severity.ERROR,
       "MEV model weights are all >= 0", _mev_nonneg, "MEV"),
]
