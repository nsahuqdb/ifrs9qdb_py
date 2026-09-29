"""The intermediate tables R's TRANSFORM checks read, built from Python's.

R validates four intermediates -- trans_l (the lending book per contract),
cm_view (the lending book per CUSTOMER, keyed on CustomerMaster), trans_i and
inv_view (the investment book) -- and its checks read specific columns of
them: cm_view$dpd_status, $watchlist_status, $restructuring_final,
$exposure_total; trans_l$rating_after_override, $is_default_final. Python's
ETL builds the same numbers in differently shaped frames, and the TRANSFORM
checks used to read those frames with fallbacks: where a column was missing
the check passed. On a book where a contract's customer is absent from
CustomerMaster, R failed three TRANSFORM checks that Python passed.

So the frames are rebuilt here in R's shape -- R/transform_lending.R,
R/lending_portfolio_view.R, R/transform_investments.R and
R/investment_portfolio_view.R, rule for rule -- and the checks port R's
literally against them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ids import as_id
from ._helpers import col, text

__all__ = ["lending_frames", "investment_frames", "apply_staging_rule_r",
           "apply_sicr_staging"]


def _threshold(static, default=60.0) -> float:
    try:
        th = static.get("staging_thresholds")
        k = text(col(th, "key"))
        v = pd.to_numeric(col(th, "value")[k == "dpd_stage2_threshold_days"],
                          errors="coerce").dropna()
        return float(v.iloc[0]) if len(v) else default
    except Exception:
        return default


def apply_staging_rule_r(dpd, restructuring_final, watchlist_status,
                         dpd_stage2_threshold) -> pd.Series:
    """R's apply_staging_rule() (R/lending_portfolio_view.R)."""
    d = pd.to_numeric(pd.Series(dpd), errors="coerce").reset_index(drop=True)
    r = pd.Series(restructuring_final).reset_index(drop=True)
    w = pd.Series(watchlist_status).reset_index(drop=True)
    out = pd.Series(["Stage 1"] * len(d), dtype=object)
    s2 = (r == "Restructured").fillna(False) | (w == "Watchlist").fillna(False) | \
        ((d > dpd_stage2_threshold) & (d <= 90)).fillna(False)
    out[s2.to_numpy()] = "Stage 2"
    out[(d > 90).fillna(False).to_numpy()] = "Stage 3"
    return out


def _first_lookup(keys: pd.Series, values: pd.Series, targets: pd.Series) -> pd.Series:
    """R's setNames(values, keys)[targets]: the FIRST match, NA when none."""
    k = pd.Series(values.to_numpy(), index=keys.to_numpy())
    k = k[~k.index.duplicated(keep="first")]
    return pd.Series(targets.to_numpy()).map(k)


def _csf_flag(csf, field, customer_id: pd.Series) -> pd.Series:
    """R: as.integer(setNames(as.integer(csf$<field>), csf_id)[customer_id]),
    NA -> 0."""
    if csf is None or len(csf) == 0:
        return pd.Series([0] * len(customer_id))
    c = col(csf, field)
    if c is None:
        return pd.Series([0] * len(customer_id))
    v = pd.to_numeric(c, errors="coerce")
    got = _first_lookup(as_id(col(csf, "customer_id")), v, customer_id)
    return got.fillna(0).astype(int)


def lending_frames(view: pd.DataFrame, inputs, static):
    """(trans_l, cm_view) as R builds them, from Python's lending view."""
    n = len(view)
    t = pd.DataFrame({
        "contract_id": view["contract_id"].to_numpy(),
        "customer_id": as_id(view["customer_id"]).to_numpy(),
        "account_type": view["account_type"].to_numpy(),
        "rating": view["rating"].to_numpy(),
        "rating_worst": view["rating_worst"].to_numpy(),
        # R: blank DPD and blank balance are 0 on the transformation
        "past_dues_days": pd.to_numeric(view["past_dues_days"],
                                        errors="coerce").fillna(0).to_numpy(),
        "past_dues_worst": pd.to_numeric(view["past_dues_worst"],
                                         errors="coerce").to_numpy(),
        "exposure_amount": pd.to_numeric(view["on_balance"],
                                         errors="coerce").fillna(0).to_numpy(),
    })
    ms = static.get("master_rating_scale") if static is not None else None
    hier = {}
    if ms is not None and len(ms):
        for r, h in zip(text(col(ms, "rating")), pd.to_numeric(col(ms, "hierarchy"),
                                                                errors="coerce")):
            hier.setdefault(r, h)
    t["rating_hierarchy"] = pd.Series(t["rating"]).map(hier).to_numpy()
    csf = inputs.get("CustomerStagingFlag") if hasattr(inputs, "get") else None
    t["is_watchlist"] = _csf_flag(csf, "is_watchlist", t["customer_id"]).to_numpy()
    t["is_restructured"] = _csf_flag(csf, "is_local1", t["customer_id"]).to_numpy()

    # ---- cm_view: one row per CustomerMaster customer, in its order ------
    cm = inputs.get("CustomerMaster") if hasattr(inputs, "get") else None
    cust = as_id(col(cm, "customer_id")) if cm is not None and len(cm) else \
        pd.Series([], dtype=object)
    first = t.drop_duplicates("customer_id", keep="first")
    exposure_total = cust.map(t.groupby("customer_id")["exposure_amount"].sum()).fillna(0)
    rating_current = _first_lookup(first["customer_id"], first["rating_worst"], cust)
    restr_any = t.groupby("customer_id")["is_restructured"].apply(lambda x: bool((x == 1).any()))
    watch_any = t.groupby("customer_id")["is_watchlist"].apply(lambda x: bool((x == 1).any()))

    def label(any_map, word):
        got = cust.map(any_map)
        out = pd.Series([None] * len(cust), dtype=object)
        out[got == True] = word                      # noqa: E712
        out[got == False] = ""                       # noqa: E712
        return out
    restructuring_current = label(restr_any, "Restructured")
    watchlist_status = label(watch_any, "Watchlist")
    dpd_status = _first_lookup(first["customer_id"], first["past_dues_worst"], cust)
    dpd_status = pd.to_numeric(dpd_status, errors="coerce").fillna(0)
    restructuring_final = restructuring_current.where(
        restructuring_current != "", "").where(restructuring_current.notna(), None)
    stage_final = apply_staging_rule_r(dpd_status, restructuring_final,
                                       watchlist_status, _threshold(static))
    name = col(cm, "customer_name") if cm is not None and len(cm) else None
    cm_view = pd.DataFrame({
        "customer_id": cust.to_numpy(),
        "customer_name": (text(name).to_numpy() if name is not None
                          else [""] * len(cust)),
        "exposure_total": exposure_total.to_numpy(),
        "rating_current": rating_current.to_numpy(),
        "rating_final": rating_current.to_numpy(),
        "restructuring_current": restructuring_current.to_numpy(),
        "restructuring_final": restructuring_final.to_numpy(),
        "watchlist_status": watchlist_status.to_numpy(),
        "dpd_status": dpd_status.to_numpy(),
        "stage_final": stage_final.to_numpy(),
    })
    # ---- Pass 6 back-fill onto the contracts -------------------------------
    ra = _first_lookup(cm_view["customer_id"], cm_view["rating_final"], t["customer_id"])
    sf = _first_lookup(cm_view["customer_id"], cm_view["stage_final"], t["customer_id"])
    t["rating_after_override"] = ra.to_numpy()
    t["is_default_final"] = sf.map(lambda s: np.nan if pd.isna(s) else
                                   int(s == "Stage 3")).to_numpy()
    return t, cm_view


def apply_sicr_staging(rating_hierarchy, rating_origination_hierarchy) -> pd.Series:
    """R's apply_sicr_staging() (R/investment_portfolio_view.R)."""
    h = pd.to_numeric(pd.Series(rating_hierarchy), errors="coerce").reset_index(drop=True)
    o = pd.to_numeric(pd.Series(rating_origination_hierarchy),
                      errors="coerce").reset_index(drop=True)
    mig = h - o
    out = pd.Series(["Stage 1"] * len(h), dtype=object)
    ig = h.notna() & (h > 4) & (h < 11)
    junk = h.notna() & (h >= 11)
    out[(ig & mig.notna() & (mig >= 2)).to_numpy()] = "Stage 2"
    out[(junk & mig.notna() & (mig >= 1)).to_numpy()] = "Stage 2"
    out[(h.notna() & (h <= 4)).to_numpy()] = "Stage 1"
    return out


def investment_frames(inv: pd.DataFrame, inputs, static):
    """(trans_i, inv_view) as R builds them, from Python's investment frame."""
    ms = static.get("master_rating_scale") if static is not None else None
    ext_h = {}
    if ms is not None and len(ms):
        ext = ms[text(col(ms, "rating_type")) == "External"]
        for r, h in zip(text(col(ext, "rating")),
                        pd.to_numeric(col(ext, "hierarchy"), errors="coerce")):
            ext_h.setdefault(r, h)
    ids = as_id(inv["contract_id"]).reset_index(drop=True)
    rating_current = pd.Series(inv["rating_worst"]).reset_index(drop=True)
    oi = inputs.get("OriginationInvestments") if hasattr(inputs, "get") else None
    if oi is not None and len(oi) and col(oi, "origination_rating") is not None:
        orat = text(col(oi, "origination_rating"))
        orat = orat.where(~orat.isin(["0", ""]), None)
        rat_orig = _first_lookup(as_id(col(oi, "contract_id")), orat, ids)
    else:
        rat_orig = pd.Series([None] * len(ids), dtype=object)
    h = rating_current.map(ext_h)
    oh = rat_orig.map(ext_h)
    oh = oh.where(oh.notna(), h)
    trans_i = pd.DataFrame({
        "account_id": ids.to_numpy(),
        "customer_id_inv": as_id(inv["customer_id"]).to_numpy(),
        "rating_current": rating_current.to_numpy(),
        "rating_hierarchy": h.to_numpy(),
        "rating_origination_hierarchy": oh.to_numpy(),
        "exposure_amount": pd.to_numeric(inv["on_balance"], errors="coerce")
        .fillna(0).to_numpy(),
    })
    inv_view = pd.DataFrame({
        "account_id": trans_i["account_id"],
        "rating_current": trans_i["rating_current"],
        "stage_final": apply_sicr_staging(trans_i["rating_hierarchy"],
                                          trans_i["rating_origination_hierarchy"]),
    })
    return trans_i, inv_view
