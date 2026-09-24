"""
The six reference files.

Derived from the static reference rather than copied: the static tables carry
the model's own column names and, for ratings, BOTH scales in one frame, while
LIC wants its own schema and every scale numbered.

They change when the model changes, not per quarter, but they still carry the
run's extract date so a run is self-describing.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = ["build_reference_files"]


def _col(df, *names, default=None):
    if df is None or len(df) == 0:
        return None
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        hit = low.get(n.lower())
        if hit is not None:
            return df[hit]
    if default is not None:
        return pd.Series([default] * len(df))
    return None


def _description(rating, hierarchy, rating_type) -> str:
    """The Desc label LIC expects for a rating."""
    if pd.isna(hierarchy):
        return ""
    if rating_type == 1:
        m = re.search(r"\d+", str(rating))
        return "Desc" + (m.group() if m else str(int(hierarchy)))
    return "Desc" + str(int(hierarchy))


def _rating_type_number(col) -> pd.Series:
    """Rating type as the NUMBER LIC keys on.

    The master scale spells it "Internal"/"External"; LIC wants 1/2. Both forms
    appear across the reference files, so both are accepted.
    """
    n = pd.to_numeric(col, errors="coerce")
    if n.notna().any():
        return n.astype("Int64")
    return (col.astype(str).str.strip().str.lower()
            .map({"internal": 1, "external": 2}).astype("Int64"))


def build_reference_files(static, extract_date: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # ---- Portfolios ---------------------------------------------------
    p = static.get("portfolios")
    if p is not None and len(p):
        code = _col(p, "portfolio_code", "PortfolioCode", "portfolio")
        desc = _col(p, "description", "Description")
        out["Portfolios.csv"] = pd.DataFrame({
            "ExtractDate": extract_date,
            "PortfolioCode": code.astype(str),
            "Description": (desc if desc is not None else code).astype(str),
        })

        # ---- PortfolioRatingType --------------------------------------
        rt = _col(p, "rating_type", "RatingType")
        if rt is not None:
            out["PortfolioRatingType.csv"] = pd.DataFrame({
                "ExtractDate": extract_date,
                "PortfolioCode": code.astype(str),
                "RatingType": _rating_type_number(rt),
            })

    # ---- Ratings ------------------------------------------------------
    # Both scales in one file, each numbered. The hierarchy is what contracts
    # are bucketed on, and the two scales REUSE numbers 1-21, so the rating
    # type is what keeps them apart.
    ms = static.get("master_rating_scale")
    if ms is not None and len(ms):
        rating = _col(ms, "rating", "Rating")
        rtype = _rating_type_number(_col(ms, "rating_type", "RatingType"))
        hier = pd.to_numeric(_col(ms, "hierarchy", "Hierarchy"), errors="coerce")
        r = pd.DataFrame({
            "ExtractDate": extract_date,
            "Rating": rating.astype(str),
            "RatingType": rtype,
            # The two scales number their descriptions differently, which is
            # not a quirk worth normalising: internal takes the GRADE number
            # from the rating name, so QDB 4+, QDB 4 and QDB 4- are all Desc4;
            # external has no grade number in "Aaa" and uses the HIERARCHY, so
            # Aaa is Desc1 and A1 is Desc5.
            "Description": [
                _description(r, h, t)
                for r, h, t in zip(rating, hier, rtype)],
            "Hierarchy": hier.astype("Int64"),
        })
        # One row per (type, hierarchy): the external scale lists several agency
        # grades against the same bucket, and LIC wants the bucket once.
        r = r.dropna(subset=["Hierarchy"])
        r = r[~r.duplicated(subset=["RatingType", "Hierarchy"])]
        out["Ratings.csv"] = r.sort_values(["RatingType", "Hierarchy"]
                                           ).reset_index(drop=True)

        out["RatingTypes.csv"] = pd.DataFrame({
            "ExtractDate": extract_date,
            "RatingType": [1, 2],
            "Description": ["Internal Rating", "External Rating"],
        })

    # ---- CollateralType -----------------------------------------------
    ct = static.get("collateral_types")
    if ct is not None and len(ct):
        cid = _col(ct, "collateral_type_id", "CollateralTypeId", "id")
        out["CollateralType.csv"] = pd.DataFrame({
            "ExtractDate": extract_date,
            "CollateralTypeId": pd.to_numeric(cid, errors="coerce").astype("Int64"),
            "ExternalCollateralType": pd.to_numeric(
                _col(ct, "external_collateral_type", "external_type", default=None)
                if _col(ct, "external_collateral_type", "external_type") is not None
                else cid, errors="coerce").astype("Int64"),
            "Description": _col(ct, "description", "Description",
                                "collateral_type", default="").astype(str),
            # Two decimals, as the reference writes it. Pandas would emit
            # 1.0 where LIC's file has 1.00.
            "HaircutGeneral": pd.to_numeric(
                _col(ct, "haircut_general", "haircut", default=np.nan),
                errors="coerce").map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
            "Haircut12M": _col(ct, "haircut_12m", default=""),
            "Haircut24M": _col(ct, "haircut_24m", default=""),
            "RealizationPeriod": _col(ct, "realization_period", default=""),
        })

    # ---- FxRate -------------------------------------------------------
    fx = static.get("fx_rates")
    if fx is not None and len(fx):
        out["FxRate.csv"] = pd.DataFrame({
            "ExtractDate": extract_date,
            "CurrencyCode": _col(fx, "currency_code", "currency",
                                 "CurrencyCode").astype(str),
            "FXRate": pd.to_numeric(_col(fx, "fx_rate", "rate", "FXRate"),
                                    errors="coerce"),
        })
    return out
