"""
The static reference data.

Rating scales, through-the-cycle PDs, scenario severities, portfolio and
product mappings. These change when the model changes, not per quarter, so they
ship with the package and a run freezes a copy of whatever it used.

A run's own frozen copy always wins when one is given: reproducing a past
quarter means using the reference THAT run used, not today's.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

__all__ = ["StaticReference", "load_static_reference", "PACKAGED_STATIC"]

PACKAGED_STATIC = Path(__file__).parent.parent / "static"

_FILES = {
    "ttc_pd_table": "ttc_pd_table.csv",
    "scenario_severity": "scenario_severity.csv",
    "master_rating_scale": "master_rating_scale.csv",
    "portfolios": "portfolios.csv",
    "product_portfolio_mapping": "product_portfolio_mapping.csv",
    "off_balance_products": "off_balance_products.csv",
    "staging_thresholds": "staging_thresholds.csv",
    "collateral_types": "collateral_types.csv",
    "fx_rates": "fx_rates.csv",
    "gcc_real_gdp_growth": "gcc_real_gdp_growth.csv",
    "gcc_gdp_current_prices": "gcc_gdp_current_prices.csv",
    "non_oil_gdp_history": "non_oil_gdp_history.csv",
    "industry_sector_mapping": "industry_sector_mapping.csv",
    "collective_assessment_rules": "collective_assessment_rules.csv",
    "segment_fallback_ratings": "segment_fallback_ratings.csv",
}


def _matches_type(column: pd.Series, rating_type: int) -> pd.Series:
    """Match a rating type given as either a number or a name.

    The TTC table codes it as 1 or 2; the master rating scale spells it
    "Internal" or "External". Handling both means neither file has to be
    edited to suit the other, and a mismatch cannot silently return nothing.
    """
    numeric = pd.to_numeric(column, errors="coerce")
    if numeric.notna().any():
        return numeric == rating_type
    name = {1: "internal", 2: "external"}.get(rating_type, "")
    return column.astype(str).str.strip().str.lower() == name


class StaticReference(dict):
    """The reference tables, addressable by attribute or key."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def ratings_for(self, rating_type: int) -> pd.DataFrame:
        """The rating scale for one type, ordered best first.

        The internal and external scales reuse hierarchy numbers 1-21, so
        hierarchy 10 means a different grade on each. They must never be
        combined, and asking for one at a time is how that is enforced.
        """
        s = self["master_rating_scale"]
        col = "rating_type" if "rating_type" in s.columns else "RatingType"
        out = s[_matches_type(s[col], rating_type)]
        return out.sort_values("hierarchy").reset_index(drop=True)

    def ttc_for(self, rating_type: int) -> pd.DataFrame:
        """Through-the-cycle PDs for one scale, valid rows only.

        A PD of zero is KEPT: the engine short-circuits it to a zero curve, and
        the rating still needs a bucket in the output. Negative PDs and PDs of
        one or more are dropped, because a probit cannot be taken of them and
        the result would be a silent NaN curve. The boundary matters -- filtering
        on `> 0` instead of `>= 0` loses three external grades and 3,600 rows.
        """
        t = self["ttc_pd_table"]
        col = "rating_type" if "rating_type" in t.columns else "RatingType"
        out = t[_matches_type(t[col], rating_type)].copy()
        pd_col = pd.to_numeric(out["ttc_pd"], errors="coerce")
        return out[pd_col.notna() & (pd_col >= 0) & (pd_col < 1)].reset_index(drop=True)


@lru_cache(maxsize=8)
def _read_dir(directory: str) -> tuple:
    d = Path(directory)
    items = []
    for name, filename in _FILES.items():
        p = d / filename
        if not p.is_file():
            continue
        # Some reference files carry a comment block above the header. Skipping
        # comment lines is safer than a fixed skiprows, which would break the
        # moment someone adds or removes a line of explanation.
        try:
            items.append((name, pd.read_csv(p, comment="#",
                                            skip_blank_lines=True)))
        except Exception as exc:
            raise ValueError(f"could not read {p.name}: {exc}") from exc
    return tuple(items)


def load_static_reference(directory=None) -> StaticReference:
    """Load the reference tables, from a run's frozen copy or the package."""
    d = Path(directory) if directory else PACKAGED_STATIC
    if not d.is_dir():
        d = PACKAGED_STATIC
    return StaticReference(dict(_read_dir(str(d))))
