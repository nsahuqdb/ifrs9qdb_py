"""Ids, and making them join.

A CSV column of whole numbers reads as int64 when it is complete and as float64
the moment one value is blank. ``astype(str)`` then renders the same id as
"548840" or "548840.0" depending on which. The two never match, and nothing
fails: the lookup simply returns nothing and whatever it fed goes to zero.

Every id used as a key goes through :func:`as_id`, so the join does not depend
on whether this quarter's extract happened to have a gap in that column.
"""
from __future__ import annotations

import pandas as pd

__all__ = ["as_id"]


def as_id(values) -> pd.Series:
    """A contract, customer or collateral id as a string that will JOIN.

    A CSV column of whole numbers reads as int64 when it is complete and as
    float64 the moment one value is blank, and ``astype(str)`` then renders the
    same id as "548840" or "548840.0" depending on which. The two never match,
    and nothing fails: the lookup simply returns nothing.

    That is not hypothetical. AccountCollateralAllocation carries a blank
    ContractId, so every allocation keyed as "548840.0" missed an account
    master keyed as "548840" -- all 5,932 of them. Collateral resolved to zero
    for the whole book, LGD used none of it, and the collateral stress lever
    moved the provision by exactly nothing. The R is not exposed to this
    because `as.character()` on an R integer has no trailing ".0".

    So ids are normalised once, here, and every key goes through it.
    """
    s = pd.Series(values)
    missing = s.isna()
    if pd.api.types.is_float_dtype(s):
        whole = s.notna() & (s == s.round())
        out = s.astype("object")
        out[whole] = s[whole].astype("int64").astype(str)
        out[~whole & s.notna()] = s[~whole & s.notna()].astype(str)
        s = out
    s = s.astype(str).str.strip()
    # A float that arrived as text ("548840.0") has the same problem.
    s = s.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    # pandas 2 renders a missing value as the STRING "nan", which is a
    # perfectly good dictionary key and therefore the more dangerous of the
    # two; pandas 3 keeps it as NaN. Blank either way, so a missing id cannot
    # collide with another missing id.
    s = s.replace({"nan": "", "None": "", "<NA>": "", "NaT": ""})
    return s.mask(missing, "").fillna("")
