"""Shared plumbing for the check suites.

The checks themselves should read as statements about the data, so the fiddly
parts -- finding a column whose header spelling varies between extracts,
parsing a date that arrives in three formats, collecting examples for the
report -- live here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ids import as_id

__all__ = ["ok", "fail", "col", "has", "squash", "parse_any_date",
           "dup_detail", "blank_detail", "numeric_detail", "fk_detail",
           "examples_of"]

_DATE_FORMATS = ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y",
                 "%m/%d/%y", "%Y/%m/%d")


def ok(**kw) -> dict:
    return {"passed": True, **kw}


def fail(count: int, detail: str, examples=None) -> dict:
    return {"passed": False, "count": int(count), "detail": detail,
            "examples": [str(e) for e in list(examples or [])[:10]]}


def squash(name) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def col(df, *names):
    """A column by any of its spellings, or None.

    Headers arrive uppercase from SQL*Plus, snake_case from the schema layer
    and title case from Excel. Matching on the squashed form means a check does
    not have to know which reader produced the frame.
    """
    if df is None or len(df) == 0:
        return None
    low = {squash(c): c for c in df.columns}
    for n in names:
        hit = low.get(squash(n))
        if hit is not None:
            return df[hit]
    return None


def has(inputs, key: str) -> bool:
    """Whether an input table is present AND has rows.

    A file that exists with a header and no data rows is missing for every
    practical purpose, and treating it as present is how an empty extract
    reaches a validator that then reports nothing wrong.
    """
    if inputs is None:
        return False
    try:
        df = inputs[key]
    except Exception:
        df = getattr(inputs, key, None)
    return df is not None and len(df) > 0


def parse_any_date(values) -> pd.Series:
    """Parse a date column without guessing per row.

    Tries each known format across the whole column and keeps the one that
    parses the most values. Letting pandas infer row by row is what turns
    03/04/2025 into March in one row and April in the next.
    """
    s = pd.Series(values)
    if s.empty:
        return pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    text = s.astype(str).str.strip()
    best, best_n = None, -1
    for fmt in _DATE_FORMATS:
        p = pd.to_datetime(text, format=fmt, errors="coerce")
        n = int(p.notna().sum())
        if n > best_n:
            best, best_n = p, n
    loose = pd.to_datetime(text, errors="coerce", format="mixed")
    if int(loose.notna().sum()) > best_n:
        return loose
    return best


def examples_of(series, mask, limit: int = 10) -> list:
    try:
        return [str(v) for v in pd.Series(series)[mask].head(limit)]
    except Exception:
        return []


def dup_detail(values, label: str) -> dict:
    """Uniqueness, reported with the ids that repeat rather than a bare count."""
    s = as_id(values)
    s = s[s != ""]
    if s.empty:
        return ok()
    counts = s.value_counts()
    dups = counts[counts > 1]
    if dups.empty:
        return ok()
    return fail(int(dups.sum() - len(dups)),
                f"{len(dups)} {label} value(s) appear more than once",
                examples=[f"{k} x{v}" for k, v in dups.head(10).items()])


def blank_detail(values, label: str) -> dict:
    s = pd.Series(values).astype(str).str.strip()
    bad = s.isin(["", "nan", "None", "<NA>"]) | pd.Series(values).isna()
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} row(s) have a blank {label}",
                examples=[str(i) for i in bad[bad].index[:10]])


def numeric_detail(values, label: str, allow_negative: bool = False) -> dict:
    raw = pd.Series(values)
    x = pd.to_numeric(raw.astype(str).str.replace(",", "", regex=False),
                      errors="coerce")
    n_na = int((x.isna() & raw.notna()).sum())
    n_neg = 0 if allow_negative else int((x < 0).sum())
    if n_na == 0 and n_neg == 0:
        return ok()
    bits = []
    if n_na:
        bits.append(f"{n_na} non-numeric")
    if n_neg:
        bits.append(f"{n_neg} negative")
    mask = (x.isna() & raw.notna()) | ((x < 0) if not allow_negative else False)
    return fail(n_na + n_neg, f"{label}: " + ", ".join(bits),
                examples=examples_of(raw, mask))


def fk_detail(child, parent, label: str, parent_label: str) -> dict:
    """Referential integrity, reported with the orphans.

    Blank child keys are not orphans -- a missing id is a different finding,
    and reporting it here too would double-count it.
    """
    c = as_id(child)
    c = c[c != ""]
    p = set(as_id(parent))
    if c.empty:
        return ok()
    missing = sorted(set(c) - p)
    if not missing:
        return ok()
    n = int(c.isin(missing).sum())
    return fail(n, f"{len(missing)} {label} value(s) have no {parent_label} row",
                examples=missing[:10])
