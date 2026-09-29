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
           "r_parse_date", "r_parse_dates", "resolve_input_extract_date",
           "latest_extract_date",
           "dup_detail", "blank_detail", "numeric_detail", "fk_detail",
           "examples_of", "text", "pad4"]

_DATE_FORMATS = ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y",
                 "%m/%d/%y", "%Y/%m/%d")


def ok(**kw) -> dict:
    return {"passed": True, **kw}


def fail(count: int, detail: str, examples=None) -> dict:
    return {"passed": False, "count": int(count), "detail": detail,
            "examples": [str(e) for e in list(examples or [])[:10]]}


def text(values) -> pd.Series:
    """Values as stripped strings, with every kind of missing value as "".

    pandas 2 turned a missing value into the STRING "nan" under
    ``astype(str)``; pandas 3 keeps it as a float NaN. Checks written against
    the first behaviour (``v.lower() != "nan"``) crash on the second, and a
    validator that crashes reports a failure for the wrong reason. Going
    through this one helper makes the checks indifferent to the version.
    """
    s = pd.Series(values)
    out = s.astype(object).where(s.notna(), "").astype(str).str.strip()
    return out.mask(out.isin(["nan", "NaN", "None", "<NA>", "NaT"]), "")


def pad4(values) -> pd.Series:
    """The 4-digit ISIC code a value leads with, zero-padded, else "".

    Mirrors R's industry-code resolution: the leading run of digits of the
    DESCRIPTION ("0113 Growing of vegetables") keeps the leading zero that the
    numeric INDUST column drops (113), and 113 and 0113 must compare equal.
    """
    s = text(values)
    lead = s.str.extract(r"^\D*(\d+)", expand=False).fillna("")
    ok_ = lead.str.fullmatch(r"\d{1,4}")
    return lead.where(ok_, "").map(lambda v: v.zfill(4) if v else "")


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


# R's .parse_any_date() tries these in this order, value by value.
_R_ANY_DATE_FORMATS = ("%Y-%m-%d", "%d-%b-%y", "%d-%B-%Y", "%m/%d/%Y",
                       "%d/%m/%Y", "%Y%m%d")
# R's normalise_extract_date() -- the transforms' parser -- uses this order.
_R_EXTRACT_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y",
                      "%d-%b-%y", "%d-%b-%Y", "%d-%m-%Y", "%Y/%m/%d")


def _by_formats(text: pd.Series, formats, out: pd.Series | None = None) -> pd.Series:
    """Parse ``text`` as R's as.Date(format=) loops do: each format in turn,
    applied to whatever is still unparsed. R ignores anything after the date
    (a time of day), hence ``exact=False``."""
    if out is None:
        out = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    blank = text.isna() | text.str.lower().isin(["", "nan", "nat", "none"])
    for fmt in formats:
        todo = out.isna() & ~blank
        if not todo.any():
            break
        p = pd.to_datetime(text[todo], format=fmt, errors="coerce", exact=False)
        good = p.notna()
        if good.any():
            out.loc[p.index[good]] = p[good].dt.normalize()
    return out


def r_parse_dates(values) -> pd.Series:
    """R's .parse_any_date(), vectorised as R runs it.

    A number is an Excel serial (1 to 99,999) or Unix epoch seconds (1e8 to
    1e11), by magnitude; anything else takes the first of R's formats that
    parses it (%Y-%m-%d, %d-%b-%y, %d-%B-%Y, %m/%d/%Y, %d/%m/%Y, %Y%m%d).

    Faithful to R, faults included: "31-DEC-2025" matches %d-%b-%y first and
    reads as 2020-12-31, because R (like exact=False here) ignores what
    follows the match. The schema layer types dates before the checks and the
    pipeline see them, so on real inputs this chain only meets values that
    are already dates.
    """
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.normalize()
    text = s.astype(str).str.strip().where(s.notna())
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    num = pd.to_numeric(text, errors="coerce")
    excel = num.notna() & (num >= 1) & (num < 100000)
    unix = num.notna() & (num >= 1e8) & (num < 1e11)
    if excel.any():
        out.loc[excel] = (pd.Timestamp("1899-12-30")
                          + pd.to_timedelta(np.floor(num[excel]), unit="D"))
    if unix.any():
        out.loc[unix] = pd.to_datetime(num[unix].astype("int64"),
                                       unit="s").dt.normalize()
    return _by_formats(text, _R_ANY_DATE_FORMATS, out)


def r_parse_date(value) -> pd.Timestamp | None:
    """R's .parse_any_date() for one value; None when nothing parses it."""
    if value is None:
        return None
    t = r_parse_dates([value]).iloc[0]
    return None if pd.isna(t) else pd.Timestamp(t)


def resolve_input_extract_date(inputs) -> pd.Timestamp | None:
    """The reporting date a run adopts, exactly as R's
    resolve_input_extract_date() takes it from AccountMaster's EXTRACTDA.

    R counts each distinct SPELLING once, not each row: the date written the
    most different ways wins, and a tie goes to the earliest date (table()
    sorts its names and the decreasing sort is stable). On a clean extract
    that is its one date. On a file that mixes dates it is usually the
    EARLIEST date, even when a single stray row carries it -- an R quirk this
    mirrors, so that both engines date a run alike, and which
    INPUT_extract_date_matches_run_cfg, INPUT_consistent_extract_date and
    INPUT_extract_date_plausible report. Pass the schema-typed inputs, as R
    does: EXTRACTDA is then already a date and only a raw string reaches the
    format chain.
    """
    if not has(inputs, "AccountMaster"):
        return None
    c = col(inputs["AccountMaster"], "extract_date", "EXTRACTDA")
    if c is None:
        return None
    raw = pd.Series(c).dropna()
    if raw.empty:
        return None
    uniq = pd.Series(pd.unique(raw.astype(str)))
    parsed = r_parse_dates(uniq).dropna()
    if parsed.empty:
        return None
    counts = parsed.dt.strftime("%Y-%m-%d").value_counts().to_dict()
    top = max(counts.values())
    return pd.Timestamp(min(k for k, v in counts.items() if v == top))


def latest_extract_date(frame) -> pd.Timestamp | None:
    """The latest EXTRACTDA of one file -- R's transforms anchor a lapsed
    maturity on it (max(normalise_extract_date(extract_date))). None when
    the file carries no parseable date, and then R extends nothing."""
    if frame is None or len(frame) == 0:
        return None
    c = col(frame, "extract_date", "EXTRACTDA", "EXTRACTDATE", "ExtractDate")
    if c is None:
        return None
    s = pd.Series(c)
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s.dropna()
        return pd.Timestamp(d.max()).normalize() if len(d) else None
    uniq = pd.Series(pd.unique(s.dropna().astype(str).str.strip()))
    d = _by_formats(uniq, _R_EXTRACT_FORMATS).dropna()
    return pd.Timestamp(d.max()) if len(d) else None


def examples_of(series, mask, limit: int = 10) -> list:
    try:
        return [str(v) for v in pd.Series(series)[mask].head(limit)]
    except Exception:
        return []


def dup_detail(values, label: str) -> dict:
    """Uniqueness, in R's .collect_duplicate_details() words.

    "N distinct LABEL with duplicates (M total duplicate rows): LABEL=value
    (rows i, j); ..." -- the five most repeated values, rows numbered from 1
    as R numbers them, so the two engines' messages can be compared as text.
    """
    s = as_id(values).reset_index(drop=True)
    live = s[s != ""]
    if live.empty:
        return ok()
    dup = live[live.duplicated(keep=False)]
    if dup.empty:
        return ok()
    rows = {v: [int(i) + 1 for i in idx] for v, idx in
            dup.groupby(dup, sort=True).groups.items()}
    order = sorted(rows, key=lambda v: (-len(rows[v]), v))
    total = sum(len(rows[v]) for v in order)
    shown = [f"{label}={v} (rows {', '.join(str(r) for r in rows[v])})"
             for v in order[:5]]
    more = f" ... and {len(order) - 5} more" if len(order) > 5 else ""
    return fail(total - len(order),
                f"{len(order)} distinct {label} with duplicates ({total} total "
                f"duplicate rows): {'; '.join(shown)}{more}",
                examples=order[:10])


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
