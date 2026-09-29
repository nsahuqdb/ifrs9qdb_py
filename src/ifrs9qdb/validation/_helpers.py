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
           "latest_extract_date", "extract_date_counts", "extract_date_column",
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


def r_parse_dates(values) -> pd.Series:
    """R's .parse_any_date(), the checks' parser: see
    ifrs9qdb.dates.r_parse_any_dates. Each format reads only values of its
    shape, and a date outside 1900-2200 is NA."""
    from ..dates import r_parse_any_dates
    return r_parse_any_dates(values)


def r_parse_date(value) -> pd.Timestamp | None:
    """R's .parse_any_date() for one value; None when nothing parses it."""
    if value is None:
        return None
    t = r_parse_dates([value]).iloc[0]
    return None if pd.isna(t) else pd.Timestamp(t)


def extract_date_column(frame):
    """A file's EXTRACTDA as R's checks see it. On a schema-typed frame only
    the typed ``extract_date`` counts: R's typed table carries EXTRACTDA only
    where the file's schema defines it (not Origination, RepaymentSchedule,
    CustomerStagingFlagInvestments...), and the raw column kept beside the
    typed ones here must not add a file R does not look at. A raw frame
    (a test's dict) is read by its header."""
    if frame is None or len(frame) == 0:
        return None
    if getattr(frame, "attrs", {}).get("canonical"):
        return frame["extract_date"] if "extract_date" in frame.columns else None
    return col(frame, "extract_date", "EXTRACTDA", "EXTRACTDATE")


def extract_date_counts(frame) -> pd.DataFrame:
    """Every EXTRACTDA date one file carries, with its row count: columns
    date and rows, most rows first, a tie earliest first -- R's
    .extract_date_counts(). Pass a schema-typed frame, as R does: EXTRACTDA
    is then already a date and only a raw string is parsed here."""
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                          "rows": pd.Series(dtype="int64")})
    c = extract_date_column(frame)
    if c is None:
        return empty
    s = pd.Series(c)
    if pd.api.types.is_datetime64_any_dtype(s):
        from ..dates import date_in_range
        d = date_in_range(s.dt.normalize())
    else:
        key = text(s)
        key = key.where(key != "")
        d = r_parse_dates(key)
    d = d.dropna()
    if d.empty:
        return empty
    counts = d.value_counts()
    out = pd.DataFrame({"date": pd.to_datetime(counts.index),
                        "rows": counts.to_numpy().astype("int64")})
    return out.sort_values(["rows", "date"], ascending=[False, True],
                           kind="mergesort").reset_index(drop=True)


def resolve_input_extract_date(inputs) -> pd.Timestamp | None:
    """The reporting date a run adopts, exactly as R's
    resolve_input_extract_date() takes it: the AccountMaster EXTRACTDA most
    rows carry, a tie going to the earliest. Every row should carry it;
    INPUT_extract_date_matches_run_cfg reports any that does not and
    INPUT_extract_date_plausible a date before contracts' opening dates or
    after today. Pass the schema-typed inputs, as R does."""
    if not has(inputs, "AccountMaster"):
        return None
    counts = extract_date_counts(inputs["AccountMaster"])
    if counts.empty:
        return None
    return pd.Timestamp(counts["date"].iloc[0])


def latest_extract_date(frame) -> pd.Timestamp | None:
    """The latest EXTRACTDA of one file: the anchor of a lapsed maturity only
    when the run has no reporting date (R's run_reporting_date()). None when
    the file carries no parseable date."""
    if frame is None or len(frame) == 0:
        return None
    c = col(frame, "extract_date", "EXTRACTDA", "EXTRACTDATE", "ExtractDate")
    if c is None:
        return None
    from ..dates import normalise_extract_dates
    d = normalise_extract_dates(c).dropna()
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
