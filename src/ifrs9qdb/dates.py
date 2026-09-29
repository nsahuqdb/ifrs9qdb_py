"""Dates, parsed the way the R engine parses them.

A direct port of ``.fer_parse_date`` (R/ecl_io.R). The lesson it encodes is
that an unanchored list of formats can never be trusted: a format will match a
string it has no business matching. R's ``strptime("%Y/%m/%d")`` read
"6/9/2026" as year 6, and that year-6 date flowed into MOB, the maturity, the
ECL horizon and the provision itself. So a value is CLASSIFIED by shape with an
anchored regex first, and only the format that shape can legally take is
applied.

Shapes recognised:

    YYYY-M-D / YYYY/M/D    ISO, unambiguous
    D.M.YYYY               dotted European
    YYYYMMDD               compact
    D-Mon-YY / D-Mon-YYYY  alphabetic month, unambiguous
    A/B/YY  A/B/YYYY       ambiguous numeric, resolved ONCE per column

An ambiguous numeric column is oriented as a whole: one value with a first
part above 12 makes the column day-first, one with a second part above 12
makes it month-first, both is a corrupt column (month-first, warned), neither
is month-first, the Oracle SQL*Plus convention the extracts use.

Anything that parses outside 1900-2200 is rejected to NaT, so a silently wrong
date turns loud instead of reaching the provision.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

__all__ = ["fer_parse_date", "months_between", "years_between",
           "calendar_months"]

_RX_ISO = re.compile(r"^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$")
_RX_ISO_SL = re.compile(r"^[0-9]{4}/[0-9]{1,2}/[0-9]{1,2}$")
_RX_DOT = re.compile(r"^[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}$")
_RX_COMPACT = re.compile(r"^[0-9]{8}$")
_RX_MON = re.compile(r"^[0-9]{1,2}[-/ ][A-Za-z]{3,9}[-/ ][0-9]{2,4}$")
_RX_NUM = re.compile(r"^[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4}$")
_RX_TIME = re.compile(r"[ T][0-9]{2}:[0-9]{2}(:[0-9]{2})?(\.[0-9]+)?Z?$")
_NA_WORDS = {"na", "n/a", "null", "nan", "-", "none", "nat", "<na>"}


def _two_digit_year(y: int) -> int:
    # R's %y: 00-68 -> 20xx, 69-99 -> 19xx (POSIX)
    return 2000 + y if y <= 68 else 1900 + y


def _mk(y, m, d):
    try:
        return pd.Timestamp(year=int(y), month=int(m), day=int(d))
    except (ValueError, OverflowError):
        return pd.NaT


def fer_parse_date(values, dayfirst: bool | None = None) -> pd.Series:
    """Parse a column of dates by shape, as R's ``.fer_parse_date`` does."""
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.normalize()
    txt = s.astype(object).where(s.notna(), "").astype(str).str.strip()
    txt = txt.str.replace(_RX_TIME, "", regex=True)
    live = (txt != "") & ~txt.str.lower().isin(_NA_WORDS)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if not live.any():
        return out

    def put(mask, parse):
        for i in txt.index[mask]:
            out.at[i] = parse(txt.at[i])

    def iso(t):
        y, m, d = re.split(r"[-/]", t)
        return _mk(y, m, d)

    put(live & txt.str.match(_RX_ISO), iso)
    put(live & txt.str.match(_RX_ISO_SL), iso)

    def dot(t):
        d, m, y = t.split(".")
        return _mk(y, m, d)
    put(live & txt.str.match(_RX_DOT), dot)

    def compact(t):
        return _mk(t[:4], t[4:6], t[6:8])
    put(live & txt.str.match(_RX_COMPACT), compact)

    def mon(t):
        parts = re.split(r"[-/ ]", t)
        d, mname, y = parts[0], parts[1], parts[2]
        try:
            m = pd.to_datetime(mname[:3], format="%b").month
        except (ValueError, TypeError):
            return pd.NaT
        if len(mname) > 3:
            try:
                m = pd.to_datetime(mname, format="%B").month
            except (ValueError, TypeError):
                return pd.NaT
        yy = int(y)
        if len(y) == 2:
            yy = _two_digit_year(yy)
        elif len(y) != 4:
            return pd.NaT
        return _mk(yy, m, d)
    put(live & txt.str.match(_RX_MON), mon)

    num_mask = live & txt.str.match(_RX_NUM)
    if num_mask.any():
        t = txt[num_mask].str.replace("-", "/", regex=False)
        parts = t.str.split("/", expand=True)
        a = pd.to_numeric(parts[0], errors="coerce")
        b = pd.to_numeric(parts[1], errors="coerce")
        a_big = bool((a > 12).any())
        b_big = bool((b > 12).any())
        if dayfirst is None:
            if a_big and b_big:
                warnings.warn("Date column mixes DD/MM and MM/DD orientations; "
                              "assuming MM/DD/YYYY. Check the source extract.")
                dayfirst = False
            else:
                dayfirst = a_big
        for i in t.index:
            p0, p1, y = t.at[i].split("/")
            yy = int(y)
            if len(y) == 2:
                yy = _two_digit_year(yy)
            elif len(y) == 3:
                # R's %Y reads a 3-digit year literally; treat as implausible
                out.at[i] = pd.NaT
                continue
            d, m = (p0, p1) if dayfirst else (p1, p0)
            out.at[i] = _mk(yy, m, d)

    yr = out.dt.year
    bad = out.notna() & ((yr < 1900) | (yr > 2200))
    if bad.any():
        warnings.warn(f"{int(bad.sum())} date value(s) parsed outside "
                      "1900-2200 and were set to NA.")
        out[bad] = pd.NaT
    return out


def months_between(start, end) -> pd.Series:
    """Days between two date columns over 30.4375, as R's .fer_months_between."""
    a, b = pd.Series(start), pd.Series(end)
    days = (b.to_numpy(dtype="datetime64[ns]")
            - a.to_numpy(dtype="datetime64[ns]")).astype("timedelta64[D]")
    out = days.astype(float) / 30.4375
    out[pd.isna(a).to_numpy() | pd.isna(b).to_numpy()] = np.nan
    return pd.Series(out, index=a.index)


def years_between(start, end) -> pd.Series:
    """Days between two date columns over 365.25, as R's .fer_years_between."""
    a, b = pd.Series(start), pd.Series(end)
    days = (b.to_numpy(dtype="datetime64[ns]")
            - a.to_numpy(dtype="datetime64[ns]")).astype("timedelta64[D]")
    out = days.astype(float) / 365.25
    out[pd.isna(a).to_numpy() | pd.isna(b).to_numpy()] = np.nan
    return pd.Series(out, index=a.index)


def calendar_months(start, end) -> pd.Series:
    """Whole calendar months from start to end, NaN where either is missing.

    The count the engine prices on: (year difference) * 12 + (month
    difference), before any floor is applied.
    """
    a, b = pd.to_datetime(pd.Series(start)), pd.to_datetime(pd.Series(end))
    return ((b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)).astype(float)
