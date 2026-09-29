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
           "calendar_months", "DATE_MIN", "DATE_MAX", "date_in_range",
           "schema_parse_dates", "r_parse_any_dates", "normalise_extract_dates",
           "r_text"]

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


# ---------------------------------------------------------------------------
# The input parsers: R's schema typing (.coerce_to_type(type = "date"),
# R/input_schemas.R), its checks' parser (.parse_any_date,
# R/validators_input.R) and its transforms' parser (normalise_extract_date,
# R/transform_lending.R), value for value. Each format is tried only on values
# of its shape -- as.Date() reads a prefix and ignores the rest, which is how
# an unanchored list once read "31-DEC-2025" as 2020-12-31 -- and a date
# outside 1900-2200 is NA. Each distinct value is parsed once, so a million
# rows cost what their few distinct dates cost.
# ---------------------------------------------------------------------------
import datetime as _dt
import math as _math

DATE_MIN = pd.Timestamp("1900-01-01")
DATE_MAX = pd.Timestamp("2200-12-31")
_ORIGIN = pd.Timestamp("1899-12-30")          # Excel's day 0, as R's origin
_SERIAL_MAX = (DATE_MAX - _ORIGIN).days
_EPOCH = pd.Timestamp("1970-01-01")
_UNIX_MAX = (DATE_MAX - _EPOCH).days * 86400 + 86399

_MONTH_ABB = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
     "NOV", "DEC"])}
_MONTH_FULL = {m: i + 1 for i, m in enumerate(
    ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST",
     "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"])}
# R's as.numeric() on text: a decimal or scientific number, signed
_RX_NUMBER = re.compile(r"^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$")


def date_in_range(d) -> pd.Series:
    """NA for a date outside 1900-2200, R's .date_in_range()."""
    d = pd.Series(d)
    if not pd.api.types.is_datetime64_any_dtype(d):
        d = pd.to_datetime(d, errors="coerce")
    return d.where(d.isna() | ((d >= DATE_MIN) & (d <= DATE_MAX)))


def _ts(y, m, d):
    try:
        t = pd.Timestamp(year=int(y), month=int(m), day=int(d))
    except (ValueError, OverflowError):
        return pd.NaT
    return t if DATE_MIN <= t <= DATE_MAX else pd.NaT


def _yy(y: str) -> int:
    return _two_digit_year(int(y))


def _mon(tok: str, full: bool):
    """R's %b / %B: the month's full name or its abbreviation, any case."""
    t = tok.upper()
    m = _MONTH_FULL.get(t) if full else None
    return m if m is not None else _MONTH_ABB.get(t)


# (anchored shape, [readers]); a reader takes the match and returns a date or
# NaT, and the next reader is tried only when one returns NaT.
def _rule(rx, *readers):
    return (re.compile(rx), readers)


_ISO_GUARDED = _rule(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:\D|$)",
                     lambda g: _ts(g[0], g[1], g[2]))
_MDY4_EITHER = _rule(r"^(\d{1,2})/(\d{1,2})/(\d{4})$",
                     lambda g: _ts(g[2], g[0], g[1]),     # %m/%d/%Y
                     lambda g: _ts(g[2], g[1], g[0]))     # %d/%m/%Y
_MDY2 = _rule(r"^(\d{1,2})/(\d{1,2})/(\d{2})$",
              lambda g: _ts(_yy(g[2]), g[0], g[1]))       # %m/%d/%y
_DMON2 = _rule(r"^(\d{1,2})-([A-Za-z]{3})-(\d{2})$",
               lambda g: _ts(_yy(g[2]), _mon(g[1], True) or 0, g[0]))
_DMON4_ANY = _rule(r"^(\d{1,2})-([A-Za-z]{3,9})-(\d{4})$",
                   lambda g: _ts(g[2], _mon(g[1], True) or 0, g[0]))
_DMY_DASH = _rule(r"^(\d{1,2})-(\d{1,2})-(\d{4})$",
                  lambda g: _ts(g[2], g[1], g[0]))        # %d-%m-%Y
_YMD_SLASH = _rule(r"^(\d{4})/(\d{1,2})/(\d{1,2})(?:\D|$)",
                   lambda g: _ts(g[0], g[1], g[2]))
_COMPACT = _rule(r"^(\d{4})(\d{2})(\d{2})$",
                 lambda g: _ts(g[0], g[1], g[2]))         # %Y%m%d

# R's .EXTRACT_DATE_SHAPES and .ANY_DATE_SHAPES
_EXTRACT_RULES = (_ISO_GUARDED, _MDY4_EITHER, _MDY2, _DMON2, _DMON4_ANY,
                  _DMY_DASH, _YMD_SLASH)
_ANY_RULES = _EXTRACT_RULES + (_COMPACT,)
# R's schema chain (after the serial): ISO (a prefix, no guard), M/D/YYYY,
# DD-MON-YY, DD-MON-YYYY with a 3-letter month, M/D/YY.
_SCHEMA_RULES = (
    _rule(r"^(\d{4})-(\d{1,2})-(\d{1,2})", lambda g: _ts(g[0], g[1], g[2])),
    _rule(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", lambda g: _ts(g[2], g[0], g[1])),
    _DMON2,
    _rule(r"^(\d{1,2})-([A-Za-z]{3})-(\d{4})$",
          lambda g: _ts(g[2], _mon(g[1], True) or 0, g[0])),
    _MDY2,
)


def _by_shape(t: str, rules):
    for rx, readers in rules:
        m = rx.match(t)
        if m is None:
            continue
        g = m.groups()
        for read in readers:
            d = read(g)
            if d is not pd.NaT:
                return d
    return pd.NaT


def _serial(x: float):
    """An Excel serial as R reads it: whole days from 1899-12-30."""
    if not _math.isfinite(x):
        return pd.NaT
    days = _math.floor(x)
    if days < 2 or days > _SERIAL_MAX:
        return pd.NaT
    return _ORIGIN + pd.Timedelta(days=days)


def _is_datelike(v) -> bool:
    return isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64))


def _as_date(v):
    t = pd.Timestamp(v)
    if t is pd.NaT or pd.isna(t):
        return pd.NaT
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    t = t.normalize()
    return t if DATE_MIN <= t <= DATE_MAX else pd.NaT


def _is_number(v) -> bool:
    return (isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_)))


def r_text(v):
    """A cell as R's as.character() writes it: an integral number without
    a decimal point (46182, not 46182.0), TRUE/FALSE for a logical, None
    for a missing value."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if _is_number(v):
        x = float(v)
        if _math.isnan(x):
            return None
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return f"{x:.15g}"
    if isinstance(v, float) and _math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _map_unique(values, one) -> pd.Series:
    s = pd.Series(values)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if s.empty:
        return out
    if pd.api.types.is_datetime64_any_dtype(s):
        d = s
        if getattr(d.dt, "tz", None) is not None:
            d = d.dt.tz_localize(None)
        return date_in_range(d.dt.normalize().astype("datetime64[ns]"))
    obj = s.astype(object)
    notna = obj.notna()
    if not notna.any():
        return out
    try:
        uniq = pd.unique(obj[notna])
    except TypeError:
        uniq = list(dict.fromkeys(obj[notna].tolist()))
    parsed = {}
    for u in uniq:
        try:
            parsed[u] = one(u)
        except Exception:
            parsed[u] = pd.NaT
    out.loc[notna] = pd.to_datetime(obj[notna].map(parsed), errors="coerce")
    return out


def _schema_one(v):
    if _is_datelike(v):
        return _as_date(v)
    if _is_number(v):
        x = float(v)
        return pd.NaT if _math.isnan(x) or x <= 0 else _serial(x)
    t = r_text(v)
    if t is None:
        return pd.NaT
    t = t.strip()
    if t == "" or t.lower() == "na":
        return pd.NaT
    if _RX_NUMBER.match(t) and float(t) > 0:
        return _serial(float(t))
    return _by_shape(t, _SCHEMA_RULES)


def schema_parse_dates(values) -> pd.Series:
    """R's schema typing of a date column (.coerce_to_type, R/input_schemas.R).

    An Excel serial (a number above 0, as text or not) is whole days from
    1899-12-30; text is YYYY-MM-DD (a time after it ignored), M/D/YYYY,
    DD-MON-YY, DD-MON-YYYY or M/D/YY. Nothing else is read -- not D/M/YYYY,
    not a compact 20260609 (that is a serial far beyond 2200) -- and a date
    outside 1900-2200 is NA. Every check and the calculation read the typed
    column, so this decides what a date in an extract means.
    """
    return _map_unique(values, _schema_one)


def _any_one(v):
    if _is_datelike(v):
        return _as_date(v)
    t = r_text(v)
    if t is None:
        return pd.NaT
    t = t.strip()
    if _RX_NUMBER.match(t):
        x = float(t)
        if 1 <= x < 100000:
            return _serial(x)
        if 1e8 <= x < 1e11:
            secs = _math.floor(x)
            if secs > _UNIX_MAX:
                return pd.NaT
            return _EPOCH + pd.Timedelta(days=secs // 86400)
    return _by_shape(t, _ANY_RULES)


def r_parse_any_dates(values) -> pd.Series:
    """R's .parse_any_date() (R/validators_input.R), the checks' parser.

    A number is an Excel serial (1 to 99,999) or Unix epoch seconds (1e8 to
    1e11) by magnitude; text takes the format its shape allows: YYYY-MM-DD,
    M/D/YYYY (else D/M/YYYY), M/D/YY, DD-MON-YY, DD-MON-YYYY (month
    abbreviated or in full), D-M-YYYY, YYYY/MM/DD or YYYYMMDD.
    """
    return _map_unique(values, _any_one)


def _extract_one(v):
    if _is_datelike(v):
        return _as_date(v)
    t = r_text(v)
    if t is None:
        return pd.NaT
    t = t.strip()
    return pd.NaT if t == "" else _by_shape(t, _EXTRACT_RULES)


def normalise_extract_dates(values) -> pd.Series:
    """R's normalise_extract_date() (R/transform_lending.R): the shapes of
    r_parse_any_dates without the numbers and the compact form."""
    return _map_unique(values, _extract_one)

