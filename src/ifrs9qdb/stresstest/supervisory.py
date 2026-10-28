"""Supervisory scenarios from their publishers' own tables.

The Federal Reserve publishes its stress test scenarios every February as
CSV tables: a baseline and a severely adverse scenario, each with a domestic
and an international table of quarterly paths. ``fed_scenario`` turns them
into a scenario on this system's variables, so next year's scenario is a
download (or an upload of the same four files) rather than numbers typed in:

    GDP_WLD   world GDP growth: each region's real GDP growth through the
              year (fourth quarter on fourth quarter, from the quarterly
              annualised rates), severely adverse less baseline, weighted
              by the regions' shares of world output at PPP (IMF WEO, from
              the macro store: the US; the euro area as Germany, France and
              Italy; Japan; the UK; developing Asia as China, India,
              Indonesia, Thailand, Malaysia and Pakistan)
    US_RATE   the three-month Treasury rate's yearly average, severely
              adverse less baseline; US_RATE is a yearly change, so each
              year's shock is the change in that gap

Everything else -- oil, Qatar, the GCC -- follows through the linked
effects, as for the library's Fed scenario. A scenario covers the calendar
years the tables cover in full (three for the Fed's thirteen quarters).

Other supervisors publish in other forms (the ESRB a PDF and a workbook
every second year); their scenarios are entered on the builder and saved
as supervisory.

    fed_urls(year)              the four tables' addresses
    fetch_fed(year)             download them -> {table: text}
    read_fed_table(text)        one table -> a tidy frame
    fed_scenario(year, tables)  the scenario, with its reasons and sources
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

__all__ = ["FED_TABLES", "fed_urls", "fetch_fed", "read_fed_table", "fed_scenario",
           "REGIONS_PPP", "region_weights"]

FED_PAGE = "https://www.federalreserve.gov/supervisionreg/dfa-stress-tests-{year}.htm"
FED_FILE = "https://www.federalreserve.gov/supervisionreg/files/{year}-Table_{code}.csv"
FED_TABLES = {
    "baseline_domestic": "3A_Supervisory_Baseline_Domestic",
    "baseline_international": "3B_Supervisory_Baseline_International",
    "adverse_domestic": "4A_Supervisory_Severely_Adverse_Domestic",
    "adverse_international": "4B_Supervisory_Severely_Adverse_International",
}

# The regions of the Fed's tables, and the countries whose PPP shares of
# world output weight them (those the macro store holds).
REGIONS_PPP = {
    "US": ["USA"],
    "Euro area": ["DEU", "FRA", "ITA"],
    "Developing Asia": ["CHN", "IND", "IDN", "THA", "MYS", "PAK"],
    "Japan": ["JPN"],
    "UK": ["GBR"],
}
# the column of each region's real GDP growth in the Fed's tables
_GDP_COL = {
    "US": r"^real gdp growth",
    "Euro area": r"^euro area real gdp growth",
    "Developing Asia": r"^developing asia real gdp growth",
    "Japan": r"^japan real gdp growth",
    "UK": r"^u\.?k\.? real gdp growth",
}
_RATE_COL = r"^3-month treasury"
_NAME = {"US": "the US", "Euro area": "the euro area", "Developing Asia": "developing Asia",
         "Japan": "Japan", "UK": "the UK"}


def fed_urls(year: int) -> dict:
    return {k: FED_FILE.format(year=int(year), code=c) for k, c in FED_TABLES.items()}


def fetch_fed(year: int, transport=None) -> dict:
    """The four tables of the Fed's ``year`` scenarios, as text. Raises
    ``IOError`` naming the file that could not be read."""
    from ..macro.sources import _get
    out = {}
    for k, url in fed_urls(year).items():
        body = _get(transport, url, timeout=60)
        out[k] = body.decode("utf-8-sig", "replace")
    return out


def _quarter(v) -> tuple[int, int] | None:
    s = str(v).strip()
    m = re.match(r"^(\d{4})\s*[:\-\s]?\s*Q([1-4])$", s, re.I) or \
        re.match(r"^Q([1-4])\s*[:\-\s]?\s*(\d{4})$", s, re.I)
    if not m:
        return None
    a, b = m.groups()
    return (int(a), int(b)) if len(a) == 4 else (int(b), int(a))


def read_fed_table(text: str) -> pd.DataFrame:
    """One Fed table as a frame: ``year``, ``q`` and one column per series,
    names lower-cased (``real gdp growth``, ``3-month treasury rate``, ...)."""
    d = pd.read_csv(io.StringIO(text))
    d.columns = [str(c).strip().lower() for c in d.columns]
    date = next((c for c in d.columns if c in ("date", "quarter", "period")), None)
    if date is None:
        raise ValueError("The table has no Date column.")
    yq = d[date].map(_quarter)
    d = d[yq.notna()].copy()
    d["year"] = [t[0] for t in yq[yq.notna()]]
    d["q"] = [t[1] for t in yq[yq.notna()]]
    for c in d.columns:
        if c not in (date, "year", "q", "scenario name"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.sort_values(["year", "q"]).reset_index(drop=True)


def _col(d: pd.DataFrame, pattern: str) -> str | None:
    return next((c for c in d.columns if re.search(pattern, c)), None)


def _full_years(*frames) -> list:
    """The calendar years every table covers in all four quarters."""
    yrs = None
    for d in frames:
        cnt = d.groupby("year")["q"].nunique()
        y = set(cnt[cnt == 4].index)
        yrs = y if yrs is None else yrs & y
    return sorted(yrs or [])


def _q4q4(d: pd.DataFrame, col: str, years: list) -> list:
    """Growth through each year from quarterly annualised rates."""
    out = []
    for y in years:
        g = d.loc[d["year"] == y, col].to_numpy(float)
        out.append(float((np.prod((1 + g / 100.0) ** 0.25) - 1) * 100) if len(g) == 4 else np.nan)
    return out


def _yearly_mean(d: pd.DataFrame, col: str, years: list) -> list:
    return [float(d.loc[d["year"] == y, col].mean()) for y in years]


def region_weights(store=None, vintage=None, year: int | None = None) -> dict:
    """Each region's share of world output at PPP (IMF WEO) for ``year``
    (or the latest the store has), normalised over the regions."""
    w = {}
    if store is not None:
        for r, cs in REGIONS_PPP.items():
            tot = 0.0
            for c in cs:
                try:
                    a = store.annual(f"IMF:WEO:{c}:PPPSH:A", vintage)
                except Exception:
                    a = pd.Series(dtype=float)
                a = a.dropna()
                if len(a):
                    upto = a[a.index <= year] if year is not None else a
                    tot += float((upto if len(upto) else a).iloc[-1])
            if tot > 0:
                w[r] = tot
    if len(w) < len(REGIONS_PPP):
        return {}
    s = sum(w.values())
    return {r: v / s for r, v in w.items()}


def fed_scenario(year: int, tables: dict, weights: dict | None = None) -> dict:
    """The Fed's ``year`` severely adverse scenario on this system's
    variables, as a scenario dict ready to save (category supervisory).
    ``tables`` holds the four tables' text (``FED_TABLES`` keys);
    ``weights`` each region's share of world output (``region_weights``)."""
    t = {k: read_fed_table(v) for k, v in tables.items() if v}
    miss = [k for k in FED_TABLES if k not in t]
    if miss:
        raise ValueError("Missing the Fed's tables: " + ", ".join(miss))
    bd, bi, ad, ai = (t["baseline_domestic"], t["baseline_international"],
                      t["adverse_domestic"], t["adverse_international"])
    years = _full_years(bd, bi, ad, ai)
    if not years:
        raise ValueError("The tables do not cover a full calendar year.")
    weights = weights or {}
    if not weights:
        raise ValueError("No world-output weights: the macro store has no PPP shares.")

    dev = {}
    for r, pat in _GDP_COL.items():
        dom = r == "US"
        b, a = (bd, ad) if dom else (bi, ai)
        cb, ca = _col(b, pat), _col(a, pat)
        if cb is None or ca is None:
            raise ValueError(f"No real GDP growth for {r} in the Fed's tables.")
        dev[r] = np.array(_q4q4(a, ca, years)) - np.array(_q4q4(b, cb, years))
    wsum = sum(weights[r] for r in dev)
    world = sum(weights[r] * dev[r] for r in dev) / wsum
    cb, ca = _col(bd, _RATE_COL), _col(ad, _RATE_COL)
    if cb is None or ca is None:
        raise ValueError("No three-month Treasury rate in the Fed's tables.")
    gap = np.array(_yearly_mean(ad, ca, years)) - np.array(_yearly_mean(bd, cb, years))
    rate = np.diff(np.concatenate([[0.0], gap]))

    r2 = lambda x: [round(float(v), 2) for v in x]          # noqa: E731
    gdp_w, us_r = r2(world), r2(rate)
    y0, y1 = years[0], years[-1]
    lvl = {r: float((np.prod(1 + dev[r] / 100.0) - 1) * 100) for r in dev}
    w_txt = ", ".join(f"{r} {100 * weights[r] / wsum:.0f}%" for r in dev)
    trough = float(ad.loc[ad["year"].isin(years), ca].min())
    url = FED_PAGE.format(year=int(year))
    span = f"{y0}-{y1}" if y1 != y0 else f"{y0}"
    return {
        "name": f"Fed {year} severely adverse (global)",
        "category": "supervisory", "region": "global",
        "source": f"Board of Governors of the Federal Reserve System, {year} stress test scenarios",
        "description": (f"The Federal Reserve's severely adverse scenario for {year}, from its "
                        f"published tables: a global recession with US short rates falling."),
        "history": (f"The Federal Reserve publishes a baseline and a severely adverse scenario "
                    f"for its stress test each year. In the {year} severely adverse scenario "
                    f"real GDP ends {y1} "
                    + ", ".join(f"{abs(v):.1f}% {'below' if v < 0 else 'above'} the baseline "
                                f"in {_NAME[r]}" for r, v in lvl.items())
                    + f", and the three-month Treasury rate falls to {trough:.1f}% at its "
                      "lowest."),
        "narrative": (f"World growth is {abs(gdp_w[0]):.1f} points "
                      f"{'below' if gdp_w[0] < 0 else 'above'} the forecast in Y1"
                      + "".join(f", {abs(v):.1f} {'below' if v < 0 else 'above'} in Y{j + 2}"
                                for j, v in enumerate(gdp_w[1:]) if abs(v) >= 0.05)
                      + f"; US rates are {abs(us_r[0]):.1f} points "
                        f"{'lower' if us_r[0] < 0 else 'higher'} in Y1. Oil falls with world "
                        "demand, cutting Qatar's real government spending; non-oil growth falls "
                        "with world demand and spending, then credit and property; GCC growth "
                        "falls with world demand and oil."),
        "translation": (f"From the Fed's {year} tables (baseline and severely adverse, domestic "
                        f"and international), years {span}. World GDP: each region's real GDP "
                        "growth through the year, severely adverse less baseline, weighted by "
                        f"shares of world output at PPP (IMF WEO): {w_txt}. US rate: the "
                        "three-month Treasury rate's yearly average, severely adverse less "
                        "baseline, as a yearly change."),
        "why": {"GDP_WLD": (f"The Fed's regional real GDP paths, severely adverse less baseline, "
                            f"weighted by world output shares ({w_txt})."),
                "US_RATE": ("The Fed's three-month Treasury rate, severely adverse less "
                            "baseline, yearly average, as a change each year.")},
        "book": ("Every channel, through the linked effects: the scenario sets only global "
                 "variables."),
        "limits": ("Regions outside the Fed's tables are assumed to move with the weighted "
                   "average. Oil is not in the Fed's scenario and follows world demand."),
        "references": [{"title": f"Federal Reserve, Dodd-Frank Act stress tests {year}: "
                                 "scenario data", "url": url}],
        "main": {"GDP_WLD": gdp_w, "US_RATE": us_r},
        "restructured": 0.0,
    }
