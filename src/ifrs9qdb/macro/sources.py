"""Where macro series come from, and how each is read into one shape.

Every source returns the same two tables:

    series        one row per series: series_id, source, dataset, country,
                  indicator, freq, name, unit, last_actual, release
    observations  one row per value: series_id, period, value

with periods written one way whatever the source wrote: ``2026`` for a year,
``2026-Q3`` for a quarter, ``2026-09`` for a month.

The sources:

    IMF        the IMF Data portal's SDMX 3.0 API (no key). A dataset is found
               by its dataflow id -- the portal moves some between its
               departments and renames a few, so ``datasets.csv`` lists the ids
               to try -- and its own structure gives the key's dimensions, so
               nothing here assumes their order.
    DM         the IMF DataMapper JSON API: WEO and Fiscal Monitor indicators,
               latest release only. The fallback when the SDMX API is blocked.
    DBNOMICS   DBnomics' mirror of the IMF datasets, for a network that can
               reach it but not the IMF.
    FRED       the St. Louis Fed (free key): US rates, and the IMF REO non-oil
               GDP projections it republishes.
    files      a long or wide CSV, an SDMX-CSV file downloaded from the
               portal, or a DataMapper JSON file: for QCB, the Planning
               Council, the Ministry of Finance, or any source with no API.
    STATIC     the package's own history files (non-oil GDP, GCC growth and
               nominal GDP), so a new store is never empty.

Nothing here touches a run: the store is read by the stress screens and by
the new-quarter config fill, which writes a draft config for approval.

Every network call goes through one ``transport(url, headers)`` returning
``(status, bytes)``; the default is urllib with the environment's proxy and
CA settings, and the tests pass their own.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .catalogue import countries as _countries
from .catalogue import series_id

__all__ = ["SERIES_COLS", "OBS_COLS", "norm_period", "period_year",
           "default_transport", "ImfSdmx", "DataMapper", "DBnomics", "Fred",
           "parse_sdmx_csv", "parse_datamapper", "read_long_csv",
           "read_wide_csv", "read_file", "static_seed", "empty_frames",
           "FRED_DEFAULT"]

SERIES_COLS = ["series_id", "source", "dataset", "country", "indicator", "freq",
               "name", "unit", "last_actual", "release"]
OBS_COLS = ["series_id", "period", "value"]

FRED_DEFAULT = ["QATNGDPXORPCHPT", "FEDFUNDS", "SOFR", "DGS2", "DGS10",
                "DCOILBRENTEU"]

Transport = Callable[[str, dict], "tuple[int, bytes]"]


def empty_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.DataFrame({c: pd.Series(dtype=str) for c in SERIES_COLS}),
            pd.DataFrame({"series_id": pd.Series(dtype=str),
                          "period": pd.Series(dtype=str),
                          "value": pd.Series(dtype=float)}))


# ----------------------------------------------------------------- periods --
_Q = re.compile(r"^(\d{4})[-_ ]?Q([1-4])$", re.I)
_M = re.compile(r"^(\d{4})[-_ ]?M?(\d{1,2})$", re.I)
_D = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_Y = re.compile(r"^(\d{4})$")
_FY = re.compile(r"^FY(\d{4})")


def norm_period(p, freq: str | None = None) -> str | None:
    """A period in the store's form, or None when it cannot be read.

    ``freq`` (A, Q, M) decides how a full date is read: FRED dates every
    observation on the first day of its period."""
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return None
    s = str(p).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    f = (freq or "").upper()[:1]
    m = _D.match(s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if f == "A":
            return f"{y}"
        if f == "Q":
            return f"{y}-Q{(mo - 1) // 3 + 1}"
        return f"{y}-{mo:02d}"
    if _Y.match(s):
        return s
    m = _FY.match(s)
    if m:
        return m.group(1)
    m = _Q.match(s)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"
    m = _M.match(s)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return None


def period_year(p: str) -> int | None:
    try:
        return int(str(p)[:4])
    except (TypeError, ValueError):
        return None


def _freq_of(period: str) -> str:
    if period is None:
        return ""
    if "-Q" in period:
        return "Q"
    if "-" in period:
        return "M"
    return "A"


# --------------------------------------------------------------- transport --
def default_transport(url: str, headers: dict | None = None,
                      timeout: float = 120.0) -> tuple[int, bytes]:
    """GET ``url`` with urllib: the environment's proxy (HTTPS_PROXY), and
    its CA bundle when ``IFRS9_CA_BUNDLE``, ``REQUESTS_CA_BUNDLE`` or
    ``SSL_CERT_FILE`` names one -- a bank network usually inspects TLS with
    its own certificate. Returns the status and the body; a network failure
    is status 0 with the reason as the body."""
    ca = next((os.environ[k] for k in ("IFRS9_CA_BUNDLE", "REQUESTS_CA_BUNDLE",
                                       "SSL_CERT_FILE") if os.environ.get(k)), None)
    ctx = ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "ifrs9qdb", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if hasattr(e, "read") else b""
    except Exception as e:                       # DNS, proxy refusal, TLS
        return 0, str(e).encode()


def _get(transport, url, headers=None) -> bytes:
    status, body = (transport or default_transport)(url, headers or {})
    if status != 200:
        msg = body[:300].decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
        raise IOError(f"HTTP {status} from {url.split('?')[0]}: {msg.strip()}")
    return body


def _split_label(v) -> tuple[str, str]:
    """``QAT: Qatar`` -> (QAT, Qatar); a bare code is its own label."""
    s = "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)
    code, sep, label = s.partition(": ")
    return (code.strip(), label.strip()) if sep else (s.strip(), s.strip())


# ------------------------------------------------------------------- SDMX ---
_COUNTRY_DIMS = ("COUNTRY", "REF_AREA", "JURISDICTION", "ECONOMY")
_FREQ_DIMS = ("FREQUENCY", "FREQ")
_UNIT_ATTRS = ("UNIT", "UNIT_MEASURE", "UNIT_MULT", "SCALE")


def parse_sdmx_csv(text: str, dataset: str, dims: list[str] | None = None,
                   country_dim: str | None = None, source: str = "IMF",
                   release: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """An SDMX-CSV table (the portal's download or its API's ``text/csv``)
    as the store's two tables. Headers and values may carry labels
    (``COUNTRY: Country``, ``QAT: Qatar``); both forms are read.

    The dimensions are ``dims`` when given (from the dataflow's structure),
    else every column before TIME_PERIOD that is not SDMX-CSV's own
    (STRUCTURE, STRUCTURE_ID, ACTION)."""
    if not text or not text.strip():
        return empty_frames()
    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    raw.columns = [_split_label(c)[0] for c in raw.columns]
    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError("Not an SDMX-CSV table: no TIME_PERIOD / OBS_VALUE columns.")
    cols = list(raw.columns)
    if dims is None:
        dims = [c for c in cols[:cols.index("TIME_PERIOD")]
                if c not in ("STRUCTURE", "STRUCTURE_ID", "STRUCTURE_NAME", "ACTION",
                             "DATAFLOW")]
    dims = [d for d in dims if d in raw.columns and d != "TIME_PERIOD"]
    cdim = country_dim if country_dim in dims else next(
        (d for d in _COUNTRY_DIMS if d in dims), None)
    fdim = next((d for d in _FREQ_DIMS if d in dims), None)
    other = [d for d in dims if d not in (cdim, fdim)]

    codes = {d: raw[d].map(lambda v: _split_label(v)[0]) for d in dims}
    labels = {d: raw[d].map(lambda v: _split_label(v)[1]) for d in dims}
    country = codes[cdim] if cdim else pd.Series([""] * len(raw))
    freq = codes[fdim] if fdim else pd.Series([""] * len(raw))
    if other:
        indicator = pd.Series([".".join(t) for t in zip(*[codes[d] for d in other])])
        name = pd.Series([" · ".join(t) for t in zip(*[labels[d] for d in other])])
    else:
        indicator = pd.Series(["VALUE"] * len(raw))
        name = indicator.copy()
    period = [norm_period(p, f) for p, f in zip(raw["TIME_PERIOD"], freq)]
    freq = pd.Series([f or _freq_of(p) for f, p in zip(freq, period)])
    sid = [series_id(source, dataset, c, i, f) for c, i, f in zip(country, indicator, freq)]
    value = pd.to_numeric(raw["OBS_VALUE"].str.replace(",", "", regex=False),
                          errors="coerce")
    obs = pd.DataFrame({"series_id": sid, "period": period, "value": value})
    obs = obs[obs["period"].notna() & obs["value"].notna()]

    unit = pd.Series([""] * len(raw))
    for a in _UNIT_ATTRS:
        if a in raw.columns:
            u = raw[a].map(lambda v: _split_label(v)[1])
            unit = unit.where(unit != "", u)
    last = (raw["LATEST_ACTUAL_ANNUAL_DATA"].map(lambda v: norm_period(_split_label(v)[0], "A") or "")
            if "LATEST_ACTUAL_ANNUAL_DATA" in raw.columns else pd.Series([""] * len(raw)))
    ser = pd.DataFrame({"series_id": sid, "source": source, "dataset": dataset,
                        "country": country.to_numpy(), "indicator": indicator.to_numpy(),
                        "freq": freq.to_numpy(), "name": name.to_numpy(),
                        "unit": unit.to_numpy(), "last_actual": last.to_numpy(),
                        "release": release})
    # one row per series: the last non-blank unit and last-actual seen for it
    ser = (ser.replace({"": np.nan}).groupby("series_id", sort=True).last()
           .reset_index().fillna(""))
    ser = ser[ser["series_id"].isin(set(obs["series_id"]))]
    return ser[SERIES_COLS].reset_index(drop=True), obs.reset_index(drop=True)


class ImfSdmx:
    """The IMF Data portal's SDMX 3.0 REST API.

    ``resolve`` finds a dataflow by any of its candidate ids and reads its
    dimensions from its own data structure; ``fetch`` builds the key from
    those -- the countries in the country dimension, the frequency in the
    frequency dimension, every other dimension open -- and reads the answer
    as SDMX-CSV with labels. Releases are the dataflow's version."""

    BASE = "https://api.imf.org/external/sdmx/3.0"

    def __init__(self, transport: Transport | None = None, base: str | None = None,
                 key: str | None = None):
        self.transport = transport or default_transport
        self.base = (base or os.environ.get("IFRS9_IMF_SDMX") or self.BASE).rstrip("/")
        self.key = key or os.environ.get("IFRS9_IMF_KEY")

    def _headers(self, accept: str) -> dict:
        h = {"Accept": accept}
        if self.key:
            h["Ocp-Apim-Subscription-Key"] = self.key
        return h

    def resolve(self, candidates) -> dict:
        ids = [c.strip() for c in str(candidates).split("|") if c.strip()]
        errors = []
        for cid in ids:
            url = (f"{self.base}/structure/dataflow/*/{urllib.parse.quote(cid)}/~"
                   "?references=datastructure")
            try:
                j = json.loads(_get(self.transport, url, self._headers("application/json")))
            except Exception as e:
                errors.append(f"{cid}: {e}")
                continue
            data = j.get("data", j)
            flows = data.get("dataflows") or []
            if not flows:
                errors.append(f"{cid}: no dataflow in the answer")
                continue
            fl = flows[0]
            dims = []
            for s in data.get("dataStructures") or []:
                dl = ((s.get("dataStructureComponents") or {}).get("dimensionList") or {})
                ds = dl.get("dimensions") or []
                dims = [d["id"] for d in sorted(ds, key=lambda d: d.get("position", 0))]
                if dims:
                    break
            name = fl.get("name")
            if isinstance(name, dict):
                name = name.get("en") or next(iter(name.values()), "")
            return {"id": fl.get("id", cid), "agency": fl.get("agencyID", "IMF"),
                    "version": fl.get("version", "~"), "dims": dims,
                    "name": name or cid}
        raise IOError("; ".join(errors) or "no dataflow id given")

    def fetch(self, row: dict, country_codes: list[str], start: int | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """One dataset of ``datasets.csv`` for the countries given."""
        meta = self.resolve(row["candidates"] or row["dataset"])
        cdim_spec = str(row.get("country_dim") or "COUNTRY")
        fixed = {}
        if "=" in cdim_spec:                       # REF_AREA=W00: a fixed area
            k, v = cdim_spec.split("=", 1)
            fixed[k.strip()] = v.strip()
            cdim = k.strip()
        else:
            cdim = cdim_spec.strip()
        dims = meta["dims"]
        if dims and cdim not in dims:
            cdim = next((d for d in _COUNTRY_DIMS if d in dims), cdim)
        parts = []
        for d in dims:
            if d in fixed:
                parts.append(fixed[d])
            elif d == cdim:
                parts.append("+".join(country_codes))
            elif d in _FREQ_DIMS and row.get("frequency"):
                parts.append(str(row["frequency"]))
            else:
                parts.append("*")
        key = ".".join(parts) if parts else "*"
        q = {}
        y0 = start or (int(row["start"]) if str(row.get("start") or "").isdigit() else None)
        if y0:
            q["c[TIME_PERIOD]"] = f"ge:{y0}"
        url = (f"{self.base}/data/dataflow/{meta['agency']}/{meta['id']}/~/{key}"
               + ("?" + urllib.parse.urlencode(q, safe="[]:") if q else ""))
        body = _get(self.transport, url, self._headers(
            "application/vnd.sdmx.data+csv;version=2.0.0;labels=both"))
        ser, obs = parse_sdmx_csv(body.decode("utf-8-sig", "replace"), row["dataset"],
                                  dims=dims or None, country_dim=cdim,
                                  release=str(meta.get("version", "")))
        return ser, obs, {"dataflow": f"{meta['agency']}:{meta['id']}({meta['version']})",
                          "url": url}


# -------------------------------------------------------------- DataMapper --
def parse_datamapper(j: dict, indicators: dict | None = None,
                     dataset: str = "WEO") -> tuple[pd.DataFrame, pd.DataFrame]:
    """A DataMapper answer (``{"values": {IND: {ISO: {year: v}}}}``) as the
    store's tables. ``indicators`` is the indicator list's
    ``{code: {label, unit, dataset}}`` for names and units."""
    vals = (j or {}).get("values") or {}
    srows, orows = [], []
    for ind, by_c in vals.items():
        info = (indicators or {}).get(ind) or {}
        ds = (info.get("dataset") or dataset or "WEO").upper()
        for iso, by_y in (by_c or {}).items():
            sid = series_id("DM", ds, iso, ind, "A")
            srows.append({"series_id": sid, "source": "DM", "dataset": ds,
                          "country": iso, "indicator": ind, "freq": "A",
                          "name": info.get("label") or ind,
                          "unit": info.get("unit") or "", "last_actual": "",
                          "release": "latest"})
            for y, v in (by_y or {}).items():
                if v is None:
                    continue
                orows.append({"series_id": sid, "period": norm_period(y, "A"),
                              "value": float(v)})
    if not srows:
        return empty_frames()
    ser = pd.DataFrame(srows).drop_duplicates("series_id")
    obs = pd.DataFrame(orows)
    return ser[SERIES_COLS], obs.dropna(subset=["period"])


class DataMapper:
    """The IMF DataMapper API: WEO and Fiscal Monitor indicators, latest
    release, no key. Used when the SDMX API cannot be reached."""

    BASE = "https://www.imf.org/external/datamapper/api/v1"

    def __init__(self, transport: Transport | None = None, base: str | None = None):
        self.transport = transport or default_transport
        self.base = (base or self.BASE).rstrip("/")

    def indicators(self) -> dict:
        j = json.loads(_get(self.transport, f"{self.base}/indicators"))
        return j.get("indicators") or {}

    def fetch(self, country_codes: list[str], datasets=("WEO", "FM"),
              indicators: list[str] | None = None, chunk: int = 20
              ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        info = self.indicators()
        want = [k for k, v in info.items()
                if (indicators is None or k in indicators)
                and str((v or {}).get("dataset", "")).upper() in {d.upper() for d in datasets}]
        sers, obss = [], []
        for i in range(0, len(want), chunk):
            for ind in want[i:i + chunk]:
                url = f"{self.base}/{ind}/{'/'.join(country_codes)}"
                try:
                    j = json.loads(_get(self.transport, url))
                except Exception:
                    continue
                s, o = parse_datamapper(j, info)
                sers.append(s)
                obss.append(o)
        if not sers:
            s, o = empty_frames()
            return s, o, {"indicators": 0}
        return (pd.concat(sers, ignore_index=True), pd.concat(obss, ignore_index=True),
                {"indicators": len(want)})


# --------------------------------------------------------------- DBnomics ---
class DBnomics:
    """DBnomics' mirror of the IMF datasets (``api.db.nomics.world``).

    The mirror's dimension names and country codes differ from the portal's
    (``REF_AREA`` with two-letter codes in most, ``weo-country`` with three in
    the WEO), so each dataset's dimensions are read first and the countries
    sent in whichever form the dataset uses."""

    BASE = "https://api.db.nomics.world/v22"

    def __init__(self, transport: Transport | None = None, base: str | None = None):
        self.transport = transport or default_transport
        self.base = (base or self.BASE).rstrip("/")

    def fetch(self, dataset: str, ds_code: str, codes3: list[str], codes2: list[str],
              limit: int = 1000, max_pages: int = 50
              ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        meta = json.loads(_get(self.transport, f"{self.base}/datasets/IMF/{ds_code}"))
        dsd = ((meta.get("datasets") or {}).get("docs") or [{}])[0]
        order = dsd.get("dimensions_codes_order") or []
        dvals = dsd.get("dimensions_values_labels") or {}
        cdim = next((d for d in order if d.lower() in
                     ("ref_area", "weo-country", "country", "geo")), None)
        if cdim is None:
            raise IOError(f"DBnomics {ds_code}: no country dimension")
        known = set((dvals.get(cdim) or {}).keys()) if isinstance(dvals.get(cdim), dict) else set()
        sel = [c for c in codes3 if c in known] or [c for c in codes2 if c in known] or codes2
        dims = json.dumps({cdim: sel})
        srows, orows = [], []
        for page in range(max_pages):
            q = urllib.parse.urlencode({"observations": 1, "limit": limit,
                                        "offset": page * limit, "dimensions": dims})
            j = json.loads(_get(self.transport, f"{self.base}/series/IMF/{ds_code}?{q}"))
            docs = ((j.get("series") or {}).get("docs")) or []
            for d in docs:
                dd = d.get("dimensions") or {}
                ctry = str(dd.get(cdim, ""))
                freq = str(dd.get("FREQ", dd.get("frequency", d.get("@frequency", ""))))[:1].upper()
                ind = ".".join(str(dd[k]) for k in order if k not in (cdim, "FREQ", "frequency") and k in dd)
                sid = series_id("DBNOMICS", dataset, ctry, ind or d.get("series_code", ""), freq)
                srows.append({"series_id": sid, "source": "DBNOMICS", "dataset": dataset,
                              "country": ctry, "indicator": ind, "freq": freq,
                              "name": d.get("series_name", ind), "unit": "",
                              "last_actual": "", "release": str(d.get("indexed_at", ""))[:10]})
                for p, v in zip(d.get("period") or [], d.get("value") or []):
                    if v is None or v == "NA":
                        continue
                    orows.append({"series_id": sid, "period": norm_period(p, freq),
                                  "value": float(v)})
            if len(docs) < limit:
                break
        if not srows:
            s, o = empty_frames()
            return s, o, {"series": 0}
        ser = pd.DataFrame(srows).drop_duplicates("series_id")[SERIES_COLS]
        obs = pd.DataFrame(orows).dropna(subset=["period"])
        return ser, obs, {"series": len(ser)}


# -------------------------------------------------------------------- FRED ---
class Fred:
    """FRED series by id; needs a free key (``FRED_API_KEY``)."""

    BASE = "https://api.stlouisfed.org/fred"

    def __init__(self, key: str | None = None, transport: Transport | None = None):
        self.key = key or os.environ.get("FRED_API_KEY", "")
        self.transport = transport or default_transport

    def fetch(self, ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        if not self.key:
            raise IOError("FRED needs an API key (FRED_API_KEY).")
        srows, orows, missing = [], [], []
        for sid0 in ids:
            q = urllib.parse.urlencode({"series_id": sid0, "api_key": self.key,
                                        "file_type": "json"})
            try:
                info = json.loads(_get(self.transport, f"{self.BASE}/series?{q}"))
                s = (info.get("seriess") or [{}])[0]
                obs = json.loads(_get(self.transport,
                                      f"{self.BASE}/series/observations?{q}"))
            except Exception:
                missing.append(sid0)
                continue
            freq = str(s.get("frequency_short", "A"))[:1].upper()
            freq = {"D": "M", "W": "M", "B": "M", "S": "A"}.get(freq, freq)
            sid = series_id("FRED", sid0, "", sid0, freq)
            srows.append({"series_id": sid, "source": "FRED", "dataset": sid0,
                          "country": "", "indicator": sid0, "freq": freq,
                          "name": s.get("title", sid0), "unit": s.get("units", ""),
                          "last_actual": "", "release": str(s.get("last_updated", ""))[:10]})
            vals = []
            for o in obs.get("observations") or []:
                v = pd.to_numeric(o.get("value"), errors="coerce")
                if pd.notna(v):
                    vals.append((norm_period(o.get("date"), freq), float(v)))
            # daily and weekly series are averaged to the month
            df = pd.DataFrame(vals, columns=["period", "value"]).dropna()
            df = df.groupby("period", as_index=False)["value"].mean()
            for p, v in zip(df["period"], df["value"]):
                orows.append({"series_id": sid, "period": p, "value": v})
        if not srows:
            s, o = empty_frames()
            return s, o, {"missing": missing}
        return (pd.DataFrame(srows)[SERIES_COLS], pd.DataFrame(orows),
                {"missing": missing})


# ------------------------------------------------------------------- files ---
def _frame(srows, orows):
    if not srows:
        return empty_frames()
    ser = pd.DataFrame(srows).drop_duplicates("series_id")
    for c in SERIES_COLS:
        if c not in ser.columns:
            ser[c] = ""
    obs = pd.DataFrame(orows)
    obs = obs[obs["period"].notna() & obs["value"].notna()]
    return ser[SERIES_COLS].reset_index(drop=True), obs.reset_index(drop=True)


def read_long_csv(path_or_text, source: str = "FILE", dataset: str = "FILE",
                  country: str = "", unit: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """A long table: ``period, value`` and either ``series_id`` or
    ``indicator`` (with optional ``country``, ``frequency``/``freq``,
    ``name``, ``unit``, ``source``, ``dataset``)."""
    t = _read_text_table(path_or_text)
    cols = {c.lower().strip(): c for c in t.columns}
    need = {"period", "value"}
    if not need <= set(cols):
        raise ValueError("A long table needs period and value columns.")
    srows, orows = [], []
    for _, r in t.iterrows():
        g = lambda k, d="": str(r[cols[k]]).strip() if k in cols and str(r[cols[k]]).strip() not in ("", "nan") else d
        freq = (g("frequency") or g("freq") or "").upper()[:1]
        per = norm_period(g("period"), freq or None)
        if per is None:
            continue
        freq = freq or _freq_of(per)
        if "series_id" in cols and g("series_id"):
            sid = g("series_id")
            parts = (sid.split(":") + [""] * 5)[:5]
            src, ds, ctry, ind, fq = parts
        else:
            src, ds = g("source", source), g("dataset", dataset)
            ctry, ind, fq = g("country", country), g("indicator", "VALUE"), freq
            sid = series_id(src, ds, ctry, ind, fq)
        srows.append({"series_id": sid, "source": src, "dataset": ds, "country": ctry,
                      "indicator": ind, "freq": fq or freq, "name": g("name", ind),
                      "unit": g("unit", unit), "last_actual": "", "release": ""})
        orows.append({"series_id": sid, "period": per,
                      "value": pd.to_numeric(g("value"), errors="coerce")})
    return _frame(srows, orows)


def read_wide_csv(path_or_text, source: str, dataset: str, country: str = "",
                  freq: str | None = None, unit: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """A wide table: the first column the period, one column per indicator
    (its header the indicator code, optionally ``CODE: name``)."""
    t = _read_text_table(path_or_text)
    pcol = t.columns[0]
    srows, orows = [], []
    for col in t.columns[1:]:
        code, label = _split_label(col)
        code = re.sub(r"[^A-Za-z0-9_.-]+", "_", code).strip("_") or "VALUE"
        for p, v in zip(t[pcol], t[col]):
            per = norm_period(p, freq)
            val = pd.to_numeric(str(v).replace(",", ""), errors="coerce")
            if per is None or pd.isna(val):
                continue
            fq = (freq or _freq_of(per)).upper()[:1]
            sid = series_id(source, dataset, country, code, fq)
            srows.append({"series_id": sid, "source": source, "dataset": dataset,
                          "country": country, "indicator": code, "freq": fq,
                          "name": label, "unit": unit, "last_actual": "", "release": ""})
            orows.append({"series_id": sid, "period": per, "value": float(val)})
    return _frame(srows, orows)


def _read_text_table(path_or_text) -> pd.DataFrame:
    if isinstance(path_or_text, (str, Path)) and Path(str(path_or_text)).is_file():
        p = Path(path_or_text)
        if p.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(p, dtype=str).fillna("")
        text = p.read_text(encoding="utf-8-sig")
    else:
        text = str(path_or_text)
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(lines)), dtype=str,
                       keep_default_na=False)


def read_file(path, source: str = "FILE", dataset: str = "FILE", country: str = "",
              freq: str | None = None, unit: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Any supported file, by what it holds: a DataMapper JSON, an SDMX-CSV
    (TIME_PERIOD and OBS_VALUE columns), a long table (period and value), or
    a wide table (period first, one column per indicator)."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return parse_datamapper(json.loads(p.read_text(encoding="utf-8")), dataset=dataset)
    if p.suffix.lower() in (".xlsx", ".xls"):
        t = pd.read_excel(p, dtype=str).fillna("")
        text = t.to_csv(index=False)
    else:
        text = p.read_text(encoding="utf-8-sig")
    head = next((ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")), "")
    cols = [_split_label(c)[0].strip().lower() for c in next(csv.reader([head]))]
    if "time_period" in cols and "obs_value" in cols:
        return parse_sdmx_csv(text, dataset, source=source)
    if "period" in cols and "value" in cols:
        return read_long_csv(text, source, dataset, country, unit)
    return read_wide_csv(text, source, dataset, country, freq, unit)


# ------------------------------------------------------------------ static ---
def static_seed(static_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The package's own history files as store series, so a store built on
    a network that reaches nothing still has the PD models' variables:

        non_oil_gdp_history.csv     STATIC:NONOIL:QAT:NGDP_NOIL_RPCH:A
        gcc_real_gdp_growth.csv     STATIC:GCC_GDP:<ISO>:NGDP_RPCH:A
        gcc_gdp_current_prices.csv  STATIC:GCC_NGDPD:<ISO>:NGDPD:A
    """
    d = Path(static_dir)
    by_name = {r["name"]: r["code"] for _, r in _countries().iterrows()}
    srows, orows = [], []

    def read(name):
        f = d / name
        if not f.is_file():
            return None
        lines = [ln for ln in f.read_text(encoding="utf-8").splitlines()
                 if not ln.startswith("#")]
        return pd.read_csv(io.StringIO("\n".join(lines)))

    t = read("non_oil_gdp_history.csv")
    if t is not None and {"year", "value"} <= set(t.columns):
        sid = series_id("STATIC", "NONOIL", "QAT", "NGDP_NOIL_RPCH", "A")
        srows.append({"series_id": sid, "source": "STATIC", "dataset": "NONOIL",
                      "country": "QAT", "indicator": "NGDP_NOIL_RPCH", "freq": "A",
                      "name": "Non-oil real GDP growth", "unit": "Percent change",
                      "last_actual": str(int(t["year"].max())), "release": "package"})
        orows += [{"series_id": sid, "period": str(int(y)), "value": float(v)}
                  for y, v in zip(t["year"], t["value"]) if pd.notna(v)]
    for fname, ds, ind, label, unit in (
            ("gcc_real_gdp_growth.csv", "GCC_GDP", "NGDP_RPCH",
             "Real GDP growth", "Percent change"),
            ("gcc_gdp_current_prices.csv", "GCC_NGDPD", "NGDPD",
             "GDP, current prices", "U.S. dollars, billions")):
        t = read(fname)
        if t is None or not {"country", "year", "value"} <= set(t.columns):
            continue
        for ctry, g in t.groupby("country"):
            iso = by_name.get(str(ctry).strip(), str(ctry).strip())
            sid = series_id("STATIC", ds, iso, ind, "A")
            srows.append({"series_id": sid, "source": "STATIC", "dataset": ds,
                          "country": iso, "indicator": ind, "freq": "A",
                          "name": label, "unit": unit,
                          "last_actual": str(int(g["year"].max())), "release": "package"})
            orows += [{"series_id": sid, "period": str(int(y)), "value": float(v)}
                      for y, v in zip(g["year"], g["value"]) if pd.notna(v)]
    return _frame(srows, orows)
