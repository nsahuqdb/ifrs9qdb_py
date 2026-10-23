"""What the macro store holds: the datasets, the countries and the concepts.

Three small tables, packaged here and in the R package's ``inst/macro``
byte for byte, and overridable per project by a file of the same name in the
project's ``config/macro`` folder:

    datasets.csv   the IMF datasets fetched, by group, with the dataflow ids
                   to try (the portal renames some), the frequency, the
                   country dimension and the first year
    countries.csv  the countries fetched and why each is there
    concepts.csv   the variables a scenario and the macro model work in: each
                   resolves to ONE stored series, the first of its candidates
                   present, in one unit (% change, points, % of GDP)

A series id is ``SOURCE:DATASET:COUNTRY:INDICATOR:FREQ`` -- the IMF's WEO
real GDP growth for Qatar is ``IMF:WEO:QAT:NGDP_RPCH:A``. A candidate is a
glob on that id, or ``name~REGEX@GLOB``: a series whose id matches the glob
and whose name matches the regular expression (for series whose code is not
known in advance, such as the bank NPL ratio in the soundness indicators).
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pandas as pd

__all__ = ["DATA_DIR", "read_table", "datasets", "countries", "concepts",
           "concept", "match_candidates", "series_id", "split_series_id",
           "CONCEPT_UNITS"]

DATA_DIR = Path(__file__).parent / "data"

CONCEPT_UNITS = ("% change", "points", "% of GDP", "% of loans", "USD bn")


def read_table(name: str, project_root=None) -> pd.DataFrame:
    """One catalogue table: the project's ``config/macro/<name>`` when it has
    one, else the packaged default. Every column is read as text, blanks as
    empty strings, so the R and Python readings agree."""
    cands = []
    if project_root:
        cands.append(Path(project_root) / "config" / "macro" / name)
    cands.append(DATA_DIR / name)
    p = next(c for c in cands if c.is_file())
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def datasets(project_root=None, groups=None, only_default: bool = False) -> pd.DataFrame:
    d = read_table("datasets.csv", project_root)
    if groups:
        d = d[d["group"].isin(list(groups))]
    if only_default:
        d = d[d["fetch"].str.lower() == "yes"]
    return d.reset_index(drop=True)


def countries(project_root=None) -> pd.DataFrame:
    return read_table("countries.csv", project_root)


def concepts(project_root=None) -> pd.DataFrame:
    c = read_table("concepts.csv", project_root)
    c["core"] = c["core"].str.lower().eq("yes")
    c["persistence"] = pd.to_numeric(c["persistence"], errors="coerce").fillna(0.0)
    c["default_sd"] = pd.to_numeric(c["default_sd"], errors="coerce")
    return c


def concept(cid: str, project_root=None) -> dict | None:
    c = concepts(project_root)
    hit = c[c["concept"] == cid]
    return None if hit.empty else hit.iloc[0].to_dict()


def series_id(source: str, dataset: str, country: str, indicator: str,
              freq: str) -> str:
    """The store's id for a series. Colons inside a part would break the
    split, so they are replaced."""
    parts = [str(x or "").replace(":", "_") for x in
             (source, dataset, country, indicator, freq)]
    return ":".join(parts)


def split_series_id(sid: str) -> dict:
    p = (str(sid).split(":") + [""] * 5)[:5]
    return dict(zip(("source", "dataset", "country", "indicator", "freq"), p))


def match_candidates(candidates: str, series: pd.DataFrame) -> str | None:
    """The first candidate present in ``series`` (a store's series table):
    the id of the series it names, or None. Candidates are tried in order;
    within one, the first matching id in sorted order is taken, so the
    answer does not depend on how the table happens to be ordered."""
    if series is None or series.empty or not candidates:
        return None
    ids = series["series_id"].astype(str)
    names = series["name"].astype(str) if "name" in series.columns else ids
    for cand in [c.strip() for c in str(candidates).split("|") if c.strip()]:
        if cand.startswith("name~"):
            body = cand[5:]
            rx, _, glob = body.partition("@")
            glob = glob or "*"
            try:
                pat = re.compile(rx)
            except re.error:
                continue
            ok = [i for i, n in zip(ids, names)
                  if fnmatch.fnmatchcase(i, glob) and pat.search(n)]
        else:
            ok = [i for i in ids if fnmatch.fnmatchcase(i, cand)]
        if ok:
            return sorted(ok)[0]
    return None
