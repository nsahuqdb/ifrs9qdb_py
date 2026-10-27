"""The scenario library (``data/library.yml``): pre-built scenarios with
their story, their source and -- for the historical ones -- the episode in
the data.

    library()          the scenarios, as ``Scenario`` objects
    derive(ref, sys)   a historical shock's numbers recomputed from the store
    check_library(sys) every historical number recomputed on a vintage, next
                       to the one the library holds
    episode(id, sys)   how every variable moved through an episode, for the
                       history chart
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .scenarios import Scenario

__all__ = ["LIBRARY_PATH", "library", "library_entry", "derive", "check_library",
           "episode", "derivation", "CATEGORIES", "REGIONS"]

LIBRARY_PATH = Path(__file__).parent / "data" / "library.yml"

# how the library is grouped on screen, in this order
CATEGORIES = {"history": "History", "supervisory": "Supervisory",
              "hypothetical": "What-if", "current": "Current", "mine": "Saved"}
# where a scenario's shock starts
REGIONS = {"global": "Global", "gcc": "GCC", "qatar": "Qatar"}


def _words(x) -> str:
    return " ".join(str(x or "").split())


@lru_cache(maxsize=4)
def _read(path: str, mtime: float) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _raw(path=None) -> dict:
    p = Path(path or LIBRARY_PATH)
    return _read(str(p), p.stat().st_mtime)


def library_entry(sid: str, path=None) -> dict | None:
    for e in _raw(path).get("scenarios") or []:
        if str(e.get("id")) == str(sid):
            return e
    return None


def library(H: int = 5, path=None) -> list[Scenario]:
    """Every library scenario, its numbers as the library holds them."""
    out = []
    for e in _raw(path).get("scenarios") or []:
        main = {}
        for v, ref in (e.get("main") or {}).items():
            vals = [None if x is None else float(x) for x in (ref.get("values") or [])]
            main[v] = (vals + [None] * H)[:H]
        out.append(Scenario(
            id=str(e["id"]), name=str(e["name"]), category=str(e.get("category", "history")),
            description=_words(e.get("description")),
            source=str(e.get("source", "")), main=main,
            restructured=float(e.get("restructured") or 0.0),
            episode=list(e["episode"]) if e.get("episode") else None,
            window=list(e["window"]) if e.get("window") else None,
            translation=_words(e.get("translation")),
            tail=bool(e.get("tail", False)), refs={v: _ref_text(r) for v, r in
                                                   (e.get("main") or {}).items()},
            region=str(e.get("region", "global")), history=_words(e.get("history")),
            narrative=_words(e.get("narrative")),
            why={str(k): _words(v) for k, v in (e.get("why") or {}).items()},
            book=_words(e.get("book")), limits=_words(e.get("limits")),
            references=[{"title": str(r.get("title", "")), "url": str(r.get("url", ""))}
                        for r in (e.get("references") or []) if isinstance(r, dict)]))
    return out


def _ref_text(ref: dict) -> str:
    """How a shock was derived, in words."""
    how = ref.get("how")
    if not how:
        return "set by the source"
    yrs = sorted(set(ref.get("years") or []))
    span = f"{yrs[0]}" if len(yrs) == 1 else f"{yrs[0]}–{yrs[-1]}"
    where = ""
    if ref.get("countries"):
        cs = ref["countries"]
        where = f", {'average of ' if len(cs) > 1 else ''}{', '.join(cs)}"
    src = f" ({ref['from']})" if ref.get("from") else ""
    then = ", then held" if ref.get("then") else ""
    if how == "actual":
        return f"as in {span}{where}{src}{then}"
    return f"{span} less {ref.get('base')}{where}{src}{then}"


# ------------------------------------------------------------- the data --
def _values(system, var: str, countries=None) -> pd.Series:
    """A variable's yearly values: global, Qatar's, or the average over
    ``countries`` -- outturns and projections both (an episode can be the
    current year)."""
    p = system.panel
    if var not in p:
        return pd.Series(dtype=float)
    if (system.meta.get(var) or {}).get("series"):
        rows = p[p["c"] == "G"]
    else:
        rows = p[p["c"].isin(countries or ["QAT"])]
    return rows.groupby("y")[var].mean().dropna()


def derive(ref: dict, system) -> list | None:
    """A shock's numbers from the store: the episode's own values (``how:
    actual``) or their change from the year before the shock (``how:
    change``). ``None`` when the store does not cover the years."""
    how = ref.get("how")
    if not how:
        return None
    s = _values(system, ref.get("from") or ref.get("var"), ref.get("countries"))
    yrs = [int(y) for y in ref.get("years") or []]
    if not yrs or any(y not in s.index for y in yrs):
        return None
    then = [round(float(x), 2) for x in ref.get("then") or []]
    if how == "actual":
        return [round(float(s[y]), 2) for y in yrs] + then
    b = int(ref["base"])
    if b not in s.index:
        return None
    return [round(float(s[y] - s[b]), 2) for y in yrs] + then


def derivation(sid: str, system, path=None) -> pd.DataFrame:
    """How each shock of a library scenario was derived, year by year:
    the episode's year, the value recorded then, the reference it is
    measured against (the year before the shock, for a growth rate) and
    the shock that results. A supervisory scenario's numbers are the
    source's own."""
    e = library_entry(sid, path)
    if e is None:
        return pd.DataFrame()
    rows = []
    for v, ref in (e.get("main") or {}).items():
        held = [float(x) for x in ref.get("values") or []]
        how = ref.get("how")
        if not how:
            for j, x in enumerate(held):
                rows.append({"variable": v, "label": system.label(v), "unit": system.unit(v),
                             "y": j + 1, "year": None, "recorded": None, "reference": None,
                             "reference_year": None, "shock": x, "rule": "source"})
            continue
        s = _values(system, ref.get("from") or v, ref.get("countries"))
        b = ref.get("base")
        for j, yr in enumerate(ref.get("years") or []):
            rec = float(s[yr]) if yr in s.index else None
            refv = float(s[int(b)]) if (how == "change" and b is not None
                                        and int(b) in s.index) else None
            rows.append({"variable": v, "label": system.label(v), "unit": system.unit(v),
                         "y": j + 1, "year": int(yr), "recorded": rec,
                         "reference": refv if how == "change" else 0.0,
                         "reference_year": int(b) if how == "change" else None,
                         "shock": held[j] if j < len(held) else None,
                         "rule": how})
        n = len(ref.get("years") or [])
        for k, x in enumerate(ref.get("then") or []):
            rows.append({"variable": v, "label": system.label(v), "unit": system.unit(v),
                         "y": n + k + 1, "year": None, "recorded": None, "reference": None,
                         "reference_year": None, "shock": float(x), "rule": "then"})
    return pd.DataFrame(rows)


def check_library(system, path=None) -> pd.DataFrame:
    """Every historical shock recomputed on the system's vintage, next to
    the library's numbers."""
    rows = []
    for e in _raw(path).get("scenarios") or []:
        for v, ref in (e.get("main") or {}).items():
            if not ref.get("how"):
                continue
            got = derive({**ref, "var": v}, system)
            held = [float(x) for x in ref.get("values") or []]
            gap = (float(np.max(np.abs(np.asarray(got) - np.asarray(held))))
                   if got is not None and len(got) == len(held) else np.nan)
            rows.append({"scenario": e["id"], "variable": v, "how": _ref_text(ref),
                         "library": held, "store": got, "largest_gap": gap})
    return pd.DataFrame(rows)


def episode(sid: str, system, path=None) -> dict:
    """How every variable moved through a historical scenario's window:
    year, value and whether it is an outturn, with the episode's years."""
    e = library_entry(sid, path)
    if e is None or not e.get("window"):
        return {"ok": False, "reason": "Not a historical scenario."}
    w0, w1 = (int(x) for x in e["window"])
    out = []
    for v in system.order:
        d = system.display(v)
        d = d[(d["year"] >= w0) & (d["year"] <= w1)]
        for r in d.itertuples(index=False):
            out.append({"variable": v, "label": system.label(v), "unit": system.unit(v),
                        "year": int(r.year), "value": float(r.value), "actual": bool(r.actual),
                        "set": v in (e.get("main") or {})})
    return {"ok": True, "id": sid, "episode": list(e.get("episode") or []),
            "window": [w0, w1], "series": pd.DataFrame(out)}
