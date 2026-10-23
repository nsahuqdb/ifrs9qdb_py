"""The macro store: every series fetched, kept by vintage.

A vintage is one refresh, frozen: what each source said on that day. A new
refresh writes a new vintage and never edits an old one, so a config filled
from the store, or a scenario run on it, names the vintage it used and can be
rebuilt from it later -- an IMF revision never changes a past quarter.

On disk, under the store's folder (``<project>/macro_store`` by default):

    vintages/<id>/series.csv            one row per series (sources.SERIES_COLS)
    vintages/<id>/observations.csv.gz   series_id, period, value
    vintages/<id>/meta.yml              when, by whom, the fetch log, the parent

Plain CSV, so the R engine reads the same store and a person can open it.

A refresh that fetches only some datasets carries the rest forward from the
latest vintage: a series is replaced when its (source, dataset) was fetched
again, and kept otherwise. A dataset that failed is therefore not lost, and
the log says it is the earlier vintage's copy.
"""
from __future__ import annotations

import datetime as _dt
import getpass
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .catalogue import concepts as _concepts
from .catalogue import match_candidates
from .sources import OBS_COLS, SERIES_COLS, empty_frames, period_year

__all__ = ["MacroStore", "annualise", "to_concept_unit", "DEFAULT_STORE_DIR"]

DEFAULT_STORE_DIR = "macro_store"


def _who() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def annualise(obs: pd.DataFrame, freq: str, how: str = "mean") -> pd.Series:
    """One value a year from a series' observations (period, value).

    A year is used only when it is complete -- four quarters, twelve months
    -- so a year in progress never enters a history as though it were
    whole. ``how`` is ``mean`` (prices, rates, indices) or ``last`` (a stock
    at the year's end)."""
    if obs is None or obs.empty:
        return pd.Series(dtype=float)
    y = obs["period"].map(period_year)
    v = pd.to_numeric(obs["value"], errors="coerce")
    t = pd.DataFrame({"year": y, "period": obs["period"], "value": v}).dropna()
    if t.empty:
        return pd.Series(dtype=float)
    f = (freq or "A").upper()[:1]
    if f == "A":
        s = t.groupby("year")["value"].last()
    else:
        need = 4 if f == "Q" else 12
        g = t.sort_values("period").groupby("year")
        n = g["value"].size()
        s = g["value"].mean() if how == "mean" else g["value"].last()
        s = s[n >= need]
    s.index = s.index.astype(int)
    return s.sort_index().astype(float)


def to_concept_unit(annual: pd.Series, transform: str) -> pd.Series:
    """A yearly series in its concept's unit: ``pct_change`` turns a level
    (a price, an index, spending) into its % change on the year before,
    over consecutive years only; ``none`` keeps it."""
    if transform == "pct_change":
        s = annual.sort_index()
        prev = s.shift(1)
        gap = pd.Series(s.index, index=s.index).diff()
        out = 100.0 * (s / prev - 1.0)
        return out[(gap == 1) & prev.notna() & (prev != 0)].astype(float)
    if transform == "diff":
        s = annual.sort_index()
        gap = pd.Series(s.index, index=s.index).diff()
        return s.diff()[gap == 1].astype(float)
    return annual.astype(float)


class MacroStore:
    """A folder of vintages. ``MacroStore(root).series()`` is the latest."""

    def __init__(self, root):
        self.root = Path(root)

    # ------------------------------------------------------------ vintages --
    @property
    def vdir(self) -> Path:
        return self.root / "vintages"

    def vintages(self) -> pd.DataFrame:
        rows = []
        if self.vdir.is_dir():
            for d in sorted(self.vdir.iterdir()):
                m = d / "meta.yml"
                if not m.is_file():
                    continue
                try:
                    meta = yaml.safe_load(m.read_text(encoding="utf-8")) or {}
                except Exception:
                    meta = {}
                rows.append({"vintage": d.name, "created": meta.get("created", ""),
                             "by": meta.get("by", ""), "label": meta.get("label", ""),
                             "parent": meta.get("parent") or "",
                             "series": int(meta.get("n_series") or 0),
                             "observations": int(meta.get("n_obs") or 0)})
        cols = ["vintage", "created", "by", "label", "parent", "series", "observations"]
        return pd.DataFrame(rows, columns=cols)

    def latest(self) -> str | None:
        v = self.vintages()
        return None if v.empty else str(v["vintage"].iloc[-1])

    def _resolve(self, vintage) -> str | None:
        return vintage or self.latest()

    def meta(self, vintage=None) -> dict:
        v = self._resolve(vintage)
        if not v:
            return {}
        p = self.vdir / v / "meta.yml"
        if not p.is_file():
            return {}
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    def _new_id(self) -> str:
        base = "v" + _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        vid, k = base, 1
        while (self.vdir / vid).exists():
            k += 1
            vid = f"{base}-{k}"
        return vid

    def save(self, series: pd.DataFrame, obs: pd.DataFrame, label: str = "",
             log: list | None = None, carry_forward: bool = True,
             by: str | None = None) -> str:
        """Write a new vintage from what was just fetched; the rest is
        carried forward from the latest vintage when ``carry_forward``.
        Returns the new vintage's id."""
        series = series.copy() if series is not None else empty_frames()[0]
        obs = obs.copy() if obs is not None else empty_frames()[1]
        for c in SERIES_COLS:
            if c not in series.columns:
                series[c] = ""
        series = series[SERIES_COLS].fillna("").astype(str)
        obs = obs[OBS_COLS].copy()
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs = obs.dropna(subset=["period", "value"])
        parent = self.latest()
        if carry_forward and parent:
            old_s, old_o = self.series(parent), self.observations(parent)
            fresh = set(zip(series["source"], series["dataset"])) | \
                set(series["series_id"])
            keep = ~old_s.apply(lambda r: (r["source"], r["dataset"]) in fresh
                                or r["series_id"] in fresh, axis=1) \
                if len(old_s) else pd.Series(dtype=bool)
            kept = old_s[keep] if len(old_s) else old_s
            series = pd.concat([kept, series], ignore_index=True)
            obs = pd.concat([old_o[old_o["series_id"].isin(set(kept["series_id"]))], obs],
                            ignore_index=True)
        series = series.drop_duplicates("series_id", keep="last")
        obs = obs.drop_duplicates(["series_id", "period"], keep="last")
        obs = obs[obs["series_id"].isin(set(series["series_id"]))]
        series = series.sort_values("series_id").reset_index(drop=True)
        obs = obs.sort_values(["series_id", "period"]).reset_index(drop=True)

        vid = self._new_id()
        d = self.vdir / vid
        d.mkdir(parents=True, exist_ok=False)
        series.to_csv(d / "series.csv", index=False)
        obs.to_csv(d / "observations.csv.gz", index=False, compression="gzip",
                   float_format="%.10g")
        meta = {"vintage": vid,
                "created": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "by": by or _who(), "label": label or "", "parent": parent,
                "n_series": int(len(series)), "n_obs": int(len(obs)),
                "log": list(log or [])}
        (d / "meta.yml").write_text(yaml.safe_dump(meta, sort_keys=False,
                                                   allow_unicode=True), encoding="utf-8")
        return vid

    # ---------------------------------------------------------------- read --
    def series(self, vintage=None) -> pd.DataFrame:
        v = self._resolve(vintage)
        if not v or not (self.vdir / v / "series.csv").is_file():
            return empty_frames()[0]
        return pd.read_csv(self.vdir / v / "series.csv", dtype=str,
                           keep_default_na=False)

    def observations(self, vintage=None, ids=None) -> pd.DataFrame:
        v = self._resolve(vintage)
        if not v or not (self.vdir / v / "observations.csv.gz").is_file():
            return empty_frames()[1]
        o = pd.read_csv(self.vdir / v / "observations.csv.gz", dtype={"series_id": str,
                                                                      "period": str})
        if ids is not None:
            o = o[o["series_id"].isin(set(ids))]
        return o.reset_index(drop=True)

    def search(self, text: str = "", vintage=None, limit: int = 200,
               source: str | None = None, country: str | None = None) -> pd.DataFrame:
        """Series whose id, name, dataset or country contain every word of
        ``text`` (case ignored), with their first and last period."""
        s = self.series(vintage)
        if s.empty:
            return s
        hay = (s["series_id"] + " " + s["name"] + " " + s["dataset"] + " " +
               s["country"] + " " + s["unit"]).str.lower()
        keep = pd.Series(True, index=s.index)
        for w in [w for w in re.split(r"\s+", str(text or "").lower()) if w]:
            keep &= hay.str.contains(re.escape(w), regex=True)
        if source:
            keep &= s["source"] == source
        if country:
            keep &= s["country"] == country
        hit = s[keep].head(int(limit)).copy()
        if hit.empty:
            return hit
        o = self.observations(vintage, hit["series_id"])
        span = o.groupby("series_id")["period"].agg(["min", "max", "size"])
        hit = hit.join(span, on="series_id").rename(
            columns={"min": "first", "max": "last", "size": "values"})
        return hit.reset_index(drop=True)

    def annual(self, sid: str, vintage=None, how: str = "mean") -> pd.Series:
        s = self.series(vintage)
        row = s[s["series_id"] == sid]
        if row.empty:
            return pd.Series(dtype=float)
        return annualise(self.observations(vintage, [sid]), row["freq"].iloc[0], how)

    # ------------------------------------------------------------ concepts --
    def resolve_concepts(self, vintage=None, project_root=None) -> pd.DataFrame:
        """Each concept with the series it resolves to (blank when none of
        its candidates is in the store), and that series' span."""
        c = _concepts(project_root)
        s = self.series(vintage)
        sid = [match_candidates(cands, s) for cands in c["candidates"]]
        out = c[["concept", "label", "unit", "transform", "model_variable", "core"]].copy()
        out["series_id"] = [x or "" for x in sid]
        info = s.set_index("series_id") if len(s) else None
        out["source"] = [info.loc[x, "source"] if x and info is not None else "" for x in sid]
        out["last_actual"] = [info.loc[x, "last_actual"] if x and info is not None else ""
                              for x in sid]
        first, last = [], []
        for x, tr in zip(sid, c["transform"]):
            if not x:
                first.append(None)
                last.append(None)
                continue
            a = to_concept_unit(self.annual(x, vintage), tr)
            first.append(int(a.index.min()) if len(a) else None)
            last.append(int(a.index.max()) if len(a) else None)
        out["first_year"] = first
        out["last_year"] = last
        return out

    def concept_annual(self, cid: str, vintage=None, project_root=None,
                       series_id: str | None = None) -> pd.DataFrame:
        """A concept's yearly values in its own unit -- year, value, actual
        -- from the series it resolves to (or ``series_id``). ``actual`` is
        the year being at or before the series' last actual year where the
        source says one (the WEO does), else every year."""
        c = _concepts(project_root)
        row = c[c["concept"] == cid]
        if row.empty:
            raise KeyError(f"No concept {cid!r}")
        row = row.iloc[0]
        s = self.series(vintage)
        sid = series_id or match_candidates(row["candidates"], s)
        if not sid:
            return pd.DataFrame({"year": pd.Series(dtype=int),
                                 "value": pd.Series(dtype=float),
                                 "actual": pd.Series(dtype=bool)})
        a = to_concept_unit(self.annual(sid, vintage), row["transform"])
        la = s.loc[s["series_id"] == sid, "last_actual"]
        la = period_year(la.iloc[0]) if len(la) and str(la.iloc[0]).strip() else None
        out = pd.DataFrame({"year": a.index.astype(int), "value": a.to_numpy()})
        out["actual"] = True if la is None else out["year"] <= la
        out.attrs["series_id"] = sid
        return out.reset_index(drop=True)

    def concept_matrix(self, cids, vintage=None, project_root=None,
                       actual_only: bool = True) -> pd.DataFrame:
        """Several concepts side by side, one row a year (NaN where a
        concept has no value)."""
        cols = {}
        for cid in cids:
            a = self.concept_annual(cid, vintage, project_root)
            if actual_only:
                a = a[a["actual"]]
            cols[cid] = pd.Series(a["value"].to_numpy(), index=a["year"].to_numpy())
        if not cols:
            return pd.DataFrame()
        m = pd.DataFrame(cols).sort_index()
        m.index.name = "year"
        return m

    def concept_sd(self, cid: str, vintage=None, project_root=None,
                   min_years: int = 8) -> float:
        """The standard deviation of a concept's actual yearly values, or its
        catalogue default when the store has fewer than ``min_years``."""
        c = _concepts(project_root)
        row = c[c["concept"] == cid]
        dflt = float(row["default_sd"].iloc[0]) if len(row) and \
            np.isfinite(row["default_sd"].iloc[0]) else float("nan")
        a = self.concept_annual(cid, vintage, project_root)
        a = a[a["actual"]]["value"].dropna()
        if len(a) >= min_years:
            sd = float(a.std(ddof=1))
            # a series that barely moves would make every shock to it huge
            if np.isfinite(sd) and sd > 1e-6 * max(1.0, float(a.abs().mean())):
                return sd
        return dflt
