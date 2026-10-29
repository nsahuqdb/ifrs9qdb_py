"""A new quarter's macro inputs, filled from the store.

``propose`` reads a config (a snapshot's, or the live one), the store and the
reporting date, and returns what the new config would hold, with every value
that differs from the config it starts from and the checks that held or
failed. ``apply`` writes it into a DRAFT config snapshot, where it is tested
and approved like any other edit. Nothing here runs a calculation or touches
a run. What it fills:

    model_inputs.yml
      mev_forecasts                non-oil GDP growth, the internal PD
                                   model's driver, five years
      external_gcc_country_growth  each GCC country's real GDP growth and
      external_gcc_country_prices  nominal GDP (USD bn), five years
      internal_scenario_weights    explicit_weights: the V2 weights of the
                                   new history and forecast, the figures
                                   auto_non_oil_gdp_cdf works out at run time
    static/non_oil_gdp_history.csv every year of the file the source has
                                   revised, and the years it now has before
                                   the forecast
    static/gcc_real_gdp_growth.csv, gcc_gdp_current_prices.csv
                                   the same for each GCC country (IMF WEO)

The five-year rule. The forecast starts at the first year with no confirmed
annual figure and runs five years: in 2026Q3, 2026 is not confirmed, so the
forecast is 2026-2030 and the IMF's 2031 is not used. "Confirmed" is the
latest of the IMF's own marker (the WEO's last actual year) and the last year
in the non-oil history file. A first year before the reporting year is
allowed -- a year can still be unconfirmed a quarter after it ends -- and is
reported, so nobody approves it without seeing it.

History. A history is only taken from a source that keeps a record: one that
marks its own last actual year (the WEO), the IMF's other datasets (the
regional outlook, whose non-oil series the history file is built from, gives
no marker, so its years before the forecast are its record) or a Planning and
Statistics Authority table. A projection republished elsewhere (FRED) never
enters a history. A file keeps its first year: earlier years in the store are
not added.

Units. The PD model is calibrated in percentage points (2.9, not 0.029): a
forecast that looks like fractions against a history in points is flagged
and scaled, which is the mistake that once separated our PD curve from
LIC's. Nominal GDP is in billions of US dollars; weights must be positive and
every country present.

Precision. Every value is written as the store holds it: nothing is rounded.
"""
from __future__ import annotations

import datetime as _dt
import re
import warnings as _warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .catalogue import countries as _countries
from .store import MacroStore

__all__ = ["propose", "apply", "render_model_inputs", "replace_top_level_block", "GCC",
           "first_forecast_year"]

GCC = ["Bahrain", "Kuwait", "Oman", "Qatar", "Saudi Arabia", "United Arab Emirates"]
_RECORD_SOURCES = {"IMF", "DM", "NPC"}
_TOL = 1e-9                 # a change smaller than this is the same value


def _iso(name: str) -> str:
    c = _countries()
    hit = c[c["name"] == name]
    return hit["code"].iloc[0] if len(hit) else name


def _read_values(path: Path) -> pd.DataFrame:
    """A static file's rows, its values as numbers: a column written with
    percent signs ("9.06%", as the history files are) is read in the unit
    the file's header declares."""
    from ..etl.static_ref import _declared_units, _read_percent_signs
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, comment="#", skip_blank_lines=True,
                     float_precision="round_trip")
    return _read_percent_signs(df, _declared_units(path))


def first_forecast_year(reporting_date: str, last_confirmed: int | None) -> dict:
    """The first forecast year and why: the first unconfirmed year, never
    after the reporting year."""
    ry = int(str(reporting_date)[:4])
    if last_confirmed is None:
        return {"first": ry, "reporting_year": ry, "last_confirmed": None,
                "rule": f"No confirmed year is known, so the forecast starts in "
                        f"the reporting year, {ry}."}
    first = min(int(last_confirmed) + 1, ry)
    rule = (f"{int(last_confirmed)} is the last confirmed year, so the forecast "
            f"starts in {first} and runs to {first + 4}.")
    if first < ry:
        rule += (f" {first} is before the reporting year {ry}: add it to the "
                 "history once it is published.")
    return {"first": first, "reporting_year": ry, "last_confirmed": int(last_confirmed),
            "rule": rule}


def _concept_values(store, cid, years, vintage, project_root):
    a = store.concept_annual(cid, vintage, project_root)
    sid = a.attrs.get("series_id", "")
    m = dict(zip(a["year"], a["value"]))
    return [m.get(y, np.nan) for y in years], sid, a


def _records(store: MacroStore, vintage) -> set:
    """The series whose past years are a record (see the module notes)."""
    ser = store.series(vintage)
    if not len(ser):
        return set()
    marked = ser["last_actual"].astype(str).str.strip() != ""
    return set(ser.loc[marked | ser["source"].isin(_RECORD_SOURCES), "series_id"])


def _diff(new, old) -> float:
    """``new - old`` to twelve decimals: the subtraction's binary residue
    (2.899624 - 2.902 = -0.0023759999999999337) is not a change."""
    new, old = float(new), float(old)
    return round(new - old, 12) if np.isfinite(new) and np.isfinite(old) else np.nan


def _history_update(cur: pd.DataFrame, rec: pd.DataFrame, keys: list) -> dict:
    """A history file after a source's record of it: every year the record
    has from the file's first year on replaces or joins the file's rows;
    rows the record lacks are kept. Returns the new rows and what moved."""
    cols = keys + ["value"]

    def tidy(d):
        d = d[cols].copy() if len(d) else pd.DataFrame({c: [] for c in cols})
        d = d.dropna(subset=["year", "value"])
        d["year"] = d["year"].astype(int)
        d["value"] = d["value"].astype(float)
        if "country" in d:
            d["country"] = d["country"].astype(str)
        return d

    cur, rec = tidy(cur), tidy(rec)
    if len(cur) and len(rec):
        if keys == ["year"]:
            rec = rec[rec["year"] >= cur["year"].min()]
        else:
            first = cur.groupby("country")["year"].min()
            rec = rec[rec["year"] >= rec["country"].map(first).fillna(-1)]
    j = rec.merge(cur, on=keys, how="left", suffixes=("", "_old"))
    new_row = j["value_old"].isna()
    # a value the source gives to ten digits is the file's value, not a revision
    moved = ~new_row & ~np.isclose(j["value"].astype(float), j["value_old"].astype(float),
                                   rtol=_TOL, atol=_TOL)
    take = j.loc[new_row | moved, cols]
    rows = (pd.concat([cur, take], ignore_index=True)
            .drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True))
    rows["year"] = rows["year"].astype(int)
    ch = j[new_row | moved].rename(columns={"value_old": "current", "value": "proposed"})
    ch["change"] = [_diff(a, b) for a, b in zip(ch["proposed"], ch["current"])]
    return {"rows": rows, "changes": ch.reset_index(drop=True),
            "added": int(new_row.sum()), "revised": int(moved.sum()),
            "changed": int(new_row.sum() + moved.sum()),
            "largest": float(ch["change"].abs().max()) if moved.any() else 0.0}


def propose(store: MacroStore, config_dir, static_dir, reporting_date: str,
            vintage: str | None = None, project_root=None,
            first_year: int | None = None, n_years: int | None = None,
            nonoil_series: str | None = None) -> dict:
    """What the new quarter's config would hold. See the module notes."""
    from ..etl.macro import compute_internal_scenario_weights
    config_dir, static_dir = Path(config_dir), Path(static_dir)
    vintage = vintage or store.latest()
    if not vintage:
        return {"ok": False, "reason": "The macro store is empty: refresh it first."}
    mi_text = (config_dir / "model_inputs.yml").read_text(encoding="utf-8")
    mi = yaml.safe_load(mi_text) or {}
    model = yaml.safe_load((config_dir / "model.yml").read_text(encoding="utf-8")) or {}
    n = int(n_years or ((model.get("horizons") or {}).get("n_forecasts") or 5))
    notes, checks = [], []
    records = _records(store, vintage)

    # --- confirmed years and the five-year rule
    hist = _read_values(static_dir / "non_oil_gdp_history.csv")
    if len(hist):
        hist = hist[["year", "value"]].dropna(subset=["year"])
        hist["year"] = hist["year"].astype(int)
        hist["value"] = pd.to_numeric(hist["value"], errors="coerce")
    hist_last = int(hist["year"].max()) if len(hist) else None
    nonoil = store.concept_annual("NONOIL_GDP_QAT", vintage, project_root,
                                  series_id=nonoil_series)
    weo_q = store.concept_annual("GDP_QAT", vintage, project_root)
    marks = [hist_last]
    for a in (nonoil, weo_q):
        if len(a) and not a["actual"].all():
            marks.append(int(a.loc[a["actual"], "year"].max()) if a["actual"].any() else None)
    last_conf = max([m for m in marks if m is not None], default=None)
    rule = first_forecast_year(reporting_date, last_conf)
    if first_year:
        rule["first"] = int(first_year)
        rule["rule"] = f"The forecast starts in {first_year}, as set, and runs to {int(first_year) + n - 1}."
    years = list(range(rule["first"], rule["first"] + n))

    # --- non-oil forecast (the internal model's driver)
    sid_no = nonoil.attrs.get("series_id", "") if len(nonoil) else ""
    m = dict(zip(nonoil["year"], nonoil["value"])) if len(nonoil) else {}
    vals = [m.get(y, np.nan) for y in years]
    if sid_no and not any(np.isfinite(v) for v in vals):
        notes.append(f"{sid_no} has no values for {years[0]}-{years[-1]} (it is a "
                     "history, not a forecast): the current non-oil forecast is kept.")
        sid_no = ""
    if not sid_no:
        if not notes:
            notes.append("No non-oil GDP forecast in the store (REO or FRED): the "
                         "current forecast is kept.")
    else:
        miss = [y for y, v in zip(years, vals) if not np.isfinite(v)]
        if miss:
            notes.append(f"Non-oil GDP has no value for {', '.join(map(str, miss))} "
                         f"in {sid_no}: the last available year is repeated.")
            last = next((v for v in reversed(vals) if np.isfinite(v)), np.nan)
            fill = []
            for v in vals:
                fill.append(v if np.isfinite(v) else (fill[-1] if fill else last))
            vals = fill
    h_vals = hist["value"].to_numpy(float) if len(hist) else np.array([])
    in_points = len(h_vals) > 1 and np.nanmean(np.abs(h_vals)) > 1

    def fractions(v) -> bool:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        return bool(in_points and len(v) and np.max(np.abs(v)) < 0.2 and (v != 0).any())

    if sid_no and fractions(vals):
        vals = [100.0 * v for v in vals]
        notes.append("The store's non-oil forecast looked like fractions against a "
                     "history in points; it was multiplied by 100.")

    # the column of mev_forecasts that holds non-oil GDP
    col = 1
    for vid, v in (model.get("variables") or {}).items():
        if vid == "VAR_NON_OIL_GDP_GROWTH":
            col = int(((v or {}).get("forecast_source") or {}).get("column_index", 1))
    fc_old = (mi.get("mev_forecasts") or {}).get("forecasts") or {}
    fc_keys = sorted(fc_old, key=lambda k: int(k)) if fc_old else list(range(1, n + 1))
    fc_new = {}
    for i, k in enumerate(fc_keys[:n] if len(fc_keys) >= n else range(1, n + 1)):
        row = [float(x) for x in (fc_old.get(k) or fc_old.get(str(k)) or [])]
        if not row:
            row = [0.0] * max(col, 1)
        while len(row) < col:
            row.append(0.0)
        if sid_no:
            row[col - 1] = float(vals[i])
        fc_new[int(k)] = row
    if not sid_no and fractions([fc_new[k][col - 1] for k in fc_new]):
        for k in fc_new:
            fc_new[k][col - 1] *= 100.0
        notes.append("The current non-oil forecast is entered as fractions (0.029 for "
                     "2.9%) against a history in points: it is multiplied by 100. The PD "
                     "model reads 0.029 as 0.029% growth.")
    nonoil_new = [fc_new[k][col - 1] for k in sorted(fc_new)]
    checks.append({"check": "Non-oil GDP in percentage points",
                   "ok": not fractions(nonoil_new)})

    # --- GCC growth and nominal GDP (the external model's driver and weights)
    gg_old = mi.get("external_gcc_country_growth") or {}
    gp_old = mi.get("external_gcc_country_prices") or {}
    gg_new, gp_new, gsrc = {}, {}, {}
    for name in GCC:
        iso = _iso(name)
        g, sg, _ = _concept_values(store, f"GDP_{iso}", years, vintage, project_root)
        p, sp, _ = _concept_values(store, f"NGDPD_{iso}", years, vintage, project_root)
        if sg and all(np.isfinite(g)):
            gg_new[name] = [float(x) for x in g]
            gsrc[name] = sg
        else:
            gg_new[name] = [float(x) for x in (gg_old.get(name) or [])]
            notes.append(f"{name}: no WEO growth for {years[0]}-{years[-1]} in the "
                         "store; the current path is kept.")
        if sp and all(np.isfinite(p)):
            gp_new[name] = [float(x) for x in p]
        else:
            gp_new[name] = [float(x) for x in (gp_old.get(name) or [])]
            notes.append(f"{name}: no WEO nominal GDP for {years[0]}-{years[-1]}; "
                         "the current weights are kept.")
    checks.append({"check": "Every GCC country has a growth path",
                   "ok": all(len(v) == n for v in gg_new.values())})
    checks.append({"check": "GCC nominal GDP is positive (weights add up)",
                   "ok": all(len(v) == n and all(x > 0 for x in v) for v in gp_new.values())})

    # --- history: the source's record of every year before the forecast
    rec = pd.DataFrame(columns=["year", "value"])
    sid_h = nonoil.attrs.get("series_id", "") if len(nonoil) else ""
    if sid_h in records:
        rec = nonoil[nonoil["actual"] & (nonoil["year"] < years[0])][["year", "value"]]
    elif sid_h:
        notes.append(f"{sid_h} is not a record of past years, so the non-oil history "
                     "is kept as it is.")
    h_up = _history_update(hist, rec, ["year"])
    h_up["series_id"] = sid_h if sid_h in records else ""
    gcc_hist = {}
    for fname, prefix in (("gcc_real_gdp_growth.csv", "GDP"),
                          ("gcc_gdp_current_prices.csv", "NGDPD")):
        cur = _read_values(static_dir / fname)
        rows, srcs = [], set()
        for name in GCC:
            a = store.concept_annual(f"{prefix}_{_iso(name)}", vintage, project_root)
            sid = a.attrs.get("series_id", "")
            if not len(a) or sid not in records or str(sid).startswith("STATIC"):
                continue
            srcs.add(sid)
            a = a[a["actual"] & (a["year"] < years[0])]
            rows += [{"country": name, "year": int(y), "value": float(v)}
                     for y, v in zip(a["year"], a["value"])]
        new = pd.DataFrame(rows, columns=["country", "year", "value"])
        up = _history_update(cur, new, ["country", "year"]) if len(new) else \
            {"rows": cur, "changes": pd.DataFrame(), "added": 0, "revised": 0,
             "changed": 0, "largest": 0.0}
        up["sources"] = sorted(srcs)
        gcc_hist[fname] = up

    # --- scenario weights: the V2 weights of the new history and forecast
    iw = mi.get("internal_scenario_weights") or {}
    sev = _read_values(static_dir / "scenario_severity.csv")
    w_new = {}
    h_after = h_up["rows"]["value"].to_numpy(float) if len(h_up["rows"]) else np.array([])
    if len(sev) and len(h_after) >= 2:
        nw = int(iw.get("n_forecast_years", 2))
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            try:
                w_new = compute_internal_scenario_weights(h_after, nonoil_new[:nw], sev)
            except ValueError as exc:
                notes.append(str(exc))
        notes += [str(c.message) for c in caught]
    w_new = {k: float(v) for k, v in w_new.items()}
    w_old = {k: float(v) for k, v in (iw.get("explicit_weights") or {}).items()}
    ew = mi.get("external_scenario_weights") or {}
    if ew.get("mode", "explicit") == "explicit":
        notes.append("The external scenario weights are explicit in this config and are "
                     "not recomputed; with mode auto_gcc_cdf the run works them out from "
                     "the GCC path.")

    # --- the change list
    def rows_for(section, item, old, new):
        out = []
        for i, y in enumerate(years):
            o = old[i] if i < len(old) else np.nan
            v = new[i] if i < len(new) else np.nan
            out.append({"section": section, "item": item, "year": y, "current": o,
                        "proposed": v,
                        "change": _diff(v, o)})
        return out

    changes = []
    old_no = [float((fc_old.get(k) or fc_old.get(str(k)) or [np.nan] * col)[col - 1])
              for k in fc_keys[:n]] if fc_old else [np.nan] * n
    changes += rows_for("Forecast", "Non-oil GDP growth, Qatar", old_no, nonoil_new)
    for name in GCC:
        changes += rows_for("Forecast", f"Real GDP growth, {name}",
                            gg_old.get(name) or [], gg_new[name])
    for name in GCC:
        changes += rows_for("Forecast", f"Nominal GDP (USD bn), {name}",
                            gp_old.get(name) or [], gp_new[name])
    for r in h_up["changes"].itertuples():
        changes.append({"section": "History", "item": "Non-oil GDP growth, Qatar",
                        "year": int(r.year), "current": r.current, "proposed": r.proposed,
                        "change": _diff(r.proposed, r.current)})
    for k in w_new:
        o = w_old.get(k, np.nan)
        changes.append({"section": "Scenario weights", "item": f"Internal: {k}",
                        "year": None, "current": o, "proposed": w_new[k],
                        "change": _diff(w_new[k], o)})
    ch = pd.DataFrame(changes)
    if len(ch):
        a, b = ch["current"].astype(float), ch["proposed"].astype(float)
        moved = ~(np.isclose(a, b, rtol=_TOL, atol=_TOL) | (a.isna() & b.isna()))

    release = store.meta(vintage).get("created", "")
    return {
        "ok": True, "vintage": vintage, "release": release,
        "years": years, "rule": rule["rule"], "first_year": years[0],
        "last_confirmed": rule["last_confirmed"],
        "nonoil": {"series_id": sid_no, "values": nonoil_new, "column": col},
        "mev_forecasts": fc_new,
        "gcc_growth": gg_new, "gcc_prices": gp_new, "gcc_sources": gsrc,
        "history_nonoil": h_up,
        "history_gcc": gcc_hist,
        "weights": {"internal": {"mode": iw.get("mode", "explicit"),
                                 "n_forecast_years": int(iw.get("n_forecast_years", 2)),
                                 "current": w_old, "proposed": w_new},
                    "external": {"mode": ew.get("mode", "explicit")}},
        "changes": ch, "changed": int(moved.sum()) if len(ch) else 0,
        "warnings": notes, "checks": pd.DataFrame(checks),
        "model_inputs_text": mi_text,
    }


def replace_top_level_block(text: str, key: str, block: str) -> str:
    """Replace one top-level YAML key's block (its line and everything
    indented under it) with ``block``, keeping every other line -- and every
    comment -- as written. A key that is not there is appended."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(key)}\s*:", ln)),
                 None)
    new = block.rstrip("\n").splitlines()
    if start is None:
        return "\n".join(lines + new) + "\n"
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t", "-")) or not lines[end].strip()):
        end += 1
    # trailing blank lines belong to the next block
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[:start] + new + lines[end:]) + "\n"


def _dump(key, value) -> str:
    return yaml.safe_dump({key: value}, sort_keys=False, default_flow_style=False,
                          allow_unicode=True)


def _weights_block(old: dict, proposed: dict, vintage: str) -> str:
    """The internal_scenario_weights block with the new explicit weights, its
    mode and every other setting kept."""
    b = dict(old or {})
    b["explicit_weights"] = {k: float(v) for k, v in proposed.items()}
    lines = _dump("internal_scenario_weights", b).splitlines()
    i = next(j for j, ln in enumerate(lines) if ln.startswith("  explicit_weights:"))
    note = ["  # The V2 weights of the history and forecast filled from the macro store",
            f"  # (vintage {vintage}), unrounded. Used only when mode is explicit;",
            "  # auto_non_oil_gdp_cdf works out the same weights at run time."]
    return "\n".join(lines[:i] + note + lines[i:]) + "\n"


def render_model_inputs(p: dict) -> str:
    """The new model_inputs.yml text: the forecast blocks and the explicit
    internal weights replaced, a ``macro_source`` block naming where they
    came from, every other line and comment kept."""
    t = p["model_inputs_text"]
    t = replace_top_level_block(t, "external_gcc_country_growth",
                                _dump("external_gcc_country_growth", p["gcc_growth"]))
    t = replace_top_level_block(t, "external_gcc_country_prices",
                                _dump("external_gcc_country_prices", p["gcc_prices"]))
    t = replace_top_level_block(t, "mev_forecasts",
                                _dump("mev_forecasts", {"forecasts": p["mev_forecasts"]}))
    w = (p.get("weights") or {}).get("internal") or {}
    if w.get("proposed"):
        old = (yaml.safe_load(p["model_inputs_text"]) or {}).get("internal_scenario_weights")
        t = replace_top_level_block(t, "internal_scenario_weights",
                                    _weights_block(old, w["proposed"], p["vintage"]))
    src = {"vintage": p["vintage"], "store_created": p["release"],
           "first_forecast_year": int(p["first_year"]),
           "years": [int(y) for y in p["years"]],
           "non_oil_gdp": p["nonoil"]["series_id"],
           "non_oil_history": (p.get("history_nonoil") or {}).get("series_id", ""),
           "gcc": sorted(set(p["gcc_sources"].values())),
           "filled": _dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    t = replace_top_level_block(t, "macro_source", _dump("macro_source", src))
    return t


def _header(lines: list, source: str, vintage: str) -> list:
    """A history file's comment header, saying where its values now come
    from; every other line kept."""
    want = {"source": source, "extracted_from": f"macro store vintage {vintage}",
            "last_refreshed": _dt.date.today().isoformat()}
    out, seen = [], set()
    for ln in lines or []:
        m = re.match(r"\s*#\s*([A-Za-z_]+)\s*:", ln)
        k = m.group(1).lower() if m else None
        if k in want:
            out.append(f"# {k}: {want[k]}")
            seen.add(k)
        else:
            out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return out + [f"# {k}: {v}" for k, v in want.items() if k not in seen]


def apply(p: dict, snapshot_label: str, snapshots_root=None, edited_by=None) -> dict:
    """Write a proposal into a DRAFT snapshot: model_inputs.yml, and the
    history files the source has revised or extended."""
    from ..snapshots import read_static_csv_with_header, save_snapshot_csv, save_snapshot_yaml
    from ..snapshots import snapshot_dir
    if not p.get("ok"):
        return {"ok": False, "message": p.get("reason", "Nothing to apply.")}
    out = []
    r = save_snapshot_yaml(snapshot_label, "config/model_inputs.yml",
                           render_model_inputs(p), snapshots_root=snapshots_root,
                           edited_by=edited_by)
    if not r.get("ok"):
        return r
    out.append(r["message"])
    d = snapshot_dir(snapshot_label, snapshots_root)
    files = [("static/non_oil_gdp_history.csv", p.get("history_nonoil") or {})]
    files += [(f"static/{f}", h) for f, h in (p.get("history_gcc") or {}).items()]
    for rel, h in files:
        if not h.get("changed") or not len(h.get("rows", [])):
            continue
        src = h.get("series_id") or ", ".join(h.get("sources") or [])
        head = read_static_csv_with_header(d / rel)["comment_header"]
        r = save_snapshot_csv(snapshot_label, rel, h["rows"], snapshots_root=snapshots_root,
                              edited_by=edited_by,
                              comment_header=_header(head, src, p["vintage"]))
        out.append(r.get("message", ""))
    return {"ok": True, "message": "; ".join(m for m in out if m)}
