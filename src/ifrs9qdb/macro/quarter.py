"""A new quarter's macro inputs, filled from the store.

``propose`` reads a config (a snapshot's, or the live one), the store and the
reporting date, and returns what the new config would hold -- the non-oil GDP
forecast, the GCC growth and nominal GDP paths, and the confirmed years to add
to the history files -- with every value that differs from the config it
starts from, and the checks that held or failed. ``apply`` writes it into a
DRAFT config snapshot, where it is tested and approved like any other edit.
Nothing here runs a calculation or touches a run.

The five-year rule. The forecast starts at the first year with no confirmed
annual figure and runs five years: in 2026Q3, 2026 is not confirmed, so the
forecast is 2026-2030 and the IMF's 2031 is not used. "Confirmed" is the
latest of the IMF's own marker (the WEO's last actual year) and the last year
in the non-oil history file. A first year before the reporting year is
allowed -- a year can still be unconfirmed a quarter after it ends -- and is
reported, so nobody approves it without seeing it.

Units. The PD model is calibrated in percentage points (2.9, not 0.029): a
forecast that looks like fractions against a history in points is flagged
and scaled, which is the mistake that once separated our PD curve from
LIC's. Nominal GDP weights must be positive and every country present.
"""
from __future__ import annotations

import datetime as _dt
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .catalogue import countries as _countries
from .store import MacroStore

__all__ = ["propose", "apply", "replace_top_level_block", "GCC",
           "first_forecast_year"]

GCC = ["Bahrain", "Kuwait", "Oman", "Qatar", "Saudi Arabia", "United Arab Emirates"]


def _iso(name: str) -> str:
    c = _countries()
    hit = c[c["name"] == name]
    return hit["code"].iloc[0] if len(hit) else name


def _read_static(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(lines)))


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


def propose(store: MacroStore, config_dir, static_dir, reporting_date: str,
            vintage: str | None = None, project_root=None,
            first_year: int | None = None, n_years: int | None = None,
            nonoil_series: str | None = None) -> dict:
    """What the new quarter's config would hold. See the module notes."""
    config_dir, static_dir = Path(config_dir), Path(static_dir)
    vintage = vintage or store.latest()
    if not vintage:
        return {"ok": False, "reason": "The macro store is empty: refresh it first."}
    mi_text = (config_dir / "model_inputs.yml").read_text(encoding="utf-8")
    mi = yaml.safe_load(mi_text) or {}
    model = yaml.safe_load((config_dir / "model.yml").read_text(encoding="utf-8")) or {}
    n = int(n_years or ((model.get("horizons") or {}).get("n_forecasts") or 5))
    warnings, checks = [], []

    # --- confirmed years and the five-year rule
    hist = _read_static(static_dir / "non_oil_gdp_history.csv")
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
    vals, sid_no, _ = _concept_values(store, "NONOIL_GDP_QAT", years, vintage, project_root)
    if nonoil_series:
        a = store.concept_annual("NONOIL_GDP_QAT", vintage, project_root,
                                 series_id=nonoil_series)
        m = dict(zip(a["year"], a["value"]))
        vals, sid_no = [m.get(y, np.nan) for y in years], nonoil_series
    if sid_no and not any(np.isfinite(v) for v in vals):
        warnings.append(f"{sid_no} has no values for {years[0]}-{years[-1]} (it is a "
                        "history, not a forecast): the current non-oil forecast is kept.")
        sid_no = ""
    if not sid_no:
        if not warnings:
            warnings.append("No non-oil GDP forecast in the store (REO or FRED): the "
                            "current forecast is kept.")
    else:
        miss = [y for y, v in zip(years, vals) if not np.isfinite(v)]
        if miss:
            warnings.append(f"Non-oil GDP has no value for {', '.join(map(str, miss))} "
                            f"in {sid_no}: the last available year is repeated.")
            last = next((v for v in reversed(vals) if np.isfinite(v)), np.nan)
            fill = []
            for v in vals:
                fill.append(v if np.isfinite(v) else (fill[-1] if fill else last))
            vals = fill
        h = pd.to_numeric(hist.get("value", pd.Series(dtype=float)), errors="coerce")
        if len(h) and np.nanmean(np.abs(h)) > 1 and np.nanmax(np.abs(vals)) < 0.5:
            vals = [100.0 * v for v in vals]
            warnings.append("The non-oil forecast looked like fractions against a "
                            "history in points; it was multiplied by 100.")
        checks.append({"check": "Non-oil GDP in percentage points",
                       "ok": bool(np.nanmax(np.abs(vals)) >= 0.5 or np.nanmax(np.abs(vals)) == 0)})

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
        if sid_no:
            while len(row) < col:
                row.append(0.0)
            row[col - 1] = float(vals[i])
        fc_new[int(k)] = row

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
            warnings.append(f"{name}: no WEO growth for {years[0]}-{years[-1]} in the "
                            "store; the current path is kept.")
        if sp and all(np.isfinite(p)):
            gp_new[name] = [float(x) for x in p]
        else:
            gp_new[name] = [float(x) for x in (gp_old.get(name) or [])]
            warnings.append(f"{name}: no WEO nominal GDP for {years[0]}-{years[-1]}; "
                            "the current weights are kept.")
    checks.append({"check": "Every GCC country has a growth path",
                   "ok": all(len(v) == n for v in gg_new.values())})
    checks.append({"check": "GCC nominal GDP is positive (weights add up)",
                   "ok": all(len(v) == n and all(x > 0 for x in v) for v in gp_new.values())})

    # --- history files: the confirmed years the store has and the file lacks
    # Only a series that marks its own last actual year (the IMF's do, a
    # Planning Council table imported as actuals does) can add history: a
    # projection republished elsewhere (FRED) would otherwise pass as fact.
    hist_add = pd.DataFrame(columns=["year", "value"])
    ser = store.series(vintage)
    marked = set(ser.loc[ser["last_actual"].astype(str).str.strip() != "", "series_id"]) \
        | set(ser.loc[ser["source"].isin(["NPC"]), "series_id"])
    if len(nonoil) and hist_last is not None and nonoil.attrs.get("series_id") in marked:
        conf = nonoil[nonoil["actual"] & (nonoil["year"] > hist_last) &
                      (nonoil["year"] < years[0])]
        hist_add = conf[["year", "value"]].reset_index(drop=True)
    gcc_hist = {}
    for fname, prefix in (("gcc_real_gdp_growth.csv", "GDP"),
                          ("gcc_gdp_current_prices.csv", "NGDPD")):
        cur = _read_static(static_dir / fname)
        rows = []
        for name in GCC:
            a = store.concept_annual(f"{prefix}_{_iso(name)}", vintage, project_root)
            if not len(a) or str(a.attrs.get("series_id", "")).startswith("STATIC"):
                continue
            a = a[a["actual"] & (a["year"] < years[0])]
            rows += [{"country": name, "year": int(y), "value": float(v)}
                     for y, v in zip(a["year"], a["value"])]
        new = pd.DataFrame(rows, columns=["country", "year", "value"])
        changed = 0
        if len(new) and len(cur):
            j = new.merge(cur, on=["country", "year"], how="left", suffixes=("", "_old"))
            changed = int(((j["value_old"].isna()) |
                           (np.abs(j["value"] - j["value_old"]) > 1e-9)).sum())
        gcc_hist[fname] = {"rows": new, "changed": changed if len(cur) else len(new)}

    # --- the change list
    def rows_for(item, old, new):
        out = []
        for i, y in enumerate(years):
            o = old[i] if i < len(old) else np.nan
            v = new[i] if i < len(new) else np.nan
            out.append({"item": item, "year": y, "current": o, "proposed": v,
                        "change": (v - o) if np.isfinite(o) and np.isfinite(v) else np.nan})
        return out

    changes = []
    old_no = [float((fc_old.get(k) or fc_old.get(str(k)) or [np.nan] * col)[col - 1])
              for k in fc_keys[:n]] if fc_old else [np.nan] * n
    changes += rows_for("Non-oil GDP growth, Qatar", old_no,
                        [fc_new[k][col - 1] for k in sorted(fc_new)])
    for name in GCC:
        changes += rows_for(f"Real GDP growth, {name}", gg_old.get(name) or [], gg_new[name])
    for name in GCC:
        changes += rows_for(f"Nominal GDP (USD bn), {name}", gp_old.get(name) or [], gp_new[name])
    ch = pd.DataFrame(changes)

    release = store.meta(vintage).get("created", "")
    return {
        "ok": True, "vintage": vintage, "release": release,
        "years": years, "rule": rule["rule"], "first_year": years[0],
        "last_confirmed": rule["last_confirmed"],
        "nonoil": {"series_id": sid_no, "values": [fc_new[k][col - 1] for k in sorted(fc_new)],
                   "column": col},
        "mev_forecasts": fc_new,
        "gcc_growth": gg_new, "gcc_prices": gp_new, "gcc_sources": gsrc,
        "history_nonoil_add": hist_add,
        "history_gcc": gcc_hist,
        "changes": ch, "changed": int((ch["change"].abs() > 1e-9).sum()) if len(ch) else 0,
        "warnings": warnings, "checks": pd.DataFrame(checks),
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


def render_model_inputs(p: dict) -> str:
    """The new model_inputs.yml text: the forecast blocks replaced, a
    ``macro_source`` block naming where they came from, every other line
    and comment kept."""
    t = p["model_inputs_text"]
    t = replace_top_level_block(t, "external_gcc_country_growth",
                                _dump("external_gcc_country_growth", p["gcc_growth"]))
    t = replace_top_level_block(t, "external_gcc_country_prices",
                                _dump("external_gcc_country_prices", p["gcc_prices"]))
    t = replace_top_level_block(t, "mev_forecasts",
                                _dump("mev_forecasts", {"forecasts": p["mev_forecasts"]}))
    src = {"vintage": p["vintage"], "store_created": p["release"],
           "first_forecast_year": int(p["first_year"]),
           "years": [int(y) for y in p["years"]],
           "non_oil_gdp": p["nonoil"]["series_id"],
           "gcc": sorted(set(p["gcc_sources"].values())),
           "filled": _dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    t = replace_top_level_block(t, "macro_source", _dump("macro_source", src))
    return t


def apply(p: dict, snapshot_label: str, snapshots_root=None, edited_by=None) -> dict:
    """Write a proposal into a DRAFT snapshot: model_inputs.yml, and the
    history files where the store has confirmed years they lack."""
    from ..snapshots import (read_static_csv_with_header, save_snapshot_csv,
                             save_snapshot_yaml, snapshot_dir)
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
    add = p["history_nonoil_add"]
    if len(add):
        cur = read_static_csv_with_header(d / "static" / "non_oil_gdp_history.csv")["data"]
        new = (pd.concat([cur, add], ignore_index=True)
               .drop_duplicates("year", keep="last").sort_values("year"))
        r = save_snapshot_csv(snapshot_label, "static/non_oil_gdp_history.csv", new,
                              snapshots_root=snapshots_root, edited_by=edited_by)
        out.append(r.get("message", ""))
    for fname, h in (p.get("history_gcc") or {}).items():
        if not h["changed"] or not len(h["rows"]):
            continue
        cur = read_static_csv_with_header(d / "static" / fname)["data"]
        new = (pd.concat([cur, h["rows"]], ignore_index=True)
               .drop_duplicates(["country", "year"], keep="last")
               .sort_values(["country", "year"]))
        r = save_snapshot_csv(snapshot_label, f"static/{fname}", new,
                              snapshots_root=snapshots_root, edited_by=edited_by)
        out.append(r.get("message", ""))
    return {"ok": True, "message": "; ".join(m for m in out if m)}
