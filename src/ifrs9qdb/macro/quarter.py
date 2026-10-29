"""A config's macro inputs, filled from the store when asked.

``propose`` reads a config (a snapshot's, or the live one), the store and the
reporting date, and returns what the config would hold with the chosen
rating models' variables taken from the store -- their forecasts and their
history -- and every value that moves. ``apply`` writes it into a DRAFT
config snapshot, where it is tested and approved like any other edit.
Nothing here runs a calculation or touches a run.

Nothing is named here. A rating model (internal or external, by its
``rating_type`` in model.yml) brings its ``mev_components``; each variable's
own entry in model.yml says where its forecast lives in model_inputs.yml
(``forecast_source``: a column of a yearly table, or one path per group
weighted by another variable) and which file holds its history
(``history_file``; a weight variable's file is the one whose header names
it). The catalogue (macro/data/concepts.csv, ``model_variable``) links each
variable -- or each group of it, ``VARIABLE:group`` -- to a series in the
store. A model change is a change to model.yml and the catalogue, not here.

What is not chosen stays as it is in the file, as does a variable the store
does not hold. A chosen model's scenario weights move to the automatic mode
(worked out at run time from the history and forecast) unless they are kept.

The five-year rule. The forecast starts at the first year with no confirmed
annual figure and runs ``horizons.n_forecasts`` years: in 2026Q3, 2026 is not
confirmed, so the forecast is 2026-2030. "Confirmed" is the latest of the
sources' own markers (the WEO's last actual year) and the last year in the
history files. It is never after the reporting year.

History. A history is only taken from a source that keeps a record: one that
marks its own last actual year (the WEO), the IMF's other datasets (the
regional outlook gives no marker, so its years before the forecast are its
record) or a Planning and Statistics Authority table. A projection
republished elsewhere (FRED) never enters a history. A file keeps its first
year.

Units. The store's values are in the units the config takes (growth in
percentage points, nominal GDP in USD bn); a forecast that looks like
fractions against a history in points is flagged. Every value is written as
the store holds it: nothing is rounded.
"""
from __future__ import annotations

import copy
import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..etl.macro import EXTERNAL_AUTO, INTERNAL_AUTO
from .catalogue import concepts as _concepts
from .store import MacroStore

__all__ = ["propose", "apply", "render_model_inputs", "replace_top_level_block",
           "first_forecast_year", "MODELS", "model_variables"]

_RECORD_SOURCES = {"IMF", "DM", "NPC"}
_TOL = 1e-9                 # a change smaller than this is the same value

# The two rating models, by rating_type, and the weights block each has in
# model_inputs.yml with its automatic mode.
MODELS = {"internal": "Internal rated", "external": "External rated"}
_RATING = {"internal": "internal", "external": "external"}
_WEIGHTS = {"internal": ("internal_scenario_weights", INTERNAL_AUTO),
            "external": ("external_scenario_weights", EXTERNAL_AUTO)}
# the catalogue's links written before they named the model variable
_OLD_LINK = {"group": "GCC", "weight": "GCCW"}


# ---------------------------------------------------------------- reading --
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


def _header_of(path: Path) -> dict:
    """A static file's ``# key: value`` header lines."""
    out = {}
    for ln in Path(path).read_text(encoding="utf-8-sig").splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            break
        m = re.match(r"#\s*([A-Za-z_]+)\s*:\s*(.*)$", s)
        if m:
            out[m.group(1).lower()] = m.group(2).strip()
    return out


def _file_for(static_dir: Path, variable: str, given=None) -> str | None:
    """The history file of a variable: the one model.yml gives, else the
    static file whose header names the variable."""
    if given:
        return Path(str(given)).name
    for p in sorted(Path(static_dir).glob("*.csv")):
        if _header_of(p).get("variable_id", "").split(",")[0].strip() == variable:
            return p.name
    return None


def _get(d: dict, path: str):
    for k in str(path).split("."):
        d = (d or {}).get(k) if isinstance(d, dict) else None
    return d


def _set(d: dict, path: str, value) -> None:
    keys = str(path).split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def model_variables(model: dict, key: str) -> list:
    """The variables of a rating model ("internal" or "external"), in the
    order model.yml lists them."""
    out = []
    for m in (model.get("models") or {}).values():
        if str((m or {}).get("rating_type", "")).lower().startswith(_RATING[key]):
            for c in (m.get("mev_components") or []):
                v = c.get("variable")
                if v and v not in out:
                    out.append(v)
    return out


def _links(cat: pd.DataFrame, variable: str, old: str | None = None) -> dict:
    """The catalogue's concepts for a variable: {None: concept} for a single
    series, {group: concept} for one per group."""
    out = {}
    for _, r in cat.iterrows():
        base, _, grp = str(r["model_variable"] or "").partition(":")
        if base == variable or (old and base == old and grp):
            out[grp or None] = r["concept"]
    return out


def _plan(model: dict, cat: pd.DataFrame, static_dir: Path, key: str) -> list:
    """What a rating model's variables take from the store, one entry per
    variable (a weighted variable brings its weight variable too)."""
    vcfg = model.get("variables") or {}
    out = []
    for v in model_variables(model, key):
        c = vcfg.get(v) or {}
        fs = c.get("forecast_source") or {}
        if fs.get("aggregation"):
            grp = c.get("history_group_column") or "country"
            out.append({"model": key, "variable": v, "grouped": True, "path": fs.get("path"),
                        "group_col": grp, "units": c.get("units"),
                        "history": _file_for(static_dir, v, c.get("history_file")),
                        "concepts": _links(cat, v, _OLD_LINK["group"])})
            w = (c.get("history_aggregation") or {}).get("weight_variable")
            if w and fs.get("weight_path"):
                out.append({"model": key, "variable": w, "grouped": True,
                            "path": fs.get("weight_path"), "group_col": grp, "units": "",
                            "history": _file_for(static_dir, w),
                            "concepts": _links(cat, w, _OLD_LINK["weight"]), "weight": True})
        elif fs.get("column_index"):
            out.append({"model": key, "variable": v, "grouped": False, "path": fs.get("path"),
                        "column": int(fs["column_index"]), "units": c.get("units"),
                        "label": c.get("display_name") or v,
                        "history": _file_for(static_dir, v, c.get("history_file")),
                        "concepts": _links(cat, v)})
    return out


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
    grp = keys[0] if len(keys) > 1 else None

    def tidy(d):
        d = d[cols].copy() if len(d) and all(c in d for c in cols) else \
            pd.DataFrame({c: [] for c in cols})
        d = d.dropna(subset=["year", "value"])
        d["year"] = d["year"].astype(int)
        d["value"] = d["value"].astype(float)
        if grp:
            d[grp] = d[grp].astype(str)
        return d

    cur, rec = tidy(cur), tidy(rec)
    if len(cur) and len(rec):
        if grp is None:
            rec = rec[rec["year"] >= cur["year"].min()]
        else:
            first = cur.groupby(grp)["year"].min()
            rec = rec[rec["year"] >= rec[grp].map(first).fillna(-1)]
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
            "changed": int(new_row.sum() + moved.sum())}


# ---------------------------------------------------------------- propose --
def propose(store: MacroStore, config_dir, static_dir, reporting_date: str,
            vintage: str | None = None, project_root=None,
            first_year: int | None = None, n_years: int | None = None,
            models=None, keep_weights=None) -> dict:
    """What the config would hold with ``models`` (keys of ``MODELS``; both
    when None) filled from the store, the weights of the models in
    ``keep_weights`` kept as they are and the others set to automatic. See
    the module notes."""
    config_dir, static_dir = Path(config_dir), Path(static_dir)
    vintage = vintage or store.latest()
    if not vintage:
        return {"ok": False, "reason": "The macro store is empty: refresh it first."}
    mi_text = (config_dir / "model_inputs.yml").read_text(encoding="utf-8")
    mi = yaml.safe_load(mi_text) or {}
    model = yaml.safe_load((config_dir / "model.yml").read_text(encoding="utf-8")) or {}
    n = int(n_years or ((model.get("horizons") or {}).get("n_forecasts") or 5))
    chosen = [k for k in MODELS if k in set(MODELS if models is None else models)]
    keep = set(keep_weights or [])
    cat = _concepts(project_root)
    records = _records(store, vintage)
    notes, checks = [], []

    plan = [e for k in chosen for e in _plan(model, cat, static_dir, k)]
    series = {}

    def annual(cid):
        if cid not in series:
            series[cid] = store.concept_annual(cid, vintage, project_root)
        return series[cid]

    # --- the five-year rule: the latest confirmed year anywhere in the plan
    marks = []
    for e in plan:
        if e["history"] and not e["grouped"]:
            h = _read_values(static_dir / e["history"])
            if len(h) and "year" in h:
                marks.append(int(pd.to_numeric(h["year"], errors="coerce").max()))
        for cid in e["concepts"].values():
            a = annual(cid)
            if len(a) and not a["actual"].all() and a["actual"].any():
                marks.append(int(a.loc[a["actual"], "year"].max()))
    rule = first_forecast_year(reporting_date, max(marks) if marks else None)
    if first_year:
        rule["first"] = int(first_year)
        rule["rule"] = (f"The forecast starts in {first_year}, as set, and runs to "
                        f"{int(first_year) + n - 1}.")
    years = list(range(rule["first"], rule["first"] + n))
    label = lambda cid: str(cat.loc[cat["concept"] == cid, "label"].iloc[0])  # noqa: E731

    new_mi = copy.deepcopy(mi)
    touched, fc_rows, hist_rows, files, used, kept = [], [], [], {}, set(), []

    def fc(model_key, name, old, new):
        for i, y in enumerate(years):
            o = float(old[i]) if i < len(old) and old[i] is not None else np.nan
            v = float(new[i]) if i < len(new) else np.nan
            fc_rows.append({"model": MODELS[model_key], "variable": name, "year": y,
                            "current": o, "proposed": v, "change": _diff(v, o)})

    for e in plan:
        mk = e["model"]
        # --- the forecast
        if not e["grouped"]:
            cid = e["concepts"].get(None)
            table = _get(mi, e["path"]) or {}
            keys = sorted(table, key=lambda k: int(k))[:n] if table else list(range(1, n + 1))
            col = e["column"]
            old = []
            for k in keys:
                row = table.get(k) or table.get(str(k)) or []
                old.append(float(row[col - 1]) if len(row) >= col else np.nan)
            if not cid:
                kept.append(e["label"])
                continue
            a = annual(cid)
            m = dict(zip(a["year"], a["value"]))
            vals = [m.get(y, np.nan) for y in years]
            if not any(np.isfinite(v) for v in vals):
                kept.append(e["label"])
            else:
                if not all(np.isfinite(v) for v in vals):
                    last = None
                    for i, v in enumerate(vals):
                        last = v if np.isfinite(v) else last
                        vals[i] = last if last is not None else v
                    notes.append(f"{e['label']}: the store stops before {years[-1]}; its "
                                 "last year is repeated.")
                new_table = copy.deepcopy(_get(new_mi, e["path"]) or {})
                for i, k in enumerate(keys if len(keys) == n else range(1, n + 1)):
                    row = list(new_table.get(k) or new_table.get(str(k)) or [])
                    row += [0.0] * (col - len(row))
                    row[col - 1] = float(vals[i])
                    new_table.pop(str(k), None)
                    new_table[int(k)] = row
                _set(new_mi, e["path"], new_table)
                touched.append(str(e["path"]).split(".")[0])
                used.add(a.attrs.get("series_id", ""))
                fc(mk, e["label"], old, vals)
        else:
            cur = _get(mi, e["path"]) or {}
            new_paths = copy.deepcopy(cur)
            for grp, cid in e["concepts"].items():
                if grp is None:
                    continue
                a = annual(cid)
                m = dict(zip(a["year"], a["value"]))
                vals = [m.get(y, np.nan) for y in years]
                if not len(a) or not all(np.isfinite(v) for v in vals):
                    kept.append(label(cid))
                    continue
                new_paths[grp] = [float(v) for v in vals]
                used.add(a.attrs.get("series_id", ""))
                fc(mk, label(cid), cur.get(grp) or [], vals)
            if new_paths != cur:
                _set(new_mi, e["path"], new_paths)
                touched.append(str(e["path"]).split(".")[0])
            if e.get("weight"):
                checks.append({"check": f"{MODELS[mk]}: every weight is positive",
                               "ok": all(len(v) == n and all(x > 0 for x in v)
                                         for v in new_paths.values())})

        # --- the history
        if not e["history"]:
            continue
        cur = _read_values(static_dir / e["history"])
        rows = []
        for grp, cid in e["concepts"].items():
            a = annual(cid)
            if not len(a) or a.attrs.get("series_id") not in records \
                    or str(a.attrs.get("series_id", "")).startswith("STATIC"):
                continue
            a = a[a["actual"] & (a["year"] < years[0])]
            rows += [{**({e.get("group_col"): grp} if grp else {}), "year": int(y),
                      "value": float(v)} for y, v in zip(a["year"], a["value"])]
        if not rows:
            continue
        keys = [e["group_col"], "year"] if e["grouped"] else ["year"]
        up = _history_update(cur, pd.DataFrame(rows), keys)
        if up["changed"]:
            srcs = sorted({annual(c).attrs.get("series_id", "") for c in e["concepts"].values()
                           if annual(c).attrs.get("series_id") in records})
            files[f"static/{e['history']}"] = {"rows": up["rows"], "changed": up["changed"],
                                               "source": ", ".join(srcs)}
            for r in up["changes"].to_dict(orient="records"):
                name = label(e["concepts"][r[e["group_col"]]]) if e["grouped"] else e["label"]
                hist_rows.append({"model": MODELS[mk], "variable": name, "year": int(r["year"]),
                                  "current": r["current"], "proposed": r["proposed"],
                                  "change": r["change"]})

    # --- the scenario weights: automatic unless kept
    weights = []
    for k in chosen:
        block, auto = _WEIGHTS[k]
        cur_mode = str((mi.get(block) or {}).get("mode", "explicit"))
        if k not in keep and cur_mode not in auto:
            weights.append({"model": MODELS[k], "block": block, "current": cur_mode,
                            "proposed": auto[0]})

    # --- the fractions slip: a forecast in fractions against a history in points
    for e in plan:
        if e["grouped"] or e.get("units") != "percentage_points" or not e["history"]:
            continue
        h = _read_values(static_dir / e["history"])
        hv = pd.to_numeric(h.get("value", pd.Series(dtype=float)), errors="coerce").dropna()
        t = _get(new_mi, e["path"]) or {}
        fv = np.array([float((t.get(k) or [np.nan] * e["column"])[e["column"] - 1])
                       for k in t], dtype=float)
        fv = fv[np.isfinite(fv)]
        if len(hv) > 1 and hv.abs().mean() > 1 and len(fv) and np.abs(fv).max() < 0.2 \
                and (fv != 0).any():
            notes.append(f"{e['label']}: the forecast looks like fractions (0.029 for 2.9%) "
                         "against a history in points. The PD model reads 0.029 as 0.029% "
                         "growth.")

    cols = ["model", "variable", "year", "current", "proposed", "change"]
    fcd = pd.DataFrame(fc_rows, columns=cols)
    hd = pd.DataFrame(hist_rows, columns=cols)

    def n_moved(d):
        if not len(d):
            return 0
        a, b = d["current"].astype(float), d["proposed"].astype(float)
        return int((~(np.isclose(a, b, rtol=_TOL, atol=_TOL) | (a.isna() & b.isna()))).sum())

    release = store.meta(vintage).get("created", "")
    return {
        "ok": True, "vintage": vintage, "release": release, "models": chosen,
        "keep_weights": sorted(keep & set(chosen)),
        "years": years, "rule": rule["rule"], "first_year": years[0],
        "last_confirmed": rule["last_confirmed"],
        "forecast": fcd, "history": hd, "weights": weights,
        "changed": n_moved(fcd) + n_moved(hd) + len(weights),
        "warnings": notes, "checks": pd.DataFrame(checks, columns=["check", "ok"]),
        "kept": kept,
        "series": sorted(s for s in used if s),
        "model_inputs_text": mi_text,
        "_model_inputs": new_mi, "_touched": list(dict.fromkeys(touched)), "_files": files,
    }


# ----------------------------------------------------------------- writing --
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


def _set_mode(text: str, block: str, mode: str) -> str:
    """``block``'s ``mode:`` line set to ``mode``, every other line of the
    block -- the explicit weights, the comments -- kept."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(block)}\s*:", ln)),
                 None)
    if start is None:
        return "\n".join(lines + [f"{block}:", f"  mode: {mode}"]) + "\n"
    i = start + 1
    while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
        if re.match(r"^\s+mode\s*:", lines[i]):
            ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
            lines[i] = f"{ind}mode: {mode}"
            return "\n".join(lines) + "\n"
        i += 1
    lines.insert(start + 1, f"  mode: {mode}")
    return "\n".join(lines) + "\n"


def _dump(key, value) -> str:
    return yaml.safe_dump({key: value}, sort_keys=False, default_flow_style=False,
                          allow_unicode=True)


def render_model_inputs(p: dict) -> str:
    """The new model_inputs.yml text: the blocks the chosen models fill and
    the weights' mode replaced, a ``macro_source`` block naming where they
    came from, every other line and comment kept."""
    t = p["model_inputs_text"]
    for key in p.get("_touched") or []:
        t = replace_top_level_block(t, key, _dump(key, p["_model_inputs"][key]))
    for w in p.get("weights") or []:
        t = _set_mode(t, w["block"], w["proposed"])
    src = {"vintage": p["vintage"], "store_created": p["release"],
           "models": [MODELS[k] for k in p.get("models") or []],
           "first_forecast_year": int(p["first_year"]),
           "years": [int(y) for y in p["years"]],
           "series": list(p.get("series") or []),
           "date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    src = {k: v for k, v in src.items() if v not in ("", [], None)}
    return replace_top_level_block(t, "macro_source", _dump("macro_source", src))


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
    """Write a proposal into a DRAFT snapshot: model_inputs.yml when a
    forecast or a weights mode moves, and the history files that move."""
    from ..snapshots import read_static_csv_with_header, save_snapshot_csv, save_snapshot_yaml
    from ..snapshots import snapshot_dir
    if not p.get("ok"):
        return {"ok": False, "message": p.get("reason", "Nothing to apply.")}
    out = []
    if p.get("_touched") or p.get("weights"):
        r = save_snapshot_yaml(snapshot_label, "config/model_inputs.yml",
                               render_model_inputs(p), snapshots_root=snapshots_root,
                               edited_by=edited_by)
        if not r.get("ok"):
            return r
        out.append(r["message"])
    d = snapshot_dir(snapshot_label, snapshots_root)
    for rel, h in (p.get("_files") or {}).items():
        head = read_static_csv_with_header(d / rel)["comment_header"]
        r = save_snapshot_csv(snapshot_label, rel, h["rows"], snapshots_root=snapshots_root,
                              edited_by=edited_by,
                              comment_header=_header(head, h["source"], p["vintage"]))
        out.append(r.get("message", ""))
    return {"ok": True, "message": "; ".join(m for m in out if m) or "nothing to write"}
