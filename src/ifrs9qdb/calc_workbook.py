"""
One customer's ECL as an Excel calculator: the run's inputs, and every step
from them to the provision as a live formula.

``customer_ecl_workbook(run_dir, customer_id)`` returns the .xlsx. A reviewer
opens it, filters nothing, and can follow each figure of the report back to
the files the run priced:

    Lines       the customer's positions as the run read them (AccountMaster,
                CustomerStagingFlag), every value an input
    Overrides   the rating, stage and restructuring overrides applied to the
                customer in the run, with who, when and why
    Staging     the staging rules, the manual stage and customer contagion
    LGD         collateral value x allocation x (1 - haircut), then the LGD
    PD internal the internal PiT PD model on its forecasts: scaling factor per
                scenario and year, then per bucket the annual PiT, cumulative
                and marginal PD, weighted over the scenarios
    PD external the same for externally rated positions, from GCC growth and
                the Basel formula, with weights that change by year
    PD curves   the rating -> bucket map and the monthly cumulative PD curves:
                calculated (one per rating scale and bucket) beside the run's
                StPD.csv, the difference, and which of the two the ECL uses
    EAD         months to maturity, the amortisation shape and the horizon
    ECL         month by month: EAD x LGD x marginal PD / (1 + EIR)^(t/12)
    Summary     the provision per position and in total, against the report

THE BOUNDARY
    The workbook starts from the run's LIC input files (the Output folder),
    its frozen config and its override log -- the same point the report is
    priced from. How the ETL built those files from the raw extracts is the
    ETL's own record (validation and readiness reports), not repeated here.
    The PD curves are calculated from the run's model, forecasts and scenario
    weights (the weights as the run resolved them), and compared with the
    run's StPD.csv; when the two differ the ECL uses the run's. Two inputs are
    taken as the run has them: a repayment schedule's monthly EAD
    (LifeTimeParameterOther) and the overlays.

ONE ANSWER
    The figures come from :func:`etl.report.compute_ecl`'s own inputs
    (:func:`ecl_frame`, the run's :class:`EclContext`) and the formulas
    follow it rule for rule, so the workbook's total equals the report's to
    the cent. The Summary says so per position; a difference is a finding.

"""
from __future__ import annotations

import io
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .ids import as_id

__all__ = ["customer_ecl_facts", "write_customer_workbook",
           "customer_ecl_workbook", "run_customers"]

MAX_MONTH = 600
# the repayment shapes, as a reader would name them
SHAPE_LABEL = {"bullet": "Bullet", "linear": "Equal instalments",
               "annuity": "Annuity"}
# column headers as the workbook shows them (internal name -> shown)
SHOWN = {
    "Manual stage (its digits)": "Manual stage",
    "DPD rule": "DPD over threshold (up to 90)",
    "Any local flag (or watchlist)": "Watchlist or local flag",
    "Stage 3 rule": "DPD over 90 or default",
    "Rule stage": "Stage by the rules",
    "After manual": "Stage after manual",
    "Customer's worst in book": "Customer's worst stage",
    "Final stage": "Stage",
    "Why": "Reason",
    "Source": "EAD source",
    "Months to maturity (N)": "Months to maturity",
    "Shape": "Repayment type",
    "f": "Months between payments",
    "d": "Grace months before repayments",
    "n payments": "Number of payments",
    "First payment p1": "First payment in month",
    "Monthly rate r": "Monthly interest rate",
    "Path length": "Months of exposure",
    "Horizon (months)": "Months counted in ECL",
    "Horizon": "Months counted",
    "IsLocal1": "IsLocal1 (restructured)",
}


def _shown(cols):
    return [SHOWN.get(c, c) for c in cols]          # EclConfig.max_month: the PD curves run to month 600
S1_MONTHS = 12           # Stage 1 looks twelve months ahead
MIN_MONTHS = 3           # months for a facility at or past maturity


# ======================================================================
# Facts: everything the workbook shows, gathered with the engine's own
# functions so the numbers are the ones the report priced.
# ======================================================================
def _out_dir(run_dir: Path) -> Path:
    return run_dir / "Output" if (run_dir / "Output").is_dir() else run_dir


def _run_root(run_dir) -> Path:
    p = Path(run_dir)
    return p.parent if p.name == "Output" and not (p / "Output").is_dir() else p


def _str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "<na>", "nat") else s


def _pay_type(v) -> str:
    """resolve_ead_shape()'s reading of a payment type."""
    s = _str(v)
    return s[:-2] if s.endswith(".0") else s


def _manual_stage(v):
    """classify_stage_report()'s reading of AccountMaster.Stage: its digits,
    when they are 1, 2 or 3."""
    s = re.sub(r"[^0-9]", "", _str(v))
    try:
        n = int(s)
    except ValueError:
        return None
    return n if n in (1, 2, 3) else None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _date(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    t = pd.Timestamp(v)
    return t.to_pydatetime().replace(tzinfo=None)


def _reference_names(out_dir: Path) -> dict:
    """Customer id -> name from the run's contract reference (the LIC customer
    master leaves the name blank)."""
    p = Path(out_dir).parent / "reference" / "contracts.csv"
    if not p.is_file():
        return {}
    try:
        r = pd.read_csv(p, dtype=str, keep_default_na=False)
    except Exception:
        return {}
    return {a: b for a, b in zip(as_id(r["CustomerId"]), r["CustomerName"]) if b}


def run_customers(run_dir) -> pd.DataFrame:
    """Every customer in a run's books: id, name, positions, book(s) -- what
    the app's customer picker offers."""
    from .etl.report import read_output_csv, _chr, _col
    out_dir = _out_dir(_run_root(run_dir))
    ref = _reference_names(out_dir)
    rows = []
    for suffix, book in (("_1", "Lending"), ("_2", "Investments")):
        am = read_output_csv(out_dir, f"AccountMaster{suffix}.csv")
        if am is None or len(am) == 0:
            continue
        cm = read_output_csv(out_dir, f"CustomerMaster{suffix}.csv")
        cust = as_id(_chr(_col(am, "CustomerId")))
        names = {}
        if cm is not None and _col(cm, "CustomerName") is not None:
            k = as_id(_chr(_col(cm, "CustomerId")))
            for a, b in zip(k, _chr(_col(cm, "CustomerName"))):
                names.setdefault(a, _str(b))
        bal = pd.to_numeric(_col(am, "OnBalance"), errors="coerce")
        d = pd.DataFrame({"customer_id": cust, "on_balance": bal})
        d = d[d["customer_id"].notna() & (d["customer_id"] != "")]
        g = d.groupby("customer_id").agg(positions=("on_balance", "size"),
                                         on_balance=("on_balance", "sum"))
        g["book"] = book
        g["customer_name"] = [names.get(c) or ref.get(c, "") for c in g.index]
        rows.append(g.reset_index())
    if not rows:
        return pd.DataFrame(columns=["customer_id", "customer_name", "book",
                                     "positions", "on_balance"])
    d = pd.concat(rows, ignore_index=True)
    out = d.groupby("customer_id", as_index=False).agg(
        customer_name=("customer_name", lambda s: next((x for x in s if x), "")),
        book=("book", lambda s: " + ".join(dict.fromkeys(s))),
        positions=("positions", "sum"), on_balance=("on_balance", "sum"))
    return out.sort_values("on_balance", ascending=False).reset_index(drop=True)


def _versions(root: Path) -> dict:
    """The config version and calculator version the run recorded: its
    manifest's run metadata, else the saved config it was started from."""
    import json
    import yaml
    meta, snap = {}, {}
    p = root / "reports" / "manifest.json"
    if p.is_file():
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            meta, snap = m.get("run_metadata") or {}, m.get("snapshot") or {}
        except Exception:
            pass
    cfg = meta.get("config_version") or snap.get("label")
    if not cfg:
        cu = root / "config_used" / "config_used.yml"
        try:
            cfg = (yaml.safe_load(cu.read_text(encoding="utf-8")) or {}).get(
                "snapshot_label") if cu.is_file() else None
        except Exception:
            cfg = None
    calc = meta.get("calculator_version")
    if calc and meta.get("calculator_label"):
        calc = f"{calc} ({meta['calculator_label']})"
    return {"config_version": cfg or "not recorded (the live config, not a saved version)",
            "calculator_version": calc or "not recorded"}


def _report_rows(out_dir: Path, name: str) -> pd.DataFrame | None:
    p = out_dir / name
    if not p.is_file():
        return None
    return pd.read_csv(p, low_memory=False, dtype=str, keep_default_na=False)


def _overlay_report(out_dir: Path) -> tuple[pd.DataFrame | None, str | None]:
    """The overlaid report the app's analytics read (the first by name)."""
    ov = sorted(out_dir.glob("FinalEclReport_overlay_*.csv"))
    if not ov:
        return None, None
    return _report_rows(out_dir, ov[0].name), ov[0].name


def _overrides(run_root: Path, customer: str) -> pd.DataFrame:
    cols = ["kind", "value", "prior_value", "reason", "created_by",
            "created_at", "effect"]
    effect = {
        "rating": "Every contract of the customer takes this rating "
                  "(AccountMaster_1.Rating).",
        "stage": "Stage 3 sets IsDefault, Stage 2 sets IsLocal3 "
                 "(CustomerStagingFlag_1).",
        "restructuring": "Restructured sets IsLocal1 (CustomerStagingFlag_1).",
    }
    rows = []
    for kind in ("rating", "stage", "restructuring"):
        p = run_root / "overrides" / f"{kind}_overrides.csv"
        if not p.is_file():
            continue
        try:
            d = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception:
            continue
        if "customer_id" not in d.columns:
            continue
        hit = d[as_id(d["customer_id"]) == customer]
        for _, r in hit.iterrows():
            rows.append({"kind": kind, "value": r.get("value", ""),
                         "prior_value": r.get("prior_value", ""),
                         "reason": r.get("reason", ""),
                         "created_by": r.get("created_by", ""),
                         "created_at": r.get("created_at", ""),
                         "effect": effect[kind]})
    return pd.DataFrame(rows, columns=cols)


def _collateral(out_dir: Path, cids: list[str]) -> dict:
    """The allocation rows of these contracts, and the collateral and type
    rows they point at (first occurrence, as the engine looks them up)."""
    from .etl.report import read_output_csv, _chr, _col, _num as rnum
    alloc = read_output_csv(out_dir, "AccountCollateralAllocation.csv")
    coll = read_output_csv(out_dir, "Collateral.csv")
    ctype = read_output_csv(out_dir, "CollateralType.csv")
    empty = {"alloc": pd.DataFrame(columns=["contract", "collateral", "share"]),
             "coll": pd.DataFrame(columns=["collateral", "type", "value"]),
             "types": pd.DataFrame(columns=["type", "haircut", "name"])}
    if alloc is None or coll is None or ctype is None:
        return empty
    a = pd.DataFrame({"contract": _chr(_col(alloc, "ContractId")),
                      "collateral": _chr(_col(alloc, "CollateralId")),
                      "share": rnum(_col(alloc, "AllocationPercentage"))})
    a = a[a["contract"].isin(set(cids))].reset_index(drop=True)
    c = pd.DataFrame({"collateral": _chr(_col(coll, "CollateralId")),
                      "type": _chr(_col(coll, "CollateralTypeId")),
                      "value": rnum(_col(coll, "CollateralValue"))})
    c = c[c["collateral"].notna()].drop_duplicates("collateral", keep="first")
    c = c[c["collateral"].isin(set(a["collateral"].dropna()))]
    name_col = next((n for n in ("CollateralTypeName", "Name", "Description")
                     if _col(ctype, n) is not None), None)
    t = pd.DataFrame({"type": _chr(_col(ctype, "CollateralTypeId")),
                      "haircut": rnum(_col(ctype, "HaircutGeneral")),
                      "name": (_chr(_col(ctype, name_col)) if name_col
                               else pd.Series([None] * len(ctype)))})
    t = t[t["type"].notna()].drop_duplicates("type", keep="first")
    t = t[t["type"].isin(set(c["type"].dropna()))]
    return {"alloc": a, "coll": c.reset_index(drop=True),
            "types": t.reset_index(drop=True)}


def customer_ecl_facts(run_dir, customer_id) -> dict:
    """Everything the calculator shows for one customer of a run."""
    from . import __version__
    from .etl.report import (build_ecl_context, compute_ecl, ecl_frame,
                             read_output_csv, run_pricing_config, _chr, _col,
                             _portfolio_map, _KEYSEP)
    from .stress import staging_threshold
    from .engine import ead_fallback_rules

    root = _run_root(run_dir)
    out_dir = _out_dir(root)
    cust = as_id(pd.Series([customer_id])).iloc[0]
    static, model_cfg = run_pricing_config(root)
    pmap = _portfolio_map(static)
    thr = float(staging_threshold(out_dir))
    ctx = build_ecl_context(out_dir, model_cfg=model_cfg)
    frame = ecl_frame(out_dir, pmap, thr)
    if frame is None:
        raise FileNotFoundError(f"No AccountMaster output in {out_dir}")
    sel = frame[as_id(frame["customer"]) == cust]
    if len(sel) == 0:
        raise KeyError(f"Customer {customer_id!r} has no position in this run")
    priced = compute_ecl(ctx, sel)

    # the run's report, row for row: the report and the frame keep the
    # AccountMaster order, lending then investments
    rep = _report_rows(out_dir, "FinalEclReport.csv")
    ovr, ovr_name = _overlay_report(out_dir)

    def rep_get(df, i, col):
        if df is None or col not in df.columns or i >= len(df):
            return None
        return df.iloc[i][col]

    # raw columns the frame does not keep
    am = {s: read_output_csv(out_dir, f"AccountMaster{s}.csv") for s in ("_1", "_2")}
    cm = {s: read_output_csv(out_dir, f"CustomerMaster{s}.csv") for s in ("_1", "_2")}
    n1 = 0 if am["_1"] is None else len(am["_1"])

    lines = []
    for k, (i, r) in enumerate(sel.iterrows(), start=1):
        s = r["suffix"]
        j = i if s == "_1" else i - n1
        a = am[s]
        stage_raw = _str(_chr(_col(a, "Stage")).iloc[j]) if _col(a, "Stage") is not None else ""
        if rep is not None and i < len(rep):
            rc = as_id(pd.Series([rep.iloc[i]["Contract Id"]])).iloc[0]
            if rc != r["contract"]:
                rep_i = None
            else:
                rep_i = i
        else:
            rep_i = None
        p = priced.loc[i]
        lines.append({
            "line": k, "book": r["book"], "suffix": s, "cid": _str(r["cid"]),
            "contract": _str(r["contract"]), "customer": cust,
            "account_type": _str(r["account_type"]),
            "portfolio": _str(r["portfolio"]),
            "rating_type": _str(r["rating_type"]), "rating": _str(r["rating"]),
            "d_extract": _date(r["d_extract"]), "d_maturity": _date(r["d_maturity"]),
            "on_balance": _num(r["on_balance"]), "off_balance": _num(r["off_balance"]),
            "eir": _num(r["eir"]), "dpd": _num(r["dpd"]),
            "payment_type": _pay_type(r["payment_type"]),
            "pay_freq": _num(r["pay_freq"]), "deferral": _num(r["deferral"]),
            "nir": _num(r["nir"]), "stage_raw": stage_raw,
            "stage_manual": _manual_stage(stage_raw),
            **{f: _num(r[f]) for f in ("IsWatchlist", "IsDefault", "IsDefaultInGCC",
                                       "IsInsolvency", "IsLocal1", "IsLocal2",
                                       "IsLocal3", "IsLocal4", "IsLocal5",
                                       "IsLocal6")},
            "schedule": ctx.schedules.get(r["cid"]),
            "stage_engine": int(r["stage"]),
            "ecl_engine": _num(p["ecl"]),
            "rep_stage": _num(rep_get(rep, rep_i, "Ifrs Stage")) if rep_i is not None else None,
            # the model figure: the overlay waterfall's first column when the
            # report has one, else the provision column
            "rep_ecl": _num(rep_get(rep, rep_i, "Ecl Model Onbal")
                            if rep is not None and "Ecl Model Onbal" in rep.columns
                            and _str(rep_get(rep, rep_i, "Ecl Model Onbal")) != ""
                            else rep_get(rep, rep_i, "Cla Amount Onbal"))
            if rep_i is not None else None,
            "rep_pdlife": _num(rep_get(rep, rep_i, "PD Lifetime Value")) if rep_i is not None else None,
            "rep_lgd": _num(rep_get(rep, rep_i, "LGD Rate")) if rep_i is not None else None,
            # the overlay: the overlaid report the analytics read when there
            # is one, else the report's own waterfall
            "overlay_id": _str(rep_get(ovr if ovr is not None else rep, rep_i,
                                       "Overlay Id")) if rep_i is not None else "",
            "overlay": _num(rep_get(ovr if ovr is not None else rep, rep_i,
                                    "Overlay Amount")) if rep_i is not None else None,
            "rep_final": _num(rep_get(ovr if ovr is not None else rep, rep_i,
                                      "Cla Amount Onbal")) if rep_i is not None else None,
        })

    # customer name, from the book(s) the customer is in
    name = ""
    for s in ("_1", "_2"):
        c = cm[s]
        if c is None or _col(c, "CustomerName") is None:
            continue
        hit = c[as_id(_chr(_col(c, "CustomerId"))) == cust]
        if len(hit):
            name = _str(_chr(_col(hit, "CustomerName")).iloc[0])
            break
    name = name or _reference_names(out_dir).get(cust, "")

    # ratings -> buckets, as the curve lookup reads Ratings.csv
    rat = read_output_csv(out_dir, "Ratings.csv")
    rmap = []
    if rat is not None:
        rd = pd.DataFrame({"rt": _chr(_col(rat, "RatingType")).astype(str),
                           "rating": _chr(_col(rat, "Rating")).astype(str),
                           "bucket": _chr(_col(rat, "Hierarchy"))})
        want = {("2" if (l["rating_type"].startswith("External")
                         or l["rating_type"] == "2") else "1", l["rating"])
                for l in lines}
        seen = set()
        for row in rd.itertuples():
            key = (row.rt, row.rating)
            if key in want and key not in seen:
                seen.add(key)
                rmap.append({"rt": row.rt, "rating": row.rating,
                             "bucket": _num(row.bucket)})

    # the curve each line prices on: its portfolio and its rating's bucket
    bucket_of = {(m["rt"], m["rating"]): m["bucket"] for m in rmap}
    curves = []
    for l in lines:
        rt = "2" if (l["rating_type"].startswith("External")
                     or l["rating_type"] == "2") else "1"
        b = bucket_of.get((rt, l["rating"]))
        bkey = None if b is None else (str(int(b)) if float(b).is_integer() else str(b))
        l["rt"], l["bucket"] = rt, b
        key = (l["portfolio"], bkey)
        l["curve_key"] = (f"{l['portfolio']} | bucket {bkey}"
                          if bkey is not None else None)
        if bkey is None or any(c["key"] == key for c in curves):
            continue
        stpd = ctx.stpd.get(f"{l['portfolio']}{_KEYSEP}{bkey}")
        curves.append({"key": key, "label": f"{l['portfolio']} | bucket {bkey}",
                       "portfolio": l["portfolio"], "bucket": b, "rt": int(rt),
                       "stpd": None if stpd is None else np.asarray(stpd, dtype=float)})

    rules, default = ead_fallback_rules(model_cfg)
    pdm = _pd_model(root, static, lines, curves)
    return {
        "run_id": root.name, "run_dir": str(root),
        "customer_id": cust, "customer_name": name,
        "engine": f"ifrs9qdb (python) {__version__}",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **_versions(root),
        "report_file": "FinalEclReport.csv" if rep is not None else None,
        "overlay_file": ovr_name,
        "settings": {"threshold": thr, "lgd_base": ctx.lgd_base,
                     "lgd_floor": ctx.lgd_floor, "zero_lgd": ctx.cfg.zero_exposure_lgd,
                     "stage3": ctx.stage3, "cap": bool(ctx.cap),
                     "rules": [(None if pf is None else str(pf), str(pt), str(sh))
                               for pf, pt, sh in rules],
                     "default_shape": str(default)},
        "lines": lines, "overrides": _overrides(root, cust),
        "collateral": _collateral(out_dir, [l["cid"] for l in lines]),
        "ratings_map": rmap, "curves": curves, "pd_model": pdm,
    }


CALC, RUN = "Calculated", "Run's StPD"      # the two sources of the PD curves


def _gcc_years(static, n: int) -> list:
    """The years of the GDP-weighted GCC growth history (gcc_weighted_history
    keeps the values only): from 1982, at least two countries with growth and
    GDP. Blank labels when they do not line up with the ``n`` values."""
    from .etl.macro import GCC_HISTORY_FIRST_YEAR
    try:
        g, p = static.get("gcc_real_gdp_growth"), static.get("gcc_gdp_current_prices")
        lo = lambda d: {c.lower(): c for c in d.columns}
        gl, pl = lo(g), lo(p)
        a = pd.DataFrame({"year": pd.to_numeric(g[gl["year"]], errors="coerce"),
                          "country": g[gl["country"]],
                          "v": pd.to_numeric(g[gl.get("value", gl.get("growth"))],
                                             errors="coerce")}).dropna()
        b = pd.DataFrame({"year": pd.to_numeric(p[pl["year"]], errors="coerce"),
                          "country": p[pl["country"]],
                          "w": pd.to_numeric(p[pl.get("value", pl.get("gdp"))],
                                             errors="coerce")}).dropna()
        m = a[a["year"] >= GCC_HISTORY_FIRST_YEAR].merge(b, on=["year", "country"])
        n_by = m.groupby("year").size()
        years = [int(y) for y in n_by[n_by >= 2].index]
        return years if len(years) == n else [None] * n
    except Exception:
        return [None] * n


def _pd_model(root: Path, static, lines: list, curves: list) -> dict | None:
    """What the PD tabs compute from: the run's PiT PD model, its forecasts
    and scenario weights, the regional growth the external scale follows, and
    the TTC PD of each bucket the customer uses. The curves the same chain
    gives (build_stpd_from_static) are compared with the run's StPD: the
    workbook prices on the calculated curves when they are the run's, else
    on the run's own (a run whose StPD came from elsewhere)."""
    from .etl.macro import (_gcc_inputs, build_stpd_from_static,
                            resolve_scenario_weights)
    from .etl.model_registry import mev_model_weights, resolve_model, run_model_id
    from .etl.pipeline import load_model_config
    from .etl.report import _KEYSEP
    if static is None:
        return None
    cu = root / "config_used" / "config"
    try:
        model_cfg, inputs = load_model_config(cu if cu.is_dir() else None)
        mid = run_model_id(root)
        resolved = resolve_model(model_cfg, mid)
        w_mev = mev_model_weights(resolved, inputs)
        fc = inputs["mev_forecasts"]["forecasts"]
        years = sorted(fc, key=lambda k: int(k))
        hz = model_cfg.get("horizons", {}) or {}
        n_fc = int(hz.get("n_forecasts", model_cfg.get("n_forecasts", len(years))))
        max_mat = int(hz.get("max_maturity", model_cfg.get("max_maturity", 50)))
        scen = static["scenario_severity"]
        names = [str(x) for x in scen["scenario"]]
        w_int, w_ext = resolve_scenario_weights(static, model_cfg, inputs)
        gcc_hist, ext_fc = _gcc_inputs(static, inputs, n_fc)
        calc = build_stpd_from_static(static, model_cfg, inputs, "", model_id=mid)
    except Exception as exc:                      # noqa: BLE001
        return {"ok": False, "source": RUN,
                "reason": f"The PD model could not be rebuilt from the run's config "
                          f"({exc}); the ECL uses the run's StPD."}

    ext_ok = bool(np.asarray(gcc_hist).size) and ext_fc is not None
    pf = static["portfolios"]
    pf_col = "portfolio_code" if "portfolio_code" in pf.columns else pf.columns[0]
    rt_col = next((c for c in pf.columns if "rating" in c.lower()), None)
    from .etl.macro import _matching_type
    buckets, calc_curves = {1: [], 2: []}, {}
    for rt in (1, 2):
        used = sorted({l["bucket"] for l in lines if l.get("rt") == str(rt)
                       and l.get("bucket") is not None}, key=float)
        if not used:
            continue
        ttc = static.ttc_for(rt)
        rmap = static.ratings_for(rt)
        hier = dict(zip(rmap["rating"].astype(str),
                        pd.to_numeric(rmap["hierarchy"], errors="coerce")))
        by_b = {}
        for r in ttc.itertuples():
            b = hier.get(str(r.rating))
            if b is not None and np.isfinite(b) and b not in by_b:
                by_b[float(b)] = (str(r.rating), float(r.ttc_pd))
        pfs = (pf.loc[_matching_type(pf[rt_col], rt), pf_col].tolist()
               if rt_col is not None else pf[pf_col].tolist())
        for b in used:
            rating, ttc_pd = by_b.get(float(b), (None, None))
            bkey = str(int(b)) if float(b).is_integer() else str(b)
            buckets[rt].append({"bucket": b, "key": bkey, "rating": rating, "ttc": ttc_pd})
            sub = calc[(calc["PortfolioCode"].isin(pfs))
                       & (calc["PDBucketDim1"].astype(float) == float(b))]
            if len(sub):
                sub = sub[sub["PortfolioCode"] == sub["PortfolioCode"].iloc[0]]
                calc_curves[(rt, bkey)] = np.concatenate(
                    [[0.0], sub.sort_values("MonthLifetime")["PDLifetime"].to_numpy(float)])

    # the run's curves against the calculated ones
    worst, missing = 0.0, []
    for c in curves:
        bkey = str(int(c["bucket"])) if float(c["bucket"]).is_integer() else str(c["bucket"])
        mine = calc_curves.get((c["rt"], bkey))
        if c["stpd"] is None:
            continue
        if mine is None or (c["rt"] == 2 and not ext_ok):
            missing.append(c["label"])
            continue
        n = min(len(mine), len(c["stpd"]))
        worst = max(worst, float(np.max(np.abs(mine[:n] - c["stpd"][:n]))))
    ok = not missing and worst <= 1e-10
    reason = ("" if ok else
              f"No calculated curve for {', '.join(missing)}: the ECL uses the run's StPD."
              if missing else
              f"The run's StPD differs from the calculated curves (by up to {worst:.3g}): "
              "the ECL uses the run's StPD; switch to Calculated to price on the model.")
    pe = w_ext["per_year"]
    return {
        "ok": True, "source": CALC if ok else RUN, "reason": reason,
        "largest_difference": worst,
        "model_id": resolved["model"]["id"],
        "mevs": [{"name": m.get("name") or m.get("variable_id"),
                  "intercept": float(m["intercept"]), "coefficient": float(m["coefficient"]),
                  "sd": float(m["standard_deviation"]),
                  "unit": float(m["stress_unit_multiplier"]), "weight": float(w)}
                 for m, w in zip(resolved["model"]["mevs"], w_mev)],
        "anchor": float(model_cfg["ttc_anchor_pd"]), "n_fc": n_fc, "max_mat": max_mat,
        "years": [int(y) for y in years[:n_fc]],
        "forecasts": [[float(v) for v in fc[y]] for y in years[:n_fc]],
        "scenarios": list(zip(names, pd.to_numeric(scen["severity_z"]).astype(float))),
        "w_int": [float(w_int[n]) for n in names],
        "int_mode": (inputs.get("internal_scenario_weights") or {}).get("mode", "explicit"),
        "ext_ok": ext_ok,
        "gcc_hist": [float(v) for v in np.asarray(gcc_hist, dtype=float)],
        "gcc_years": _gcc_years(static, int(np.asarray(gcc_hist).size)),
        "ext_fc": [] if ext_fc is None else [float(v) for v in np.asarray(ext_fc, float)],
        "w_ext": [[float(pe.iloc[i][n]) for n in names] for i in range(len(pe))],
        "ext_mode": (inputs.get("external_scenario_weights") or {}).get("mode", "explicit"),
        "buckets": buckets, "calc_curves": calc_curves,
    }


# ======================================================================
# The workbook
# ======================================================================
def _col_letter(c: int) -> str:
    s = ""
    while c:
        c, r = divmod(c - 1, 26)
        s = chr(65 + r) + s
    return s


def _a(r: int, c: int, sheet: str | None = None, fixed: bool = True) -> str:
    """A cell address, absolute by default, on another sheet when named."""
    ref = (f"${_col_letter(c)}${r}" if fixed else f"{_col_letter(c)}{r}")
    return f"'{sheet}'!{ref}" if sheet else ref


def _rng(r1, c1, r2, c2, sheet=None) -> str:
    ref = f"${_col_letter(c1)}${r1}:${_col_letter(c2)}${r2}"
    return f"'{sheet}'!{ref}" if sheet else ref


class _Book:
    """A thin layer over openpyxl: inputs blue, formulas black, one header
    style, number formats by kind."""

    FMT = {"amt": "#,##0.00", "pct": "0.0000%", "pd": "0.000000000",
           "rate": "0.000000", "int": "0", "date": "yyyy-mm-dd", "num": "General",
           "x": "0.0000"}

    def __init__(self):
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        self.wb = Workbook()
        self.wb.remove(self.wb.active)
        self.Font, self.Fill, self.Align = Font, PatternFill, Alignment
        self.f_in = Font(color="0000FF")
        self.f_hdr = Font(bold=True, color="FFFFFF")
        self.fill_hdr = PatternFill("solid", fgColor="305496")
        self.f_title = Font(bold=True, size=14)
        self.f_sub = Font(bold=True, size=11)
        self.f_note = Font(italic=True, color="595959")
        self.fill_key = PatternFill("solid", fgColor="FFF2CC")

    def sheet(self, name):
        return self.wb.create_sheet(name)

    def put(self, ws, r, c, v, kind=None, inp=False, bold=False, key=False):
        cell = ws.cell(r, c)
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        cell.value = v
        if kind:
            cell.number_format = self.FMT[kind]
        if inp and not (isinstance(v, str) and v.startswith("=")):
            cell.font = self.f_in
        if bold:
            cell.font = self.Font(bold=True)
        if key:
            cell.fill = self.fill_key
            cell.font = self.Font(bold=True)
        return cell

    def header(self, ws, r, c, labels):
        for k, lab in enumerate(labels):
            cell = ws.cell(r, c + k, lab)
            cell.font, cell.fill = self.f_hdr, self.fill_hdr
            cell.alignment = self.Align(wrap_text=True, vertical="top")

    def title(self, ws, text, note=None):
        ws.cell(1, 1, text).font = self.f_title
        if note:
            ws.cell(2, 1, note).font = self.f_note

    def note(self, ws, r, c, text):
        ws.cell(r, c, text).font = self.f_note

    def widths(self, ws, w: dict):
        for col, width in w.items():
            ws.column_dimensions[col].width = width


def write_customer_workbook(facts: dict, dest=None):
    """Write the calculator for ``facts`` (:func:`customer_ecl_facts`) to
    ``dest`` (a path or a binary buffer); returns the bytes when ``dest`` is
    None."""
    B = _Book()
    L = facts["lines"]
    st = facts["settings"]

    pdm = facts.get("pd_model") or {}
    scales = {l.get("rt") for l in L}
    ws_sum = B.sheet("Summary")
    ws_ln = B.sheet("Lines")
    ws_ov = B.sheet("Overrides")
    ws_stg = B.sheet("Staging")
    ws_lgd = B.sheet("LGD")
    ws_pi = B.sheet("PD internal") if pdm.get("ok") and "1" in scales else None
    ws_pe = (B.sheet("PD external") if pdm.get("ok") and pdm.get("ext_ok")
             and "2" in scales else None)
    ws_pd = B.sheet("PD curves")
    ws_ead = B.sheet("EAD")
    ws_ecl = B.sheet("ECL")
    ws_set = B.sheet("Settings")
    # what a reviewer does not need to open: the settings, and the overrides
    # tab when the customer has none (Excel: right-click a tab, Unhide)
    ws_set.sheet_state = "hidden"
    if len(facts["overrides"]) == 0:
        ws_ov.sheet_state = "hidden"

    # ---------------------------------------------------------- Settings
    B.title(ws_set, "Settings the run priced with",
            "From the run's frozen config (config_used). Blue cells are inputs; "
            "change one and every formula follows.")
    S = {}
    rows = [
        ("thr", "Stage 2 DPD threshold (days, DPD > threshold)", st["threshold"],
         "int", "staging_thresholds.csv; 60 when the run froze none"),
        ("s3dpd", "Stage 3 DPD (days, DPD > this)", 90, "int",
         "fixed in the staging rule"),
        ("lgd_base", "LGD base", st["lgd_base"], "x", "model.yml ecl.lgd_base"),
        ("lgd_floor", "Unsecured floor", st["lgd_floor"], "x",
         "model.yml ecl.lgd_unsecured_floor: at least this share is unsecured"),
        ("zero_lgd", "LGD at zero or missing exposure", st["zero_lgd"], "x", ""),
        ("stage3", "Stage 3 method", st["stage3"], None,
         "full_outstanding: Stage 3 ECL = on-balance exposure; zero: nil"),
        ("cap", "Cap ECL at on-balance exposure", bool(st["cap"]), None,
         "model.yml ecl.cap_ecl_at_exposure"),
        ("s1m", "Stage 1 horizon (months)", S1_MONTHS, "int", "12-month ECL"),
        ("minm", "Months for a facility at or past maturity", MIN_MONTHS, "int",
         "LIC's floor for a run-off facility"),
        ("maxm", "Last month of the PD curves", MAX_MONTH, "int", ""),
    ]
    B.header(ws_set, 4, 1, ["Setting", "Value", "Note"])
    r = 5
    for key, lab, val, kind, note in rows:
        B.put(ws_set, r, 1, lab)
        B.put(ws_set, r, 2, val, kind, inp=True)
        B.note(ws_set, r, 3, note)
        S[key] = _a(r, 2, "Settings")
        r += 1
    r += 1
    ws_set.cell(r, 1, "EAD fallback: amortisation shape for a position with "
                      "no repayment schedule").font = B.f_sub
    B.note(ws_set, r + 1, 1, "First a rule naming the position's portfolio "
           "and payment type, then a rule with no portfolio, else the default. "
           "A facility with 3 months or less to run is a bullet.")
    B.header(ws_set, r + 2, 1, ["Portfolio", "Payment type", "Repayment type",
                                 "Lookup key"])
    r0 = r + 3
    for k, (pf, pt, sh) in enumerate(st["rules"]):
        rr = r0 + k
        B.put(ws_set, rr, 1, pf or "", inp=True)
        B.put(ws_set, rr, 2, pt, inp=True)
        B.put(ws_set, rr, 3, SHAPE_LABEL.get(sh, SHAPE_LABEL["bullet"]), inp=True)
        B.put(ws_set, rr, 4, f"={_a(rr, 1, fixed=False)}&\"|\"&{_a(rr, 2, fixed=False)}")
    r1 = r0 + max(len(st["rules"]), 1) - 1
    S["rule_key"] = _rng(r0, 4, r1, 4, "Settings")
    S["rule_shape"] = _rng(r0, 3, r1, 3, "Settings")
    B.put(ws_set, r1 + 2, 1, "Default repayment type")
    B.put(ws_set, r1 + 2, 2, SHAPE_LABEL.get(st["default_shape"], SHAPE_LABEL["bullet"]),
          inp=True)
    S["default_shape"] = _a(r1 + 2, 2, "Settings")
    B.widths(ws_set, {"A": 48, "B": 18, "C": 90, "D": 22})

    # ---------------------------------------------------------- Lines
    B.title(ws_ln, "The customer's positions, as the run read them",
            "AccountMaster_1/_2 and CustomerStagingFlag_1/_2 from the run's "
            "Output folder; a flag is the customer's, repeated on each of its "
            "positions in that book. Every value here is an input.")
    ln_cols = ["Line", "Book", "Contract", "Customer", "Account type",
               "Portfolio", "Rating type", "Rating", "Extract date",
               "Maturity date", "On balance", "Off balance", "EIR",
               "Past due days", "Payment type", "Payment frequency (months)",
               "Deferral (months)", "Nominal interest rate",
               "AccountMaster.Stage", "Manual stage (its digits)",
               "IsWatchlist", "IsDefault", "IsDefaultInGCC", "IsInsolvency",
               "IsLocal1", "IsLocal2", "IsLocal3", "IsLocal4", "IsLocal5",
               "IsLocal6", "Repayment schedule", "Rating bucket", "PD curve"]
    C = {n: i + 1 for i, n in enumerate(ln_cols)}
    HR = 4
    B.header(ws_ln, HR, 1, _shown(ln_cols))
    LR = {}                                        # line -> row on Lines
    for k, l in enumerate(L):
        rr = HR + 1 + k
        LR[l["line"]] = rr
        vals = [l["line"], l["book"], l["cid"], l["customer"], l["account_type"],
                l["portfolio"], l["rating_type"], l["rating"], l["d_extract"],
                l["d_maturity"], l["on_balance"], l["off_balance"], l["eir"],
                l["dpd"], l["payment_type"], l["pay_freq"], l["deferral"],
                l["nir"], l["stage_raw"], l["stage_manual"],
                l["IsWatchlist"], l["IsDefault"], l["IsDefaultInGCC"],
                l["IsInsolvency"], l["IsLocal1"], l["IsLocal2"], l["IsLocal3"],
                l["IsLocal4"], l["IsLocal5"], l["IsLocal6"],
                "Yes" if l["schedule"] is not None else "No"]
        kinds = [None, None, None, None, None, None, None, None, "date", "date",
                 "amt", "amt", "rate", "int", None, "int", "int", "rate", None,
                 "int"] + ["int"] * 10 + [None]
        for c, (v, kd) in enumerate(zip(vals, kinds), start=1):
            B.put(ws_ln, rr, c, v, kd, inp=c > 1)
    S["ln_first"], S["ln_last"] = HR + 1, HR + len(L)

    # ---------------------------------------------------------- PD
    wm = {1: _write_pd_internal(B, ws_pi, pdm) if ws_pi is not None else {},
          2: _write_pd_external(B, ws_pe, pdm) if ws_pe is not None else {}}
    pm = _write_pd(B, ws_pd, facts, wm)
    # the rating's bucket and the PD curve it reads, on Lines, by formula: the
    # calculated curve of its scale and bucket, or the run's of its portfolio
    for l in L:
        rr = LR[l["line"]]
        rt = (f"IF(OR(LEFT({_a(rr, C['Rating type'], fixed=False)},8)=\"External\","
              f"{_a(rr, C['Rating type'], fixed=False)}=\"2\"),\"External\",\"Internal\")")
        rg = _a(rr, C["Rating"], fixed=False)
        B.put(ws_ln, rr, C["Rating bucket"],
              f"=IF(COUNTIFS({pm['rm_scale']},{rt},{pm['rm_rating']},{rg})=0,\"\","
              f"SUMIFS({pm['rm_bucket']},{pm['rm_scale']},{rt},{pm['rm_rating']},{rg}))",
              "int")
        bk = _a(rr, C["Rating bucket"], fixed=False)
        B.put(ws_ln, rr, C["PD curve"],
              f"=IF({bk}=\"\",\"\",IF({pm['source']}=\"{CALC}\",{rt},"
              f"{_a(rr, C['Portfolio'], fixed=False)})&\" | bucket \"&{bk})")
    B.widths(ws_ln, {"A": 6, "B": 11, "C": 14, "D": 12, "E": 12, "F": 16,
                     "G": 11, "H": 10, "I": 12, "J": 12, "K": 16, "L": 14,
                     "M": 11, "N": 9})

    def ln(line, name, fixed=True):
        return _a(LR[line], C[name], "Lines", fixed)

    # ---------------------------------------------------------- Overrides
    B.title(ws_ov, "Overrides applied to this customer in the run",
            "From the run's overrides folder (reviewed at the pause step). They "
            "are already in the Lines tab's rating and flags: this is the record "
            "of what changed and why.")
    B.header(ws_ov, 4, 1, ["Kind", "New value", "Model value (before)",
                           "Reason", "By", "When", "Where it shows"])
    ov = facts["overrides"]
    if len(ov) == 0:
        B.note(ws_ov, 5, 1, "No rating, stage or restructuring override for this "
                            "customer in this run.")
    for k, row in enumerate(ov.itertuples(index=False)):
        for c, v in enumerate([row.kind, row.value, row.prior_value, row.reason,
                               row.created_by, row.created_at, row.effect], 1):
            B.put(ws_ov, 5 + k, c, v, inp=True)
    rr = 7 + max(len(ov), 1)
    ws_ov.cell(rr, 1, "Manual stage in AccountMaster.Stage").font = B.f_sub
    B.note(ws_ov, rr + 1, 1, "A stage written into the account master by the "
           "extract. It replaces the rule stage unless the rules say Stage 3; "
           "see the Staging tab.")
    man = [l for l in L if l["stage_manual"] is not None]
    B.header(ws_ov, rr + 2, 1, ["Line", "Contract", "AccountMaster.Stage"])
    if not man:
        B.note(ws_ov, rr + 3, 1, "None on this customer's positions.")
    for k, l in enumerate(man):
        B.put(ws_ov, rr + 3 + k, 1, f"={ln(l['line'], 'Line')}")
        B.put(ws_ov, rr + 3 + k, 2, f"={ln(l['line'], 'Contract')}")
        B.put(ws_ov, rr + 3 + k, 3, f"={ln(l['line'], 'AccountMaster.Stage')}")
    B.widths(ws_ov, {"A": 15, "B": 14, "C": 20, "D": 60, "E": 12, "F": 26,
                     "G": 60})

    # ---------------------------------------------------------- Staging
    B.title(ws_stg, "Stage, rule by rule",
            "classify_stage_report(): DPD over the threshold (and not over 90), "
            "the watchlist or any local flag -> 2; Tasdeer -> 2; DPD over 90 or "
            "the default flag -> 3; a manual stage replaces the rule stage "
            "unless that is 3; then a customer with any Stage 2 or 3 position "
            "in the book lifts its Stage 1 positions to 2 (Tasdeer excepted).")
    sg_cols = ["Line", "Contract", "Book", "Portfolio", "DPD", "DPD rule",
               "Watchlist", "Any local flag (or watchlist)", "Tasdeer",
               "Stage 3 rule", "Rule stage", "Manual stage", "After manual",
               "Customer's worst in book", "Final stage", "Why",
               "Report stage", "Check"]
    G = {n: i + 1 for i, n in enumerate(sg_cols)}
    B.header(ws_stg, 4, 1, _shown(sg_cols))
    GR = {}
    for k, l in enumerate(L):
        GR[l["line"]] = 5 + k
    books = {}
    for l in L:
        books.setdefault(l["suffix"], []).append(GR[l["line"]])
    for l in L:
        rr, x = GR[l["line"]], l["line"]
        g = lambda name: _a(rr, G[name], fixed=False)
        dpd = ln(x, "Past due days")
        B.put(ws_stg, rr, G["Line"], f"={ln(x, 'Line')}")
        B.put(ws_stg, rr, G["Contract"], f"={ln(x, 'Contract')}")
        B.put(ws_stg, rr, G["Book"], f"={ln(x, 'Book')}")
        B.put(ws_stg, rr, G["Portfolio"], f"={ln(x, 'Portfolio')}")
        B.put(ws_stg, rr, G["DPD"], f"=IF(ISNUMBER({dpd}),{dpd},\"\")", "int")
        B.put(ws_stg, rr, G["DPD rule"],
              f"=AND(ISNUMBER({dpd}),N({dpd})>{S['thr']},N({dpd})<={S['s3dpd']})")
        B.put(ws_stg, rr, G["Watchlist"], f"={ln(x, 'IsWatchlist')}=1")
        flags = ",".join(f"{ln(x, f)}=1" for f in
                         ("IsWatchlist", "IsLocal1", "IsLocal2", "IsLocal3",
                          "IsLocal4", "IsLocal5", "IsLocal6"))
        B.put(ws_stg, rr, G["Any local flag (or watchlist)"], f"=OR({flags})")
        B.put(ws_stg, rr, G["Tasdeer"], f"={ln(x, 'Portfolio')}=\"Tasdeer\"")
        B.put(ws_stg, rr, G["Stage 3 rule"],
              f"=OR(AND(ISNUMBER({dpd}),N({dpd})>{S['s3dpd']}),{ln(x, 'IsDefault')}=1)")
        B.put(ws_stg, rr, G["Rule stage"],
              f"=IF({g('Stage 3 rule')},3,IF(OR({g('DPD rule')},{g('Watchlist')},"
              f"{g('Any local flag (or watchlist)')},{g('Tasdeer')}),2,1))", "int")
        man = ln(x, "Manual stage (its digits)")
        B.put(ws_stg, rr, G["Manual stage"],
              f"=IF(ISNUMBER({man}),{man},\"\")", "int")
        B.put(ws_stg, rr, G["After manual"],
              f"=IF({g('Stage 3 rule')},3,IF(ISNUMBER({g('Manual stage')}),"
              f"{g('Manual stage')},{g('Rule stage')}))", "int")
        rows_b = books[l["suffix"]]
        B.put(ws_stg, rr, G["Customer's worst in book"],
              f"=MAX({_rng(min(rows_b), G['After manual'], max(rows_b), G['After manual'])})",
              "int")
        worst = _a(rr, G["Customer's worst in book"], fixed=False)
        B.put(ws_stg, rr, G["Final stage"],
              f"=IF(AND({g('After manual')}=1,{worst}>=2,"
              f"NOT({g('Tasdeer')})),2,{g('After manual')})", "int", key=True)
        why = (f"=IF({g('Stage 3 rule')},IF({ln(x, 'IsDefault')}=1,\"Default flag\","
               f"\"DPD over \"&{S['s3dpd']}),IF(ISNUMBER({g('Manual stage')}),"
               f"\"Manual stage \"&{g('Manual stage')},IF({g('DPD rule')},"
               f"\"DPD over \"&{S['thr']},IF({g('Watchlist')},\"Watchlist\","
               f"IF({g('Any local flag (or watchlist)')},\"Local flag\","
               f"IF({g('Tasdeer')},\"Tasdeer (collective)\","
               f"IF({g('Final stage')}=2,\"Contagion: the customer has a Stage \"&"
               f"{worst}&\" position\","
               f"\"Performing\")))))))")
        B.put(ws_stg, rr, G["Why"], why)
        B.put(ws_stg, rr, G["Report stage"], l["rep_stage"], "int", inp=True)
        B.put(ws_stg, rr, G["Check"],
              f"=IF({g('Report stage')}=\"\",\"\",IF({g('Final stage')}="
              f"{g('Report stage')},\"OK\",\"DIFFERS\"))")
    B.widths(ws_stg, {"A": 6, "B": 14, "C": 11, "D": 16, "E": 7, "F": 9,
                      "G": 10, "H": 12, "I": 9, "J": 10, "K": 9, "L": 9,
                      "M": 9, "N": 11, "O": 9, "P": 42, "Q": 9, "R": 9})

    def stage(line):
        return _a(GR[line], G["Final stage"], "Staging")

    # ---------------------------------------------------------- LGD
    lg = _write_lgd(B, ws_lgd, facts, L, ln, S)

    # ---------------------------------------------------------- EAD
    ead = _write_ead(B, ws_ead, facts, L, ln, stage, S)

    # ---------------------------------------------------------- ECL
    ecl = _write_ecl(B, ws_ecl, facts, L, ln, lg, ead, pm, S)

    # ---------------------------------------------------------- Summary
    _write_summary(B, ws_sum, facts, L, ln, stage, lg, ead, ecl, pm, S, GR, G)

    buf = io.BytesIO()
    B.wb.calculation.fullCalcOnLoad = True
    B.wb.save(buf)
    data = buf.getvalue()
    if dest is None:
        return data
    if hasattr(dest, "write"):
        dest.write(data)
        return None
    Path(dest).write_bytes(data)
    return None


def _trunc3(x: str) -> str:
    """Truncated (not rounded) to 3 significant digits, as PERCENTRANK.EXC
    returns its rank: x is a number in (0, 1)."""
    return f"ROUNDDOWN({x},2-INT(LOG10({x})))"


def _write_scale_blocks(B, ws, r, pdm, rt, sf_cell, weight_of, pit_formula, note):
    """The annual chain for each bucket of one rating scale the customer
    uses: per scenario the PiT PD by year (the forecast years from the
    scaling factor, later years reverting to the TTC PD in log space), the
    cumulative PD by survival and the marginal, then the scenario-weighted
    marginal the monthly curve is built from. Returns ({bucket key: the
    weighted-marginal range}, the next free row)."""
    names = [n for n, _ in pdm["scenarios"]]
    n_s, mm = len(names), pdm["max_mat"]
    n_fy = pdm["n_fc"] if rt == 1 else len(pdm["ext_fc"])
    ws.cell(r, 1, "TTC PD by bucket").font = B.f_sub
    hdr = ["Bucket", "Rating (TTC table)", "TTC PD"] + (["Asset correlation R"] if rt == 2 else [])
    B.header(ws, r + 1, 1, hdr)
    ttc_cell, rr_cell = {}, {}
    for k, bk in enumerate(pdm["buckets"][rt]):
        rr = r + 2 + k
        B.put(ws, rr, 1, bk["bucket"], "int", inp=True)
        B.put(ws, rr, 2, bk["rating"] or "no TTC PD", inp=True)
        B.put(ws, rr, 3, bk["ttc"], "pd", inp=True)
        ttc_cell[bk["key"]] = _a(rr, 3)
        if rt == 2:
            t = _a(rr, 3)
            B.put(ws, rr, 4, f"=0.24-0.12*(1-EXP(-50*{t}))/(1-EXP(-50))", "x")
            rr_cell[bk["key"]] = _a(rr, 4)
    r = r + 3 + len(pdm["buckets"][rt])
    B.note(ws, r, 1, note)
    r += 2
    out = {}
    for bk in pdm["buckets"][rt]:
        if bk["ttc"] is None:
            continue
        T = ttc_cell[bk["key"]]
        ws.cell(r, 1, f"Bucket {bk['key']} ({bk['rating']}), TTC PD "
                      f"{bk['ttc']:.6g}").font = B.f_sub
        hr = r + 1
        cols = ["Year"]
        for nm in names:
            cols += [f"{nm}: PiT PD", f"{nm}: cumulative", f"{nm}: marginal"]
        cols += ["Weighted marginal"]
        B.header(ws, hr, 1, cols)
        first = hr + 1
        for y in range(1, mm + 1):
            rr = first + y - 1
            B.put(ws, rr, 1, y, "int")
            terms = []
            for k, nm in enumerate(names):
                cp, cc, cm = 2 + 3 * k, 3 + 3 * k, 4 + 3 * k
                pit = _a(rr, cp, fixed=False)
                if y <= n_fy:
                    f = pit_formula(T, rr_cell.get(bk["key"]), sf_cell(k, y))
                else:
                    anc = _a(first + n_fy - 1, cp)
                    f = (f"=LN(({y}-{n_fy})/({mm}-{n_fy})*(EXP({T})-EXP({anc}))"
                         f"+EXP({anc}))")
                B.put(ws, rr, cp, f, "pd")
                prev = None if y == 1 else _a(rr - 1, cc, fixed=False)
                B.put(ws, rr, cc, f"=1-(1-{pit})" if prev is None
                      else f"=1-(1-{prev})*(1-{pit})", "pd")
                B.put(ws, rr, cm, f"={_a(rr, cc, fixed=False)}" if prev is None
                      else f"={_a(rr, cc, fixed=False)}-{prev}", "pd")
                terms.append(f"{weight_of(k, y)}*{_a(rr, cm, fixed=False)}")
            B.put(ws, rr, 2 + 3 * n_s, "=" + "+".join(terms), "pd", key=True)
        out[bk["key"]] = _rng(first, 2 + 3 * n_s, first + mm - 1, 2 + 3 * n_s, ws.title)
        r = first + mm + 1
    return out, r


def _write_pd_internal(B, ws, pdm) -> dict:
    """The internal scale: the run's PiT PD model on its forecasts."""
    B.title(ws, f"PD internal: the internal rating model ({pdm['model_id']})",
            "Each scenario moves each variable by its severity x its standard "
            "deviation; the variable's PD is EXP(x)/(EXP(x)+1) with x = intercept + "
            "coefficient x (variable / unit multiplier); its scaling factor is "
            "NORMSINV(PD) - NORMSINV(TTC anchor PD), and the variables combine by "
            "their weights. Blue cells are the run's inputs (model.yml, "
            "model_inputs.yml, the static reference).")
    mev, names = pdm["mevs"], [n for n, _ in pdm["scenarios"]]
    r = 4
    ws.cell(r, 1, "Model").font = B.f_sub
    B.header(ws, r + 1, 1, ["Variable", "Intercept", "Coefficient",
                            "Standard deviation", "Unit multiplier", "Weight"])
    M = []
    for k, m in enumerate(mev):
        rr = r + 2 + k
        B.put(ws, rr, 1, m["name"], inp=True)
        for c, key in enumerate(("intercept", "coefficient", "sd", "unit", "weight"), 2):
            B.put(ws, rr, c, m[key], "num", inp=True)
        M.append({key: _a(rr, c) for c, key in
                  enumerate(("intercept", "coefficient", "sd", "unit", "weight"), 2)})
    r = r + 2 + len(mev) + 1
    B.put(ws, r, 1, "TTC anchor PD")
    B.put(ws, r, 2, pdm["anchor"], "num", inp=True)
    anchor = _a(r, 2)
    B.put(ws, r + 1, 1, "Forecast years modelled")
    B.put(ws, r + 1, 2, pdm["n_fc"], "int", inp=True)
    B.put(ws, r + 2, 1, "Years in the curve")
    B.put(ws, r + 2, 2, pdm["max_mat"], "int", inp=True)
    r += 4
    ws.cell(r, 1, "Forecasts").font = B.f_sub
    B.header(ws, r + 1, 1, ["Year"] + [m["name"] for m in mev])
    F = {}
    for i, (y, row) in enumerate(zip(pdm["years"], pdm["forecasts"])):
        rr = r + 2 + i
        B.put(ws, rr, 1, y, "int", inp=True)
        for k, v in enumerate(row):
            B.put(ws, rr, 2 + k, v, "num", inp=True)
            F[(i + 1, k)] = _a(rr, 2 + k)
    r = r + 2 + len(pdm["years"]) + 1
    ws.cell(r, 1, "Scenarios").font = B.f_sub
    B.header(ws, r + 1, 1, ["Scenario", "Severity z", "Weight"])
    Z, W = [], []
    for k, ((nm, z), w) in enumerate(zip(pdm["scenarios"], pdm["w_int"])):
        rr = r + 2 + k
        B.put(ws, rr, 1, nm, inp=True)
        B.put(ws, rr, 2, z, "num", inp=True)
        B.put(ws, rr, 3, w, "num", inp=True)
        Z.append(_a(rr, 2))
        W.append(_a(rr, 3))
    B.put(ws, r + 2 + len(names), 1, "Total")
    B.put(ws, r + 2 + len(names), 3, f"=SUM({W[0]}:{W[-1]})", "num")
    B.note(ws, r + 3 + len(names), 1,
           "The weights the run used" + (": worked out by the run from where the "
           "forecast sits on the history (mode auto)." if pdm["int_mode"] != "explicit"
           else ", as written in model_inputs.yml (mode explicit)."))
    r = r + 5 + len(names)
    ws.cell(r, 1, "Scaling factor by scenario and year").font = B.f_sub
    cols = ["Scenario", "Year"]
    for m in mev:
        cols += [f"{m['name']}: stressed", f"{m['name']}: PD", f"{m['name']}: SF"]
    cols += ["Scaling factor"]
    B.header(ws, r + 1, 1, cols)
    SF = {}
    rr = r + 2
    for k, nm in enumerate(names):
        for y in range(1, pdm["n_fc"] + 1):
            B.put(ws, rr, 1, nm)
            B.put(ws, rr, 2, y, "int")
            parts = []
            for j, m in enumerate(M):
                cs, cp, cf = 3 + 3 * j, 4 + 3 * j, 5 + 3 * j
                B.put(ws, rr, cs, f"={F[(y, j)]}+{Z[k]}*{m['sd']}*{m['unit']}", "num")
                x = f"({m['intercept']}+{m['coefficient']}*{_a(rr, cs, fixed=False)}/{m['unit']})"
                B.put(ws, rr, cp, f"=EXP({x})/(EXP({x})+1)", "pd")
                B.put(ws, rr, cf, f"=NORMSINV({_a(rr, cp, fixed=False)})-NORMSINV({anchor})",
                      "num")
                parts.append(f"{m['weight']}*{_a(rr, cf, fixed=False)}")
            B.put(ws, rr, 3 + 3 * len(M), "=" + "+".join(parts), "num", key=True)
            SF[(k, y)] = _a(rr, 3 + 3 * len(M))
            rr += 1
    out, _ = _write_scale_blocks(
        B, ws, rr + 1, pdm, 1, lambda k, y: SF[(k, y)], lambda k, y: W[k],
        lambda T, R, sf: f"=IF({T}=0,0,NORMSDIST(NORMSINV({T})+{sf}))",
        "PiT PD in a forecast year = NORMSDIST(NORMSINV(TTC PD) + scaling factor); "
        "after the last forecast year it reverts to the TTC PD in log space: "
        "LN(f x (EXP(TTC PD) - EXP(PiT PD of the last forecast year)) + EXP(that PiT "
        "PD)), f rising from 0 to 1 by the last year. Cumulative PD = 1 - product of "
        "(1 - PiT PD); the scenarios are weighted on the marginal.")
    B.widths(ws, {"A": 30, **{_col_letter(c): 16 for c in range(2, 20)}})
    return out


def _write_pd_external(B, ws, pdm) -> dict:
    """The external scale: GDP-weighted GCC growth and the Basel formula."""
    B.title(ws, "PD external: the external rating model (GCC growth)",
            "Externally rated positions follow regional growth: GCC real growth, "
            "weighted by each country's GDP. Each scenario moves the regional "
            "forecast by its severity x the history's standard deviation; the "
            "scaling factor is NORMSINV of where that lands in the history "
            "(PERCENTRANK.EXC, truncated to 3 significant digits; 1/(n+1) below the "
            "lowest year and n/(n+1) above the highest). Blue cells are the run's "
            "inputs.")
    names = [n for n, _ in pdm["scenarios"]]
    h, fc = pdm["gcc_hist"], pdm["ext_fc"]
    r = 4
    ws.cell(r, 1, "Regional growth history (%)").font = B.f_sub
    B.header(ws, r + 1, 1, ["Year", "Growth"])
    for k, (y, v) in enumerate(zip(pdm["gcc_years"], h)):
        B.put(ws, r + 2 + k, 1, y, "int", inp=True)
        B.put(ws, r + 2 + k, 2, v, "num", inp=True)
    H = _rng(r + 2, 2, r + 1 + len(h), 2)
    c = 4
    stats = [("Years", f"=COUNT({H})", "int"), ("Standard deviation", f"=STDEV({H})", "num"),
             ("Lowest", f"=MIN({H})", "num"), ("Highest", f"=MAX({H})", "num")]
    St = {}
    for k, (lab, f, kd) in enumerate(stats):
        B.put(ws, r + 1 + k, c, lab, bold=True)
        B.put(ws, r + 1 + k, c + 1, f, kd)
        St[lab] = _a(r + 1 + k, c + 1)
    N, SD = St["Years"], St["Standard deviation"]
    r2 = r + 7
    ws.cell(r2, 4, "Regional growth forecast (%)").font = B.f_sub
    B.header(ws, r2 + 1, 4, ["Year", "Growth"])
    Fc = []
    for k, v in enumerate(fc):
        B.put(ws, r2 + 2 + k, 4, k + 1, "int")
        B.put(ws, r2 + 2 + k, 5, v, "num", inp=True)
        Fc.append(_a(r2 + 2 + k, 5))
    r3 = r2 + 3 + len(fc)
    ws.cell(r3, 4, "Scenario weights by forecast year").font = B.f_sub
    B.header(ws, r3 + 1, 4, ["Year"] + names + ["Total"])
    WR = []
    for i, row in enumerate(pdm["w_ext"]):
        rr = r3 + 2 + i
        B.put(ws, rr, 4, i + 1, "int")
        for k, w in enumerate(row):
            B.put(ws, rr, 5 + k, w, "num", inp=True)
        B.put(ws, rr, 5 + len(names), f"=SUM({_rng(rr, 5, rr, 4 + len(names))})", "num")
        WR.append(rr)
    ra = r3 + 2 + len(WR)
    B.put(ws, ra, 4, "Later years (average)")
    for k in range(len(names)):
        B.put(ws, ra, 5 + k, f"=AVERAGE({_rng(WR[0], 5 + k, WR[-1], 5 + k)})", "num")
    B.put(ws, ra, 5 + len(names), f"=SUM({_rng(ra, 5, ra, 4 + len(names))})", "num")
    B.note(ws, ra + 1, 4,
           "The weights the run used" + (": worked out by the run for each year from "
           "where that year's forecast sits on the history (mode auto)."
           if pdm["ext_mode"] != "explicit" else ", as written in model_inputs.yml.")
           + " Years after the last forecast year take the average.")

    def weight_of(k, y):
        row = WR[y - 1] if y <= len(WR) else ra
        return _a(row, 5 + k)

    r = max(r + 2 + len(h), ra + 2) + 1
    ws.cell(r, 1, "Scaling factor by scenario and year").font = B.f_sub
    B.header(ws, r + 1, 1, ["Scenario", "Severity z", "Year", "Stressed growth",
                            "History at or below it", "Percentile rank",
                            "Scaling factor"])
    SF = {}
    rr = r + 2
    for k, (nm, z) in enumerate(pdm["scenarios"]):
        for y in range(1, len(fc) + 1):
            B.put(ws, rr, 1, nm)
            B.put(ws, rr, 2, z, "num", inp=True)
            B.put(ws, rr, 3, y, "int")
            x = _a(rr, 4, fixed=False)
            j = _a(rr, 5, fixed=False)
            B.put(ws, rr, 4, f"={Fc[y - 1]}+{_a(rr, 2, fixed=False)}*{SD}", "num")
            B.put(ws, rr, 5, f"=SUMPRODUCT(--({H}<={x}))", "int")
            inner = (f"({j}+({x}-SMALL({H},{j}))/(SMALL({H},{j}+1)-SMALL({H},{j})))"
                     f"/({N}+1)")
            lo, hi = f"(1/({N}+1))", f"({N}/({N}+1))"
            B.put(ws, rr, 6, f"=IF({x}<{St['Lowest']},{_trunc3(lo)},IF({j}>={N},"
                             f"{_trunc3(hi)},{_trunc3(inner)}))", "num")
            B.put(ws, rr, 7, f"=NORMSINV({_a(rr, 6, fixed=False)})", "num", key=True)
            SF[(k, y)] = _a(rr, 7)
            rr += 1
    out, _ = _write_scale_blocks(
        B, ws, rr + 1, pdm, 2, lambda k, y: SF[(k, y)], weight_of,
        lambda T, R, sf: (f"=IF({T}<=0,0,IF({T}>=1,1,NORMSDIST((NORMSINV({T})"
                          f"-SQRT({R})*{sf})/SQRT(1-{R}))))"),
        "PiT PD in a forecast year: the Basel formula NORMSDIST((NORMSINV(TTC PD) - "
        "SQRT(R) x scaling factor) / SQRT(1 - R)), so higher growth gives a lower PD; "
        "R = 0.24 - 0.12 x (1 - EXP(-50 x TTC PD)) / (1 - EXP(-50)). After the last "
        "forecast year the PD reverts to the TTC PD in log space. The weights change "
        "by year (above).")
    B.widths(ws, {"A": 22, **{_col_letter(c): 16 for c in range(2, 20)}})
    return out


def _write_pd(B, ws, facts, wm: dict) -> dict:
    """The monthly cumulative PD curves: the calculated ones (from the PD
    internal and PD external tabs), one per rating scale and bucket the
    customer uses, beside the run's own (StPD.csv) for each portfolio and
    bucket, with the largest difference; and the switch between the two."""
    pdm = facts.get("pd_model") or {}
    calc_ok = bool(pdm.get("ok"))
    B.title(ws, "PD curves: cumulative PD by month",
            "A year's weighted marginal PD (PD internal / PD external tabs) is spread "
            "evenly over its twelve months and summed, capped at 1. A position reads "
            "the curve of its rating scale and bucket (the PD curve column on Lines). "
            "The run's own curves (StPD.csv) are shown beside them for the check; the "
            "switch below chooses which ones the ECL prices on.")
    B.put(ws, 4, 1, "PD curves used in the ECL", bold=True)
    sw = B.put(ws, 4, 2, pdm.get("source") or RUN, inp=True, key=True)
    from openpyxl.worksheet.datavalidation import DataValidation
    dv = DataValidation(type="list", formula1=f'"{CALC},{RUN}"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add(sw)
    B.note(ws, 4, 3, pdm.get("reason") or (
        "The calculated curves equal the run's StPD: the ECL prices on the calculation."))
    out = {"source": _a(4, 2, "PD curves")}
    r = 6
    ws.cell(r, 1, "Rating buckets (Ratings.csv)").font = B.f_sub
    B.header(ws, r + 1, 1, ["Rating scale", "Rating", "Bucket"])
    rm = facts["ratings_map"]
    for k, m in enumerate(rm):
        rr = r + 2 + k
        B.put(ws, rr, 1, "External" if str(m["rt"]) == "2" else "Internal", inp=True)
        B.put(ws, rr, 2, m["rating"], inp=True)
        B.put(ws, rr, 3, m["bucket"], "int", inp=True)
    last = r + 1 + max(len(rm), 1)
    out.update({"rm_scale": _rng(r + 2, 1, last, 1, "PD curves"),
                "rm_rating": _rng(r + 2, 2, last, 2, "PD curves"),
                "rm_bucket": _rng(r + 2, 3, last, 3, "PD curves")})

    calc = []                                   # (label, rt, bucket key)
    if calc_ok:
        for rt, scale in ((1, "Internal"), (2, "External")):
            if rt == 2 and not pdm["ext_ok"]:
                continue
            for bk in pdm["buckets"][rt]:
                calc.append((f"{scale} | bucket {bk['key']}", rt, bk["key"]))
    runs = facts["curves"]
    nC = len(calc) + len(runs)
    r = last + 2
    hr = r + 3                                  # the curve labels (the keys)
    first = hr + 1
    ws.cell(r, 1, "Cumulative PD by month").font = B.f_sub
    if calc:
        B.put(ws, hr - 1, 2, "Calculated", bold=True)
    if runs:
        B.put(ws, hr - 1, 2 + len(calc), "As the run has it (StPD.csv)", bold=True)
        B.put(ws, hr - 2, 1, "Largest difference", bold=True)
    B.header(ws, hr, 1, ["Month"] + [c[0] for c in calc] + [c["label"] for c in runs])
    for m in range(0, MAX_MONTH + 1):
        rr = first + m
        B.put(ws, rr, 1, m, "int")
        for k, (lab, rt, bkey) in enumerate(calc):
            col = 2 + k
            if bkey not in wm.get(rt, {}):
                continue                        # a bucket with no TTC PD
            if m == 0:
                B.put(ws, rr, col, 0, "pd")
                continue
            B.put(ws, rr, col,
                  f"=MIN({_a(rr - 1, col, fixed=False)}+INDEX({wm[rt][bkey]},"
                  f"MIN(ROUNDUP({_a(rr, 1, fixed=False)}/12,0),{pdm['max_mat']}))/12,1)",
                  "pd")
        for k, c in enumerate(runs):
            sp = c["stpd"]
            v = float(sp[m]) if sp is not None and m < len(sp) else None
            B.put(ws, rr, 2 + len(calc) + k, v, "pd", inp=True)
    # the run's curve against the calculated one of its scale and bucket
    where = {(rt, bkey): 2 + k for k, (_, rt, bkey) in enumerate(calc)}
    for k, c in enumerate(runs):
        bkey = str(int(c["bucket"])) if float(c["bucket"]).is_integer() else str(c["bucket"])
        cc, col = where.get((c["rt"], bkey)), 2 + len(calc) + k
        if cc is None or c["stpd"] is None:
            B.put(ws, hr - 2, col, "no calculated curve")
            continue
        B.put(ws, hr - 2, col,
              f"=SUMPRODUCT(MAX(ABS({_rng(first, col, first + MAX_MONTH, col)}-"
              f"{_rng(first, cc, first + MAX_MONTH, cc)})))", "pd")
    out["keys"] = _rng(hr, 2, hr, 1 + max(nC, 1), "PD curves")
    out["used"] = _rng(first, 2, first + MAX_MONTH, 1 + max(nC, 1), "PD curves")
    B.widths(ws, {"A": 26, **{_col_letter(c): 22 for c in range(2, 2 + max(nC, 3))}})
    return out


def _write_lgd(B, ws, facts, L, ln, S) -> dict:
    co = facts["collateral"]
    a, c, t = co["alloc"], co["coll"], co["types"]
    B.title(ws, "LGD: collateral to loss given default",
            "Net collateral per allocation = value x allocation x (1 - haircut) "
            "(nil when any of the three is missing). LGD = base x MAX(floor, "
            "(exposure - net collateral) / exposure); the floor keeps at least "
            "that share unsecured. Zero or missing exposure takes the setting's "
            "LGD.")
    r = 4
    ws.cell(r, 1, "Collateral (Collateral.csv)").font = B.f_sub
    B.header(ws, r + 1, 1, ["Collateral id", "Type", "Value"])
    cr0 = r + 2
    for k, row in enumerate(c.itertuples(index=False)):
        B.put(ws, cr0 + k, 1, row.collateral, inp=True)
        B.put(ws, cr0 + k, 2, row.type, inp=True)
        B.put(ws, cr0 + k, 3, _num(row.value), "amt", inp=True)
    cr1 = cr0 + max(len(c), 1) - 1
    ws.cell(r, 5, "Collateral types (CollateralType.csv)").font = B.f_sub
    B.header(ws, r + 1, 5, ["Type", "Name", "Haircut"])
    for k, row in enumerate(t.itertuples(index=False)):
        B.put(ws, cr0 + k, 5, row.type, inp=True)
        B.put(ws, cr0 + k, 6, _str(row.name), inp=True)
        B.put(ws, cr0 + k, 7, _num(row.haircut), "pct", inp=True)
    tr1 = cr0 + max(len(t), 1) - 1
    r = max(cr1, tr1) + 3
    ws.cell(r, 1, "Allocations (AccountCollateralAllocation.csv)").font = B.f_sub
    B.header(ws, r + 1, 1, ["Contract", "Collateral id", "Allocation",
                            "Type", "Value", "Haircut", "Net collateral"])
    ar0 = r + 2
    for k, row in enumerate(a.itertuples(index=False)):
        rr = ar0 + k
        B.put(ws, rr, 1, row.contract, inp=True)
        B.put(ws, rr, 2, row.collateral, inp=True)
        B.put(ws, rr, 3, _num(row.share), "pct", inp=True)
        idr = _a(rr, 2, fixed=False)
        B.put(ws, rr, 4, f"=IFERROR(INDEX({_rng(cr0, 2, cr1, 2)},MATCH({idr},"
                         f"{_rng(cr0, 1, cr1, 1)},0)),\"\")")
        B.put(ws, rr, 5, f"=IFERROR(INDEX({_rng(cr0, 3, cr1, 3)},MATCH({idr},"
                         f"{_rng(cr0, 1, cr1, 1)},0)),\"\")", "amt")
        B.put(ws, rr, 6, f"=IFERROR(INDEX({_rng(cr0, 7, tr1, 7)},MATCH("
                         f"{_a(rr, 4, fixed=False)},{_rng(cr0, 5, tr1, 5)},0)),\"\")",
              "pct")
        v, s_, h = (_a(rr, i, fixed=False) for i in (5, 3, 6))
        B.put(ws, rr, 7, f"=IF(COUNT({v},{s_},{h})=3,{v}*{s_}*(1-{h}),0)", "amt")
    ar1 = ar0 + max(len(a), 1) - 1
    r = ar1 + 3
    ws.cell(r, 1, "LGD per position").font = B.f_sub
    B.header(ws, r + 1, 1, ["Line", "Contract", "On balance", "Net collateral",
                            "Collateral cover", "LGD", "Report LGD (Stage 1/2)"])
    out = {}
    for k, l in enumerate(L):
        rr = r + 2 + k
        x = l["line"]
        B.put(ws, rr, 1, f"={ln(x, 'Line')}")
        B.put(ws, rr, 2, f"={ln(x, 'Contract')}")
        onb = ln(x, "On balance")
        B.put(ws, rr, 3, f"=IF(ISNUMBER({onb}),{onb},0)", "amt")
        B.put(ws, rr, 4, f"=SUMIFS({_rng(ar0, 7, ar1, 7)},{_rng(ar0, 1, ar1, 1)},"
                         f"{_a(rr, 2, fixed=False)})", "amt")
        e, n = _a(rr, 3, fixed=False), _a(rr, 4, fixed=False)
        B.put(ws, rr, 5, f"=IF({e}>0,{n}/{e},\"\")", "pct")
        B.put(ws, rr, 6, f"=IF({e}>0,{S['lgd_base']}*MAX({S['lgd_floor']},"
                         f"({e}-{n})/{e}),{S['zero_lgd']})", "pct", key=True)
        B.put(ws, rr, 7, l["rep_lgd"], "pct", inp=True)
        out[x] = {"lgd": _a(rr, 6, "LGD"), "net": _a(rr, 4, "LGD"),
                  "onb": _a(rr, 3, "LGD"), "cover": _a(rr, 5, "LGD")}
    B.widths(ws, {"A": 16, "B": 16, "C": 16, "D": 16, "E": 14, "F": 14, "G": 14})
    return out


def _write_ead(B, ws, facts, L, ln, stage, S) -> dict:
    B.title(ws, "EAD: how the exposure runs down, and how many months count",
            "With a repayment schedule, the monthly exposure is the schedule's "
            "(LifeTimeParameterOther.csv, blue on the ECL tab). Without one it is "
            "estimated: the repayment type comes from the Settings rules (Bullet: "
            "the balance stays until maturity; Equal instalments: the balance "
            "falls by the same amount at each payment; Annuity: level payments "
            "including interest). Payments fall every 'months between payments', "
            "counted back from maturity, after any grace months. A part month "
            "counts as a whole one, and a facility at or past maturity runs 3 "
            "months as a bullet. Months counted in ECL: up to 12 at Stage 1, the "
            "whole exposure at Stages 2 and 3.")
    cols = ["Line", "Contract", "Source", "Months to maturity (N)", "Shape",
            "f", "d", "n payments", "First payment p1",
            "Monthly rate r", "Path length", "Stage", "Horizon (months)"]
    E = {n: i + 1 for i, n in enumerate(cols)}
    B.header(ws, 4, 1, _shown(cols))
    out = {}
    for k, l in enumerate(L):
        rr = 5 + k
        x = l["line"]
        g = lambda name: _a(rr, E[name], fixed=False)
        B.put(ws, rr, E["Line"], f"={ln(x, 'Line')}")
        B.put(ws, rr, E["Contract"], f"={ln(x, 'Contract')}")
        sched = l["schedule"]
        B.put(ws, rr, E["Source"], "Repayment schedule" if sched is not None
              else "Estimated (no schedule)")
        mat, ext = ln(x, "Maturity date"), ln(x, "Extract date")
        mm = (f"((YEAR({mat})-YEAR({ext}))*12+MONTH({mat})-MONTH({ext})+"
              f"IF(DAY({mat})>DAY({ext}),1,0))")
        B.put(ws, rr, E["Months to maturity (N)"],
              f"=IF(OR({mat}=\"\",{ext}=\"\"),{S['minm']},IF({mm}>0,{mm},{S['minm']}))",
              "int")
        N = g("Months to maturity (N)")
        pf, pt = ln(x, "Portfolio"), ln(x, "Payment type")
        shape = (f"IFERROR(INDEX({S['rule_shape']},MATCH({pf}&\"|\"&{pt},"
                 f"{S['rule_key']},0)),IFERROR(INDEX({S['rule_shape']},MATCH("
                 f"\"|\"&{pt},{S['rule_key']},0)),{S['default_shape']}))")
        B.put(ws, rr, E["Shape"],
              f"=IF({g('Source')}=\"Repayment schedule\",\"As scheduled\","
              f"IF({N}<={S['minm']},\"{SHAPE_LABEL['bullet']}\",{shape}))")
        pfq, dfr = ln(x, "Payment frequency (months)"), ln(x, "Deferral (months)")
        B.put(ws, rr, E["f"], f"=MAX(1,IF(ISNUMBER({pfq}),TRUNC({pfq}),1))", "int")
        B.put(ws, rr, E["d"], f"=MAX(0,MIN(IF(ISNUMBER({dfr}),TRUNC({dfr}),0),"
                              f"MAX(1,{N})-1))", "int")
        B.put(ws, rr, E["n payments"],
              f"=MAX(1,ROUNDUP(MAX(1,MAX(1,{N})-{g('d')})/{g('f')},0))", "int")
        B.put(ws, rr, E["First payment p1"],
              f"=MAX(1,{N})-({g('n payments')}-1)*{g('f')}", "int")
        nir = ln(x, "Nominal interest rate")
        B.put(ws, rr, E["Monthly rate r"], f"=IF(ISNUMBER({nir}),{nir}/12,0)", "rate")
        if sched is not None:
            B.put(ws, rr, E["Path length"], len(sched), "int")
        else:
            B.put(ws, rr, E["Path length"], f"=MIN(MAX(1,{N}),{S['maxm']})", "int")
        B.put(ws, rr, E["Stage"], f"={stage(x)}", "int")
        B.put(ws, rr, E["Horizon (months)"],
              f"=MIN(IF({g('Stage')}=1,MIN({S['s1m']},{g('Path length')}),"
              f"{g('Path length')}),{S['maxm']})", "int", key=True)
        out[x] = {c: _a(rr, E[c], "EAD") for c in cols}
    B.widths(ws, {"A": 6, "B": 14, "C": 22, "D": 11, "E": 18, "F": 12, "G": 13,
                  "H": 11, "I": 11, "J": 12, "K": 11, "L": 7, "M": 12})
    return out


def _write_ecl(B, ws, facts, L, ln, lg, ead, pm, S) -> dict:
    B.title(ws, "ECL month by month",
            "ECL in a month = EAD x LGD x PD in the month / discount factor, "
            "where the PD in the month is the cumulative PD at its end less at "
            "its start (the position's PD curve, PD curves tab) and the discount "
            "factor is (1 + EIR)^(month / 12). A position's ECL is the sum over "
            "the months counted (Summary tab), capped at the exposure; Stage 3 "
            "follows the Settings.")
    cols = ["Line", "Contract", "Month", "EAD", "Counted in ECL",
            "Cumulative PD at start", "Cumulative PD at end", "PD in the month",
            "LGD", "EIR", "Discount factor", "ECL in the month"]
    B.header(ws, 4, 1, cols)
    rr = 5
    out = {}
    for l in L:
        x = l["line"]
        e = ead[x]
        sched = l["schedule"]
        if sched is not None:
            n = min(len(sched), MAX_MONTH)
        else:
            n = min(max(1, _fallback_n(l)), MAX_MONTH)
        start = rr
        onb = ln(x, "On balance")
        curve = f"MATCH({ln(x, 'PD curve')},{pm['keys']},0)"
        for t in range(n):
            B.put(ws, rr, 1, x, "int")
            B.put(ws, rr, 2, l["cid"])
            B.put(ws, rr, 3, t, "int")
            T = _a(rr, 3, fixed=False)
            if sched is not None:
                B.put(ws, rr, 4, float(sched[t]), "amt", inp=True)
            else:
                paid = (f"IF({T}<{e['First payment p1']},0,MIN({e['n payments']},"
                        f"INT(({T}-{e['First payment p1']})/{e['f']})+1))")
                rp = f"({e['Monthly rate r']}*{e['f']})"
                lin, ann = SHAPE_LABEL["linear"], SHAPE_LABEL["annuity"]
                coef = (f"IF({e['Shape']}=\"{lin}\",MAX(0,1-{paid}/{e['n payments']}),"
                        f"IF(AND({e['Shape']}=\"{ann}\",{e['Monthly rate r']}>1E-9),"
                        f"IF({paid}>={e['n payments']},0,MAX(0,((1+{rp})^{e['n payments']}"
                        f"-(1+{rp})^{paid})/((1+{rp})^{e['n payments']}-1))),"
                        f"IF({e['Shape']}=\"{ann}\",MAX(0,1-{paid}/{e['n payments']}),1)))")
                B.put(ws, rr, 4, f"=IF(ISNUMBER({onb}),{onb},0)*{coef}", "amt")
            B.put(ws, rr, 5, f"={T}<{e['Horizon (months)']}")
            B.put(ws, rr, 6, f"=INDEX({pm['used']},{T}+1,{curve})", "pd")
            B.put(ws, rr, 7, f"=INDEX({pm['used']},{T}+2,{curve})", "pd")
            B.put(ws, rr, 8, f"={_a(rr, 7, fixed=False)}-{_a(rr, 6, fixed=False)}", "pd")
            B.put(ws, rr, 9, f"={lg[x]['lgd']}", "pct")
            eir = ln(x, "EIR")
            B.put(ws, rr, 10, f"={eir}", "rate")
            B.put(ws, rr, 11, f"=IF(ISNUMBER({eir}),(1+{eir})^({T}/12),NA())", "rate")
            B.put(ws, rr, 12,
                  f"=IF({_a(rr, 5, fixed=False)},{_a(rr, 4, fixed=False)}*"
                  f"{_a(rr, 9, fixed=False)}*{_a(rr, 8, fixed=False)}/"
                  f"{_a(rr, 11, fixed=False)},0)", "amt")
            rr += 1
        out[x] = (start, rr - 1)
    out["_col_line"] = _rng(5, 1, max(rr - 1, 5), 1, "ECL")
    out["_col_ecl"] = _rng(5, 12, max(rr - 1, 5), 12, "ECL")
    ws.auto_filter.ref = f"A4:L{max(rr - 1, 5)}"
    B.widths(ws, {"A": 6, "B": 14, "C": 8, "D": 16, "E": 10, "F": 14, "G": 14,
                  "H": 14, "I": 10, "J": 10, "K": 12, "L": 16})
    return out


def _fallback_n(l) -> int:
    from .engine import months_to_maturity
    return months_to_maturity(l["d_maturity"], l["d_extract"])


def _write_summary(B, ws, facts, L, ln, stage, lg, ead, ecl, pm, S, GR, G):
    B.title(ws, f"ECL calculator: customer {facts['customer_id']}"
                + (f" ({facts['customer_name']})" if facts["customer_name"] else ""))
    info = [
        ("Run", facts["run_id"]),
        ("Extract date", L[0]["d_extract"] if L else None),
        ("Config version", facts["config_version"]),
        ("Calculator version", facts["calculator_version"]),
        ("Engine", facts["engine"]),
        ("Report compared", facts["report_file"] or "none in the run"),
        ("Overlay report", facts["overlay_file"] or "none"),
        ("PD curves", f"={pm['source']}&\" (PD curves tab)\""),
        ("Generated", facts["generated_at"]),
    ]
    for k, (lab, v) in enumerate(info):        # label overflows into B:C
        B.put(ws, 3 + k, 1, lab, bold=True)
        c = B.put(ws, 3 + k, 4, v, "date" if lab == "Extract date" else None)
        c.alignment = B.Align(horizontal="left")
    B.note(ws, 12, 1, "How to read it: Lines (inputs) -> Staging -> LGD -> PD "
           "internal / PD external -> PD curves -> EAD -> ECL. Blue cells are the "
           "run's inputs; black cells are formulas. Change an input and the "
           "provision follows; the report column stays as booked. The settings (and "
           "the overrides, when there are none) are on hidden tabs.")
    ws.cell(21, 1, "Per position").font = B.f_sub
    hr = 22
    cols = ["Line", "Book", "Contract", "Portfolio", "Rating", "Stage", "Why",
            "On balance", "Net collateral", "LGD", "PD lifetime", "Horizon",
            "ECL (this workbook)", "ECL (report)", "Difference", "Overlay",
            "ECL final", "Report final", "Check"]
    U = {n: i + 1 for i, n in enumerate(cols)}
    B.header(ws, hr, 1, _shown(cols))
    first = hr + 1
    for k, l in enumerate(L):
        rr = first + k
        x = l["line"]
        g = lambda name: _a(rr, U[name], fixed=False)
        B.put(ws, rr, U["Line"], f"={ln(x, 'Line')}")
        B.put(ws, rr, U["Book"], f"={ln(x, 'Book')}")
        B.put(ws, rr, U["Contract"], f"={ln(x, 'Contract')}")
        B.put(ws, rr, U["Portfolio"], f"={ln(x, 'Portfolio')}")
        B.put(ws, rr, U["Rating"], f"={ln(x, 'Rating')}")
        B.put(ws, rr, U["Stage"], f"={stage(x)}", "int")
        B.put(ws, rr, U["Why"], f"={_a(GR[x], G['Why'], 'Staging')}")
        B.put(ws, rr, U["On balance"], f"={lg[x]['onb']}", "amt")
        B.put(ws, rr, U["Net collateral"], f"={lg[x]['net']}", "amt")
        B.put(ws, rr, U["LGD"], f"={lg[x]['lgd']}", "pct")
        e = ead[x]
        h = (f"MIN(MAX(0,IF({g('Stage')}=1,MIN({S['s1m']},{e['Months to maturity (N)']}),"
             f"{e['Months to maturity (N)']})),{S['maxm']})")
        B.put(ws, rr, U["PD lifetime"],
              f"=IFERROR(INDEX({pm['used']},{h}+1,MATCH({ln(x, 'PD curve')},"
              f"{pm['keys']},0)),\"\")", "pd")
        B.put(ws, rr, U["Horizon"], f"={e['Horizon (months)']}", "int")
        a, b = ecl[x]
        tot = f"SUM({_rng(a, 12, b, 12, 'ECL')})"
        onb = g("On balance")
        B.put(ws, rr, U["ECL (this workbook)"],
              f"=IF({g('Stage')}=3,IF({S['stage3']}=\"zero\",0,MAX(0,{onb})),"
              f"IF(AND({S['cap']},{onb}>0),MIN({tot},{onb}),{tot}))", "amt", key=True)
        B.put(ws, rr, U["ECL (report)"], l["rep_ecl"], "amt", inp=True)
        B.put(ws, rr, U["Difference"],
              f"=IF({g('ECL (report)')}=\"\",\"\",{g('ECL (this workbook)')}-"
              f"{g('ECL (report)')})", "amt")
        B.put(ws, rr, U["Overlay"], l["overlay"] if l["overlay"] is not None else 0,
              "amt", inp=True)
        B.put(ws, rr, U["ECL final"], f"={g('ECL (this workbook)')}+{g('Overlay')}",
              "amt")
        B.put(ws, rr, U["Report final"], l["rep_final"], "amt", inp=True)
        tol = lambda a, b: f"ABS({a}-{b})<=MAX(0.01,1E-9*ABS({b}))"
        B.put(ws, rr, U["Check"],
              f"=IF({g('ECL (report)')}=\"\",\"not in report\",IF(AND("
              f"{tol(g('ECL (this workbook)'), g('ECL (report)'))},OR("
              f"{g('Report final')}=\"\",{tol(g('ECL final'), g('Report final'))})),"
              f"\"OK\",\"DIFFERS\"))")
    last = first + max(len(L), 1) - 1
    # totals by stage
    ws.cell(14, 1, "Totals").font = B.f_sub
    B.header(ws, 15, 1, ["Stage", "Positions", "On balance",
                         "ECL (this workbook)", "ECL (report)", "Difference",
                         "ECL final", "Report final"])
    col = lambda n: _rng(first, U[n], last, U[n])
    for k, s in enumerate([1, 2, 3, "Total"]):
        rr = 16 + k
        B.put(ws, rr, 1, s, bold=s == "Total")
        if s == "Total":
            f = lambda n: f"=SUM({_rng(16, n, 18, n)})"
            for c in range(2, 9):
                B.put(ws, rr, c, f(c), "int" if c == 2 else "amt", bold=True)
            continue
        B.put(ws, rr, 2, f"=COUNTIF({col('Stage')},{s})", "int")
        for c, n in ((3, "On balance"), (4, "ECL (this workbook)"),
                     (5, "ECL (report)"), (7, "ECL final"), (8, "Report final")):
            B.put(ws, rr, c, f"=SUMIFS({col(n)},{col('Stage')},{s})", "amt")
        B.put(ws, rr, 6, f"={_a(rr, 4, fixed=False)}-{_a(rr, 5, fixed=False)}", "amt")
    from openpyxl.formatting.rule import CellIsRule
    rng = f"{_col_letter(U['Check'])}{first}:{_col_letter(U['Check'])}{last}"
    ws.conditional_formatting.add(rng, CellIsRule(
        operator="equal", formula=['"OK"'],
        fill=B.Fill("solid", fgColor="C6EFCE")))
    ws.conditional_formatting.add(rng, CellIsRule(
        operator="equal", formula=['"DIFFERS"'],
        fill=B.Fill("solid", fgColor="FFC7CE")))
    B.widths(ws, {"A": 8, "B": 12, "C": 16, "D": 18, "E": 10, "F": 7, "G": 34,
                  "H": 16, "I": 16, "J": 9, "K": 12, "L": 9, "M": 18, "N": 18,
                  "O": 12, "P": 12, "Q": 16, "R": 16, "S": 10})


def customer_ecl_workbook(run_dir, customer_id, dest=None):
    """The calculator for one customer of a run: bytes, or written to
    ``dest``."""
    return write_customer_workbook(customer_ecl_facts(run_dir, customer_id), dest)
