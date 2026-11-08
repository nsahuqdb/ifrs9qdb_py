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

import functools
import io
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .ids import as_id

__all__ = ["customer_ecl_facts", "write_customer_workbook",
           "customer_ecl_workbook", "run_customers", "run_contracts",
           "contract_ecl_facts", "contract_ecl_workbook"]

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
    "Line": "No.",
    "Original contract": "Contract (source id)",
    "Rating set by": "How the rating was set",
    "Stage": "Stage",
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
    """Customer id -> name: from the run's contract reference (engine 0.45
    and later), else from the customer masters of the input folder the run
    read (the LIC customer master leaves the name blank)."""
    p = Path(out_dir).parent / "reference" / "contracts.csv"
    if p.is_file():
        try:
            r = pd.read_csv(p, dtype=str, keep_default_na=False)
            if "CustomerName" in r.columns and r["CustomerName"].str.strip().ne("").any():
                return {a: b for a, b in zip(as_id(r["CustomerId"]), r["CustomerName"])
                        if b}
        except Exception:
            pass
    return _input_names(str(Path(out_dir).parent))


@functools.lru_cache(maxsize=16)
def _input_names(run_root: str) -> dict:
    """Customer id -> name from the CustomerMaster extracts in the input
    folder the run's manifest names; {} when that folder is gone."""
    from .etl.read_inputs import INPUT_SPECS, read_input
    root = Path(run_root)
    d = None
    for m in (root / "reports" / "manifest.json", root / "manifest.json"):
        if m.is_file():
            try:
                import json
                d = json.loads(m.read_text(encoding="utf-8")).get("input_dir")
            except Exception:
                d = None
            break
    if not d or not Path(d).is_dir():
        return {}
    names = {}
    for spec in INPUT_SPECS:
        if spec.name not in ("CustomerMaster", "CustomerMasterInvestments"):
            continue
        for f in (spec.file, *spec.alternatives):
            fp = Path(d) / f
            if not fp.is_file():
                continue
            try:
                df = read_input(fp)
            except Exception:
                break
            cols = {str(c).strip().upper(): c for c in df.columns}
            if "CUSTOMERID" in cols and "CUSTOMERNAME" in cols:
                ids = as_id(df[cols["CUSTOMERID"]].astype(str))
                for a, b in zip(ids, df[cols["CUSTOMERNAME"]]):
                    b = _str(b)
                    if a and b:
                        names.setdefault(a, b)
            break
    return names


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


def run_contracts(run_dir) -> pd.DataFrame:
    """Every position in a run's books: contract (LIC id and source id),
    customer id and name, product, book and on-balance exposure, largest
    first -- what the app's contract picker offers."""
    from .etl.report import read_output_csv, _chr, _col, _contract_reference
    out_dir = _out_dir(_run_root(run_dir))
    names = _reference_names(out_dir)
    rows = []
    for suffix, book in (("_1", "Lending"), ("_2", "Investments")):
        am = read_output_csv(out_dir, f"AccountMaster{suffix}.csv")
        if am is None or len(am) == 0:
            continue
        cm = read_output_csv(out_dir, f"CustomerMaster{suffix}.csv")
        own = {}
        if cm is not None and _col(cm, "CustomerName") is not None:
            for a, b in zip(as_id(_chr(_col(cm, "CustomerId"))),
                            _chr(_col(cm, "CustomerName"))):
                if _str(b):
                    own.setdefault(a, _str(b))
        contract = _chr(_col(am, "ContractId"))
        ref = _contract_reference(out_dir, suffix, contract)
        cust = as_id(_chr(_col(am, "CustomerId")))
        d = pd.DataFrame({
            "book": book, "contract_id": contract.map(_str),
            "contract_original": (ref["ContractIdOriginal"].map(_str)
                                  if ref is not None else ""),
            "customer_id": cust,
            "customer_name": [own.get(c) or names.get(c, "") for c in cust],
            "product": _chr(_col(am, "AccountType")).map(_str),
            "on_balance": pd.to_numeric(_col(am, "OnBalance"), errors="coerce")})
        rows.append(d[d["contract_id"] != ""])
    if not rows:
        return pd.DataFrame(columns=["book", "contract_id", "contract_original",
                                     "customer_id", "customer_name", "product",
                                     "on_balance"])
    d = pd.concat(rows, ignore_index=True)
    return d.sort_values("on_balance", ascending=False,
                         na_position="last").reset_index(drop=True)


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
                             _contract_reference, _portfolio_map, _KEYSEP)
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
    # what the ETL kept beside the run: the source contract id, the facility's
    # own days past due and the inputs of its rating (reference/contracts.csv)
    ref = {s: _contract_reference(out_dir, s, _chr(_col(a, "ContractId")))
           for s, a in am.items() if a is not None}
    recorded = any(r.attrs.get("recorded") for r in ref.values())

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
            "contract_original": _str(ref[s]["ContractIdOriginal"].iloc[j]),
            "dpd_own": _num(ref[s]["PastDueDays"].iloc[j]),
            "rating_input": _str(ref[s]["RatingInput"].iloc[j]),
            "sector": _str(ref[s]["Sector"].iloc[j]),
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

    # ratings -> buckets, as the curve lookup reads Ratings.csv: every grade,
    # so a rating changed in the workbook still finds its bucket
    rat = read_output_csv(out_dir, "Ratings.csv")
    rmap = []
    if rat is not None:
        rd = pd.DataFrame({"rt": _chr(_col(rat, "RatingType")).astype(str),
                           "rating": _chr(_col(rat, "Rating")).astype(str),
                           "bucket": _chr(_col(rat, "Hierarchy"))})
        seen = set()
        for row in rd.itertuples():
            key = (row.rt, row.rating)
            if key not in seen and _num(row.bucket) is not None:
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
        "rating": _rating_tables(static, recorded),
    }


def _rating_tables(static, recorded: bool) -> dict:
    """The run's rating assumptions: the program lending ratings by days past
    due, the fallback rating of each segment, and the master rating scale
    (internal grades first, by hierarchy)."""
    from .etl.lending import _master_scale
    from .etl.transform import pick
    out = {"recorded": bool(recorded), "rules": [], "fallback": [], "scale": []}
    if static is None:
        return out
    rules = static.get("collective_assessment_rules")
    if rules is not None and len(rules):
        r = pd.DataFrame({"sector": pick(rules, "sector", "Sector").astype(str),
                          "dpd": pd.to_numeric(pick(rules, "dpd_threshold"),
                                               errors="coerce"),
                          "rating": pick(rules, "rating", "Rating").astype(str)})
        r = r.sort_values(["sector", "dpd"], kind="stable")
        out["rules"] = [tuple(x) for x in r.itertuples(index=False)]
    fb = static.get("segment_fallback_ratings")
    if fb is not None and len(fb):
        out["fallback"] = list(zip(pick(fb, "segment", "Segment").astype(str),
                                   pick(fb, "fallback_rating",
                                        "FallbackRating").astype(str)))
    sc = _master_scale(static).dropna(subset=["hierarchy"])
    sc = pd.concat([sc[sc["rating_type"] == "Internal"].sort_values("hierarchy",
                                                                    kind="stable"),
                    sc[sc["rating_type"] != "Internal"]])
    out["scale"] = [(a, b, int(c)) for a, b, c in
                    sc[["rating", "rating_type", "hierarchy"]].itertuples(index=False)]
    return out


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
    from .etl.macro import EXTERNAL_AUTO, INTERNAL_AUTO, internal_weight_inputs
    int_mode = (inputs.get("internal_scenario_weights") or {}).get("mode", "explicit")
    ext_mode = (inputs.get("external_scenario_weights") or {}).get("mode", "explicit")
    iw = None
    if int_mode in INTERNAL_AUTO:
        w = internal_weight_inputs(inputs, static, model_cfg)
        h = np.asarray(w["history"], dtype=float)
        var = w["variable"]
        nm = next((m.get("name") or m.get("variable_id") for m in resolved["model"]["mevs"]
                   if m.get("variable_id") == var), var or "the main variable")
        iw = {"history": [float(v) for v in h[np.isfinite(h)]],
              "forecasts": [float(v) for v in w["forecasts"]], "n": int(w["n"]),
              "column": int(w["column"]),
              "in_table": w["path"] == "mev_forecasts.forecasts", "name": nm}
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
        "int_mode": int_mode, "int_weights": iw,
        "ext_auto": ext_mode in EXTERNAL_AUTO and not (
            (inputs.get("external_scenario_weights") or {}).get("explicit_weights_per_year")),
        "ext_ok": ext_ok,
        "gcc_hist": [float(v) for v in np.asarray(gcc_hist, dtype=float)],
        "gcc_years": _gcc_years(static, int(np.asarray(gcc_hist).size)),
        "ext_fc": [] if ext_fc is None else [float(v) for v in np.asarray(ext_fc, float)],
        "w_ext": [[float(pe.iloc[i][n]) for n in names] for i in range(len(pe))],
        "ext_mode": ext_mode,
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

    def put(self, ws, r, c, v, kind=None, inp=False, bold=False, key=False,
            align=None):
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
        if align or kind == "int":
            cell.alignment = self.Align(horizontal=align or "center")
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
    None.

    The tabs follow the provision from start to end: Facilities (the inputs),
    Rating (how each rating was set), Staging (the customer's stage, then each
    facility's), the PD (model, scenario weights, curves), LGD, EAD and the ECL
    month by month; the Summary on top, the settings on a hidden tab."""
    B = _Book()
    L = facts["lines"]
    # a contract workbook: Facilities, Rating and Staging keep every facility
    # of the customer (the customer's worst facility sets its rating and any
    # Stage 2 or 3 facility lifts the others); the rest show the contract
    focus = set(facts.get("focus") or [])
    LF = [l for l in L if l["line"] in focus] if focus else L
    st = facts["settings"]
    R: dict = {}                     # (line, column) -> its cell, sheet-qualified

    pdm = facts.get("pd_model") or {}
    scales = {l.get("rt") for l in L}
    ws_sum = B.sheet("Summary")
    ws_fac = B.sheet("Facilities")
    ws_rat = B.sheet("Rating")
    ws_stg = B.sheet("Staging")
    ws_pi = B.sheet("PD internal") if pdm.get("ok") and "1" in scales else None
    ws_pe = (B.sheet("PD external") if pdm.get("ok") and pdm.get("ext_ok")
             and "2" in scales else None)
    ws_pd = B.sheet("PD curves")
    ws_lgd = B.sheet("LGD")
    ws_ead = B.sheet("EAD")
    ws_ecl = B.sheet("ECL")
    ws_set = B.sheet("Settings")
    ws_set.sheet_state = "hidden"      # Excel: right-click a tab, Unhide

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

    # ---------------------------------------------------------- Facilities
    _write_facilities(B, ws_fac, L, R, focus)

    def ln(line, name, fixed=True):
        return R[(line, name)]

    # ---------------------------------------------------------- PD
    wm = {1: _write_pd_internal(B, ws_pi, pdm) if ws_pi is not None else {},
          2: _write_pd_external(B, ws_pe, pdm) if ws_pe is not None else {}}
    pm = _write_pd(B, ws_pd, facts, wm)

    # ---------------------------------------------------------- Rating
    _write_rating(B, ws_rat, facts, L, R, pm)

    # ---------------------------------------------------------- Staging
    _write_staging(B, ws_stg, facts, L, R, S)

    def stage(line):
        return R[(line, "Stage")]

    # ---------------------------------------------------------- LGD
    lg = _write_lgd(B, ws_lgd, facts, LF, ln, S)

    # ---------------------------------------------------------- EAD
    ead = _write_ead(B, ws_ead, facts, LF, ln, stage, S)

    # ---------------------------------------------------------- ECL
    ecl = _write_ecl(B, ws_ecl, facts, LF, ln, lg, ead, pm, S)

    # ---------------------------------------------------------- Summary
    _write_summary(B, ws_sum, facts, LF, ln, stage, lg, ead, ecl, pm, S)

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


# ======================================================================
# Facilities, Rating, Staging: the customer's inputs, how each rating was
# set, and the stage -- the customer's, then each facility's.
# ======================================================================
FAC_COLS = ["Line", "Book", "Contract", "Original contract", "Customer",
            "Account type", "Portfolio", "Rating type", "Rating", "Extract date",
            "Maturity date", "On balance", "Off balance", "EIR", "Facility DPD",
            "Past due days", "Payment type", "Payment frequency (months)",
            "Deferral (months)", "Nominal interest rate", "AccountMaster.Stage",
            "Repayment schedule"]
FAC_SHOWN = {
    "Line": "No.", "Contract": "Contract (LIC id)",
    "Original contract": "Contract (source id)", "Customer": "Customer id",
    "Account type": "Product", "Rating type": "Rating scale",
    "Rating": "Rating in the run", "Extract date": "Reporting date",
    "On balance": "On balance (QAR)", "Off balance": "Off balance (QAR)",
    "EIR": "Effective interest rate", "Facility DPD": "Days past due",
    "Past due days": "Customer's worst days past due",
    "Payment frequency (months)": "Months between payments",
    "Deferral (months)": "Grace months",
    "AccountMaster.Stage": "Stage in the account master",
}


def _write_facilities(B, ws, L, R, focus=frozenset()) -> None:
    B.title(ws, "Facilities: the customer's positions as the run read them",
            "The run's account master (the LIC file), with the source contract id "
            "and each facility's own days past due kept by the ETL. Every value "
            "here is an input."
            + (" The contract this workbook is for is marked in yellow; the "
               "customer's other facilities are here because they set its "
               "rating and can lift its stage." if focus else ""))
    C = {n: i + 1 for i, n in enumerate(FAC_COLS)}
    B.header(ws, 4, 1, [FAC_SHOWN.get(c, c) for c in FAC_COLS])
    kinds = {"Extract date": "date", "Maturity date": "date", "On balance": "amt",
             "Off balance": "amt", "EIR": "rate", "Facility DPD": "int",
             "Past due days": "int", "Payment frequency (months)": "int",
             "Deferral (months)": "int", "Nominal interest rate": "rate"}
    for k, l in enumerate(L):
        rr = 5 + k
        vals = {"Line": l["line"], "Book": l["book"], "Contract": l["cid"],
                "Original contract": l.get("contract_original") or "",
                "Customer": l["customer"], "Account type": l["account_type"],
                "Portfolio": l["portfolio"], "Rating type": l["rating_type"],
                "Rating": l["rating"], "Extract date": l["d_extract"],
                "Maturity date": l["d_maturity"], "On balance": l["on_balance"],
                "Off balance": l["off_balance"], "EIR": l["eir"],
                "Facility DPD": l.get("dpd_own"), "Past due days": l["dpd"],
                "Payment type": l["payment_type"],
                "Payment frequency (months)": l["pay_freq"],
                "Deferral (months)": l["deferral"], "Nominal interest rate": l["nir"],
                "AccountMaster.Stage": l["stage_raw"],
                "Repayment schedule": "Yes" if l["schedule"] is not None else "No"}
        for name in FAC_COLS:
            B.put(ws, rr, C[name], vals[name], kinds.get(name), inp=name != "Line",
                  key=l["line"] in focus and name in ("Line", "Contract"))
            R[(l["line"], name)] = _a(rr, C[name], "Facilities")
    B.widths(ws, {"A": 6, "B": 11, "C": 16, "D": 20, "E": 13, "F": 14, "G": 16,
                  "H": 10, "I": 11, "J": 12, "K": 12, "L": 16, "M": 15, "N": 11,
                  "O": 9, "P": 13, "Q": 10, "R": 12, "S": 9, "T": 11, "U": 13,
                  "V": 11})


def _name(B, name: str, ref: str) -> str:
    """A workbook-level name for a range, so a formula reads
    VLOOKUP(x, FallbackRatings, 2, FALSE) rather than a cell address."""
    from openpyxl.workbook.defined_name import DefinedName
    B.wb.defined_names[name] = DefinedName(name, attr_text=ref)
    return name


def _ov_text(o) -> str:
    """An override as one line: was, reason, who, when."""
    bits = [f"was {o.prior_value}" if _str(o.prior_value) else "",
            _str(o.reason), _str(o.created_by), _str(o.created_at)[:16]]
    return "; ".join(b for b in bits if b)


def _write_rating(B, ws, facts, L, R, pm) -> None:
    """How each facility's rating was set, the customer's rating, the
    override, and the rating's PD bucket and curve."""
    rt = facts.get("rating") or {}
    recorded = bool(rt.get("recorded"))
    B.title(ws, "Rating: how each facility's rating was set",
            "Lending: the rating in the customer master when it is a grade on the "
            "rating scale; otherwise a program lending customer takes the program "
            "rating for its worst days past due; otherwise the fallback rating of "
            "its segment. The customer's rating is its worst facility's, unless the "
            "run overrode it. Investments: the extract's rating when it is on the "
            "external scale, otherwise the fallback.")
    # ---- the assumptions, at the bottom, named
    rules, fb, scale = rt.get("rules") or [], rt.get("fallback") or [], rt.get("scale") or []
    lend = [l for l in L if l["suffix"] == "_1"]
    inv = [l for l in L if l["suffix"] == "_2"]
    n_lend = (len(lend) + 11) if (lend and recorded) else (9 if lend else 0)
    top = 4 + n_lend + ((len(inv) + 3) if inv else 0) + len(L) + 4
    r = top
    ws.cell(r, 1, "Assumptions").font = B.f_title
    r += 2
    ws.cell(r, 1, "Program lending ratings").font = B.f_sub
    B.header(ws, r + 1, 1, ["Sector", "Days past due from", "Rating"])
    ladders = {}                                   # sector -> named range
    rr = r + 2
    for sector in dict.fromkeys(x[0] for x in rules):
        first = rr
        for sec, d, g in [x for x in rules if x[0] == sector]:
            B.put(ws, rr, 1, sec, inp=True)
            B.put(ws, rr, 2, d, "int", inp=True)
            B.put(ws, rr, 3, g, inp=True)
            rr += 1
        nm = "Program_" + re.sub(r"[^A-Za-z0-9]", "", sector.replace("Sector", ""))
        ladders[sector] = _name(B, nm, _rng(first, 2, rr - 1, 3, "Rating"))
    ws.cell(r, 5, "Fallback ratings").font = B.f_sub
    B.header(ws, r + 1, 5, ["Segment", "Rating"])
    for k, (seg, g) in enumerate(fb):
        B.put(ws, r + 2 + k, 5, seg, inp=True)
        B.put(ws, r + 2 + k, 6, g, inp=True)
    fbr = _name(B, "FallbackRatings",
                _rng(r + 2, 5, r + 1 + max(len(fb), 1), 6, "Rating"))
    ws.cell(r, 8, "Rating scale").font = B.f_sub
    B.header(ws, r + 1, 8, ["Rating", "Scale", "Hierarchy"])
    for k, (g, t, h) in enumerate(scale):
        B.put(ws, r + 2 + k, 8, g, inp=True)
        B.put(ws, r + 2 + k, 9, t, inp=True)
        B.put(ws, r + 2 + k, 10, h, "int", inp=True)
    last = r + 1 + max(len(scale), 1)
    sr = _name(B, "ScaleRating", _rng(r + 2, 8, last, 8, "Rating"))
    stp = _name(B, "ScaleType", _rng(r + 2, 9, last, 9, "Rating"))
    sh = _name(B, "ScaleHierarchy", _rng(r + 2, 10, last, 10, "Rating"))
    n_int = sum(1 for x in scale if x[1] == "Internal")
    ig = _name(B, "InternalGrades", _rng(r + 2, 8, r + 1 + max(n_int, 1), 8, "Rating"))
    rm = facts["ratings_map"]
    ws.cell(r, 12, "Rating buckets (Ratings.csv)").font = B.f_sub
    B.header(ws, r + 1, 12, ["Rating scale", "Rating", "Bucket"])
    for k, m in enumerate(rm):
        B.put(ws, r + 2 + k, 12, "External" if str(m["rt"]) == "2" else "Internal",
              inp=True)
        B.put(ws, r + 2 + k, 13, m["rating"], inp=True)
        B.put(ws, r + 2 + k, 14, m["bucket"], "int", inp=True)
    lb = r + 1 + max(len(rm), 1)
    bs = _name(B, "BucketScale", _rng(r + 2, 12, lb, 12, "Rating"))
    br = _name(B, "BucketRating", _rng(r + 2, 13, lb, 13, "Rating"))
    bb = _name(B, "BucketNumber", _rng(r + 2, 14, lb, 14, "Rating"))

    ov = facts["overrides"]
    rov = ov[ov["kind"] == "rating"] if len(ov) else ov
    final = {}                                     # line -> final rating cell
    setby = {}
    r = 4
    # ---- lending: rating per facility, then the customer's
    if lend and not recorded:
        ws.cell(r, 1, "Lending").font = B.f_sub
        o = rov.iloc[-1] if len(rov) else None
        B.header(ws, r + 1, 1, ["Customer rating", "Value", "Detail"])
        B.put(ws, r + 2, 1, "Rating override in the run")
        B.put(ws, r + 2, 2, "" if o is None else _str(o.value), inp=True)
        B.put(ws, r + 2, 3, "" if o is None else _ov_text(o))
        B.put(ws, r + 3, 1, "Customer rating", bold=True)
        B.put(ws, r + 3, 2, f"={R[(lend[0]['line'], 'Rating')]}", key=True)
        B.put(ws, r + 3, 3, "The run's rating: the run did not record how it was set "
                            "(made before engine 0.46).")
        for l in lend:
            final[l["line"]] = _a(r + 3, 2, "Rating")
            setby[l["line"]] = _a(r + 3, 3, "Rating")
        r = r + 9
    elif lend:
        ws.cell(r, 1, "Lending").font = B.f_sub
        cols = ["No.", "Contract", "Product", "Rating in the customer master",
                "A grade on the rating scale", "Sector", "Program lending",
                "Customer's worst days past due", "Program rating",
                "Fallback segment", "Fallback rating", "Facility rating", "Set by",
                "Hierarchy"]
        K = {n: i + 1 for i, n in enumerate(cols)}
        B.header(ws, r + 1, 1, cols)
        r0 = r + 2
        r1 = r0 + len(lend) - 1
        dpd_rng = _rng(r0, 8, r1, 8)
        own = ",".join(R[(l["line"], "Facility DPD")] for l in lend)
        sector_names = {}
        for sec in ladders:
            short = re.sub(r"\s*Sector\s*$", "", sec, flags=re.I)
            sector_names[sec] = sorted({short.lower(),
                                        re.sub("Lifestock", "Livestock", short,
                                               flags=re.I).lower()})
        for k, l in enumerate(lend):
            rr = r0 + k
            x = l["line"]
            g = lambda n: _a(rr, K[n], fixed=False)
            B.put(ws, rr, 1, f"={R[(x, 'Line')]}", align="left")
            B.put(ws, rr, 2, f"={R[(x, 'Contract')]}")
            B.put(ws, rr, 3, f"={R[(x, 'Account type')]}")
            B.put(ws, rr, K["Rating in the customer master"], l.get("rating_input") or "",
                  inp=True)
            cmr = g("Rating in the customer master")
            B.put(ws, rr, K["A grade on the rating scale"],
                  f"=IF(AND({cmr}<>\"\",SUMPRODUCT(--({sr}={cmr}))>0),\"Yes\",\"No\")")
            B.put(ws, rr, K["Sector"], l.get("sector") or "", inp=True)
            sec = f"LOWER(TRIM({g('Sector')}))"
            alln = [n for v in sector_names.values() for n in v]
            B.put(ws, rr, K["Program lending"],
                  "=IF(OR(" + ",".join(f"{sec}=\"{n}\"" for n in alln) + "),\"Yes\",\"No\")"
                  if alln else "No")
            B.put(ws, rr, K["Customer's worst days past due"],
                  f"=MAX(0,{own})" if recorded else f"={R[(x, 'Past due days')]}", "int")
            d = g("Customer's worst days past due")
            prog = "\"\""
            for sec_name, rng in reversed(list(ladders.items())):
                cond = "OR(" + ",".join(f"{sec}=\"{n}\"" for n in sector_names[sec_name]) + ")"
                prog = (f"IF({cond},IFERROR(VLOOKUP({d},{rng},2,TRUE),"
                        f"INDEX({rng},1,2)),{prog})")
            B.put(ws, rr, K["Program rating"],
                  f"=IF({g('Program lending')}=\"Yes\",{prog},\"\")")
            B.put(ws, rr, K["Fallback segment"],
                  f"=IF(AND({g('A grade on the rating scale')}=\"No\","
                  f"{g('Program lending')}=\"No\"),IF(LOWER(TRIM({g('Product')}))="
                  f"\"al dhameen\",\"Al Dhameen Customers\","
                  f"\"Unrated Customer (Internal Rating)\"),\"\")")
            B.put(ws, rr, K["Fallback rating"],
                  f"=IF({g('Fallback segment')}=\"\",\"\",IFERROR(VLOOKUP("
                  f"{g('Fallback segment')},{fbr},2,FALSE),\"\"))")
            B.put(ws, rr, K["Facility rating"],
                  f"=IF({g('A grade on the rating scale')}=\"Yes\",{cmr},"
                  f"IF({g('Program lending')}=\"Yes\",{g('Program rating')},"
                  f"{g('Fallback rating')}))", key=True)
            B.put(ws, rr, K["Set by"],
                  f"=IF({g('A grade on the rating scale')}=\"Yes\",\"Customer master\","
                  f"IF({g('Program lending')}=\"Yes\",\"Program lending: \"&"
                  f"{g('Sector')}&\", \"&{d}&\" days past due\","
                  f"\"Fallback: \"&{g('Fallback segment')}))")
            B.put(ws, rr, K["Hierarchy"],
                  f"=IFERROR(INDEX({sh},MATCH({g('Facility rating')},{sr},0)),\"\")", "int")
        hier = _rng(r0, K["Hierarchy"], r1, K["Hierarchy"])
        sb = _rng(r0, K["Set by"], r1, K["Set by"])
        r = r1 + 2
        B.header(ws, r, 1, ["Customer rating", "Value", "Detail"])
        o = rov.iloc[-1] if len(rov) else None
        rows = [
            ("Worst facility hierarchy", f"=MAX({hier})", "int", ""),
            ("Worst facility rating", f"=IFERROR(INDEX({ig},{_a(r + 1, 2)}),\"\")",
             None, f"=IFERROR(INDEX({sb},MATCH({_a(r + 1, 2)},{hier},0)),\"\")"),
            ("Rating override in the run", "" if o is None else _str(o.value), None,
             "" if o is None else _ov_text(o)),
            ("Customer rating",
             f"=IF({_a(r + 3, 2)}<>\"\",{_a(r + 3, 2)},{_a(r + 2, 2)})", None,
             f"=IF({_a(r + 3, 2)}<>\"\",\"Override in the run\",{_a(r + 2, 3)})"),
            ("Rating in the run (account master)", f"={R[(lend[0]['line'], 'Rating')]}",
             None, ""),
            ("Check", f"=IF({_a(r + 4, 2)}={_a(r + 5, 2)},\"OK\",\"DIFFERS\")", None, ""),
        ]
        for k, (lab, v, kd, det) in enumerate(rows):
            B.put(ws, r + 1 + k, 1, lab, bold=lab == "Customer rating")
            B.put(ws, r + 1 + k, 2, v, kd, inp=lab == "Rating override in the run",
                  key=lab == "Customer rating")
            B.put(ws, r + 1 + k, 3, det)
        cust_rating = _a(r + 4, 2, "Rating")
        for l in lend:
            final[l["line"]] = cust_rating
            setby[l["line"]] = _a(r + 4, 3, "Rating")
        r = r + 8
    # ---- investments: per account
    if inv:
        ws.cell(r, 1, "Investments").font = B.f_sub
        cols = ["No.", "Contract", "Rating in the extract",
                "A grade on the external scale", "Fallback rating", "Rating",
                "Set by", "Rating in the run", "Check"]
        B.header(ws, r + 1, 1, cols)
        for k, l in enumerate(inv):
            rr = r + 2 + k
            x = l["line"]
            g = lambda c: _a(rr, c, fixed=False)
            B.put(ws, rr, 1, f"={R[(x, 'Line')]}", align="left")
            B.put(ws, rr, 2, f"={R[(x, 'Contract')]}")
            B.put(ws, rr, 3, l.get("rating_input") or "", inp=True)
            B.put(ws, rr, 4, f"=IF(SUMPRODUCT(({sr}={g(3)})*({stp}=\"External\"))>0,"
                             f"\"Yes\",\"No\")")
            B.put(ws, rr, 5, f"=IFERROR(VLOOKUP(\"Investment Portfolio\",{fbr},2,FALSE),\"\")")
            if recorded:
                B.put(ws, rr, 6, f"=IF({g(4)}=\"Yes\",{g(3)},{g(5)})", key=True)
                B.put(ws, rr, 7, f"=IF({g(4)}=\"Yes\",\"Investment extract\","
                                 f"\"Fallback: Investment Portfolio\")")
            else:
                B.put(ws, rr, 6, f"={R[(x, 'Rating')]}", key=True)
                B.put(ws, rr, 7, "Not recorded by the run")
            B.put(ws, rr, 8, f"={R[(x, 'Rating')]}")
            B.put(ws, rr, 9, f"=IF({g(6)}={g(8)},\"OK\",\"DIFFERS\")")
            final[x] = _a(rr, 6, "Rating")
            setby[x] = _a(rr, 7, "Rating")
        r = r + 3 + len(inv)
    # ---- rating -> bucket -> PD curve
    ws.cell(r, 1, "PD bucket and curve").font = B.f_sub
    B.header(ws, r + 1, 1, ["No.", "Contract", "Rating", "Rating scale", "Bucket",
                            "PD curve"])
    for k, l in enumerate(L):
        rr = r + 2 + k
        x = l["line"]
        g = lambda c: _a(rr, c, fixed=False)
        B.put(ws, rr, 1, f"={R[(x, 'Line')]}", align="left")
        B.put(ws, rr, 2, f"={R[(x, 'Contract')]}")
        B.put(ws, rr, 3, f"={final[x]}")
        typ = R[(x, "Rating type")]
        B.put(ws, rr, 4, f"=IF(OR(LEFT({typ},8)=\"External\",{typ}=\"2\"),"
                         f"\"External\",\"Internal\")")
        hit = f"({bs}={g(4)})*({br}={g(3)})"
        B.put(ws, rr, 5, f"=IF(SUMPRODUCT({hit})=0,\"\",SUMPRODUCT({hit}*{bb}))", "int")
        B.put(ws, rr, 6, f"=IF({g(5)}=\"\",\"\",IF({pm['source']}=\"{CALC}\",{g(4)},"
                         f"{R[(x, 'Portfolio')]})&\" | bucket \"&{g(5)})")
        R[(x, "Final rating")] = _a(rr, 3, "Rating")
        R[(x, "Rating bucket")] = _a(rr, 5, "Rating")
        R[(x, "PD curve")] = _a(rr, 6, "Rating")
        R[(x, "Rating set by")] = setby[x]
    B.widths(ws, {"A": 34, "B": 16, "C": 22, "D": 16, "E": 36, "F": 14, "G": 12,
                  "H": 14, "I": 12, "J": 30, "K": 12, "L": 16, "M": 42, "N": 10})


def _write_staging(B, ws, facts, L, R, S) -> None:
    """The customer's stage from its criteria, then each facility's."""
    thr = facts["settings"]["threshold"]
    B.title(ws, "Stage: the customer's stage, then each facility's",
            f"A customer is staged on its worst facility: more than 90 days past due "
            f"or the default flag gives Stage 3; more than {thr:g} days past due, the "
            f"watchlist or a restructuring gives Stage 2; otherwise Stage 1. A stage "
            f"override in the run can only worsen it. Each facility takes the "
            f"customer's stage, and a Tasdeer facility is at least Stage 2.")
    ov = facts["overrides"]

    def last(kind):
        o = ov[ov["kind"] == kind] if len(ov) else ov
        return o.iloc[-1] if len(o) else None

    yes = lambda v: "Yes" if _num(v) == 1 else "No"
    r = 4
    for suffix, book in (("_1", "Lending"), ("_2", "Investments")):
        bl = [l for l in L if l["suffix"] == suffix]
        if not bl:
            continue
        f0 = bl[0]
        ws.cell(r, 1, f"{book}: the customer").font = B.f_sub
        B.header(ws, r + 1, 1, ["Criterion", "Value", "Stage it gives", "Detail"])
        so, ro = (last("stage"), last("restructuring")) if suffix == "_1" else (None, None)
        other = any(_num(f0.get(f"IsLocal{i}")) == 1 for i in (2, 4, 5, 6))
        crit = [
            ("dpd", "Days past due (worst facility)",
             "=MAX(0," + ",".join(R[(l['line'], 'Past due days')] for l in bl) + ")",
             "int",
             lambda v: f"=IF(N({v})>{S['s3dpd']},3,IF(N({v})>{S['thr']},2,1))", ""),
            ("watch", "Watchlist", yes(f0.get("IsWatchlist")), None,
             lambda v: f"=IF({v}=\"Yes\",2,1)", ""),
            ("restr", "Restructured", yes(f0.get("IsLocal1")), None,
             lambda v: f"=IF({v}=\"Yes\",2,1)",
             "" if ro is None else f"Override in the run: {_str(ro.value) or 'none'}; "
                                   + _ov_text(ro)),
            ("default", "Default", yes(f0.get("IsDefault")), None,
             lambda v: f"=IF({v}=\"Yes\",3,1)", ""),
            ("lic2", "Stage 2 in the LIC staging file", yes(f0.get("IsLocal3")), None,
             lambda v: f"=IF({v}=\"Yes\",2,1)",
             "Set by the ETL from the criteria above, or a stage override"),
        ]
        if other:
            crit.append(("other", "Other Stage 2 flag in the LIC staging file", "Yes",
                         None, lambda v: f"=IF({v}=\"Yes\",2,1)", ""))
        crit.append(("override", "Stage override in the run",
                     "" if so is None else _str(so.value), None,
                     lambda v: (f"=IF({v}=\"\",1,IFERROR(VALUE(SUBSTITUTE(LOWER({v}),"
                                f"\"stage \",\"\")),1))"),
                     "" if so is None else _ov_text(so)))
        C = {}
        for k, (key, lab, val, kd, rule, det) in enumerate(crit):
            rr = r + 2 + k
            B.put(ws, rr, 1, lab)
            B.put(ws, rr, 2, val, kd, inp=key != "dpd")
            B.put(ws, rr, 3, rule(_a(rr, 2, fixed=False)), "int")
            B.put(ws, rr, 4, det)
            C[key] = (_a(rr, 2), _a(rr, 3))
        rs = r + 2 + len(crit)
        cust = _a(rs, 2)
        B.put(ws, rs, 1, "Customer stage", bold=True)
        B.put(ws, rs, 2, f"=MAX({_rng(r + 2, 3, rs - 1, 3)})", "int", key=True)
        why2 = []
        for key, text in (("override", "Stage override in the run"),
                          ("dpd", f"More than \"&{S['thr']}&\" days past due"),
                          ("watch", "Watchlist"), ("restr", "Restructured"),
                          ("lic2", "Stage 2 in the LIC staging file"),
                          ("other", "Other Stage 2 flag in the LIC staging file")):
            if key in C:
                why2.append((C[key][1], text))
        w2 = "\"Performing\""
        for cell, text in reversed(why2):
            w2 = f"IF({cell}=2,\"{text}\",{w2})"
        w3 = (f"IF({C['dpd'][1]}=3,\"More than \"&{S['s3dpd']}&\" days past due\","
              f"IF({C['override'][1]}=3,\"Stage override in the run\",\"Default\"))")
        B.put(ws, rs + 1, 1, "Why", bold=True)
        B.put(ws, rs + 1, 2, f"=IF({cust}=3,{w3},IF({cust}=2,{w2},\"Performing\"))")
        why = _a(rs + 1, 2)
        # ---- each facility: the customer's criteria on the facility's own
        # days past due (the same for every facility of a customer in an ETL
        # run), the Tasdeer floor, a stage written in the account master, and
        # the customer's worst facility lifting a Stage 1 one
        r = rs + 3
        ws.cell(r, 1, f"{book}: each facility").font = B.f_sub
        cols = ["No.", "Contract", "Portfolio", "Customer's worst days past due",
                "Tasdeer",
                "Stage in the account master", "Customer stage", "Facility stage",
                "Stage", "Reason", "Stage in the report", "Check"]
        K = {n: i + 1 for i, n in enumerate(cols)}
        B.header(ws, r + 1, 1, cols)
        r0, r1 = r + 2, r + 1 + len(bl)
        flag2 = [C[k][0] for k in ("watch", "restr", "lic2", "other") if k in C]
        for k, l in enumerate(bl):
            rr = r0 + k
            x = l["line"]
            g = lambda n: _a(rr, K[n], fixed=False)
            B.put(ws, rr, 1, f"={R[(x, 'Line')]}", align="left")
            B.put(ws, rr, 2, f"={R[(x, 'Contract')]}")
            B.put(ws, rr, 3, f"={R[(x, 'Portfolio')]}")
            B.put(ws, rr, 4, f"={R[(x, 'Past due days')]}", "int")
            B.put(ws, rr, 5, f"=IF({g('Portfolio')}=\"Tasdeer\",\"Yes\",\"No\")")
            man = R[(x, "AccountMaster.Stage")]
            B.put(ws, rr, 6, f"=IFERROR(VALUE(SUBSTITUTE(LOWER({man}),\"stage \",\"\")),\"\")",
                  "int")
            B.put(ws, rr, 7, f"={cust}", "int")
            d = g("Customer's worst days past due")
            t, m = g("Tasdeer"), g("Stage in the account master")
            s3 = f"OR(N({d})>{S['s3dpd']},{C['default'][0]}=\"Yes\")"
            s2 = ("OR(AND(N({d})>{thr},N({d})<={s3d}),".format(d=d, thr=S["thr"], s3d=S["s3dpd"])
                  + ",".join(f"{c}=\"Yes\"" for c in flag2) + ")")
            okm = f"AND(ISNUMBER({m}),{m}>=1,{m}<=3)"
            B.put(ws, rr, 8, f"=IF({s3},3,IF({okm},{m},IF(OR({s2},{t}=\"Yes\"),2,1)))", "int")
            f8 = g("Facility stage")
            B.put(ws, rr, 9, f"=IF(AND({f8}=1,MAX({_rng(r0, 8, r1, 8)})>=2,"
                             f"{t}<>\"Yes\"),2,{f8})", "int", key=True)
            s9 = g("Stage")
            own2 = (f"IF(AND(N({d})>{S['thr']},N({d})<={S['s3dpd']}),\"More than \"&"
                    f"{S['thr']}&\" days past due\","
                    + "".join(f"IF({c}=\"Yes\",\"{lab}\"," for c, lab in zip(
                        flag2, [{"watch": "Watchlist", "restr": "Restructured",
                                 "lic2": "Stage 2 in the LIC staging file",
                                 "other": "Other Stage 2 flag in the LIC staging file"}[kk]
                                for kk in ("watch", "restr", "lic2", "other") if kk in C]))
                    + "\"\"" + ")" * len(flag2) + ")")
            B.put(ws, rr, 10,
                  f"=IF({s9}=3,IF(N({d})>{S['s3dpd']},\"More than \"&{S['s3dpd']}&"
                  f"\" days past due\",\"Default\"),IF({s9}=2,IF({okm},\"Stage \"&{m}&"
                  f"\" in the account master\",IF({s2},{own2},IF({t}=\"Yes\","
                  f"\"Tasdeer (assessed collectively)\",\"Another facility of the "
                  f"customer is Stage 2 or 3\"))),\"Performing\"))")
            B.put(ws, rr, 11, l["rep_stage"], "int", inp=True)
            B.put(ws, rr, 12, f"=IF({g('Stage in the report')}=\"\",\"\",IF({s9}="
                              f"{g('Stage in the report')},\"OK\",\"DIFFERS\"))")
            R[(x, "Stage")] = _a(rr, 9, "Staging")
            R[(x, "Stage reason")] = _a(rr, 10, "Staging")
        r = r1 + 3
    B.widths(ws, {"A": 40, "B": 16, "C": 16, "D": 14, "E": 9, "F": 13, "G": 10,
                  "H": 10, "I": 8, "J": 44, "K": 10, "L": 9})


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


def _weight_bands(B, ws, r, c0, names, zs, Zc, cdf, years) -> dict:
    """The scenario weights, as the scenario probability workbook works them
    out: the scenarios in order of severity, each owning a band; per year the
    normal CDF at each band edge (``cdf(i, y)``, a formula without '='), a
    scenario's weight the CDF between its edges -- below the central scenario
    up to its edge, above it from its edge -- and the central scenario the
    rest. Returns {(scenario, year): weight cell} and {scenario: average}."""
    order = sorted(range(len(names)), key=lambda i: zs[i])
    central = min(range(len(order)), key=lambda p: abs(zs[order[p]]))
    n = len(order)
    hdr = ["Scenario", "Severity z"]
    for y in years:
        hdr += [f"Year {y}: CDF at the edge", f"Year {y}: weight"]
    hdr += ["Weight (average)"]
    B.header(ws, r, c0, hdr)
    W, A = {}, {}
    for p_, i in enumerate(order):
        rr = r + 1 + p_
        B.put(ws, rr, c0, names[i])
        B.put(ws, rr, c0 + 1, f"={Zc[i]}", "num")
        for yi, y in enumerate(years):
            cc, wc = c0 + 2 + 2 * yi, c0 + 3 + 2 * yi
            B.put(ws, rr, cc, "=" + cdf(i, y), "num")
            me = _a(rr, cc, fixed=False)
            if p_ == central:
                above = f"SUM({_rng(r + 1, wc, rr - 1, wc)})" if p_ > 0 else "0"
                below = (f"SUM({_rng(rr + 1, wc, r + n, wc)})" if p_ < n - 1 else "0")
                f = f"=1-{above}-{below}"
            elif p_ < central:
                f = f"={me}" if p_ == 0 else f"={me}-{_a(rr - 1, cc, fixed=False)}"
            else:
                f = f"=1-{me}" if p_ == n - 1 else f"={_a(rr + 1, cc, fixed=False)}-{me}"
            B.put(ws, rr, wc, f, "num")
            W[(i, y)] = _a(rr, wc)
        ac = c0 + 2 + 2 * len(years)
        B.put(ws, rr, ac, f"=AVERAGE({','.join(W[(i, y)] for y in years)})", "num",
              key=True)
        A[i] = _a(rr, ac)
    return {"w": W, "avg": A, "rows": n}


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
    iw = pdm.get("int_weights")
    for k, ((nm, z), w) in enumerate(zip(pdm["scenarios"], pdm["w_int"])):
        rr = r + 2 + k
        B.put(ws, rr, 1, nm, inp=True)
        B.put(ws, rr, 2, z, "num", inp=True)
        B.put(ws, rr, 3, w, "num", inp=iw is None)
        Z.append(_a(rr, 2))
        W.append(_a(rr, 3))
    B.put(ws, r + 2 + len(names), 1, "Total")
    B.put(ws, r + 2 + len(names), 3, f"=SUM({W[0]}:{W[-1]})", "num")
    wrows = [r + 2 + k for k in range(len(names))]
    if iw is None:
        B.note(ws, r + 3 + len(names), 1,
               "The weights as written in model_inputs.yml (mode explicit).")
        r = r + 5 + len(names)
    else:
        # ---- the weights, worked out (mode auto)
        r = r + 4 + len(names)
        ws.cell(r, 1, "Scenario weights").font = B.f_sub
        B.note(ws, r + 1, 1,
               f"Each scenario owns a band of the history of {iw['name']}, cut at mean "
               f"+ standard deviation x severity. Its weight is the chance growth falls "
               f"in its band, centred on the forecast with the history's standard "
               f"deviation, averaged over the first {iw['n']} forecast years; the "
               f"central scenario takes the rest.")
        r += 3
        B.header(ws, r, 1, ["History"])
        hv = iw["history"]
        for k, v in enumerate(hv):
            B.put(ws, r + 1 + k, 1, v, "num", inp=True)
        Hi = _rng(r + 1, 1, r + len(hv), 1)
        B.put(ws, r, 3, "Mean", bold=True)
        B.put(ws, r, 4, f"=AVERAGE({Hi})", "num")
        B.put(ws, r + 1, 3, "Standard deviation", bold=True)
        B.put(ws, r + 1, 4, f"=STDEV({Hi})", "num")
        mu, sd = _a(r, 4), _a(r + 1, 4)
        fcell = {}
        for y in range(1, iw["n"] + 1):
            B.put(ws, r + 1 + y, 3, f"Forecast, year {y}", bold=True)
            if iw["in_table"] and (y, iw["column"] - 1) in F:
                B.put(ws, r + 1 + y, 4, f"={F[(y, iw['column'] - 1)]}", "num")
            else:
                B.put(ws, r + 1 + y, 4, iw["forecasts"][y - 1], "num", inp=True)
            fcell[y] = _a(r + 1 + y, 4)
        zs = [z for _, z in pdm["scenarios"]]
        bands = _weight_bands(
            B, ws, r + 3 + iw["n"], 3, names, zs, Z,
            lambda i, y: f"NORMDIST({mu}+{sd}*{Z[i]},{fcell[y]},{sd},TRUE)",
            list(range(1, iw["n"] + 1)))
        for k in range(len(names)):
            B.put(ws, wrows[k], 3, f"={bands['avg'][k]}", "num")
        r = max(r + 1 + len(hv), r + 4 + iw["n"] + bands["rows"]) + 2
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
    auto = bool(pdm.get("ext_auto"))
    for i, row in enumerate(pdm["w_ext"]):
        rr = r3 + 2 + i
        B.put(ws, rr, 4, i + 1, "int")
        for k, w in enumerate(row):
            B.put(ws, rr, 5 + k, w, "num", inp=not auto)
        B.put(ws, rr, 5 + len(names), f"=SUM({_rng(rr, 5, rr, 4 + len(names))})", "num")
        WR.append(rr)
    ra = r3 + 2 + len(WR)
    rb = ra + 3
    if auto:
        # each year's weights, worked out: where that year's forecast sits
        B.put(ws, r + 5, 4, "Mean", bold=True)
        B.put(ws, r + 5, 5, f"=AVERAGE({H})", "num")
        MU = _a(r + 5, 5)
        ws.cell(rb, 4, "Scenario weights, worked out").font = B.f_sub
        zs = [z for _, z in pdm["scenarios"]]
        zc = []
        B.put(ws, rb + 1, 4, "Scenario", bold=True)
        B.put(ws, rb + 2, 4, "Severity z", bold=True)
        for k, (nm, z) in enumerate(zip(names, zs)):    # the severities, as inputs
            B.put(ws, rb + 1, 5 + k, nm)
            B.put(ws, rb + 2, 5 + k, z, "num", inp=True)
            zc.append(_a(rb + 2, 5 + k))
        yrs = list(range(1, len(WR) + 1))
        bands = _weight_bands(
            B, ws, rb + 4, 4, names, zs, zc,
            lambda i, y: f"NORMDIST({Fc[y - 1]}+{zc[i]}*{SD},{MU},{SD},TRUE)", yrs)
        for i in range(len(WR)):
            for k in range(len(names)):
                B.put(ws, WR[i], 5 + k, f"={bands['w'][(k, i + 1)]}", "num")
        rb = rb + 5 + bands["rows"]
    B.put(ws, ra, 4, "Later years (average)")
    for k in range(len(names)):
        B.put(ws, ra, 5 + k, f"=AVERAGE({_rng(WR[0], 5 + k, WR[-1], 5 + k)})", "num")
    B.put(ws, ra, 5 + len(names), f"=SUM({_rng(ra, 5, ra, 4 + len(names))})", "num")
    B.note(ws, ra + 1, 4,
           ("Worked out below for each year from where that year's forecast sits on "
            "the history." if auto else "As written in model_inputs.yml.")
           + " Years after the last forecast year take the average.")

    def weight_of(k, y):
        row = WR[y - 1] if y <= len(WR) else ra
        return _a(row, 5 + k)

    r = max(r + 2 + len(h), rb) + 1
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
            "evenly over its twelve months and summed, capped at 1. A facility reads "
            "the curve of its rating scale and bucket (Rating tab). The run's own "
            "curves (StPD.csv) are beside them for the check; the switch chooses "
            "which ones the ECL prices on.")
    B.put(ws, 4, 1, "PD curves used in the ECL", bold=True)
    sw = B.put(ws, 4, 2, pdm.get("source") or RUN, inp=True, key=True)
    from openpyxl.worksheet.datavalidation import DataValidation
    dv = DataValidation(type="list", formula1=f'"{CALC},{RUN}"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add(sw)
    B.note(ws, 4, 3, pdm.get("reason") or (
        "The calculated curves equal the run's StPD: the ECL prices on the calculation."))
    out = {"source": _a(4, 2, "PD curves")}
    last = 4

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


def _write_summary(B, ws, facts, L, ln, stage, lg, ead, ecl, pm, S):
    who = (f"customer {facts['customer_id']}"
           + (f" ({facts['customer_name']})" if facts["customer_name"] else ""))
    if facts.get("focus"):
        cs = ", ".join(dict.fromkeys(l["cid"] for l in L))
        B.title(ws, f"ECL calculator: contract {cs}, {who}")
    else:
        B.title(ws, f"ECL calculator: {who}")
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
    B.note(ws, 12, 1, "How to read it, tab by tab: Facilities (the inputs) -> Rating "
           "(how each rating was set) -> Staging (the customer's stage, then each "
           "facility's) -> PD internal / PD external (the PD model and scenario "
           "weights) -> PD curves -> LGD -> EAD -> ECL. Blue cells are the run's "
           "inputs, black cells formulas: change an input and the provision follows. "
           "The settings are on a hidden tab."
           + (" Facilities, Rating and Staging show every facility of the customer, "
              "since the customer's worst facility sets the contract's rating and any "
              "Stage 2 or 3 facility lifts it; LGD, EAD, ECL and this tab show the "
              "contract." if facts.get("focus") else ""))
    ws.cell(21, 1, "Per position").font = B.f_sub
    hr = 22
    cols = ["Line", "Book", "Contract", "Original contract", "Product", "Portfolio",
            "Rating", "Rating set by", "Stage", "Why",
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
        B.put(ws, rr, U["Original contract"], f"={ln(x, 'Original contract')}")
        B.put(ws, rr, U["Product"], f"={ln(x, 'Account type')}")
        B.put(ws, rr, U["Portfolio"], f"={ln(x, 'Portfolio')}")
        B.put(ws, rr, U["Rating"], f"={ln(x, 'Final rating')}")
        B.put(ws, rr, U["Rating set by"], f"={ln(x, 'Rating set by')}")
        B.put(ws, rr, U["Stage"], f"={stage(x)}", "int")
        B.put(ws, rr, U["Why"], f"={ln(x, 'Stage reason')}")
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
        # a position the model cannot price (no interest rate, say) reads
        # "no ECL" rather than an error, so the totals still add up
        B.put(ws, rr, U["ECL (this workbook)"],
              f"=IFERROR(IF({g('Stage')}=3,IF({S['stage3']}=\"zero\",0,MAX(0,{onb})),"
              f"IF(AND({S['cap']},{onb}>0),MIN({tot},{onb}),{tot})),\"no ECL\")",
              "amt", key=True)
        B.put(ws, rr, U["ECL (report)"], l["rep_ecl"], "amt", inp=True)
        B.put(ws, rr, U["Difference"],
              f"=IF(OR({g('ECL (report)')}=\"\",NOT(ISNUMBER({g('ECL (this workbook)')}))),"
              f"\"\",{g('ECL (this workbook)')}-{g('ECL (report)')})", "amt")
        B.put(ws, rr, U["Overlay"], l["overlay"] if l["overlay"] is not None else 0,
              "amt", inp=True)
        B.put(ws, rr, U["ECL final"],
              f"=IF(ISNUMBER({g('ECL (this workbook)')}),{g('ECL (this workbook)')}"
              f"+{g('Overlay')},\"\")", "amt")
        B.put(ws, rr, U["Report final"], l["rep_final"], "amt", inp=True)
        tol = lambda a, b: f"ABS({a}-{b})<=MAX(0.01,1E-9*ABS({b}))"
        B.put(ws, rr, U["Check"],
              f"=IF({g('ECL (report)')}=\"\",\"not in report\","
              f"IF(NOT(ISNUMBER({g('ECL (this workbook)')})),\"DIFFERS\",IF(AND("
              f"{tol(g('ECL (this workbook)'), g('ECL (report)'))},OR("
              f"{g('Report final')}=\"\",{tol(g('ECL final'), g('Report final'))})),"
              f"\"OK\",\"DIFFERS\")))")
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
    B.widths(ws, {_col_letter(U[n]): w for n, w in (
        ("Line", 7), ("Book", 11), ("Contract", 15), ("Original contract", 20),
        ("Product", 14), ("Portfolio", 15), ("Rating", 9), ("Rating set by", 34),
        ("Stage", 7), ("Why", 34), ("On balance", 15), ("Net collateral", 15),
        ("LGD", 9), ("PD lifetime", 12), ("Horizon", 9), ("ECL (this workbook)", 17),
        ("ECL (report)", 17), ("Difference", 11), ("Overlay", 11), ("ECL final", 15),
        ("Report final", 15), ("Check", 9))})


def customer_ecl_workbook(run_dir, customer_id, dest=None):
    """The calculator for one customer of a run: bytes, or written to
    ``dest``."""
    return write_customer_workbook(customer_ecl_facts(run_dir, customer_id), dest)


def contract_ecl_facts(run_dir, contract_id, book: str | None = None) -> dict:
    """:func:`customer_ecl_facts` for the contract's customer, with ``focus``
    on the contract's line(s)."""
    from .etl.report import read_output_csv, _chr, _col
    out_dir = _out_dir(_run_root(run_dir))
    want = as_id(pd.Series([contract_id])).iloc[0]
    cust = None
    for suffix, b in (("_1", "Lending"), ("_2", "Investments")):
        if book and book != b:
            continue
        am = read_output_csv(out_dir, f"AccountMaster{suffix}.csv")
        if am is None or len(am) == 0:
            continue
        hit = as_id(_chr(_col(am, "ContractId"))) == want
        if hit.any():
            cust = as_id(_chr(_col(am, "CustomerId")))[hit].iloc[0]
            book = b
            break
    if cust is None:
        raise KeyError(f"Contract {contract_id!r} has no position in this run")
    facts = customer_ecl_facts(run_dir, cust)
    facts["focus"] = [l["line"] for l in facts["lines"]
                      if l["contract"] == want and l["book"] == book]
    if not facts["focus"]:
        raise KeyError(f"Contract {contract_id!r} has no position in this run")
    return facts


def contract_ecl_workbook(run_dir, contract_id, book: str | None = None, dest=None):
    """The calculator for one contract of a run: its customer's facilities
    on Facilities, Rating and Staging, the contract alone from LGD to the
    Summary. Bytes, or written to ``dest``."""
    return write_customer_workbook(contract_ecl_facts(run_dir, contract_id, book),
                                   dest)
