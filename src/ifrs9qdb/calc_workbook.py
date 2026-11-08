"""
One customer's -- or one contract's -- ECL as an Excel calculator: the run's
inputs, and every step from them to the provision as a live formula.

``customer_ecl_workbook(run_dir, customer_id)`` and
``contract_ecl_workbook(run_dir, contract_id)`` return the .xlsx; the layout
(Summary, Facilities, Rating, Staging, PD, LGD, EAD, ECL, and the hidden
Assumptions and PD calculation) is :mod:`ifrs9qdb.calc_book`'s. This module
gathers the facts with the engine's own functions.

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

MAX_MONTH = 600          # EclConfig.max_month: the PD curves run to month 600
# the repayment shapes, as a reader would name them
SHAPE_LABEL = {"bullet": "Bullet", "linear": "Equal instalments",
               "annuity": "Annuity"}
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
        "books": _books(lines), "ttc_table": _ttc_table(static),
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


FLAGS = ("IsWatchlist", "IsDefault", "IsLocal1", "IsLocal2", "IsLocal3", "IsLocal4",
         "IsLocal5", "IsLocal6")


def _books(lines: list) -> dict:
    """Per book, what the customer's staging and rating read across all its
    facilities: the worst days past due, the staging flags, the Tasdeer and
    Al Dhameen facilities, a stage set in the account master on another
    facility, the rating in the customer master and the sector. A contract's
    workbook shows these as values, without the other facilities."""
    out = {}
    for s in ("_1", "_2"):
        ls = [l for l in lines if l["suffix"] == s]
        if not ls:
            continue

        def mx(f, rows=ls):
            v = [l[f] for l in rows if l.get(f) is not None]
            return max(v) if v else None
        tas = [l for l in ls if l["portfolio"] == "Tasdeer"]
        n_ad = sum(1 for l in ls if l["account_type"].strip().lower() == "al dhameen")
        out[s] = {"dpd_worst": mx("dpd") or 0, **{f: mx(f) for f in FLAGS},
                  "n_tasdeer": len(tas),
                  "manual_max": mx("stage_manual", [l for l in ls if l not in tas]),
                  "n_aldhameen": n_ad, "n_other": len(ls) - n_ad,
                  "rating_input": next((l["rating_input"] for l in ls
                                        if l.get("rating_input")), ""),
                  "sector": next((l["sector"] for l in ls if l.get("sector")), "")}
    return out


def _ttc_table(static) -> list:
    """(scale, bucket, rating, TTC PD) for both scales: the TTC PD table read
    through each rating's position, as the PD curves are built."""
    out = []
    if static is None:
        return out
    for rt, label in ((1, "Internal"), (2, "External")):
        try:
            ttc, rmap = static.ttc_for(rt), static.ratings_for(rt)
        except Exception:
            continue
        hier = dict(zip(rmap["rating"].astype(str),
                        pd.to_numeric(rmap["hierarchy"], errors="coerce")))
        seen = set()
        for r in ttc.itertuples():
            b = hier.get(str(r.rating))
            if b is None or not np.isfinite(b) or b in seen:
                continue
            seen.add(b)
            out.append((label, int(b) if float(b).is_integer() else float(b),
                        str(r.rating), float(r.ttc_pd)))
    return out


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
    None. The layout is :mod:`ifrs9qdb.calc_book`'s."""
    from .calc_book import write_book
    return write_book(facts, dest)


# ======================================================================
# Helpers the layout (calc_book) uses
# ======================================================================
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


def _trunc3(x: str) -> str:
    """Truncated (not rounded) to 3 significant digits, as PERCENTRANK.EXC
    returns its rank: x is a number in (0, 1)."""
    return f"ROUNDDOWN({x},2-INT(LOG10({x})))"


def _fallback_n(l) -> int:
    from .engine import months_to_maturity
    return months_to_maturity(l["d_maturity"], l["d_extract"])


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
