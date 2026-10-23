"""How a scenario reaches the provision: every move becomes a ``reprice``
input, and the book is priced once with all of them, through the same
``etl.report.compute_ecl`` the run used.

The layers, in the order they are applied and shown in the waterfall:

  1 PD model drivers   the paths of the variables the PD models use (non-oil
                       GDP for the internal book, each GCC country's growth
                       for the external book) go into the run's frozen config
                       and the PD chain is rebuilt (``stress.pd_chain``)
  2 credit factor      every other variable moves each segment's systematic
                       factor by its loadings; each year's PD moves in probit
                       space by -sqrt(rho / (1 - rho)) x the factor's move
                       (``stress.shift_curve``)
  3 ratings, stages    the stressed 12-month PD's implied grade on the
                       contract's scale, plus the direct downgrades below: a
                       Stage 1 contract implied to fall ``sicr_notches`` or
                       more moves to Stage 2
  4 collateral         property and equity collateral revalued by their
                       index's largest cumulative move, so LGD moves through
                       the engine's own formula
  5 rates              floating-rate debt service: notches per point of the
                       highest rate rise; the EIR too, when the config says
  6 exposure           drawdown on undrawn limits when credit tightens
  7 external book      sovereign and bank notches from oil and the fiscal
                       balance
  8 parameter levers   a ``StressSpec`` on top (``stress.spec_levers``)

Nothing here changes the engine or a run: the baseline is the run's own
report repriced (or, for a run priced by an earlier version of the
calculation, its config rebuilt on this one -- as the macro screen does).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..macro.catalogue import concepts as _concepts
from ..stress import (REPRICE_KEYS, StressSpec, _ingredients, frozen_config,
                      pd_chain, reprice, shift_curve, spec_levers, stress_scope,
                      summarise_repricing)
from .config import read_stress_config, segment_loadings
from .scenarios import Scenario, scenario_paths

__all__ = ["RunContext", "prepare", "transmit", "LAYERS", "layer_inputs",
           "collateral_shares", "basel_rho", "customer_sectors"]

LAYERS = [("drivers", "PD model drivers"), ("factor", "Credit factor"),
          ("staging", "Ratings and stages"), ("collateral", "Collateral"),
          ("rates", "Rates"), ("exposure", "Exposure"),
          ("external", "External book"), ("levers", "Parameter levers")]


def _dt_year() -> int:
    import datetime as _dt
    return _dt.date.today().year


def basel_rho(pd12) -> np.ndarray:
    """The Basel corporate asset correlation at a 12-month PD, as the
    external PD chain uses."""
    p = np.clip(np.asarray(pd12, dtype=float), 1e-6, 0.999)
    return 0.24 - 0.12 * (1 - np.exp(-50 * p)) / (1 - np.exp(-50))


def customer_sectors(run_path, static=None) -> tuple[dict, str]:
    """Customer id -> sector, and where it came from.

    A run written by this version keeps ``stress/customer_sectors.csv``
    (the pipeline's own industry lookup). For an earlier run the industry
    extract is read again from the input folder its audit log names, with
    the run's frozen industry mapping -- right as long as that folder has not
    been overwritten by a later quarter's files, which is said."""
    run_path = Path(run_path)
    p = run_path / "stress" / "customer_sectors.csv"
    if p.is_file():
        t = pd.read_csv(p, dtype=str, keep_default_na=False)
        if {"CustomerId", "Sector"} <= set(t.columns):
            return ({k.strip(): v.strip() for k, v in zip(t["CustomerId"], t["Sector"])
                     if v.strip()}, "run")
    try:
        import json
        from ..etl.lending import _customer_sector
        from ..etl.read_inputs import INPUT_SPECS, read_input, resolve_input_path
        line = (run_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0]
        idir = Path(json.loads(line).get("input_dir", ""))
        spec = next(s for s in INPUT_SPECS if s.name == "IndustryCode")
        f = resolve_input_path(idir, spec) if idir.is_dir() else None
        if f is None or static is None:
            return {}, ""
        ind = read_input(f)
        ind.columns = [str(c).strip().upper().replace(" ", "") for c in ind.columns]
        ids = ind.get("CUSTOMERID")
        if ids is None:
            return {}, ""
        ids = ids.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
        ind["CUSTOMERID"] = ids
        sec = _customer_sector(ids.drop_duplicates().reset_index(drop=True), ind,
                               static.get("industry_sector_mapping"))
        return ({k: v for k, v in zip(ids.drop_duplicates(), sec) if isinstance(v, str)},
                "inputs")
    except Exception:
        return {}, ""


def collateral_shares(out_dir, patterns: dict) -> dict:
    """For each concept, each contract's share of its net collateral (value
    x allocation x (1 - haircut), as the engine nets it) whose type's
    description matches the concept's pattern."""
    from ..etl.report import _chr, _col, _num, read_output_csv
    alloc = read_output_csv(out_dir, "AccountCollateralAllocation.csv")
    coll = read_output_csv(out_dir, "Collateral.csv")
    ctype = read_output_csv(out_dir, "CollateralType.csv")
    if alloc is None or coll is None or ctype is None:
        return {}
    tid = _chr(_col(ctype, "CollateralTypeId")).to_numpy()
    hc = pd.Series(_num(_col(ctype, "HaircutGeneral")).to_numpy(), index=tid)
    hc = hc[~hc.index.duplicated(keep="first")]
    desc = pd.Series(_chr(_col(ctype, "Description")).fillna("").to_numpy(), index=tid)
    desc = desc[~desc.index.duplicated(keep="first")]
    cid = _chr(_col(coll, "CollateralId")).to_numpy()
    cval = pd.Series(_num(_col(coll, "CollateralValue")).to_numpy(), index=cid)
    cval = cval[~cval.index.duplicated(keep="first")]
    ctyp = pd.Series(_chr(_col(coll, "CollateralTypeId")).to_numpy(), index=cid)
    ctyp = ctyp[~ctyp.index.duplicated(keep="first")]
    a_cid = _chr(_col(alloc, "ContractId"))
    a_clid = _chr(_col(alloc, "CollateralId"))
    a_pct = _num(_col(alloc, "AllocationPercentage"))
    typ = a_clid.map(ctyp)
    v = a_clid.map(cval).astype(float)
    h = typ.map(hc).astype(float)
    contrib = (v * a_pct * (1 - h)).where(lambda x: np.isfinite(x), 0.0)
    d = typ.map(desc).fillna("")
    keep = a_cid.notna()
    tot = contrib[keep].groupby(a_cid[keep]).sum()
    out = {}
    for concept, pat in (patterns or {}).items():
        try:
            rx = re.compile(pat)
        except re.error:
            continue
        hit = d.map(lambda s: bool(rx.search(str(s)))) & keep
        part = contrib[hit].groupby(a_cid[hit]).sum()
        share = (part / tot.reindex(part.index)).where(lambda x: np.isfinite(x), 0.0)
        out[concept] = share.clip(0, 1).to_dict()
    return out


@dataclass
class RunContext:
    """Everything about a run a scenario needs, read once."""
    inputs: object
    report: pd.DataFrame
    run_path: Path
    cfg: dict
    model: object
    H: int
    first_year: int
    ing: pd.DataFrame
    segment: np.ndarray
    external: np.ndarray
    rho: np.ndarray
    pd12: np.ndarray
    ladders: dict
    grade0: np.ndarray
    coll_share: dict
    drivers_internal: dict      # concept -> (column, multiplier)
    drivers_external: dict      # concept -> country
    base_curves: dict | None
    base: pd.DataFrame
    notes: list = field(default_factory=list)

    @property
    def internal_rows(self) -> np.ndarray:
        return ~self.external


def _curve12(cm) -> float:
    if cm is None or len(cm) == 0:
        return float("nan")
    c = np.asarray(cm, dtype=float)
    return float(c[min(12, c.size - 1)])


def _ladders(static) -> dict:
    t = static.get("ttc_pd_table") if isinstance(static, dict) else None
    out = {}
    if t is None or not len(t):
        return out
    tt = pd.DataFrame(t)
    rt = pd.to_numeric(tt["rating_type"], errors="coerce")
    pdv = pd.to_numeric(tt["ttc_pd"], errors="coerce")
    for r in sorted(rt.dropna().unique()):
        v = np.sort(np.unique(pdv[(rt == r) & (pdv > 0)].to_numpy()))
        out[int(r)] = v
    return out


def _grade(p, ladder) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    if ladder is None or len(ladder) == 0:
        return np.zeros(p.shape, dtype=int)
    lp = np.log(ladder)
    return np.abs(np.log(p)[:, None] - lp[None, :]).argmin(axis=1)


def _rt(v) -> int:
    t = str(v)
    if t.startswith("External") or t in ("2", "2.0"):
        return 2
    return 1


def prepare(inputs, report: pd.DataFrame, run_path, store, model=None,
            cfg: dict | None = None, project_root=None) -> RunContext:
    """Read a run for stress scenarios: its contracts, segments, curves,
    collateral, PD model drivers and the macro model."""
    from ..macro.var import fit_macro_model
    run_path = Path(run_path)
    cfg = cfg or read_stress_config(project_root)
    H = int(cfg.get("horizon_years", 5))
    ing = _ingredients(inputs, report)
    out_dir = inputs.out_dir
    notes = []

    fz = frozen_config(run_path)
    sect, how = customer_sectors(run_path, fz.get("static") if fz.get("ok") else None)
    ext_pf = set(inputs.external_portfolios())
    external = (ing["portfolio"].isin(ext_pf) |
                ing["rating_type"].map(_rt).eq(2)).to_numpy()
    if sect:
        seg = np.array([sect.get(str(c).strip(), "Others") for c in ing["customer"]], dtype=object)
        if how == "inputs":
            notes.append("Sectors were read from the input folder this run was built "
                         "from; they are this run's only if that folder still holds "
                         "its files.")
    else:
        seg = ing["portfolio"].astype(str).to_numpy().astype(object)
        notes.append("This run has no customer sectors (CustomerSector.csv), so the "
                     "credit factor is set by portfolio, on the default loadings.")
    seg[external] = (cfg.get("factor") or {}).get("external_segment", "External book")

    ctx = inputs.ctx
    base_curves = None
    ch = pd_chain(inputs, report, run_path, weight_mode="hold")
    if ch.get("ok") and ch.get("base_curves") is not None:
        base_curves = ch["base_curves"]
        notes.append("This run was priced by another version of the calculation, so "
                     "scenarios are measured from its config rebuilt on this one.")
        ctx = ctx.copy(stpd=base_curves)
    curves = [ctx.pd_curve(r.portfolio, r.rating, r.rating_type)
              for r in ing[["portfolio", "rating", "rating_type"]].itertuples(index=False)]
    pd12 = np.array([_curve12(c) for c in curves])
    rho_cfg = (cfg.get("factor") or {}).get("rho", "basel")
    rho = basel_rho(np.nan_to_num(pd12, nan=0.01)) if str(rho_cfg) == "basel" \
        else np.full(len(ing), float(rho_cfg))

    ladders = _ladders(fz["static"]) if fz.get("ok") else {}
    rts = ing["rating_type"].map(_rt).to_numpy()
    grade0 = np.zeros(len(ing), dtype=int)
    for r, lad in ladders.items():
        m = rts == r
        if m.any():
            grade0[m] = _grade(np.nan_to_num(pd12[m], nan=1e-4), lad)

    # PD model drivers, from the frozen model definition
    di, de = {}, {}
    cat = _concepts(project_root)
    if fz.get("ok"):
        model_cfg = fz["model"] or {}
        varcfg = model_cfg.get("variables") or {}
        active = set()
        for m in (model_cfg.get("models") or {}).values():
            if str((m or {}).get("rating_type", "")).lower().startswith("internal"):
                for comp in (m.get("mev_components") or []):
                    if float(comp.get("weight") or 0) != 0:
                        active.add(comp.get("variable"))
        for _, r in cat.iterrows():
            mv = str(r["model_variable"] or "")
            if mv.startswith("GCC:"):
                de[r["concept"]] = mv[4:]
            elif mv in varcfg:
                fs = (varcfg[mv] or {}).get("forecast_source") or {}
                col = fs.get("column_index")
                mult = float((varcfg[mv] or {}).get("stress_unit_multiplier") or 1.0)
                if col:
                    di[r["concept"]] = (int(col), mult, mv in active)
    if model is None:
        model = fit_macro_model(store, project_root=project_root,
                                min_years=int((cfg.get("macro_model") or {}).get("min_years", 12)),
                                tightness=float((cfg.get("macro_model") or {}).get("tightness", 0.2)),
                                cross=float((cfg.get("macro_model") or {}).get("cross", 0.5)),
                                start_year=int((cfg.get("macro_model") or {}).get("start_year", 1990)),
                                extra=list(cat["concept"]))

    base = reprice(inputs, report, pd_curves=base_curves)

    mi = (fz.get("raw") or {}) if fz.get("ok") else {}
    first = int(((mi.get("macro_source") or {}).get("first_forecast_year")) or 0) or None
    if first is None:
        ed = pd.to_datetime(report["extract_date"], errors="coerce").dropna() \
            if "extract_date" in report.columns else []
        first = int(ed.iloc[0].year) if len(ed) else _dt_year()
    rc = RunContext(inputs=inputs, report=report, run_path=run_path, cfg=cfg, model=model,
                    H=H, first_year=first, ing=ing, segment=seg, external=external,
                    rho=rho, pd12=pd12, ladders=ladders, grade0=grade0,
                    coll_share=collateral_shares(out_dir, cfg.get("collateral") or {}),
                    drivers_internal=di, drivers_external=de,
                    base_curves=base_curves, base=base, notes=notes)
    return rc


# ------------------------------------------------------------------ layers --
def _sd(rc, c) -> float:
    s = rc.model.sd.get(c) if hasattr(rc.model, "sd") else None
    return float(s) if s and np.isfinite(s) and s > 0 else 1.0


def _level_move(dev) -> float:
    """A % change path's largest cumulative level move, in per cent."""
    lvl = np.cumprod(1.0 + np.asarray(dev, float) / 100.0) - 1.0
    j = int(np.argmax(np.abs(lvl))) if lvl.size else 0
    return float(100.0 * lvl[j]) if lvl.size else 0.0


def driver_edits(rc: RunContext, paths: dict) -> tuple[dict, dict]:
    mev, gcc = {}, {}
    for c, (col, mult, _active) in rc.drivers_internal.items():
        p = np.asarray(paths.get(c, np.zeros(rc.H)), float)
        if np.any(np.abs(p) > 1e-12):
            mev[col] = list(np.asarray(mev.get(col, np.zeros(rc.H))) + p / mult)
    for c, country in rc.drivers_external.items():
        p = np.asarray(paths.get(c, np.zeros(rc.H)), float)
        if np.any(np.abs(p) > 1e-12):
            gcc[country] = list(p)
    return mev, gcc


def follow_rates(rc: RunContext, paths: dict) -> dict:
    """Qatar's rate follows the US rate through the peg: when a scenario
    moves only the US rate, the domestic one takes its path."""
    rcfg = rc.cfg.get("rates") or {}
    v, f = rcfg.get("variable"), rcfg.get("follows")
    if v and f and f in paths and np.any(np.asarray(paths[f]) != 0) and \
            not np.any(np.asarray(paths.get(v, 0.0)) != 0):
        paths = dict(paths)
        paths[v] = np.asarray(paths[f], float)
    return paths


def factor_moves(rc: RunContext, paths: dict) -> dict:
    """Each segment's factor move per year: sum of loading x move / sd,
    leaving out the variables the PD model already uses."""
    paths = follow_rates(rc, paths)
    driver_c = {c for c, v in rc.drivers_internal.items() if v[2]}
    out = {}
    for seg in pd.unique(rc.segment):
        ext = seg == (rc.cfg.get("factor") or {}).get("external_segment", "External book")
        load = segment_loadings(rc.cfg, seg, ext)
        skip = set(rc.drivers_external) if ext else driver_c
        z = np.zeros(rc.H)
        for c, b in load.items():
            if c in skip or c not in paths:
                continue
            z += b * np.asarray(paths[c], float)[:rc.H] / _sd(rc, c)
        out[seg] = z
    return out


def factor_shift(rc: RunContext, zmoves: dict) -> np.ndarray:
    """Per contract, each year's probit shift of its PD."""
    k = np.sqrt(rc.rho / (1 - rc.rho))
    z = np.vstack([zmoves.get(s, np.zeros(rc.H)) for s in rc.segment])
    return -k[:, None] * z


def stressed_pd12(rc: RunContext, pd_curves=None, shift=None) -> np.ndarray:
    """Each contract's 12-month PD on the curves given (the run's, or a
    rebuilt chain's) with its yearly probit shift, at its own rating."""
    ctx = rc.inputs.ctx.copy(stpd=pd_curves) if pd_curves is not None else rc.inputs.ctx
    out = np.empty(len(rc.ing))
    memo: dict = {}
    sh = None if shift is None else np.asarray(shift, float)
    for i, r in enumerate(rc.ing[["portfolio", "rating", "rating_type"]]
                          .itertuples(index=False)):
        cm = ctx.pd_curve(r.portfolio, r.rating, r.rating_type)
        if cm is None:
            out[i] = np.nan
            continue
        if sh is None or not np.any(sh[i] != 0):
            out[i] = _curve12(cm)
            continue
        key = (id(cm), sh[i].tobytes())
        if key not in memo:
            memo[key] = _curve12(shift_curve(cm, sh[i]))
        out[i] = memo[key]
    return out


def implied_stage(rc: RunContext, pd12s, direct=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stages after the scenario's downgrades: the implied fall of the
    12-month PD on the contract's scale plus any direct notches; a Stage 1
    contract down ``sicr_notches`` or more goes to Stage 2. Returns the
    stages, which moved, and the implied notches."""
    rts = rc.ing["rating_type"].map(_rt).to_numpy()
    g1 = rc.grade0.copy()
    for r, lad in rc.ladders.items():
        m = rts == r
        if m.any():
            g1[m] = _grade(np.nan_to_num(np.asarray(pd12s)[m], nan=1e-4), lad)
    implied = np.maximum(0, g1 - rc.grade0)
    notches = implied + (0 if direct is None else np.asarray(direct))
    sicr = int((rc.cfg.get("staging") or {}).get("sicr_notches", 3))
    stage = rc.ing["stage"].fillna(2).astype(int).to_numpy().copy()
    to2 = (stage == 1) & (notches >= sicr)
    stage[to2] = 2
    return stage, to2, implied


def rate_notches(rc: RunContext, paths: dict) -> dict:
    """The rates channel: the rate path used, its highest rise, the notches
    and the contracts they reach."""
    H, rcfg = rc.H, rc.cfg.get("rates") or {}
    rv = np.asarray(paths.get(rcfg.get("variable", "QA_RATE"), np.zeros(H)), float)
    if not np.any(rv != 0) and rcfg.get("follows"):
        rv = np.asarray(paths.get(rcfg["follows"], np.zeros(H)), float)
    rise = float(max(0.0, rv.max())) if rv.size else 0.0
    n = int(round(float(rcfg.get("notches_per_point", 0.0)) * rise))
    rows = rc.ing["portfolio"].isin(list(rcfg.get("portfolios") or [])).to_numpy() \
        & ~rc.external
    return {"path": rv, "rise": rise, "notches": n, "rows": rows}


def external_notches(rc: RunContext, paths: dict) -> dict:
    ecfg = rc.cfg.get("external") or {}
    idx = 0.0
    for c, w in (ecfg.get("notches_per_sd") or {}).items():
        p = np.asarray(paths.get(c, np.zeros(rc.H)), float)
        idx += float(w) * float(min(0.0, (p / _sd(rc, c)).min() if p.size else 0.0))
    n = int(min(int(ecfg.get("max_notches", 6)), max(0, round(-idx))))
    return {"index": idx, "notches": n}


def collateral_multiplier(rc: RunContext, paths: dict) -> tuple[np.ndarray, dict]:
    n = len(rc.ing)
    mult = np.ones(n)
    moves = {}
    for c, shares in rc.coll_share.items():
        p = paths.get(c)
        if p is None or not np.any(np.asarray(p) != 0):
            continue
        mv = _level_move(p)
        moves[c] = mv
        mult += rc_share(rc, c) * mv / 100.0
    return mult, moves


def rc_share(rc: RunContext, concept: str) -> np.ndarray:
    """Each contract's share of net collateral moving with a concept."""
    key = f"_share_{concept}"
    cache = rc.__dict__.setdefault("_cache", {})
    if key not in cache:
        shares = rc.coll_share.get(concept) or {}
        cache[key] = np.array([shares.get(str(x), 0.0) if isinstance(x, str) else 0.0
                               for x in rc.ing["cid"]])
    return cache[key]


def drawdown(rc: RunContext, paths: dict) -> tuple[np.ndarray, float]:
    xcfg = rc.cfg.get("ead") or {}
    v = xcfg.get("variable", "CREDIT_QAT")
    cp = np.asarray(paths.get(v, np.zeros(rc.H)), float)
    zc = float((cp / _sd(rc, v)).min()) if cp.size else 0.0
    pct = -float(xcfg.get("drawdown_per_sd", 0.0)) * zc if zc < 0 else 0.0
    rows = rc.ing["portfolio"].isin(list(xcfg.get("portfolios") or [])).to_numpy()
    return rows, pct


def _family(r: str) -> str:
    """Moody's-style grades (Baa1) and S&P-style ones (BBB+) share an
    external scale's hierarchy; a downgrade keeps the notation it starts in."""
    return "m" if re.search(r"[a-z0-9]", str(r)) else "s"


def notch_grade(scale, rating: str, k: int) -> str:
    """``k`` grades down a scale's HIERARCHY (not its list of labels, which
    holds two notations per grade on the external scale), clamped at both
    ends, in the rating's own notation where the scale has it."""
    if scale is None or rating not in scale.ratings or not k:
        return rating
    h = scale.hierarchy[scale.ratings.index(rating)]
    hs = sorted({x for x in scale.hierarchy if x is not None})
    if h not in hs:
        return rating
    h2 = hs[min(max(hs.index(h) + int(k), 0), len(hs) - 1)]
    cands = [r for r, x in zip(scale.ratings, scale.hierarchy) if x == h2]
    same = [r for r in cands if _family(r) == _family(rating)]
    return (same or cands or [rating])[0]


def notch_ratings(rc: RunContext, rating, rows, k: int):
    rat = np.asarray(rating, dtype=object).copy()
    if k <= 0:
        return rat
    for i in np.where(rows)[0]:
        rat[i] = notch_grade(rc.inputs.scale_for(rc.ing["rating_type"].iloc[i]), rat[i], k)
    return rat


def layer_inputs(rc: RunContext, scn: Scenario, paths: dict) -> tuple[list, dict]:
    """The scenario's ``reprice`` inputs, layer by layer: a list of
    (layer, cumulative inputs), and what each layer did."""
    ing, H, cfg = rc.ing, rc.H, rc.cfg
    n = len(ing)
    info = {}
    cum: dict = {}
    if rc.base_curves is not None:
        cum["pd_curves"] = rc.base_curves
    steps = []

    # 1 drivers
    mev, gcc = driver_edits(rc, paths)
    if mev or gcc or scn.scenario_weights:
        ch = pd_chain(rc.inputs, rc.report, rc.run_path, mev_deltas=mev, gcc_deltas=gcc,
                      weight_mode=scn.weights if scn.weights in ("hold", "auto") else "hold",
                      scenario_weights=scn.scenario_weights)
        if not ch.get("ok"):
            raise RuntimeError(ch.get("reason", "The PD chain could not be rebuilt."))
        cum["pd_curves"] = ch["curves"]
        info["drivers"] = {"internal": {str(k): v for k, v in mev.items()},
                           "external": gcc}
    steps.append(("drivers", dict(cum)))

    # 2 credit factor
    zm = factor_moves(rc, paths)
    shift = factor_shift(rc, zm)
    if np.any(shift != 0):
        cum["pd_shift"] = shift
    info["factor"] = {s: [float(x) for x in v] for s, v in zm.items()}
    steps.append(("factor", dict(cum)))

    # direct downgrades (rates and external), needed for staging
    rn = rate_notches(rc, paths)
    en = external_notches(rc, paths)
    direct = np.zeros(n, dtype=int)
    direct[rn["rows"]] += rn["notches"]
    direct[rc.external] += en["notches"]
    info["rates"] = {"rise_points": rn["rise"], "notches": rn["notches"]}
    info["external"] = {"index": en["index"], "notches": en["notches"]}

    # 3 ratings and stages: the implied fall of the 12-month PD
    pd12s = stressed_pd12(rc, cum.get("pd_curves"), cum.get("pd_shift"))
    stage, to2, implied = implied_stage(rc, pd12s, direct)
    if to2.any():
        cum["stage"] = stage
    info["staging"] = {"to_stage_2": int(to2.sum()),
                       "implied_notches_mean": float(np.mean(implied)) if n else 0.0}
    steps.append(("staging", dict(cum)))

    # 4 collateral
    mult, moves = collateral_multiplier(rc, paths)
    if moves:
        cn = ing["collateral_net"].fillna(0.0).to_numpy().astype(float)
        cum["collateral_net"] = np.maximum(0.0, cn * mult)
    info["collateral"] = moves
    steps.append(("collateral", dict(cum)))

    # 5 rates: the notches, and the discount rate when configured
    rating = ing["rating"].astype(str).to_numpy().astype(object)
    if rn["notches"] > 0:
        rating = notch_ratings(rc, rating, rn["rows"], rn["notches"])
        cum["rating"] = rating
    rv = rn["path"]
    if (cfg.get("rates") or {}).get("discount") and rv.size and rv[0] != 0:
        es = np.zeros(n)
        es[rn["rows"]] = rv[0] / 100.0
        cum["eir_shift"] = es
    steps.append(("rates", dict(cum)))

    # 6 exposure: drawdown when credit tightens
    rows, pct = drawdown(rc, paths)
    if pct > 0:
        ob = ing["on_balance"].to_numpy(dtype=float).copy()
        ob[rows] *= 1.0 + pct / 100.0
        cum["on_balance"] = ob
    info["exposure"] = {"drawdown_pct": pct}
    steps.append(("exposure", dict(cum)))

    # 7 external book notches
    if en["notches"] > 0:
        cum["rating"] = notch_ratings(rc, cum.get("rating", rating), rc.external, en["notches"])
    steps.append(("external", dict(cum)))

    # 8 parameter levers on top
    if scn.levers:
        spec = StressSpec.from_dict(scn.levers)
        scoped = stress_scope(ing, spec)
        lev = spec_levers(rc.inputs, ing, spec, scoped, start=dict(cum))
        if not lev.get("ok"):
            raise RuntimeError(lev.get("reason", "The levers could not be applied."))
        for k in REPRICE_KEYS:
            if k in lev and lev[k] is not None:
                cum[k] = lev[k]
        info["levers"] = {"notes": lev.get("notes", []), "defaulted": lev.get("defaulted", [])}
    steps.append(("levers", dict(cum)))
    return steps, info


def _price(rc, kw) -> pd.DataFrame:
    return reprice(rc.inputs, rc.report, **{k: v for k, v in kw.items() if k in REPRICE_KEYS})


def transmit(rc: RunContext, scn: Scenario, waterfall: bool = True,
             paths: dict | None = None) -> dict:
    """Price a scenario. Returns the totals, the waterfall by layer, the
    paths, the severity, and the split by book, segment, stage, portfolio
    and customer."""
    if isinstance(scn, dict):
        scn = Scenario.from_dict(scn)
    sp = scenario_paths(scn, rc.model, rc.H) if paths is None else \
        {"paths": paths, "given": paths, "severity": rc.model.severity(paths, rc.H)}
    try:
        steps, info = layer_inputs(rc, scn, sp["paths"])
    except RuntimeError as e:
        return {"ok": False, "reason": str(e)}
    before = rc.base
    after = _price(rc, steps[-1][1])
    s = summarise_repricing(before, after)
    wf = []
    if waterfall:
        prev = s["before"]
        last_kw = None
        for key, kw in steps:
            if last_kw is not None and _same_kw(kw, last_kw):
                tot = prev
            else:
                r = _price(rc, kw)
                ok = before["ecl"].notna() & r["ecl"].notna()
                tot = float(r.loc[ok, "ecl"].sum())
            label = dict(LAYERS)[key]
            wf.append({"layer": label, "key": key, "change": tot - prev, "ecl": tot})
            prev, last_kw = tot, kw
        # the last step is the full scenario: make the bars add up exactly
        if wf:
            wf[-1]["change"] += s["after"] - wf[-1]["ecl"]
            wf[-1]["ecl"] = s["after"]
    det = s["detail"].copy()
    seg = pd.Series(rc.segment, index=rc.ing.index)
    book = pd.Series(np.where(rc.external, "Externally rated", "Internally rated"),
                     index=rc.ing.index)
    det["segment"] = seg.reindex(det.index).to_numpy()
    det["book"] = book.reindex(det.index).to_numpy()

    def roll(col):
        g = det.groupby(col).agg(contracts=("contract", "size"), exposure=("exposure", "sum"),
                                 before=("ecl_before", "sum"), after=("ecl_after", "sum"))
        g["change"] = g["after"] - g["before"]
        g["change_pct"] = 100 * g["change"] / g["before"].where(g["before"] != 0)
        return g.reset_index().sort_values("change", key=abs, ascending=False)

    by_stage = (det.groupby("stage_after").agg(contracts=("contract", "size"),
                                               exposure=("exposure", "sum"),
                                               ecl=("ecl_after", "sum"))
                .reset_index().rename(columns={"stage_after": "stage"}))
    yrs = list(range(rc.first_year, rc.first_year + rc.H))
    prow = []
    for c, v in sp["paths"].items():
        v = np.asarray(v, float)
        if not np.any(np.abs(v) > 1e-9):
            continue
        for j, y in enumerate(yrs):
            prow.append({"concept": c, "year": y, "move": float(v[j]) if j < v.size else 0.0,
                         "given": c in sp["given"]})
    return {
        "ok": True, "name": scn.name,
        "before": s["before"], "after": s["after"], "delta": s["delta"],
        "pct": 100 * s["delta"] / s["before"] if s["before"] else float("nan"),
        "severity": sp["severity"], "years": yrs,
        "paths": pd.DataFrame(prow, columns=["concept", "year", "move", "given"]),
        "waterfall": pd.DataFrame(wf, columns=["layer", "key", "change", "ecl"]),
        "by_book": roll("book"), "by_segment": roll("segment"),
        "by_portfolio": roll("portfolio"), "by_stage": by_stage,
        "migration": s["migration"], "by_customer": s["by_customer"],
        "moved": s["moved"], "customers_moved": s["customers_moved"],
        "info": info, "notes": list(rc.notes), "detail": det,
    }


def _same_kw(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        if a[k] is b[k]:
            continue
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.shape != y.shape or not np.array_equal(x, y):
                return False
        elif x != y:
            return False
    return True
